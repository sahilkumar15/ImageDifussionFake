# code/DiffusionFake/train.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DiffusionFake training entrypoint (PyTorch Lightning)

Fixes included:
1) Robust dataset key normalization -> always outputs:
   source/target/hint/txt/label
2) Robust image shape normalization:
   per-sample: CHW, batch: BCHW
3) YAML-driven hyperparams:
   train.epochs, train.lr, train.logger_freq
4) DDP-safe WandB init + code snapshot only on rank 0
5) Checkpoint monitor key matches logged metric ("t/loss_epoch")
6) Safer DDP strategy selection + optional resuming
"""

import os
import shutil
import random
from typing import Any, Dict

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import wandb
from pytorch_lightning.loggers import WandbLogger

from datasets import create_dataset
from utils.logger import Logger
from utils.init import setup
from utils.parameters import get_parameters
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from utils.checkpoint import CheckpointCfg, SaveAtSchedule   # ✅ NEW


# -----------------------
# helpers
# -----------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_chw_tensor(x: Any) -> Any:
    """
    Force a single image into CHW float tensor in [-1, 1] where possible.
    Supports np.ndarray or torch.Tensor.
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        return x

    # Not an image-like tensor
    if t.ndim != 3:
        return t.float()

    # Move channel dim (size 3) to dim0 if needed: HWC -> CHW
    if t.shape[0] != 3 and 3 in t.shape:
        ch = list(t.shape).index(3)
        perm = [ch] + [i for i in range(3) if i != ch]
        t = t.permute(*perm).contiguous()

    t = t.float()

    # Normalize range
    # - uint8-like [0..255] -> [-1..1]
    # - float [0..1] -> [-1..1]
    mx = float(t.max()) if t.numel() > 0 else 0.0
    mn = float(t.min()) if t.numel() > 0 else 0.0
    if mx > 2.0:  # likely 0..255
        t = (t / 127.5) - 1.0
    elif mn >= 0.0 and mx <= 1.0:
        t = (t * 2.0) - 1.0

    return t


class KeyAdapter(Dataset):
    """
    Normalize dataset samples into the dict keys the LightningModule expects:
    source/target/hint/txt/label with CHW float tensors in [-1,1].
    """

    def __init__(self, ds: Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> Dict[str, Any]:
        s = self.ds[idx]
        # Some datasets return (dict, meta) etc.
        if isinstance(s, (list, tuple)):
            s = s[0]
        if s is None or not isinstance(s, dict):
            raise TypeError(f"Dataset must return dict, got {type(s)}")

        # Build source if missing
        if "source" not in s:
            if "hint_ori" in s:
                s["source"] = s["hint_ori"]
            elif "image" in s:
                s["source"] = s["image"]
            elif "jpg" in s:
                s["source"] = s["jpg"]
            else:
                raise KeyError(f"Cannot build source; keys={list(s.keys())}")

        # Fill missing keys
        if "target" not in s:
            s["target"] = s["source"]
        if "hint" not in s:
            s["hint"] = s["source"]
        if "txt" not in s:
            s["txt"] = ""
        if "label" not in s:
            s["label"] = 0.0

        # Normalize to CHW tensors
        s["source"] = to_chw_tensor(s["source"])
        s["target"] = to_chw_tensor(s["target"])
        s["hint"] = to_chw_tensor(s["hint"])

        for k in ("source", "target", "hint"):
            if not (isinstance(s[k], torch.Tensor) and s[k].ndim == 3 and s[k].shape[0] == 3):
                raise ValueError(f"{k} bad shape: {type(s[k])} {getattr(s[k], 'shape', None)}")

        # Numeric label
        if isinstance(s["label"], torch.Tensor):
            s["label"] = float(s["label"].item())
        else:
            s["label"] = float(s["label"])

        return s


def force_batch_bchw(batch_dict: Dict[str, Any], keys=("source", "target", "hint", "hint_ori")) -> Dict[str, Any]:
    """
    Final safety: ensure 4D tensors become BCHW.
    Lightning/model code typically expects BCHW.
    """
    for k in keys:
        if k not in batch_dict:
            continue
        x = batch_dict[k]
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            continue

        # expected: BCHW
        if x.shape[1] == 3:
            continue

        # BHWC -> BCHW
        if x.shape[-1] == 3:
            batch_dict[k] = x.permute(0, 3, 1, 2).contiguous()
            continue

        # BHCW -> BCHW (rare)
        if x.shape[2] == 3:
            batch_dict[k] = x.permute(0, 2, 1, 3).contiguous()
            continue

        raise ValueError(f"{k}: unexpected batch shape {x.shape}")
    return batch_dict


def safe_collate(batch):
    out = default_collate(batch)
    return force_batch_bchw(out)


def _get_lr_loggerfreq_from_yaml(args):
    """
    Prefer YAML if present:
      train:
        lr: ...
        logger_freq: ...
    Fallbacks:
      - args.train.learning_rate
      - defaults
    """
    lr = None
    if hasattr(args, "train"):
        if hasattr(args.train, "lr"):
            lr = args.train.lr
        elif hasattr(args.train, "learning_rate"):
            lr = args.train.learning_rate
    if lr is None:
        lr = 1e-5

    logger_freq = 300
    if hasattr(args, "train") and hasattr(args.train, "logger_freq"):
        logger_freq = int(args.train.logger_freq)

    return float(lr), int(logger_freq)


def _is_rank0(args) -> bool:
    # torchrun sets LOCAL_RANK and RANK
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"]) == 0
    return int(getattr(args, "local_rank", 0)) == 0


def main():
    # -----------------------
    # config / setup
    # -----------------------
    args = get_parameters()
    setup(args)

    seed = int(getattr(args, "seed", 3407))
    seed_everything(seed)

    # epochs must come from YAML
    if not hasattr(args, "train") or not hasattr(args.train, "epochs"):
        raise ValueError("Missing train.epochs in YAML config.")
    max_epochs = int(args.train.epochs)

    learning_rate, logger_freq = _get_lr_loggerfreq_from_yaml(args)

    # resume checkpoint (allow override)
    resume_path = (
        getattr(args, "resume_path", None)
        or getattr(args, "ckpt_path", None)
        or "./models/control_sd15_ini.ckpt"
    )

    # performance hint
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # -----------------------
    # wandb + local logger (rank0 only)
    # -----------------------
    logger = None
    wandb_logger = None

    # ✅ experiment folder from YAML (experiments/<name>)
    ckcfg = CheckpointCfg.from_args(args)

    # set exam_dir on ALL ranks deterministically (do not rely on rank0 side effects)
    args.exam_dir = os.path.join(ckcfg.exp_root, ckcfg.exp_name)
    os.makedirs(args.exam_dir, exist_ok=True)

    if _is_rank0(args):
        if getattr(args.wandb, "name", None) is None:
            # Prefer experiment.name if provided, else fall back to config filename
            exp_name = None
            if hasattr(args, "experiment") and hasattr(args.experiment, "name"):
                exp_name = args.experiment.name
            args.wandb.name = exp_name or os.path.basename(args.config).replace(".yaml", "")

        # ✅ Use PL WandbLogger (single source of truth)
        try:
            wandb_logger = WandbLogger(
                project=getattr(args.wandb, "project", None),
                name=getattr(args.wandb, "name", None),
                group=getattr(args.wandb, "group", None),
                job_type=getattr(args.wandb, "job_type", None),
                save_dir=args.exam_dir,
                log_model=False,  # keep False since you save ckpts yourself
            )
            # push full config to wandb
            wandb_logger.experiment.config.update(
                {"config_path": args.config, "seed": seed, "max_epochs": max_epochs, "lr": learning_rate},
                allow_val_change=True,
            )
            wandb_logger.experiment.save(args.config)
        except Exception:
            wandb_logger = None

        # local file logger
        logger = Logger(name="train", log_path=f"{args.exam_dir}/train.log")
        logger.info(args)
        logger.info(f"exam_dir={args.exam_dir}")
        logger.info(f"max_epochs={max_epochs}, lr={learning_rate}, logger_freq={logger_freq}, resume={resume_path}")

        # snapshot code (rank0 only)
        shutil.copytree("configs", f"{args.exam_dir}/configs", dirs_exist_ok=True)
        shutil.copy2(__file__, os.path.join(args.exam_dir, "train.py"))

    # -----------------------
    # dataloaders from factory -> wrapped
    # -----------------------
    base_train_dl = create_dataset(args, split="train")
    base_val_dl = create_dataset(args, split="val")

    train_ds = KeyAdapter(base_train_dl.dataset)
    val_ds = KeyAdapter(base_val_dl.dataset)

    # --- enforce YAML batch sizes (avoid silent mismatch if create_dataset ignores YAML) ---
    train_bs = int(args.train.batch_size) if hasattr(args, "train") and hasattr(args.train, "batch_size") else base_train_dl.batch_size
    val_bs   = int(args.val.batch_size)   if hasattr(args, "val") and hasattr(args.val, "batch_size")     else base_val_dl.batch_size

    train_workers = int(args.train.num_workers) if hasattr(args, "train") and hasattr(args.train, "num_workers") else base_train_dl.num_workers
    val_workers   = int(args.val.num_workers)   if hasattr(args, "val")   and hasattr(args.val, "num_workers")   else base_val_dl.num_workers

    train_dataloader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate,
        persistent_workers=False if train_workers == 0 else True,
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=safe_collate,
        persistent_workers=False if val_workers == 0 else True,
    )

    # --- rank0 debug: confirm scheduler config is visible to the model ---
    if _is_rank0(args):
        b = next(iter(train_dataloader))
        print("YAML scheduler cfg:", getattr(args.train, "scheduler", None))
        
        print("FINAL TRAIN BATCH KEYS:", b.keys())
        print("source:", b["source"].shape, "target:", b["target"].shape, "hint:", b["hint"].shape)
        assert b["source"].ndim == 4 and b["source"].shape[1] == 3, f"source wrong: {b['source'].shape}"

    # -----------------------
    # callbacks / ckpt dir (must exist before auto-resume check)
    # -----------------------
    img_logger = ImageLogger(batch_frequency=logger_freq, save_dir=args.exam_dir)
    ckcfg = CheckpointCfg.from_args(args)

    model_save_dir = os.path.join(args.exam_dir, "ckpt")
    os.makedirs(model_save_dir, exist_ok=True)

    # -----------------------
    # auto-resume from last.ckpt if present
    # -----------------------
    resume_ckpt = None
    if hasattr(args, "train") and hasattr(args.train, "resume_ckpt"):
        v = args.train.resume_ckpt

        if isinstance(v, str) and v.lower() == "auto":
            candidate = os.path.join(model_save_dir, "last.ckpt")
            resume_ckpt = candidate if os.path.isfile(candidate) else None
        elif v not in (None, "", "null"):
            resume_ckpt = v  # explicit path

    if _is_rank0(args):
        print("RESUME CKPT:", resume_ckpt)

    # -----------------------
    # model (load init weights only if NOT resuming)
    # -----------------------
    model = create_model("configs/diffusionfake.yaml").cpu()
    model.args = args

    # IMPORTANT: build the full control model first, including DIMF head
    model.control_model.define_feature_filter()

    if resume_ckpt is None:
        state_dict = load_state_dict(resume_path, location="cpu")

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # remove incompatible key
        state_dict.pop("cond_stage_model.transformer.text_model.embeddings.position_ids", None)

        model_state = model.state_dict()
        filtered_state = {}
        skipped_keys = []

        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                skipped_keys.append(k)

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)

        if _is_rank0(args):
            print(f"[INFO] Loaded {len(filtered_state)} matching keys from checkpoint")
            print(f"[INFO] Skipped {len(skipped_keys)} keys")
            for k in skipped_keys[:20]:
                print("   SKIPPED:", k)

            print(f"[INFO] Missing keys after load: {len(missing)}")
            for k in missing[:20]:
                print("   MISSING:", k)

            print(f"[INFO] Unexpected keys after load: {len(unexpected)}")
            for k in unexpected[:20]:
                print("   UNEXPECTED:", k)

    # Lightning module uses these attributes
    model.learning_rate = learning_rate
    model.sd_locked = True
    model.only_mid_control = False

    # ✅ best/top-k + last (from YAML)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_save_dir,
        filename=ckcfg.best_filename,
        save_top_k=ckcfg.save_top_k,
        monitor=ckcfg.monitor,
        mode=ckcfg.mode,
        save_last=ckcfg.save_last,
        auto_insert_metric_name=False,
        save_weights_only=False,  # ✅ FULL state: optimizer/scheduler/epoch/global_step
    )

    # ✅ also save schedule ckpts (every N epochs, plus explicit epochs)
    schedule_callback = SaveAtSchedule(
        dirpath=model_save_dir,
        save_every_n_epochs=ckcfg.save_every_n_epochs,
        save_epochs=ckcfg.save_epochs,
        filename_tmpl=ckcfg.every_filename,
        save_weights_only=False,   # ✅ FULL state
    )

    tqdm_bar = pl.callbacks.TQDMProgressBar(refresh_rate=10)

    # -----------------------
    # trainer (PL 1.9.5 safe, DDP-safe)
    # -----------------------
    num_visible = torch.cuda.device_count()
    want_devices = 4 if num_visible >= 4 else (1 if num_visible >= 1 else 0)

    # PL 1.9.5: best speed = bf16 if available, else 16-mixed
    if want_devices > 0 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        precision = "bf16"
    else:
        precision = "16-mixed" if want_devices > 0 else 32

    # ✅ DDP strategy (faster buckets + still safe for unused params if your graph has them)
    strategy = None
    if want_devices > 1:
        try:
            from pytorch_lightning.strategies import DDPStrategy
            strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
        except Exception:
            strategy = "ddp_find_unused_parameters_true"

    callbacks = [tqdm_bar, checkpoint_callback, schedule_callback]
    if _is_rank0(args):
        callbacks.insert(1, img_logger)
    
    accum = 1
    if hasattr(args, "train") and hasattr(args.train, "accumulate_grad_batches"):
        accum = int(args.train.accumulate_grad_batches)
    

    trainer = pl.Trainer(
        accelerator="gpu" if want_devices > 0 else "cpu",
        devices=want_devices if want_devices > 0 else None,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,                # ✅ None disables; rank0-only logger is fine
        enable_progress_bar=True,
        log_every_n_steps=50,
        enable_checkpointing=True,

        # ✅ performance knobs
        benchmark=True,                     # speed (disable if you need exact determinism)
        deterministic=False,
        num_sanity_val_steps=0,             # saves time
        gradient_clip_val=1.0,              # prevents occasional spikes
        accumulate_grad_batches=accum,          # set >1 if you want larger effective batch
        enable_model_summary=False,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_ckpt,   # ✅ TRUE resume from last.ckpt if exists
    )




if __name__ == "__main__":
    main()

# 1 GPU:
#   CUDA_VISIBLE_DEVICES=1 python train.py -c configs/train.yaml
#
# 4 GPUs (recommended):
#   CUDA_VISIBLE_DEVICES=1,5,6,7 torchrun --nproc_per_node=4 train.py -c configs/train.yaml