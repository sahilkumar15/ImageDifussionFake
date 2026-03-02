#!/usr/bin/env python
# code/DiffusionFake/eval_all.py

import os
import csv
import time
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score

from utils.init import setup
from utils.parameters import get_parameters
from cldm.model import create_model, load_state_dict
from eval_dataloaders import build_test_dataloader


# -------------------------
# metrics
# -------------------------
def compute_auc_eer(labels_np, probs_np):
    labels_np = labels_np.astype(np.float32)
    probs_np = probs_np.astype(np.float32)

    # Need both classes present
    if len(np.unique(labels_np)) < 2:
        return 0.5, 0.5

    fpr, tpr, _ = roc_curve(labels_np, probs_np, pos_label=1)
    fnr = 1.0 - tpr
    eer = float(fpr[np.nanargmin(np.abs(fpr - fnr))])
    auc = float(roc_auc_score(labels_np, probs_np))
    return auc, eer


@torch.no_grad()
def run_one_dataset(model, dataloader, threshold=0.5, device="cuda"):
    """
    Fast eval path: use the EfficientNet classifier head directly on hint images.
    This matches what you were doing in your snippet.
    """
    model.eval()
    probs_all = []
    labels_all = []

    correct = 0
    total = 0

    for batch in dataloader:
        # batch must contain: hint, label (your KeyAdapter guarantees these during training;
        # for evaluation datasets, your dataset should return same keys.)
        hint = batch["hint"].to(device, non_blocking=True).float()
        labels = batch["label"]
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device, non_blocking=True).float()
        else:
            labels = torch.tensor(labels, device=device, dtype=torch.float32)

        # classifier forward: EfficientNet -> pool -> fc -> sigmoid
        feature = model.control_model.input_hint_block.forward_features(hint)
        pooled = model.control_model.global_pool(feature).flatten(1)
        logits = model.control_model.fc(pooled).view(-1)
        probs = torch.sigmoid(logits)

        preds = (probs >= threshold).float()

        correct += (preds == labels.view(-1)).sum().item()
        total += labels.numel()

        probs_all.append(probs.detach().cpu())
        labels_all.append(labels.detach().view(-1).cpu())

    probs_np = torch.cat(probs_all, dim=0).numpy()
    labels_np = torch.cat(labels_all, dim=0).numpy()

    auc, eer = compute_auc_eer(labels_np, probs_np)
    acc = float(correct) / float(max(total, 1))
    return acc, auc, eer, int(total)


def pick_default_ckpt(args):
    """
    If eval.ckpt_path is null, try experiments/<experiment.name>/ckpt/last.ckpt
    """
    exp_root = getattr(args.experiment, "root_dir", "experiments")
    exp_name = getattr(args.experiment, "name", "exp")
    candidate = os.path.join(exp_root, exp_name, "ckpt", "last.ckpt")
    return candidate if os.path.isfile(candidate) else None


def main():
    args = get_parameters()
    setup(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- eval config from YAML --------
    if not hasattr(args, "eval"):
        raise ValueError("Missing eval: block in YAML. Add eval: {...} to configs/train.yaml")

    eval_datasets = list(getattr(args.eval, "datasets", []))
    if len(eval_datasets) == 0:
        raise ValueError("eval.datasets is empty. Put datasets list in YAML.")

    threshold = float(getattr(args.eval, "threshold", getattr(args.test, "threshold", 0.5)))
    bs = int(getattr(args.eval, "batch_size", 64))
    nw = int(getattr(args.eval, "num_workers", 4))

    ckpt_path = getattr(args.eval, "ckpt_path", None)
    if ckpt_path in (None, "", "null"):
        ckpt_path = pick_default_ckpt(args)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found. Set eval.ckpt_path. Got: {ckpt_path}")

    model_cfg = getattr(args.eval, "model_config", "configs/diffusionfake.yaml")

    out_csv = getattr(args.eval, "out_csv", None)
    if out_csv in (None, "", "null"):
        exp_root = getattr(args.experiment, "root_dir", "experiments")
        exp_name = getattr(args.experiment, "name", "exp")
        out_dir = os.path.join(exp_root, exp_name, "eval")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, "cross_dataset_eval.csv")
    else:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # -------- load model --------
    print(f"[Eval] Loading model cfg: {model_cfg}")
    model = create_model(model_cfg)

    # IMPORTANT: must call define_feature_filter (your training does this)
    model.control_model.define_feature_filter(encoder="tf_efficientnet_b4_ns")

    print(f"[Eval] Loading checkpoint: {ckpt_path}")
    model.load_state_dict(load_state_dict(ckpt_path, location=device))
    model = model.to(device)

    # -------- run datasets --------
    rows = []
    for ds_name in eval_datasets:
        print(f"\n[Eval] Dataset = {ds_name}")
        # override test loader params if your factory uses args.val/test fields:
        # easiest: set them on args before create_dataset is called.
        args.test.batch_size = bs
        args.test.num_workers = nw
        args.test.shuffle = False
        args.test.drop_last = False

        dl = build_test_dataloader(args, ds_name)

        t0 = time.time()
        acc, auc, eer, n = run_one_dataset(model, dl, threshold=threshold, device=device)
        dt = time.time() - t0

        print(f"[Eval:{ds_name}] n={n} acc={acc:.4f} auc={auc:.4f} eer={eer:.4f} time={dt/60:.2f} min")

        rows.append({
                    "model_ckpt": ckpt_path,
                    "model_config": model_cfg,
                    "dataset": ds_name,
                    "n": int(n),
                    "threshold": round(float(threshold), 3),
                    "acc": round(float(acc), 3),
                    "auc": round(float(auc), 3) if auc == auc else float("nan"),   # keeps NaN if empty dataset
                    "eer": round(float(eer), 3) if eer == eer else float("nan"),
                    "time_sec": round(float(dt), 3),
                })

    # -------- save CSV --------
    fieldnames = ["model_ckpt", "model_config", "dataset", "n", "threshold", "acc", "auc", "eer", "time_sec"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[Done] Saved: {out_csv}")


if __name__ == "__main__":
    main()
    
    
# CUDA_VISIBLE_DEVICES=1 python eval_all.py -c configs/train.yaml