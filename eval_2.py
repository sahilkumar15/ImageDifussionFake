# code/ImageDifussionFake/eval_2.py
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm

from utils.init import setup
from utils.parameters import get_parameters
from datasets import create_dataset
from cldm.model import create_model


@torch.no_grad()
def compute_metrics(probs_np, labs_np):
    probs_np = probs_np.astype(np.float64)
    labs_np = (labs_np > 0.5).astype(np.int32)

    mask = np.isfinite(probs_np) & np.isfinite(labs_np)
    probs_np = probs_np[mask]
    labs_np = labs_np[mask]

    eer = 0.5
    auc = 0.5
    best_acc = 0.0
    acc_at_eer = 0.0
    eer_thr = 0.5

    if probs_np.size > 0 and len(np.unique(labs_np)) >= 2:
        fpr, tpr, thr = roc_curve(labs_np, probs_np, pos_label=1)
        fnr = 1.0 - tpr

        idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float(fpr[idx])
        eer_thr = float(thr[idx])

        auc = float(roc_auc_score(labs_np, probs_np))

        accs = []
        for th_ in thr:
            pred = (probs_np >= th_).astype(np.int32)
            accs.append((pred == labs_np).mean())
        best_acc = float(np.max(accs)) if len(accs) else 0.0

        pred_eer = (probs_np >= eer_thr).astype(np.int32)
        acc_at_eer = float((pred_eer == labs_np).mean())

    return {
        "N": int(probs_np.size),
        "EER": eer,
        "AUC": auc,
        "BEST_ACC": best_acc,
        "ACC_AT_EER": acc_at_eer,
        "EER_THR": eer_thr,
    }


def _move_batch_to_device(batch, device):
    # your dataset returns these keys (celeb_df + ffpp loaders)
    for k in ["source", "target", "hint", "hint_ori"]:
        if k in batch and isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device, non_blocking=True)

    if "label" in batch and isinstance(batch["label"], torch.Tensor):
        batch["label"] = batch["label"].to(device, non_blocking=True)

    return batch


@torch.no_grad()
def _predict_prob_from_model(model, batch):
    """
    IMPORTANT:
    DiffusionFake forward signature is: forward(source, target, c, label)
    So we MUST go through model.get_input(batch, model.first_stage_key),
    then call model(source, target, c, labels) and parse loss_dict.
    """
    if not (hasattr(model, "get_input") and callable(getattr(model, "get_input"))):
        raise RuntimeError("Model has no get_input(); cannot evaluate.")

    source, target, c, labels = model.get_input(batch, model.first_stage_key)

    out = model(source, target, c, labels)

    # expected: (loss, loss_dict)
    if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict)):
        raise RuntimeError(f"Unexpected model output type: {type(out)} (expected (loss, dict))")

    d = out[1]

    # try probs keys first
    for k in ["v/probs", "probs", "t/probs", "pred", "preds", "v/preds", "t/preds"]:
        if k in d and torch.is_tensor(d[k]):
            x = d[k].detach().float().view(-1)
            # if it's logits, sigmoid it; if already [0,1], keep
            if x.min() < 0.0 or x.max() > 1.0:
                x = torch.sigmoid(x)
            return x.cpu()

    # then logits keys
    for k in ["v/logits", "logits", "t/logits", "v/pred_logits", "pred_logits"]:
        if k in d and torch.is_tensor(d[k]):
            return torch.sigmoid(d[k].detach().float().view(-1)).cpu()

    # if you hit this, print keys once and we’ll map the right one
    raise RuntimeError(
        "Could not find probs/logits in loss_dict.\n"
        f"Available keys (first 100): {sorted(list(d.keys()))[:100]}"
    )


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    probs_all, labs_all, paths_all = [], [], []

    for batch in tqdm(loader, desc="EVAL"):
        batch = _move_batch_to_device(batch, device)

        probs = _predict_prob_from_model(model, batch)  # CPU tensor (B,)
        probs_all.append(probs.numpy())

        # labels
        if "label" not in batch:
            raise KeyError("Batch missing 'label'")
        lab = batch["label"]
        if torch.is_tensor(lab):
            labs_all.append(lab.detach().float().view(-1).cpu().numpy())
        else:
            labs_all.append(np.array(lab, dtype=np.float32).reshape(-1))

        # paths
        if "ori_path" in batch:
            if isinstance(batch["ori_path"], (list, tuple)):
                paths_all.extend(list(batch["ori_path"]))
            else:
                paths_all.append(str(batch["ori_path"]))

    probs_np = np.concatenate(probs_all, axis=0) if len(probs_all) else np.array([], dtype=np.float64)
    labs_np = np.concatenate(labs_all, axis=0) if len(labs_all) else np.array([], dtype=np.float64)
    return probs_np, labs_np, paths_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="configs/train.yaml")
    ap.add_argument("--ckpt", required=True, help="path to trained checkpoint (.ckpt)")
    ap.add_argument("--dataset", default="celeb_df", help="dataset key in YAML: celeb_df / ffpp_rela / ...")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_csv", default=None)
    args_cli = ap.parse_args()

    # get_parameters() parses sys.argv internally, isolate it
    _orig_argv = sys.argv[:]
    sys.argv = [sys.argv[0], "-c", args_cli.config]
    args = get_parameters()
    sys.argv = _orig_argv

    args.config = args_cli.config
    setup(args)

    # choose dataset from YAML
    args.dataset.name = args_cli.dataset

    # set loader configs
    if args_cli.split == "test":
        args.test.batch_size = int(args_cli.batch_size)
        args.test.num_workers = int(args_cli.num_workers)
    elif args_cli.split == "val":
        args.val.batch_size = int(args_cli.batch_size)
        args.val.num_workers = int(args_cli.num_workers)
    else:
        args.train.batch_size = int(args_cli.batch_size)
        args.train.num_workers = int(args_cli.num_workers)

    # create model
    model_config = "configs/diffusionfake.yaml"
    if hasattr(args, "eval") and isinstance(args.eval, dict) and "model_config" in args.eval:
        model_config = args.eval["model_config"]
    model = create_model(model_config).cpu()
    model.args = args

    # load checkpoint
    ckpt = torch.load(args_cli.ckpt, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[LOAD] ckpt={args_cli.ckpt} missing={len(missing)} unexpected={len(unexpected)}")

    # optional repo hook
    if hasattr(model, "control_model") and hasattr(model.control_model, "define_feature_filter"):
        model.control_model.define_feature_filter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    base_dl = create_dataset(args, split=args_cli.split)
    loader = DataLoader(
        base_dl.dataset,
        batch_size=base_dl.batch_size,
        shuffle=False,
        num_workers=base_dl.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=getattr(base_dl, "collate_fn", None),
    )

    probs_np, labs_np, paths_all = run_eval(model, loader, device)
    metrics = compute_metrics(probs_np, labs_np)

    print(f"\n=== EVAL dataset={args_cli.dataset} split={args_cli.split} ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args_cli.out_csv is not None:
        import pandas as pd
        df = pd.DataFrame(
            {
                "path": paths_all[: len(probs_np)],
                "label": labs_np[: len(probs_np)],
                "prob": probs_np[: len(probs_np)],
            }
        )
        out_dir = os.path.dirname(args_cli.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args_cli.out_csv, index=False)
        print(f"[OK] wrote: {args_cli.out_csv}")


if __name__ == "__main__":
    main()