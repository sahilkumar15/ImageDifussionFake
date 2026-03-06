# code/ImageDifussionFake/eval_generalization.py


import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm

from utils.init import setup
from utils.parameters import get_parameters
from datasets import create_dataset
from cldm.model import create_model


# =========================
# Helpers: rounding to 3 dp
# =========================
def _round_floats(obj, ndigits=3):
    """Recursively round floats in dict/list/tuple to ndigits."""
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_round_floats(v, ndigits) for v in obj]
        return tuple(out) if isinstance(obj, tuple) else out
    if isinstance(obj, float):
        return round(obj, ndigits)
    return obj


def _round_print_metrics(m: dict, ndigits=3):
    """Return a copy rounded for printing / JSON."""
    return _round_floats(m, ndigits)


@torch.no_grad()
def compute_metrics(probs_np, labs_np):
    """
    Returns raw (full precision) metrics; we round only for JSON/printing.
    """
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
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
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
    for k in ["source", "target", "hint", "hint_ori"]:
        if k in batch and isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device, non_blocking=True)
    if "label" in batch and isinstance(batch["label"], torch.Tensor):
        batch["label"] = batch["label"].to(device, non_blocking=True)
    return batch


@torch.no_grad()
def _predict_prob_from_model(model, batch):
    if not (hasattr(model, "get_input") and callable(getattr(model, "get_input"))):
        raise RuntimeError("Model has no get_input(); cannot evaluate.")

    source, target, c, labels = model.get_input(batch, model.first_stage_key)
    out = model(source, target, c, labels)

    if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict)):
        raise RuntimeError("Unexpected model output; expected (loss, loss_dict).")

    d = out[1]

    for k in ["v/probs", "probs", "t/probs", "pred", "preds", "v/preds", "t/preds"]:
        if k in d and torch.is_tensor(d[k]):
            x = d[k].detach().float().view(-1)
            if x.min() < 0.0 or x.max() > 1.0:
                x = torch.sigmoid(x)
            return x.cpu()

    for k in ["v/logits", "logits", "t/logits", "v/pred_logits", "pred_logits"]:
        if k in d and torch.is_tensor(d[k]):
            return torch.sigmoid(d[k].detach().float().view(-1)).cpu()

    raise RuntimeError(
        f"Could not find probs/logits in loss_dict. Keys: {sorted(list(d.keys()))[:80]}"
    )


@torch.no_grad()
def run_eval_one_dataset(model, loader, device):
    model.eval()
    probs_all, labs_all, paths_all = [], [], []

    for batch in tqdm(loader, desc="EVAL", leave=False):
        batch = _move_batch_to_device(batch, device)

        probs = _predict_prob_from_model(model, batch)  # CPU tensor (B,)
        probs_all.append(probs.numpy())

        lab = batch["label"]
        labs_all.append(lab.detach().float().view(-1).cpu().numpy())

        if "ori_path" in batch:
            if isinstance(batch["ori_path"], (list, tuple)):
                paths_all.extend(list(batch["ori_path"]))
            else:
                paths_all.append(str(batch["ori_path"]))

    probs_np = np.concatenate(probs_all, axis=0) if probs_all else np.array([], dtype=np.float64)
    labs_np = np.concatenate(labs_all, axis=0) if labs_all else np.array([], dtype=np.float64)
    return probs_np, labs_np, paths_all


def _ensure_split_cfg(args, split: str, bs: int, nw: int):
    """
    Some configs load into Namespace without .test/.val/.train.
    Create them if missing so create_dataset() works.
    """
    if not hasattr(args, split) or getattr(args, split) is None:
        from argparse import Namespace
        setattr(args, split, Namespace())

    split_cfg = getattr(args, split)
    split_cfg.batch_size = int(bs)
    split_cfg.num_workers = int(nw)

    if not hasattr(split_cfg, "shuffle"):
        split_cfg.shuffle = False
    if not hasattr(split_cfg, "drop_last"):
        split_cfg.drop_last = False
    if not hasattr(split_cfg, "max_items"):
        split_cfg.max_items = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="configs/eval_all_datasets.yaml")
    ap.add_argument("--ckpt", required=True, help="path to trained checkpoint (.ckpt)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_csv", required=True, help="combined csv output (all datasets)")
    ap.add_argument("--out_metrics", required=True, help="metrics json output")
    ap.add_argument("--per_dataset_dir", default=None, help="optional dir to save per-dataset csvs")
    ap.add_argument("--round_ndigits", type=int, default=3, help="round metrics to N decimals for JSON/printing")
    ap.add_argument("--round_csv", action="store_true", help="if set, round prob/label in CSV to N decimals")
    args_cli = ap.parse_args()

    # Parse YAML via your project config loader (get_parameters reads sys.argv)
    _orig_argv = sys.argv[:]
    sys.argv = [sys.argv[0], "-c", args_cli.config]
    args = get_parameters()
    sys.argv = _orig_argv

    args.config = args_cli.config
    setup(args)

    # model config
    model_config = "configs/diffusionfake.yaml"
    if hasattr(args, "eval"):
        if isinstance(args.eval, dict) and "model_config" in args.eval:
            model_config = args.eval["model_config"]
        elif hasattr(args.eval, "model_config"):
            model_config = args.eval.model_config

    model = create_model(model_config).cpu()
    model.args = args

    # dynamic layers before loading
    if hasattr(model, "control_model") and hasattr(model.control_model, "define_feature_filter"):
        model.control_model.define_feature_filter()

    # load checkpoint
    ckpt = torch.load(args_cli.ckpt, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # eval.datasets + bs/nw from YAML
    eval_datasets = []
    if hasattr(args, "eval"):
        if isinstance(args.eval, dict):
            eval_datasets = args.eval.get("datasets", [])
            bs = int(args.eval.get("batch_size", 64))
            nw = int(args.eval.get("num_workers", 4))
        else:
            eval_datasets = getattr(args.eval, "datasets", [])
            bs = int(getattr(args.eval, "batch_size", 64))
            nw = int(getattr(args.eval, "num_workers", 4))
    else:
        bs, nw = 64, 4

    if not eval_datasets:
        raise RuntimeError("No eval.datasets found in YAML.")

    # ensure split configs exist
    _ensure_split_cfg(args, "train", bs, nw)
    _ensure_split_cfg(args, "val", bs, nw)
    _ensure_split_cfg(args, "test", bs, nw)

    # outputs
    out_csv = args_cli.out_csv
    out_metrics = args_cli.out_metrics
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_metrics), exist_ok=True)
    if args_cli.per_dataset_dir is not None:
        os.makedirs(args_cli.per_dataset_dir, exist_ok=True)

    all_rows = []
    metrics = {}
    nd = int(args_cli.round_ndigits)

    for ds_name in eval_datasets:
        print(f"\n=== DATASET: {ds_name} ({args_cli.split}) ===")

        # switch dataset
        args.dataset.name = ds_name

        # override loader settings for selected split
        split_cfg = getattr(args, args_cli.split)
        split_cfg.batch_size = int(bs)
        split_cfg.num_workers = int(nw)
        split_cfg.shuffle = False
        split_cfg.drop_last = False

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

        probs_np, labs_np, paths_all = run_eval_one_dataset(model, loader, device)

        # raw metrics for correctness
        m_raw = compute_metrics(probs_np, labs_np)
        # rounded for reporting/JSON
        m = _round_print_metrics(m_raw, nd)
        metrics[ds_name] = m

        print(f"N: {m['N']} AUC: {m['AUC']:.{nd}f} EER: {m['EER']:.{nd}f}")

        # per-dataset csv (optional)
        if args_cli.per_dataset_dir is not None:
            df_ds = pd.DataFrame({
                "dataset": ds_name,
                "split": args_cli.split,
                "path": paths_all[:len(probs_np)],
                "label": labs_np[:len(probs_np)],
                "prob": probs_np[:len(probs_np)],
            })
            if args_cli.round_csv:
                df_ds["label"] = df_ds["label"].astype(float).round(nd)
                df_ds["prob"] = df_ds["prob"].astype(float).round(nd)
            df_ds.to_csv(os.path.join(args_cli.per_dataset_dir, f"{ds_name}_{args_cli.split}.csv"), index=False)

        # combined rows
        n = min(len(probs_np), len(labs_np), len(paths_all))
        for i in range(n):
            all_rows.append({
                "dataset": ds_name,
                "split": args_cli.split,
                "path": paths_all[i],
                "label": float(labs_np[i]),
                "prob": float(probs_np[i]),
            })

    # combined dataframe
    df_all = pd.DataFrame(all_rows)
    if args_cli.round_csv:
        df_all["label"] = df_all["label"].astype(float).round(nd)
        df_all["prob"] = df_all["prob"].astype(float).round(nd)

    df_all.to_csv(out_csv, index=False)
    print(f"\n[OK] wrote combined CSV: {out_csv}")
    print("Rows:", len(df_all), "Datasets:", df_all["dataset"].nunique())

    # overall metrics
    overall_micro_raw = compute_metrics(df_all["prob"].to_numpy(dtype=np.float64),
                                        df_all["label"].to_numpy(dtype=np.float64))
    overall_micro = _round_print_metrics(overall_micro_raw, nd)

    aucs = [metrics[d]["AUC"] for d in eval_datasets if metrics[d]["N"] > 0]
    eers = [metrics[d]["EER"] for d in eval_datasets if metrics[d]["N"] > 0]
    overall_macro = {
        "N_datasets": int(len(aucs)),
        "AUC_macro": round(float(np.mean(aucs)), nd) if len(aucs) else 0.5,
        "EER_macro": round(float(np.mean(eers)), nd) if len(eers) else 0.5,
    }

    metrics["__overall_micro__"] = overall_micro
    metrics["__overall_macro__"] = overall_macro

    # dump rounded JSON
    metrics = _round_floats(metrics, nd)
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] wrote metrics JSON: {out_metrics}")
    print("\nOverall micro:", overall_micro)
    print("Overall macro:", overall_macro)


if __name__ == "__main__":
    main()