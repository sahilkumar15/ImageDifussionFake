# code/DiffusionFake/datasets/ffpp_control.py
from __future__ import annotations

import os
import glob
import random
import csv
from typing import List, Dict, Optional, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Optional RandomPatch
try:
    from .RandomPatch import RandomPatch
except Exception:
    RandomPatch = None

# FF++ path builder (fallback scan only)
try:
    from .data_structure import FaceForensicsDataStructure
except Exception:
    from data_structure import FaceForensicsDataStructure


# =========================
# Small image helpers
# =========================
def _norm_to_minus1_1(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.max() > 2.0:  # likely 0..255
        img = (img / 127.5) - 1.0
    elif img.min() >= 0.0 and img.max() <= 1.0:  # 0..1
        img = (img * 2.0) - 1.0
    return img


def _hwc_to_chw_tensor(img: np.ndarray) -> torch.Tensor:
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected HWC with 3 channels, got {img.shape}")
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()


# =========================
# CSV helpers (NEW)
# =========================
def _split_method_name(method: str) -> str:
    # Your config uses "youtube" to mean REAL.
    return "real" if method.lower() == "youtube" else method


def _maybe_join_root(data_root: str, p: str) -> str:
    """If p is relative, join with data_root. Otherwise return as-is."""
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(data_root, p))


def _video_file_to_images_dir(video_file: str) -> Optional[str]:
    """
    Convert:
      .../<comp>/videos/<name>.mp4
    to:
      .../<comp>/images/<name>/
    if it exists.
    """
    if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        return None
    videos_dir = os.path.dirname(video_file)          # .../<comp>/videos
    comp_dir = os.path.dirname(videos_dir)            # .../<comp>
    vid = os.path.splitext(os.path.basename(video_file))[0]
    cand = os.path.join(comp_dir, "images", vid)
    return cand if os.path.isdir(cand) else None


def _extract_method_from_path(p: str) -> str:
    parts = p.split(os.sep)
    if "original_sequences" in parts:
        return "real"
    if "manipulated_sequences" in parts:
        j = parts.index("manipulated_sequences")
        if j + 1 < len(parts):
            return parts[j + 1]
    return "unknown"


def _csv_path(data_root: str, csv_dirname: str, split: str) -> str:
    """
    Expected:
      {data_root}/{csv_dirname}/ffpp_train.csv
      {data_root}/{csv_dirname}/ffpp_val.csv
      {data_root}/{csv_dirname}/ffpp_test.csv
    """
    return os.path.join(data_root, csv_dirname, f"ffpp_{split}.csv")


def _read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            # normalize keys (strip spaces)
            rr = { (k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in r.items() }
            rows.append(rr)
        return rows


def _row_get_first(row: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def _row_to_video_dir(
    data_root: str,
    row: Dict[str, Any],
    compressions: str,
) -> Optional[str]:
    """
    Make this robust to different CSV schemas.

    Supported CSV columns (any one of these is sufficient):
      - images_dir / img_dir / frames_dir / folder / dir / video_dir
      - path (can be images_dir OR videos/<id>.mp4 OR a single frame path)
      - img_path / frame_path (single frame path; we take dirname as video_dir)
      - video_id + optional base_dir layout (we try to infer)
      - method (optional but recommended)
      - compression / comp (optional)
    """
    # Optional filter by compression if CSV provides it
    row_comp = _row_get_first(row, ["compression", "comp", "c"], default=None)
    if row_comp is not None and str(row_comp) != str(compressions):
        return None

    # Prefer explicit folder columns
    p = _row_get_first(
        row,
        ["images_dir", "img_dir", "frames_dir", "folder", "dir", "video_dir", "image_dir"],
        default=None,
    )
    if p:
        p = _maybe_join_root(data_root, str(p))
        if os.path.isdir(p):
            return p

    # Generic "path"
    p = _row_get_first(row, ["path"], default=None)
    if p:
        p = _maybe_join_root(data_root, str(p))
        if os.path.isdir(p):
            return p
        if os.path.isfile(p):
            # if it's a video -> map to images/<id> if available
            img_dir = _video_file_to_images_dir(p)
            if img_dir is not None:
                return img_dir
            # if it's a frame -> use its parent
            ext = os.path.splitext(p)[1].lower()
            if ext in [".jpg", ".jpeg", ".png"]:
                d = os.path.dirname(p)
                return d if os.path.isdir(d) else None

    # Frame path columns
    p = _row_get_first(row, ["img_path", "frame_path", "image_path"], default=None)
    if p:
        p = _maybe_join_root(data_root, str(p))
        if os.path.isfile(p):
            d = os.path.dirname(p)
            return d if os.path.isdir(d) else None

    # video_id fallback (best-effort)
    vid = _row_get_first(row, ["video_id", "vid", "id"], default=None)
    if vid:
        vid = str(vid)
        # Try common FF++ extracted frames layout:
        # {data_root}/original_sequences/youtube/{comp}/images/{video_id}/
        # {data_root}/manipulated_sequences/{method}/{comp}/images/{video_id}/
        method = _row_get_first(row, ["method", "manipulation", "dataset", "type"], default=None)
        method = str(method) if method is not None else None
        if method is not None and _split_method_name(method) == "real":
            cand = os.path.join(data_root, "original_sequences", "youtube", compressions, "images", vid)
            if os.path.isdir(cand):
                return cand
        if method is not None and _split_method_name(method) != "real":
            cand = os.path.join(data_root, "manipulated_sequences", method, compressions, "images", vid)
            if os.path.isdir(cand):
                return cand

    return None


# =========================
# Dataset
# =========================
class FaceForensicsRelation(Dataset):
    """
    FF++ dataset loader that uses CSV split files (NOT .txt).

    Expected files:
      {data_root}/split_csv/ffpp_train.csv
      {data_root}/split_csv/ffpp_val.csv
      {data_root}/split_csv/ffpp_test.csv

    CSV can be flexible, but MUST provide at least:
      - a folder path (images_dir/img_dir/frames_dir/...) OR
      - a generic "path" (images folder, a video file, or a frame file) OR
      - a frame path (img_path/frame_path)
    Recommended columns:
      - method (youtube/Deepfakes/Face2Face/FaceSwap/NeuralTextures)
      - compression (c23/c40) if mixing compressions in one CSV
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int,
        split: str,
        transform=None,
        base_transform=None,
        compressions: str = "c23",
        methods=None,
        has_mask: bool = False,
        balance: bool = True,
        random_patch=None,
        data_types: str = "",
        relation_data: bool = False,
        similarity: bool = False,
        use_splits: bool = True,   # kept for compatibility; now means "use CSV"
        strict_splits: bool = True,
        splits_dirname: str = "splits",  # legacy, ignored by CSV path
        similarity_path: Optional[str] = None,
        # NEW:
        csv_dirname: str = "split_csv",
    ):
        self.data_root = data_root
        self.num_frames = int(num_frames)
        self.split = split
        self.transform = transform
        self.base_transform = base_transform
        self.data_types = data_types

        self.compressions = compressions
        self.methods = methods or ["youtube", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        self.has_mask = has_mask

        # We keep these flags, but this file now uses CSV by default.
        self.use_splits = use_splits
        self.strict_splits = strict_splits
        self.splits_dirname = splits_dirname  # legacy

        self.csv_dirname = csv_dirname

        self.relation_data = relation_data
        self.simlarity = similarity
        self.similarity_path = similarity_path

        self.mode = "source"
        self.balabce = balance

        # RandomPatch optional
        if RandomPatch is not None and isinstance(random_patch, int):
            self.random_patch = RandomPatch(grid_size=random_patch)
        else:
            self.random_patch = None

        # load items
        self.real_items = self._load_items([self.methods[0]])
        self.fake_items = self._load_items(self.methods[1:])

        pos_len = len(self.real_items)
        neg_len = len(self.fake_items)
        print(f"Total number of data: {pos_len + neg_len} | pos: {pos_len}, neg: {neg_len}")

        # balance only train
        if self.split == "train" and self.balabce and pos_len > 0 and neg_len > 0:
            np.random.seed(1234)
            if pos_len > neg_len:
                self.real_items = np.random.choice(self.real_items, neg_len, replace=False).tolist()
            else:
                self.fake_items = np.random.choice(self.fake_items, pos_len, replace=False).tolist()
            image_len = len(self.real_items)
            print(f"After balance total number of data: {image_len * 2} | pos: {image_len}, neg: {image_len}")

        self.items = sorted(self.real_items + self.fake_items, key=lambda x: x["img_path"])

        if self.relation_data:
            self.realtion_item = self._load_relation_items()

        if self.simlarity:
            self._load_similarity()

    def change_mode(self, mode: str = "source"):
        self.mode = mode

    # -------------------------
    # Core: load items using CSV splits
    # -------------------------
    def _load_items(self, methods: List[str]) -> List[Dict]:
        """
        Use CSV split files to get per-video frame folders.
        """
        all_items: List[Dict] = []

        # Load split CSV once
        csv_rows: Optional[List[Dict[str, Any]]] = None
        if self.use_splits:
            csv_path = _csv_path(self.data_root, self.csv_dirname, self.split)
            if not os.path.exists(csv_path):
                msg = f"[{self.split}] split CSV missing: {csv_path}"
                if self.strict_splits:
                    raise FileNotFoundError(msg)
                else:
                    print("[WARN]", msg, "-> fallback scan ALL (not recommended)")
                    csv_rows = None
            else:
                csv_rows = _read_csv_rows(csv_path)
                print(f"[{self.split}] Using split CSV: {csv_path} (n_rows={len(csv_rows)})")

        for method in methods:
            method_norm = _split_method_name(method)  # youtube -> real
            video_dirs: List[str] = []

            if csv_rows is not None:
                # Filter rows by method if provided in CSV; otherwise, accept all rows.
                for r in csv_rows:
                    r_method = _row_get_first(r, ["method", "manipulation", "type", "dataset"], default=None)
                    if r_method is not None:
                        # match "youtube" and "real" as same
                        if _split_method_name(str(r_method)) != method_norm:
                            continue

                    vdir = _row_to_video_dir(
                        data_root=self.data_root,
                        row=r,
                        compressions=self.compressions,
                    )
                    if vdir is not None:
                        video_dirs.append(vdir)

                # De-dup but keep deterministic order
                video_dirs = sorted(list(set(video_dirs)))
                print(f"[{self.split}] method={method_norm} -> video_dirs={len(video_dirs)}")
            else:
                # Fallback scan (legacy)
                subdirs = FaceForensicsDataStructure(
                    root_dir=self.data_root,
                    methods=[method],
                    compressions=self.compressions,
                    data_types=self.data_types,
                ).get_subdirs()

                for dir_path in subdirs:
                    images_dir = os.path.join(dir_path, "images")
                    scan_root = images_dir if os.path.isdir(images_dir) else dir_path
                    for p in listdir_with_full_paths(scan_root):
                        if os.path.isdir(p):
                            video_dirs.append(p)

            # Convert each video_dir into sampled frame items
            for video_dir in video_dirs:
                if not os.path.isdir(video_dir):
                    continue
                is_real = ("original_sequences" in video_dir) or (method_norm == "real")
                label = 0.0 if is_real else 1.0
                all_items.extend(self._load_sub_items(video_dir, label))

        return all_items

    def _load_sub_items(self, video_dir: str, label: float) -> List[Dict]:
        """
        Robustly sample frames from a folder.
        Works for:
          0000.png
          001_870_0000.png
          etc.
        """
        num_frames = max(1, self.num_frames // 3) if (self.split == "train" and label == 1.0) else self.num_frames
        video_id = get_file_name(video_dir)

        frame_files = [f for f in os.listdir(video_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if len(frame_files) == 0:
            return []

        def _frame_key(name: str) -> int:
            stem = os.path.splitext(name)[0]
            toks = stem.split("_")
            try:
                return int(toks[-1])
            except Exception:
                digits = "".join([c for c in stem if c.isdigit()])
                return int(digits) if digits else 0

        frame_files = sorted(frame_files, key=_frame_key)

        ind = np.linspace(0, len(frame_files) - 1, num_frames, endpoint=True, dtype=int)
        chosen = [frame_files[i] for i in ind]

        out: List[Dict] = []
        for image_name in chosen:
            frame_id = os.path.splitext(image_name)[0].split("_")[-1]
            img_path = os.path.join(video_dir, image_name)
            method = "real" if label == 0.0 else _extract_method_from_path(img_path)

            out.append(
                {
                    "img_path": img_path,
                    "label": float(label),
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "method": method,
                }
            )
        return out

    # -------------------------
    # Optional: relation items
    # -------------------------
    def _load_relation_items(self):
        out = []

        def _pick_any_frame(folder: str):
            if not os.path.isdir(folder):
                return None
            files = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpeg"))
            return random.choice(files) if files else None

        for item in self.items:
            img_path = item["img_path"]
            label = float(item["label"])
            video_id = item["video_id"]
            frame_id = item["frame_id"]

            if label == 0.0:
                out.append(
                    {
                        "source_path": img_path,
                        "target_path": img_path,
                        "original_path": img_path,
                        "label": 0.0,
                        "video_id": video_id,
                        "frame_id": frame_id,
                    }
                )
                continue

            parts = img_path.split(os.sep)

            try:
                idx_manip = parts.index("manipulated_sequences")
                parts[idx_manip] = "original_sequences"
                if idx_manip + 1 < len(parts):
                    parts[idx_manip + 1] = "youtube"
            except ValueError:
                out.append(
                    {
                        "source_path": img_path,
                        "target_path": img_path,
                        "original_path": img_path,
                        "label": 0.0,
                        "video_id": video_id,
                        "frame_id": frame_id,
                    }
                )
                continue

            try:
                idx_images = parts.index("images")
            except ValueError:
                out.append(
                    {
                        "source_path": img_path,
                        "target_path": img_path,
                        "original_path": img_path,
                        "label": 0.0,
                        "video_id": video_id,
                        "frame_id": frame_id,
                    }
                )
                continue

            fake_pair = parts[idx_images + 1] if (idx_images + 1) < len(parts) else video_id
            if "_" in fake_pair:
                a, b = fake_pair.split("_", 1)
            elif "_" in video_id:
                a, b = video_id.split("_", 1)
            else:
                out.append(
                    {
                        "source_path": img_path,
                        "target_path": img_path,
                        "original_path": img_path,
                        "label": 0.0,
                        "video_id": video_id,
                        "frame_id": frame_id,
                    }
                )
                continue

            source_parts = parts[:]
            target_parts = parts[:]
            source_parts[idx_images + 1] = a
            target_parts[idx_images + 1] = b

            source_path = os.sep.join(source_parts)
            target_path = os.sep.join(target_parts)

            if not os.path.exists(source_path):
                alt = _pick_any_frame(os.path.dirname(source_path))
                source_path = alt if alt is not None else img_path

            if not os.path.exists(target_path):
                alt = _pick_any_frame(os.path.dirname(target_path))
                target_path = alt if alt is not None else img_path

            out.append(
                {
                    "source_path": source_path,
                    "target_path": target_path,
                    "original_path": img_path,
                    "label": 0.0,
                    "video_id": video_id,
                    "frame_id": frame_id,
                }
            )

        return out

    # -------------------------
    # Optional: similarity
    # -------------------------
    def _load_similarity(self):
        path = self.similarity_path or "simlarity.json"
        if not os.path.exists(path):
            print(f"[WARN] similarity=True but {path} not found. Disabling similarity scores.")
            self.result_dict = {}
            self.simlarity = False
            return
        import json
        with open(path, "r") as f:
            self.result_dict = json.load(f)

    # -------------------------
    # PyTorch dataset API
    # -------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image_size = 256

        img = cv2.imread(item["img_path"])
        if img is None:
            raise FileNotFoundError(item["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        img_aug = img.copy()
        if self.transform is not None:
            out = self.transform(image=img)
            img_aug = out["image"]
            if isinstance(img_aug, torch.Tensor):
                if img_aug.ndim == 3 and img_aug.shape[0] == 3:
                    img_aug = img_aug.permute(1, 2, 0).cpu().numpy()
                else:
                    img_aug = img_aug.cpu().numpy()

        img = _norm_to_minus1_1(img)
        img_aug = _norm_to_minus1_1(img_aug)

        hint_ori = _hwc_to_chw_tensor(img)
        hint = _hwc_to_chw_tensor(img_aug)

        if self.random_patch is not None and self.split == "train":
            hint = self.random_patch(hint)
            if hint.ndim != 3 or hint.shape[0] != 3:
                raise ValueError(f"random_patch output bad shape: {hint.shape}")

        if self.relation_data:
            rel = self.realtion_item[index]

            s_img = cv2.imread(rel["source_path"])
            t_img = cv2.imread(rel["target_path"])
            if s_img is None:
                raise FileNotFoundError(rel["source_path"])
            if t_img is None:
                raise FileNotFoundError(rel["target_path"])

            s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)
            t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)
            s_img = cv2.resize(s_img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            t_img = cv2.resize(t_img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

            s_img = _norm_to_minus1_1(s_img)
            t_img = _norm_to_minus1_1(t_img)

            source = _hwc_to_chw_tensor(s_img)
            target = _hwc_to_chw_tensor(t_img)

            scores = getattr(self, "result_dict", {}).get(item["img_path"], 1.0)
            if isinstance(scores, dict):
                source_score = float(scores.get("source_score", 1.0))
                target_score = float(scores.get("target_score", 1.0))
            else:
                source_score = 1.0
                target_score = 1.0

            return {
                "source": source,
                "target": target,
                "txt": "",
                "hint_ori": hint_ori,
                "hint": hint,
                "label": int(item["label"]),
                "source_score": source_score,
                "target_score": target_score,
                "ori_path": item["img_path"],
            }

        return {
            "source": hint_ori,
            "target": hint_ori,
            "txt": "",
            "hint_ori": hint_ori,
            "hint": hint,
            "label": int(item["label"]),
            "ori_path": item["img_path"],
        }


# =========================
# Tiny path utils used above
# =========================
def listdir_with_full_paths(dir_path: str) -> List[str]:
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path)]


def get_file_name(file_path: str) -> str:
    return file_path.split("/")[-1]