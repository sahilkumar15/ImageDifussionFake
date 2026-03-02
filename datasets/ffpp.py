# code/DiffusionFake/datasets/ffpp.py
# -*- coding: utf-8 -*-

import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    A = None
    ToTensorV2 = None


def _read_rgb(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _default_tf():
    if A is None:
        return None
    return A.Compose([
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_REFLECT_101),
        A.CenterCrop(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # -> [-1, 1]
        ToTensorV2()
    ])


def _apply_tf(tf, img):
    if tf is None:
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return x * 2.0 - 1.0
    return tf(image=img)["image"]


def _pick_frames_root(comp_root: str, images_subdir: str = "images") -> str:
    """
    Supports both layouts:
      (A) .../<comp>/images/<id>/<id>_0000.png
      (B) .../<comp>/<id>/<id>_0000.png   (old FF++ image layout)
    """
    cand = os.path.join(comp_root, images_subdir)
    return cand if os.path.isdir(cand) else comp_root


class FaceForensics(Dataset):
    """
    Works with:
      root/original_sequences/youtube/<comp>/images/<id>/<id>_0000.png
      root/manipulated_sequences/<method>/<comp>/images/<pair>/<pair>_0000.png

    Returns dict with keys expected by generate_weight.py:
      hint_ori, source_aug, target_aug, ori_path
    """
    def __init__(
        self,
        root: str,
        compression: str = "c23",
        methods=("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"),
        images_subdir: str = "images",
        max_items: int | None = None,
        transform_hint=None,
        transform_ref=None,
    ):
        self.root = root
        self.compression = compression
        self.methods = list(methods)
        self.images_subdir = images_subdir

        self.tf_hint = transform_hint if transform_hint is not None else _default_tf()
        self.tf_ref  = transform_ref  if transform_ref  is not None else _default_tf()

        # ---- figure out where "frames root" is for originals/manips ----
        orig_comp_root = os.path.join(root, "original_sequences", "youtube", compression)
        manip_comp_roots = {
            m: os.path.join(root, "manipulated_sequences", m, compression) for m in self.methods
        }

        self.orig_frames_root = _pick_frames_root(orig_comp_root, images_subdir=images_subdir)
        self.manip_frames_root = {
            m: _pick_frames_root(manip_comp_roots[m], images_subdir=images_subdir) for m in self.methods
        }

        self.samples = []

        # collect manipulated frames
        for m in self.methods:
            patt = os.path.join(self.manip_frames_root[m], "*", "*.png")  # e.g. .../images/000_003/*.png
            for mp in sorted(glob.glob(patt)):
                folder = os.path.basename(os.path.dirname(mp))  # "000_003"
                if "_" not in folder:
                    continue
                src, tgt = folder.split("_", 1)

                base = os.path.basename(mp).replace(".png", "")
                parts = base.split("_")
                if len(parts) < 3:
                    continue
                frame = parts[-1]  # "0000"

                # originals for same frame id
                src_ori = os.path.join(self.orig_frames_root, src, f"{src}_{frame}.png")
                tgt_ori = os.path.join(self.orig_frames_root, tgt, f"{tgt}_{frame}.png")

                if os.path.exists(src_ori) and os.path.exists(tgt_ori):
                    self.samples.append((mp, src_ori, tgt_ori))

        if max_items is not None:
            self.samples = self.samples[: int(max_items)]

        if len(self.samples) == 0:
            raise RuntimeError(
                "No FF++ samples found. Your frames are not in the expected place.\n"
                f"root={root}\ncompression={compression}\n"
                f"orig_frames_root={self.orig_frames_root}\n"
                f"manip_frames_root(example)={self.manip_frames_root.get(self.methods[0], 'N/A')}\n\n"
                "Expected (your layout):\n"
                f"  original_sequences/youtube/{compression}/{images_subdir}/000/000_0000.png\n"
                f"  manipulated_sequences/Deepfakes/{compression}/{images_subdir}/000_003/000_003_0000.png\n"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hint_path, src_path, tgt_path = self.samples[idx]

        hint = _read_rgb(hint_path)
        src  = _read_rgb(src_path)
        tgt  = _read_rgb(tgt_path)

        hint_t = _apply_tf(self.tf_hint, hint)
        src_t  = _apply_tf(self.tf_ref,  src)
        tgt_t  = _apply_tf(self.tf_ref,  tgt)

        return {
            "hint_ori": hint_t,
            "source_aug": src_t,
            "target_aug": tgt_t,
            "ori_path": hint_path
        }


def create_dataset(args, split="train"):
    # root from args or env
    root = None
    for k in ["data_root", "dataset_root", "root", "ffpp_root"]:
        if hasattr(args, k):
            root = getattr(args, k)
            break
    if root is None:
        root = os.environ.get("FFPP_ROOT", None)
    if root is None:
        raise ValueError("Could not find dataset root. Set env FFPP_ROOT or add it to args.")

    compression = getattr(args, "compression", "c23")
    bs = int(getattr(args, "batch_size", 8))
    nw = int(getattr(args, "num_workers", 4))
    max_items = getattr(args, "max_items", None)

    # IMPORTANT: your frames are inside "images/"
    images_subdir = getattr(args, "images_subdir", "images")

    methods = getattr(args, "methods", ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"])

    ds = FaceForensics(
        root=root,
        compression=compression,
        methods=methods,
        images_subdir=images_subdir,
        max_items=max_items
    )
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
