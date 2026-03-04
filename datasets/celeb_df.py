# code/ImageDifussionFake/datasets/celeb_df.py

import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


def _norm_to_minus1_1(img: np.ndarray) -> np.ndarray:
    """Convert HWC RGB image to float32 in [-1,1]. Accepts 0..255 or 0..1."""
    img = img.astype(np.float32)
    mx = float(img.max()) if img.size else 0.0
    mn = float(img.min()) if img.size else 0.0
    if mx > 2.0:  # 0..255
        img = (img / 127.5) - 1.0
    elif mn >= 0.0 and mx <= 1.0:  # 0..1
        img = (img * 2.0) - 1.0
    return img


def _hwc_to_chw_tensor(img: np.ndarray) -> torch.Tensor:
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {img.shape}")
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()


class CelebDF(data.Dataset):
    """
    CSV provides MP4 path + label + source (YouTube-real / Celeb-real / Celeb-synthesis).

    We map MP4 -> extracted frames folder:
        <data_root>/<source>-mtcnn/<vid>/*.png

    Then we expand into per-frame items (like the original DiffusionFake loader),
    uniformly subsampling up to num_frames per video.

    Returns a dict compatible with your "control" pipeline:
      source, target, hint, hint_ori, txt, label, ori_path
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        num_frames: int,
        transform=None,              # albumentations pipeline (Normalize+ToTensorV2)
        base_transform=None,
        target_transform=None,
        alb: bool = True,
        methods: str = "both",       # "real" / "fake" / "both"
        control: bool = True,
        split_csv: str = None,       # REQUIRED (your generated CSV)
        strict_csv: bool = False,
        min_frames: int = 1,
        mtcnn_suffix: str = "-mtcnn",
        image_size: int = 256,
        skip_bad_images: bool = True,
        debug_first_n_missing: int = 3,
    ):
        self.data_root = str(data_root)
        self.split = str(split)
        self.num_frames = int(num_frames)
        self.transform = transform
        self.base_transform = base_transform
        self.target_transform = target_transform
        self.alb = bool(alb)
        self.methods = str(methods)
        self.control = bool(control)

        self.split_csv = split_csv
        self.strict_csv = bool(strict_csv)
        self.min_frames = int(min_frames)
        self.mtcnn_suffix = str(mtcnn_suffix)
        self.image_size = int(image_size)
        self.skip_bad_images = bool(skip_bad_images)
        self.debug_first_n_missing = int(debug_first_n_missing)

        if self.split_csv is None:
            raise ValueError("[CelebDF] split_csv is required.")

        self.items = self._load_items_from_csv(self.split_csv)

        self.real_num = sum(1 for _, y, _ in self.items if float(y) < 0.5)
        self.fake_num = len(self.items) - self.real_num

        print(
            f"[CelebDF:{self.split}] loaded_frames={len(self.items)} "
            f"real_frames={self.real_num} fake_frames={self.fake_num} "
            f"(csv={self.split_csv})"
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path, label, _folder = self.items[index]

        img = cv2.imread(img_path)
        if img is None:
            if self.skip_bad_images:
                return self.__getitem__((index + 1) % len(self.items))
            raise FileNotFoundError(f"[CelebDF] cannot read: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        # raw image in [-1,1] (ControlNet-style)
        hint_ori = _hwc_to_chw_tensor(_norm_to_minus1_1(img))

        # augmented/normalized by albumentations transform
        if self.transform is not None:
            out = self.transform(image=img)
            hint = out["image"] if isinstance(out, dict) else out
            if not isinstance(hint, torch.Tensor):
                raise TypeError(f"[CelebDF] transform must return torch.Tensor; got {type(hint)}")
            if hint.ndim != 3 or hint.shape[0] != 3:
                raise ValueError(f"[CelebDF] hint shape bad: {hint.shape}")
        else:
            hint = hint_ori.clone()

        y = int(label)

        return {
            "source": hint_ori,
            "target": hint_ori,
            "txt": "",
            "hint_ori": hint_ori,
            "hint": hint,
            "label": y,
            "ori_path": img_path,
        }

    def _load_items_from_csv(self, csv_path: str):
        csv_path = str(csv_path)
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(self.data_root, csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[CelebDF] split_csv not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            raise ValueError(f"[CelebDF] split_csv empty: {csv_path}")

        for col in ["path", "label", "source"]:
            if col not in df.columns:
                raise ValueError(f"[CelebDF] CSV must contain column '{col}'. got={list(df.columns)}")

        items = []
        missing_mp4 = 0
        missing_folder = 0
        too_few_frames = 0
        skipped_rows = 0
        printed_missing = 0

        for _, row in df.iterrows():
            mp4_path = str(row["path"]).strip()
            if not mp4_path:
                skipped_rows += 1
                continue
            if not os.path.isabs(mp4_path):
                mp4_path = os.path.join(self.data_root, mp4_path)
            if not os.path.exists(mp4_path):
                missing_mp4 += 1
                if self.strict_csv:
                    raise FileNotFoundError(f"[CelebDF] mp4 missing: {mp4_path}")
                continue

            y = float(row["label"])
            y = 1.0 if y >= 0.5 else 0.0

            if self.methods == "real" and y == 1.0:
                continue
            if self.methods == "fake" and y == 0.0:
                continue

            src_dir = str(row["source"]).strip()
            if not src_dir:
                src_dir = os.path.basename(os.path.dirname(mp4_path))

            vid = os.path.splitext(os.path.basename(mp4_path))[0]

            # expected extraction layout:
            #   <data_root>/<source>-mtcnn/<vid>/*.png
            folder = os.path.join(self.data_root, f"{src_dir}{self.mtcnn_suffix}", vid)

            if not os.path.isdir(folder):
                missing_folder += 1
                if printed_missing < self.debug_first_n_missing:
                    printed_missing += 1
                    print(
                        f"[CelebDF:{self.split}] MISSING frames folder for mp4={mp4_path}\n"
                        f"  expected:\n"
                        f"    {folder}\n"
                    )
                if self.strict_csv:
                    raise FileNotFoundError(f"[CelebDF] frames folder not found: {folder}")
                continue

            face_paths = sorted(glob.glob(os.path.join(folder, "*.png")))
            if len(face_paths) < self.min_frames:
                too_few_frames += 1
                if self.strict_csv:
                    raise FileNotFoundError(f"[CelebDF] too few frames in {folder}: {len(face_paths)}")
                continue

            # uniform sampling per video
            if len(face_paths) > self.num_frames:
                idx = np.linspace(0, len(face_paths) - 1, self.num_frames, endpoint=True, dtype=int)
                face_paths = [face_paths[i] for i in idx]

            items.extend([[fp, y, folder] for fp in face_paths])

        print(
            f"[CelebDF:{self.split}] CSV={csv_path} rows={len(df)} "
            f"loaded_frames={len(items)} missing_mp4={missing_mp4} "
            f"missing_folder={missing_folder} too_few_frames={too_few_frames} skipped_rows={skipped_rows}"
        )
        return items