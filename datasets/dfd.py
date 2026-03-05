# datasets/dfd.py
import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

def _norm_to_minus1_1(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mx = float(img.max()) if img.size else 0.0
    mn = float(img.min()) if img.size else 0.0
    if mx > 2.0:
        img = (img / 127.5) - 1.0
    elif mn >= 0.0 and mx <= 1.0:
        img = (img * 2.0) - 1.0
    return img

def _hwc_to_chw_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

class DFD(data.Dataset):
    """
    CSV rows are videos. We map:
      <data_root>/<source>-mtcnn/<video_id>/*.png
    then expand into per-frame items with uniform sampling to num_frames.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        num_frames: int,
        transform=None,
        split_csv: str = None,
        strict_csv: bool = False,
        min_frames: int = 1,
        mtcnn_suffix: str = "-mtcnn",
        image_size: int = 256,
        skip_bad_images: bool = True,
        debug_first_n_missing: int = 3,
        methods: str = "both",
    ):
        self.data_root = str(data_root)
        self.split = str(split)
        self.num_frames = int(num_frames)
        self.transform = transform

        self.split_csv = split_csv
        self.strict_csv = bool(strict_csv)
        self.min_frames = int(min_frames)
        self.mtcnn_suffix = str(mtcnn_suffix)
        self.image_size = int(image_size)
        self.skip_bad_images = bool(skip_bad_images)
        self.debug_first_n_missing = int(debug_first_n_missing)
        self.methods = str(methods)

        if self.split_csv is None:
            raise ValueError("[DFD] split_csv is required.")

        self.items = self._load_items_from_csv(self.split_csv)

        self.real_num = sum(1 for _, y, _ in self.items if float(y) < 0.5)
        self.fake_num = len(self.items) - self.real_num

        print(
            f"[DFD:{self.split}] loaded_frames={len(self.items)} "
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
            raise FileNotFoundError(f"[DFD] cannot read: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        hint_ori = _hwc_to_chw_tensor(_norm_to_minus1_1(img))

        if self.transform is not None:
            out = self.transform(image=img)
            hint = out["image"] if isinstance(out, dict) else out
            if not isinstance(hint, torch.Tensor):
                raise TypeError(f"[DFD] transform must return torch.Tensor; got {type(hint)}")
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
            raise FileNotFoundError(f"[DFD] split_csv not found: {csv_path}")

        df = pd.read_csv(csv_path)
        for col in ["path", "label", "source"]:
            if col not in df.columns:
                raise ValueError(f"[DFD] CSV must contain '{col}'. got={list(df.columns)}")

        items = []
        missing_video = 0
        missing_folder = 0
        too_few_frames = 0
        printed_missing = 0

        for _, row in df.iterrows():
            video_path = str(row["path"]).strip()
            if not video_path:
                continue
            if not os.path.isabs(video_path):
                video_path = os.path.join(self.data_root, video_path)
            if not os.path.exists(video_path):
                missing_video += 1
                if self.strict_csv:
                    raise FileNotFoundError(f"[DFD] video missing: {video_path}")
                continue

            y = 1.0 if float(row["label"]) >= 0.5 else 0.0
            if self.methods == "real" and y == 1.0:
                continue
            if self.methods == "fake" and y == 0.0:
                continue

            src_dir = str(row["source"]).strip()
            vid = os.path.splitext(os.path.basename(video_path))[0]

            folder = os.path.join(self.data_root, f"{src_dir}{self.mtcnn_suffix}", vid)
            if not os.path.isdir(folder):
                missing_folder += 1
                if printed_missing < self.debug_first_n_missing:
                    printed_missing += 1
                    print(f"[DFD:{self.split}] MISSING folder for {video_path}\n  expected: {folder}\n")
                if self.strict_csv:
                    raise FileNotFoundError(f"[DFD] frames folder not found: {folder}")
                continue

            frame_paths = sorted(glob.glob(os.path.join(folder, "*.png")))
            if len(frame_paths) < self.min_frames:
                too_few_frames += 1
                if self.strict_csv:
                    raise FileNotFoundError(f"[DFD] too few frames in {folder}: {len(frame_paths)}")
                continue

            if len(frame_paths) > self.num_frames:
                idx = np.linspace(0, len(frame_paths) - 1, self.num_frames, endpoint=True, dtype=int)
                frame_paths = [frame_paths[i] for i in idx]

            items.extend([[fp, y, folder] for fp in frame_paths])

        print(
            f"[DFD:{self.split}] CSV={csv_path} rows={len(df)} "
            f"loaded_frames={len(items)} missing_video={missing_video} "
            f"missing_folder={missing_folder} too_few_frames={too_few_frames}"
        )
        return items