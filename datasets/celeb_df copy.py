# code/DiffusionFake/datasets/celeb_df.py
import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class CelebDF(data.Dataset):
    """
    CelebDF dataset with optional CSV split loading (FFPP-style).

    CSV expected columns:
      - path  : absolute path to .mp4 (or relative to data_root)
      - label : 0 real, 1 fake
    Optional:
      - source: "YouTube-real" / "Celeb-real" / "Celeb-synthesis"
    """

    def __init__(
        self,
        data_root,
        split,
        num_frames,
        transform=None,
        base_transform=None,
        target_transform=None,
        alb=True,
        methods="both",
        control=False,
        split_csv=None,
        strict_csv=False,     # ✅ default False for cross-dataset eval (do NOT crash)
        min_frames=1,         # ✅ allow 1+ frames (set 5 if you really want)
        mtcnn_suffix="-mtcnn",
    ):
        self.split = split
        self.frame_nums = int(num_frames)
        self.transform = transform
        self.target_transform = target_transform
        self.data_root = data_root
        self.alb = alb
        self.methods = methods
        self.control = control
        self.base_transform = base_transform

        self.split_csv = split_csv
        self.strict_csv = strict_csv
        self.min_frames = int(min_frames)
        self.mtcnn_suffix = str(mtcnn_suffix)

        if self.split_csv is not None:
            self.datas = self._load_items_from_csv(self.split_csv)
        else:
            self.datas = self._load_items_from_official_list()

        print(f"[CelebDF:{self.split}] Total frames: {len(self.datas)} | fake_frames: {self.fake_num}, real_frames: {self.real_num}")

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        img_path, target, folder = self.datas[index]

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            # for eval you want to skip bad reads rather than crash:
            raise FileNotFoundError(f"[CelebDF] Failed to read image: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.control:
            image_size = 256
            image_resized = cv2.resize(image, (image_size, image_size))
            image_resized = image_resized.astype(np.float32) / 255.0  # 0..1

            if self.transform is not None:
                hint = self.transform(image=image_resized)["image"]  # torch CHW
            else:
                hint = torch.from_numpy(image_resized).permute(2, 0, 1).float()

            hint_ori = torch.from_numpy(image_resized).permute(2, 0, 1).float() * 2.0 - 1.0

            return {
                "txt": "",
                "hint": hint,
                "hint_ori": hint_ori,
                "label": int(target),
                "path": img_path,
            }

        # non-control mode
        if self.base_transform is None:
            if self.transform is not None:
                if self.alb:
                    image = self.transform(image=image)["image"]
                else:
                    image = self.transform(img=image)
            return image, target, img_path
        else:
            image_norm = self.transform(image=image)["image"]
            small_image = self.base_transform(image=image)["image"]
            return image_norm, target, small_image, img_path

    # -------------------------
    # CSV LOADER (MP4 -> MTCNN frames)
    # -------------------------
    def _load_items_from_csv(self, csv_path: str):
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(self.data_root, csv_path)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[CelebDF] split_csv not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            raise ValueError(f"[CelebDF] split_csv is empty: {csv_path}")

        if "path" not in df.columns or "label" not in df.columns:
            raise ValueError(f"[CelebDF] CSV must have columns ['path','label']. Found: {list(df.columns)}")

        has_source = "source" in df.columns

        datas = []
        self.fake_num = 0
        self.real_num = 0
        skipped_rows = 0
        missing_folder = 0
        too_few_frames = 0

        for _, row in df.iterrows():
            mp4_path = str(row["path"]).strip()
            if not mp4_path:
                skipped_rows += 1
                continue

            # allow mp4 paths that are relative to data_root
            if not os.path.isabs(mp4_path):
                mp4_path = os.path.join(self.data_root, mp4_path)

            y = float(row["label"])
            y = 1.0 if y >= 0.5 else 0.0

            # filter by methods
            if self.methods == "real" and y == 1.0:
                continue
            if self.methods == "fake" and y == 0.0:
                continue

            # get source name
            if has_source:
                src = str(row["source"]).strip()   # YouTube-real / Celeb-real / Celeb-synthesis
            else:
                # infer from mp4 path
                if "YouTube-real" in mp4_path:
                    src = "YouTube-real"
                elif "Celeb-real" in mp4_path:
                    src = "Celeb-real"
                elif "Celeb-synthesis" in mp4_path:
                    src = "Celeb-synthesis"
                else:
                    if self.strict_csv:
                        raise ValueError(f"[CelebDF] Cannot infer source for: {mp4_path}")
                    skipped_rows += 1
                    continue

            vid = os.path.splitext(os.path.basename(mp4_path))[0]

            # ✅ Robust folder building:
            # mp4_dir = .../Celeb-DF_v2/YouTube-real
            # parent  = .../Celeb-DF_v2
            mp4_dir = os.path.dirname(mp4_path)
            parent = os.path.dirname(mp4_dir)

            # Candidate A: correct for your layout
            candA = os.path.join(parent, f"{src}{self.mtcnn_suffix}", vid)

            # Candidate B: if someone extracted frames at data_root/<src>-mtcnn/<vid>
            candB = os.path.join(self.data_root, f"{src}{self.mtcnn_suffix}", vid)

            # Candidate C: parent of parent fallback
            candC = os.path.join(os.path.dirname(parent), f"{src}{self.mtcnn_suffix}", vid)

            folder = None
            for c in (candA, candB, candC):
                if os.path.isdir(c):
                    folder = c
                    break

            if folder is None:
                if self.strict_csv:
                    raise FileNotFoundError(
                        f"[CelebDF] mtcnn folder not found for: {mp4_path}\nTried:\n  {candA}\n  {candB}\n  {candC}"
                    )
                missing_folder += 1
                continue

            face_paths = sorted(glob.glob(os.path.join(folder, "*.png")))
            if len(face_paths) < self.min_frames:
                if self.strict_csv:
                    raise FileNotFoundError(f"[CelebDF] Not enough frames in {folder} (found {len(face_paths)})")
                too_few_frames += 1
                continue

            # sample frames
            if len(face_paths) > self.frame_nums:
                idx = np.linspace(0, len(face_paths) - 1, self.frame_nums, endpoint=True, dtype=int)
                face_paths = [face_paths[i] for i in idx]

            # count frames
            if y == 1.0:
                self.fake_num += len(face_paths)
            else:
                self.real_num += len(face_paths)

            datas.extend([[fp, y, folder] for fp in face_paths])

        print(
            f"[CelebDF:{self.split}] Using split CSV: {csv_path} rows={len(df)} "
            f"loaded_frames={len(datas)} missing_folder={missing_folder} too_few_frames={too_few_frames} skipped_rows={skipped_rows}"
        )
        return datas

    # -------------------------
    # KEEP: original list loader (optional)
    # -------------------------
    def _load_items_from_official_list(self):
        raise NotImplementedError(
            "For your cross-dataset eval you said you already have CSV. "
            "Use split_csv=... in YAML and this path won't be used."
        )