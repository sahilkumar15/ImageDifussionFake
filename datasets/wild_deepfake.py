

# datasets/wild_deepfake.py
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

def _norm_to_minus1_1(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mx = float(img.max()) if img.size else 0.0
    mn = float(img.min()) if img.size else 0.0
    if mx > 2.0:                  # uint8 [0..255]
        img = (img / 127.5) - 1.0
    elif mn >= 0.0 and mx <= 1.0: # float [0..1]
        img = (img * 2.0) - 1.0
    return img

def _hwc_to_chw_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

class WildDeepfake(data.Dataset):
    """
    Image-only dataset.
    CSV must contain: path,label  (optional: dataset, split, source)
    Returns dict compatible with your pipeline:
      {source,target,txt,hint_ori,hint,label,ori_path}
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        transform=None,
        split_csv: str = None,
        strict_csv: bool = False,
        image_size: int = 256,
        skip_bad_images: bool = True,
        methods: str = "both",
    ):
        self.data_root = str(data_root)
        self.split = str(split)
        self.transform = transform

        self.split_csv = split_csv
        self.strict_csv = bool(strict_csv)
        self.image_size = int(image_size)
        self.skip_bad_images = bool(skip_bad_images)
        self.methods = str(methods)

        if self.split_csv is None:
            raise ValueError("[WildDeepfake] split_csv is required.")

        df = self._load_csv(self.split_csv)
        self.items = self._build_items(df)

        self.real_num = sum(1 for _, y in self.items if int(y) == 0)
        self.fake_num = len(self.items) - self.real_num
        print(
            f"[WildDeepfake:{self.split}] images={len(self.items)} "
            f"real={self.real_num} fake={self.fake_num} (csv={self.split_csv})"
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path, y = self.items[index]

        img = cv2.imread(img_path)
        if img is None:
            if self.skip_bad_images:
                return self.__getitem__((index + 1) % len(self.items))
            raise FileNotFoundError(f"[WildDeepfake] cannot read: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        hint_ori = _hwc_to_chw_tensor(_norm_to_minus1_1(img))

        if self.transform is not None:
            out = self.transform(image=img)
            hint = out["image"] if isinstance(out, dict) else out
            if not isinstance(hint, torch.Tensor):
                raise TypeError(f"[WildDeepfake] transform must return torch.Tensor; got {type(hint)}")
        else:
            hint = hint_ori.clone()

        return {
            "source": hint_ori,
            "target": hint_ori,
            "txt": "",
            "hint_ori": hint_ori,
            "hint": hint,
            "label": int(y),
            "ori_path": img_path,
        }

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        csv_path = str(csv_path)
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(self.data_root, csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[WildDeepfake] split_csv not found: {csv_path}")

        df = pd.read_csv(csv_path)
        for col in ["path", "label"]:
            if col not in df.columns:
                raise ValueError(f"[WildDeepfake] CSV must contain '{col}'. got={list(df.columns)}")
        return df

    def _build_items(self, df: pd.DataFrame):
        items = []
        missing = 0
        for _, row in df.iterrows():
            p = str(row["path"]).strip()
            if not p:
                continue
            if not os.path.isabs(p):
                p = os.path.join(self.data_root, p)
            if not os.path.exists(p):
                missing += 1
                if self.strict_csv:
                    raise FileNotFoundError(f"[WildDeepfake] missing image: {p}")
                continue

            y = int(float(row["label"]) >= 0.5)
            if self.methods == "real" and y == 1:
                continue
            if self.methods == "fake" and y == 0:
                continue

            items.append((p, y))

        print(f"[WildDeepfake:{self.split}] missing_images={missing}")
        return items

