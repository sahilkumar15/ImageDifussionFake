# code/DiffusionFake/datasets/dataset_factory.py

import random
import torch.utils.data as data
from torch.utils.data import Subset

from .celeb_df import CelebDF
from .ffpp_control import FaceForensicsRelation
from .transforms import create_data_transforms


def _maybe_subset(dataset, args, split: str):
    """
    Allows you to train/val/test on a smaller random subset using config:
      train.max_items: 2000
      val.max_items: 200
      test.max_items: 200
    Deterministic with args.seed.
    """
    split_cfg = getattr(args, split, None)
    if split_cfg is None:
        return dataset

    max_items = getattr(split_cfg, "max_items", None)
    if max_items is None:
        return dataset

    max_items = int(max_items)
    max_items = min(max_items, len(dataset))

    seed = int(getattr(args, "seed", 1234))
    rng = random.Random(seed + (0 if split == "train" else 1 if split == "val" else 2))

    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:max_items]

    print(f"[{split}] Using subset: {len(indices)}/{len(dataset)} (max_items={max_items})")
    return Subset(dataset, indices)


def create_dataset(args, split: str):
    # transforms
    transform = create_data_transforms(args.transform, split)

    # dataset selection
    kwargs = getattr(args.dataset, args.dataset.name)

    if args.dataset.name == "ffpp_rela":
        dataset = FaceForensicsRelation(split=split, transform=transform, **kwargs)
    elif args.dataset.name == "celeb_df":
        dataset = CelebDF(split=split, transform=transform, **kwargs)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset.name}")

    # optional subset
    dataset = _maybe_subset(dataset, args, split)

    # sampler / shuffle
    sampler = None
    if getattr(args, "distributed", False):
        sampler = data.distributed.DistributedSampler(dataset)

    # allow override in config: train.shuffle: true/false
    split_cfg = getattr(args, split)
    shuffle_cfg = getattr(split_cfg, "shuffle", None)

    if sampler is not None:
        shuffle = False
    else:
        shuffle = (split == "train") if shuffle_cfg is None else bool(shuffle_cfg)

    # loader params
    batch_size = int(getattr(split_cfg, "batch_size"))
    if getattr(args, "debug", False):
        batch_size = 4

    num_workers = int(getattr(split_cfg, "num_workers", 6))
    drop_last = bool(getattr(split_cfg, "drop_last", True))

    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    return dataloader


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="../configs/train.yaml")
    parser.add_argument("--distributed", type=int, default=0)
    args_cli = parser.parse_args()

    cfg = OmegaConf.load(args_cli.config)
    for k, v in cfg.items():
        setattr(args_cli, k, v)

    print("Dataset =>", args_cli.dataset.name)

    dl = create_dataset(args_cli, split="train")
    batch = next(iter(dl))
    print("Train batch keys:", batch.keys())
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(k, v.shape, v.dtype)