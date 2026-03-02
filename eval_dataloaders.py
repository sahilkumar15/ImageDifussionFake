# code/DiffusionFake/eval_dataloaders.py
import copy
from datasets import create_dataset

def build_test_dataloader(args, dataset_name: str):
    """
    Uses your existing dataset factory: create_dataset(args, split='test')
    We just switch args.dataset.name and return its dataloader.
    """
    # avoid mutating args permanently (safer)
    args_local = copy.deepcopy(args)
    args_local.dataset.name = dataset_name
    dl = create_dataset(args_local, split="test")
    return dl