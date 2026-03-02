import torch
import random

class RandomPatch:
    """
    Simple grid-based random patch masking on CHW tensor in [-1,1].
    grid_size: int, number of patches per side (e.g., 4 -> 4x4 grid).
    Randomly zeros out one patch.
    """
    def __init__(self, grid_size: int = 4):
        assert isinstance(grid_size, int) and grid_size > 0
        self.grid_size = grid_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C,H,W]
        if not (isinstance(x, torch.Tensor) and x.ndim == 3):
            raise ValueError(f"RandomPatch expects CHW tensor, got {type(x)} {getattr(x,'shape',None)}")

        c, h, w = x.shape
        gh = h // self.grid_size
        gw = w // self.grid_size
        if gh == 0 or gw == 0:
            return x

        i = random.randrange(self.grid_size)
        j = random.randrange(self.grid_size)

        y0, y1 = i * gh, (i + 1) * gh
        x0, x1 = j * gw, (j + 1) * gw

        out = x.clone()
        out[:, y0:y1, x0:x1] = 0.0
        return out