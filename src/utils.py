"""Generic utilities: palette conversion, sliding window, misc helpers.

Note: ``get_optimizer_params`` has moved -- each model now owns its own
parameter-grouping policy via ``BaseSegmentor.param_groups()``.  See
``MAPLE.param_groups`` for the VLP-LR-scaling pattern, or
``BaseSegmentor.param_groups`` for the trivial default.
"""
import itertools
import random
from typing import Dict, Iterator, Tuple

import numpy as np
import torch


# ============================================================
#  Palette conversion
# ============================================================

def convert_to_color(arr_2d: np.ndarray, palette: Dict[int, Tuple[int, int, int]]
                     ) -> np.ndarray:
    """Numeric class labels -> RGB image."""
    out = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for cls, color in palette.items():
        out[arr_2d == cls] = color
    return out


def convert_from_color(arr_3d: np.ndarray,
                       invert_palette: Dict[Tuple[int, int, int], int]
                       ) -> np.ndarray:
    """RGB image -> numeric class labels."""
    out = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for color, cls in invert_palette.items():
        mask = np.all(arr_3d == np.array(color).reshape(1, 1, 3), axis=2)
        out[mask] = cls
    return out


# ============================================================
#  Sliding-window inference
# ============================================================

def sliding_window(image: np.ndarray, step: int = 10,
                   window_size: Tuple[int, int] = (256, 256)
                   ) -> Iterator[Tuple[int, int, int, int]]:
    """Yield (x, y, w, h) coords covering `image` with overlap `step`."""
    H, W = image.shape[:2]
    wh, ww = window_size
    for x in range(0, H, step):
        x = min(x, H - wh)
        for y in range(0, W, step):
            y = min(y, W - ww)
            yield x, y, wh, ww


def count_sliding_window(image: np.ndarray, step: int = 10,
                         window_size: Tuple[int, int] = (256, 256)) -> int:
    return sum(1 for _ in sliding_window(image, step, window_size))


def grouper(n: int, iterable):
    """Iterate `iterable` in chunks of size `n`."""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def get_random_pos(img: np.ndarray, window_shape: Tuple[int, int]
                   ) -> Tuple[int, int, int, int]:
    """Pick a random window position inside `img`."""
    w, h = window_shape
    H, W = img.shape[-2:]
    x1 = random.randint(0, H - w - 1)
    y1 = random.randint(0, W - h - 1)
    return x1, x1 + w, y1, y1 + h


# ============================================================
#  Misc.
# ============================================================

def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Pixel-wise accuracy in percent."""
    return 100.0 * float(np.count_nonzero(pred == target)) / target.size


def batch_to_cuda(batch):
    """Move every tensor in a dict batch to CUDA; leave non-tensors alone."""
    if isinstance(batch, dict):
        return {k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_cuda(b) for b in batch)
    return batch.cuda(non_blocking=True) if isinstance(batch, torch.Tensor) else batch
