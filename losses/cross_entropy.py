"""Plain (optionally class-weighted) cross-entropy.

Thin wrapper around ``nn.CrossEntropyLoss`` so it shows up in the
registry like every other loss.

YAML:

    loss:
      name: cross_entropy
      params:
        ignore_index: 255
        weight: [1.0, 1.0, 1.0, 1.0, 5.0, 1.0]   # optional per-class weights
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from src.registries import LOSS_REGISTRY


@LOSS_REGISTRY.register('cross_entropy')
class CrossEntropyLoss(nn.CrossEntropyLoss):
    """nn.CrossEntropyLoss with YAML-friendly list-to-tensor coercion on
    the ``weight`` argument (YAML can't express a tensor)."""

    def __init__(self,
                 weight: Optional[List[float]] = None,
                 ignore_index: int = 255,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=torch.float32)
        super().__init__(weight=weight, ignore_index=ignore_index,
                         reduction=reduction, label_smoothing=label_smoothing)
