"""Plug-in segmentation losses.

Drop a file in this folder, decorate the loss class with
``@LOSS_REGISTRY.register('name')``, and reference it from a YAML:

    # losses/my_loss.py
    import torch.nn as nn
    from src.registries import LOSS_REGISTRY

    @LOSS_REGISTRY.register('my_loss')
    class MyLoss(nn.Module):
        def __init__(self, gamma=2.0):
            super().__init__()
            self.gamma = gamma
        def forward(self, logits, target):
            ...

    # configs/whatever.yaml
    loss:
      name: my_loss
      params: { gamma: 1.5 }

A loss is just an ``nn.Module`` with a ``forward(logits, target) -> Tensor``
signature.  There is no base class to inherit from -- the convention is
the contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from src.registries import LOSS_REGISTRY, auto_discover

_failures = auto_discover(__name__, Path(__file__).parent)


def build_loss(name: str, **kwargs: Any) -> nn.Module:
    """Construct a registered loss by name."""
    return LOSS_REGISTRY.build(name, **kwargs)


def available_losses() -> list:
    return LOSS_REGISTRY.names()


def import_failures() -> dict:
    return dict(_failures)


__all__ = ['LOSS_REGISTRY', 'build_loss', 'available_losses', 'import_failures']
