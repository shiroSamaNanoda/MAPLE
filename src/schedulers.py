"""LR-scheduler registry.

Pre-populated with the common ``torch.optim.lr_scheduler`` classes.
Custom schedulers register the same way as everything else::

    from src.registries import SCHEDULER_REGISTRY

    @SCHEDULER_REGISTRY.register('warmup_cosine')
    class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
        ...

YAML::

    scheduler:
      name: multistep                      # or cosine, step, plateau, none, ...
      params:
        milestones: [25, 35, 45]
        gamma: 0.1

Use ``name: none`` (or omit the section entirely) for no LR schedule.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.optim.lr_scheduler as sched

from .registries import SCHEDULER_REGISTRY


# Pre-populate from torch with friendly snake_case aliases.
_TORCH_SCHEDULERS = {
    'multistep':   sched.MultiStepLR,
    'step':        sched.StepLR,
    'exponential': sched.ExponentialLR,
    'cosine':      sched.CosineAnnealingLR,
    'cosine_wr':   sched.CosineAnnealingWarmRestarts,
    'plateau':     sched.ReduceLROnPlateau,
    'linear':      sched.LinearLR,
    'constant':    sched.ConstantLR,
    'onecycle':    sched.OneCycleLR,
    'poly':        sched.PolynomialLR,
}
for _name, _cls in _TORCH_SCHEDULERS.items():
    SCHEDULER_REGISTRY.add(_name, _cls)


def build_scheduler(name: Optional[str],
                    optimizer: torch.optim.Optimizer,
                    **kwargs: Any):
    """Construct a registered scheduler.  ``name=None`` or ``'none'``
    returns ``None`` (no schedule)."""
    if name is None or name == 'none':
        return None
    return SCHEDULER_REGISTRY.build(name, optimizer, **kwargs)


def available_schedulers() -> list:
    return SCHEDULER_REGISTRY.names()
