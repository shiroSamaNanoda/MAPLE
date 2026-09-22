"""Optimizer registry.

Pre-populated with everything in ``torch.optim``.  Custom optimizers
plug in the same way as everything else::

    from src.registries import OPTIMIZER_REGISTRY

    @OPTIMIZER_REGISTRY.register('lion')
    class Lion(torch.optim.Optimizer):
        ...

YAML::

    optimizer:
      name: sgd                            # or adam, adamw, rmsprop, lion, ...
      base_lr: 0.01                        # passed to model.param_groups(base_lr=...)
      params:                              # forwarded to the optimizer __init__
        momentum: 0.9
        weight_decay: 5.0e-4
      param_group_extras:                  # forwarded to model.param_groups(**extras)
        vlp_lr_scale: 0.1
"""
from __future__ import annotations

from typing import Any, List

import torch
import torch.optim as optim

from .registries import OPTIMIZER_REGISTRY


# Pre-populate from torch.optim.  We register both lowercase and
# original names so YAMLs can stay snake_case.
_TORCH_OPTIMIZERS = {
    'sgd':      optim.SGD,
    'adam':     optim.Adam,
    'adamw':    optim.AdamW,
    'adamax':   optim.Adamax,
    'adagrad':  optim.Adagrad,
    'adadelta': optim.Adadelta,
    'rmsprop':  optim.RMSprop,
    'asgd':     optim.ASGD,
    'lbfgs':    optim.LBFGS,
    'nadam':    optim.NAdam,
    'radam':    optim.RAdam,
}
for _name, _cls in _TORCH_OPTIMIZERS.items():
    OPTIMIZER_REGISTRY.add(_name, _cls)


def build_optimizer(name: str, params: List[dict], **kwargs: Any
                    ) -> torch.optim.Optimizer:
    """Construct a registered optimizer.

    Args:
        name:   key in OPTIMIZER_REGISTRY (e.g. 'sgd', 'adamw').
        params: parameter groups (the output of model.param_groups()).
        **kwargs: forwarded to the optimizer's ``__init__``.
    """
    return OPTIMIZER_REGISTRY.build(name, params, **kwargs)


def available_optimizers() -> list:
    return OPTIMIZER_REGISTRY.names()
