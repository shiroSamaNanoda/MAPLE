"""YAML experiment-config parsing.

A config has one section per pluggable component (model, dataset, loss,
optimizer, scheduler) plus ``training`` and ``extras``::

    model:
      name: maple
      params: { ... }

    dataset:
      name: isprs
      params: { preset: Vaihingen, data_root: /data/ISPRS, ... }

    loss:
      name: cross_entropy
      params: { ignore_index: 255 }

    optimizer:
      name: sgd
      base_lr: 0.01
      params: { momentum: 0.9, weight_decay: 5.0e-4 }
      param_group_extras: { vlp_lr_scale: 0.1 }

    scheduler:
      name: multistep
      params: { milestones: [25, 35, 45], gamma: 0.1 }

    training:
      epochs: 50
      batch_size: 4
      num_workers: 4
      save_dir: ./weights
      save_every: 1
      log_interval: 100

    extras:                                   # arbitrary model-specific flags
      safr_logger: true
      safr_log_interval: 10
      vis_dir: ./vis_safr

The loader returns a dict-shaped object so the trainer can pull each
section without knowing about specific component names.

Override individual fields from the CLI with ``--set`` dotted keys::

    python train.py --config configs/maple_vaihingen.yaml \\
        --set training.batch_size=2 optimizer.base_lr=0.005
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class ExperimentConfig:
    """Parsed YAML, one field per top-level section.

    Every section is a plain ``dict``; we don't impose schemas beyond
    requiring ``model.name`` and ``dataset.name`` because each
    registered component owns its own parameter validation (via its
    ``__init__`` signature).
    """
    model:     Dict[str, Any]
    dataset:   Dict[str, Any]
    loss:      Dict[str, Any] = field(default_factory=dict)
    optimizer: Dict[str, Any] = field(default_factory=dict)
    scheduler: Dict[str, Any] = field(default_factory=dict)
    training:  Dict[str, Any] = field(default_factory=dict)
    extras:    Dict[str, Any] = field(default_factory=dict)


def load_experiment_config(path: str,
                           overrides: Optional[Dict[str, Any]] = None
                           ) -> ExperimentConfig:
    """Parse a YAML config into an ``ExperimentConfig``.

    Args:
        path:      YAML file path.
        overrides: optional flat dict applied last with dotted keys,
                   e.g. ``{'training.batch_size': 2}``.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    if overrides:
        for dotted, val in overrides.items():
            _set_dotted(cfg, dotted, val)

    if 'model' not in cfg or 'name' not in cfg.get('model', {}):
        raise ValueError(f"{path}: missing required `model.name`")
    if 'dataset' not in cfg or 'name' not in cfg.get('dataset', {}):
        raise ValueError(f"{path}: missing required `dataset.name`")

    return ExperimentConfig(
        model     = dict(cfg['model']),
        dataset   = dict(cfg['dataset']),
        loss      = dict(cfg.get('loss', {})),
        optimizer = dict(cfg.get('optimizer', {})),
        scheduler = dict(cfg.get('scheduler', {})),
        training  = dict(cfg.get('training', {})),
        extras    = dict(cfg.get('extras', {})),
    )


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split('.')
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def parse_cli_overrides(items) -> Dict[str, Any]:
    """Parse ``['a.b=1', 'a.c=foo']`` into ``{'a.b': 1, 'a.c': 'foo'}``
    with YAML-style scalar coercion (so ``1`` becomes int, ``true`` becomes bool,
    ``[1, 2]`` becomes list, etc.)."""
    out: Dict[str, Any] = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f"--set entries must be KEY=VAL (got {item!r})")
        k, v = item.split('=', 1)
        out[k.strip()] = yaml.safe_load(v)
    return out
