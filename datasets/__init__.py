"""Plug-in datasets.

Drop a file in this folder, decorate the dataset class with
``@DATASET_REGISTRY.register('name')``, and reference it from a YAML:

    # datasets/my_dataset.py
    import torch.utils.data
    from src.registries import DATASET_REGISTRY

    @DATASET_REGISTRY.register('my_dataset')
    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, data_root, split='train', ...):
            ...
        def __getitem__(self, idx) -> dict:
            return {'image': ..., 'label': ..., ...}

    # configs/whatever.yaml
    dataset:
      name: my_dataset
      params: { data_root: /path, split: train }

The contract: subclass ``torch.utils.data.Dataset`` and return each
sample as a ``dict`` with at least ``'image'`` and ``'label'`` keys.
Add other keys (``'dsm'``, etc.) as your model needs them.  The trainer
treats the dict opaquely and passes it through to ``model.forward_train``.

A dataset class is also responsible for exposing the metadata the
framework needs for evaluation -- specifically ``num_classes``,
``labels``, ``text_prompts``, ``palette``, ``test_ids``, ``window_size``,
``stride_size``, and an ``eval_iter()`` that yields full-tile
``(image, dsm, gt_eroded)`` triples for sliding-window inference.  See
:class:`ISPRSDataset` for the concrete shape -- the generic trainer
calls these attributes by name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.registries import DATASET_REGISTRY, auto_discover

_failures = auto_discover(__name__, Path(__file__).parent)


def build_dataset(name: str, **kwargs: Any):
    """Construct a registered dataset by name."""
    return DATASET_REGISTRY.build(name, **kwargs)


def available_datasets() -> list:
    return DATASET_REGISTRY.names()


def import_failures() -> dict:
    return dict(_failures)


__all__ = ['DATASET_REGISTRY', 'build_dataset',
           'available_datasets', 'import_failures']
