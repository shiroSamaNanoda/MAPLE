"""Plug-in segmentation models.

Drop a file in this folder, decorate the top-level class with
``@MODEL_REGISTRY.register('name')``, and the framework discovers it
automatically:

    # model/my_net.py
    from src.registries import MODEL_REGISTRY
    from .base import BaseSegmentor

    @MODEL_REGISTRY.register('my_net')
    class MyNet(BaseSegmentor):
        ...

Point a YAML at it (``model: {name: my_net, params: {...}}``) and run
``python train.py --config configs/my_net.yaml``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.registries import MODEL_REGISTRY, auto_discover

from .base import BaseSegmentor


_failures = auto_discover(__name__, Path(__file__).parent, skip=('base',))


def build_model(name: str, **kwargs: Any) -> BaseSegmentor:
    """Construct a registered model by name."""
    return MODEL_REGISTRY.build(name, **kwargs)


def available_models() -> list:
    return MODEL_REGISTRY.names()


def import_failures() -> dict:
    """{module_name: exception} for files that failed to import."""
    return dict(_failures)


__all__ = ['BaseSegmentor', 'MODEL_REGISTRY',
           'build_model', 'available_models', 'import_failures']
