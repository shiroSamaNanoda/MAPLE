"""Central registries for the entire framework.

All pluggable components -- models, datasets, losses, optimizers,
schedulers -- live in registries defined here.  Each plug-in package
(``model/``, ``datasets/``, ``losses/``) imports the relevant singleton
from this module and uses ``@REGISTRY.register('name')`` to make its
classes discoverable.

Why one file?  So the plug-in packages share infrastructure but have
no inter-dependencies: ``losses/`` does not need to import ``model/``,
``datasets/`` does not need to import ``losses/``, etc.  They all only
reach back to ``src.registries``.
"""
from __future__ import annotations

import importlib
import pkgutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type


# ============================================================
#  Generic registry
# ============================================================

class Registry:
    """Map ``name -> class`` (or factory callable) with a decorator API.

    The same class is used for every registry in the framework; the
    ``scope`` argument only affects error messages and ``repr``.
    """

    def __init__(self, scope: str) -> None:
        self.scope = scope
        self._table: Dict[str, Any] = {}

    # ----- registration -------------------------------------------------

    def register(self, name: Optional[str] = None, *, override: bool = False
                 ) -> Callable[[Any], Any]:
        """Decorator.  ``name`` defaults to the class's own ``__name__``."""
        def decorator(obj: Any) -> Any:
            key = name or getattr(obj, '__name__', None)
            if key is None:
                raise ValueError(
                    f"Registering {obj!r} requires an explicit name.")
            if key in self._table and not override:
                raise KeyError(
                    f"{key!r} is already registered in {self.scope!r}; "
                    f"pass override=True to replace it.")
            self._table[key] = obj
            return obj
        return decorator

    def add(self, name: str, obj: Any, *, override: bool = False) -> Any:
        """Imperative form of ``register`` for objects you can't decorate
        (e.g. third-party classes like ``torch.optim.SGD``)."""
        return self.register(name, override=override)(obj)

    # ----- lookup -------------------------------------------------------

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Construct (or call) the entry registered under ``name``."""
        return self.get(name)(*args, **kwargs)

    def get(self, name: str) -> Any:
        if name not in self._table:
            raise KeyError(
                f"{name!r} not found in {self.scope!r}.  "
                f"Available: {self.names()}")
        return self._table[name]

    def names(self) -> List[str]:
        return sorted(self._table)

    def __contains__(self, name: str) -> bool:
        return name in self._table

    def __repr__(self) -> str:                       # pragma: no cover
        return f"Registry(scope={self.scope!r}, items={self.names()})"


# ============================================================
#  Public singletons
# ============================================================

MODEL_REGISTRY     = Registry('models')
DATASET_REGISTRY   = Registry('datasets')
LOSS_REGISTRY      = Registry('losses')
OPTIMIZER_REGISTRY = Registry('optimizers')
SCHEDULER_REGISTRY = Registry('schedulers')


# ============================================================
#  Auto-discovery shared by every plug-in package
# ============================================================

def auto_discover(package_name: str, package_path: Path,
                  skip: tuple = ()) -> Dict[str, Exception]:
    """Import every ``.py`` sibling in ``package_path``.

    Designed to be called from a plug-in package's ``__init__.py``:

        from src.registries import auto_discover
        _failures = auto_discover(__name__, Path(__file__).parent,
                                  skip=('base',))

    Any module that raises during import is recorded and a warning is
    emitted; the import itself does **not** propagate, so a missing
    third-party dep (e.g. SAM3 for MAPLE) doesn't take down sibling
    plug-ins that are independently usable.

    Returns:
        Dict ``{module_name: exception}`` of failures (empty on full success).
    """
    failures: Dict[str, Exception] = {}
    skip_set = set(skip)
    for _finder, name, _ispkg in pkgutil.iter_modules([str(package_path)]):
        if name.startswith('_') or name in skip_set:
            continue
        try:
            importlib.import_module(f'.{name}', package=package_name)
        except Exception as e:                       # pragma: no cover
            failures[name] = e
            warnings.warn(
                f"{package_name}.{name} failed to import "
                f"({type(e).__name__}: {e}); it will not appear in "
                f"the registry.  Other entries remain usable.",
                ImportWarning, stacklevel=2)
    return failures
