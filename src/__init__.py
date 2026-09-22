"""MAPLE framework core."""
from .config import ExperimentConfig, load_experiment_config, parse_cli_overrides
from .metrics import metrics
from .optimizers import build_optimizer, available_optimizers, OPTIMIZER_REGISTRY
from .registries import (
    DATASET_REGISTRY, LOSS_REGISTRY, MODEL_REGISTRY,
    OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY,
    Registry, auto_discover,
)
from .safr_logger import SAFRWeightLogger
from .schedulers import build_scheduler, available_schedulers, SCHEDULER_REGISTRY
from .trainer import evaluate, train
from .utils import (
    accuracy, batch_to_cuda,
    convert_from_color, convert_to_color,
    count_sliding_window, get_random_pos, grouper, sliding_window,
)

__all__ = [
    # core registries (re-exported for convenience)
    'MODEL_REGISTRY', 'DATASET_REGISTRY', 'LOSS_REGISTRY',
    'OPTIMIZER_REGISTRY', 'SCHEDULER_REGISTRY',
    'Registry', 'auto_discover',
    # config
    'ExperimentConfig', 'load_experiment_config', 'parse_cli_overrides',
    # builders
    'build_optimizer', 'available_optimizers',
    'build_scheduler', 'available_schedulers',
    # training
    'train', 'evaluate',
    'SAFRWeightLogger',
    # misc
    'metrics',
    'accuracy', 'batch_to_cuda',
    'convert_from_color', 'convert_to_color',
    'count_sliding_window', 'get_random_pos', 'grouper', 'sliding_window',
]
