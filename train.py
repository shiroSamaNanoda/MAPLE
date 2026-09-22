"""Generic YAML-driven training entry point.

    python train.py --config configs/maple_vaihingen.yaml
    python train.py --config configs/maple_vaihingen.yaml \\
        --set training.batch_size=2 optimizer.base_lr=0.005

Every pluggable component (model, dataset, loss, optimizer, scheduler)
is built from its own registry; this file orchestrates them but knows
nothing about any specific one.
"""
import argparse
import os
from pathlib import Path

from torch.utils.data import DataLoader

# Importing these packages triggers their auto-discovery, populating
# the registries before we look anything up.
import model       # noqa: F401   (MODEL_REGISTRY)
import datasets    # noqa: F401   (DATASET_REGISTRY)
import losses      # noqa: F401   (LOSS_REGISTRY)

from model    import build_model,    available_models
from datasets import build_dataset,  available_datasets
from losses   import build_loss,     available_losses
from src import (
    build_optimizer, build_scheduler,
    available_optimizers, available_schedulers,
    SAFRWeightLogger, load_experiment_config, parse_cli_overrides, train,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a registered model.")
    p.add_argument('--config', required=True,
                   help='Path to a YAML experiment config.')
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VAL',
                   help='Override config entries by dotted path, e.g. '
                        '--set training.batch_size=2 optimizer.base_lr=0.005')
    p.add_argument('--gpu', default='0')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    exp = load_experiment_config(args.config, parse_cli_overrides(args.set))

    print(f"[registries] models     = {available_models()}")
    print(f"[registries] datasets   = {available_datasets()}")
    print(f"[registries] losses     = {available_losses()}")
    print(f"[registries] optimizers = {available_optimizers()}")
    print(f"[registries] schedulers = {available_schedulers()}")

    # ----------------------------------------------------------------
    #  Dataset (train split) + eval handle
    # ----------------------------------------------------------------
    ds_params = dict(exp.dataset.get('params', {}))
    train_ds = build_dataset(exp.dataset['name'], split='train', **ds_params)
    eval_ds  = build_dataset(exp.dataset['name'], split='test',
                             augmentation=False, **ds_params)
    print(f"[dataset] {train_ds.name} num_classes={train_ds.num_classes}  "
          f"window={train_ds.window_size}  stride={train_ds.stride_size}")
    print(f"[dataset] train_ids={len(train_ds.data_files)}  "
          f"test_ids={len(eval_ds.test_ids)}")

    # ----------------------------------------------------------------
    #  Model
    # ----------------------------------------------------------------
    model_params = dict(exp.model.get('params', {}))
    model_params.setdefault('num_classes', train_ds.num_classes)
    net = build_model(exp.model['name'], **model_params).cuda()

    # ----------------------------------------------------------------
    #  Loss (built from registry, handed to the model)
    # ----------------------------------------------------------------
    if exp.loss:
        loss_fn = build_loss(exp.loss['name'], **exp.loss.get('params', {})).cuda()
        print(f"[loss] {exp.loss['name']}({exp.loss.get('params', {})})")
    else:
        loss_fn = None
        print("[loss] (default cross_entropy via BaseSegmentor.setup_loss)")
    net.setup_loss(loss_fn)

    s = net.param_summary()
    print(f"[model] trainable / total parameters: "
          f"{s['trainable']/1e6:.2f}M / {s['total']/1e6:.2f}M")

    # ----------------------------------------------------------------
    #  Optimizer (param groups owned by the model)
    # ----------------------------------------------------------------
    opt_cfg = exp.optimizer
    base_lr = opt_cfg.get('base_lr', 1e-2)
    opt_extras = dict(opt_cfg.get('param_group_extras', {}))
    param_groups = net.param_groups(base_lr=base_lr, **opt_extras)
    optimizer = build_optimizer(opt_cfg.get('name', 'sgd'),
                                param_groups,
                                **opt_cfg.get('params', {}))
    print(f"[optim] {opt_cfg.get('name', 'sgd')}  base_lr={base_lr}  "
          f"extras={opt_extras}")

    # ----------------------------------------------------------------
    #  Scheduler
    # ----------------------------------------------------------------
    sch_cfg = exp.scheduler
    scheduler = build_scheduler(sch_cfg.get('name'),
                                optimizer,
                                **sch_cfg.get('params', {}))
    if scheduler is not None:
        print(f"[sched] {sch_cfg.get('name')}")

    # ----------------------------------------------------------------
    #  Data loader
    # ----------------------------------------------------------------
    tr = exp.training
    train_loader = DataLoader(train_ds,
                              batch_size=tr.get('batch_size', 4),
                              num_workers=tr.get('num_workers', 4),
                              shuffle=False, pin_memory=True)

    # ----------------------------------------------------------------
    #  Optional per-iter logger (MAPLE wires SAFR weights here)
    # ----------------------------------------------------------------
    iter_logger = None
    ex = exp.extras
    if ex.get('safr_logger', False):
        iter_logger = SAFRWeightLogger(
            train_ds.num_classes, train_ds.labels,
            log_interval=ex.get('safr_log_interval', 10))

    # ----------------------------------------------------------------
    #  Train
    # ----------------------------------------------------------------
    best_miou = train(
        model=net,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_dataset=eval_ds,
        epochs=tr.get('epochs', 50),
        save_dir=tr.get('save_dir', './weights'),
        save_every=tr.get('save_every', 1),
        scheduler=scheduler,
        log_interval=tr.get('log_interval', 100),
        iter_logger=iter_logger,
    )

    if iter_logger is not None:
        vis_dir = ex.get('vis_dir', './vis_safr')
        Path(vis_dir).mkdir(parents=True, exist_ok=True)
        iter_logger.save(vis_dir)
        iter_logger.plot(vis_dir)
    print(f"[done] best mIoU = {best_miou:.4f}")


if __name__ == '__main__':
    main()
