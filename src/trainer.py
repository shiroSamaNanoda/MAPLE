"""Generic train / evaluate loops.

The trainer talks to plug-in components through stable interfaces:

  - Model:   :class:`model.BaseSegmentor`'s 4 hooks
              (forward_train, forward_test, param_groups, on_train_iter_end)
  - Dataset: a ``torch.utils.data.Dataset`` whose ``__getitem__`` returns
              a dict batch, plus a handful of metadata attributes
              (``num_classes``, ``labels``, ``text_prompts``, ``palette``,
              ``window_size``, ``stride_size``, ``drop_last_class_in_eval``)
              and an ``eval_iter()`` generator for sliding-window eval.
  - Loss:    any ``nn.Module`` with signature ``forward(logits, target)``.

The trainer does not import MAPLE / SAM3 / RemoteCLIP and has no
knowledge of DSM, text prompts, or SAFR.  ``batch['text_prompts']`` is
injected here per-batch from the dataset so models that consume it
don't have to dig into the config.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.base import BaseSegmentor

from .metrics import metrics
from .utils import accuracy, batch_to_cuda, grouper, sliding_window


# ============================================================
#  Sliding-window evaluation
# ============================================================

@torch.no_grad()
def evaluate(model: BaseSegmentor, dataset: Any,
             stride: Optional[int] = None, batch_size: int = 4,
             return_predictions: bool = False, verbose: bool = True):
    """Sliding-window evaluation using ``dataset.eval_iter()``.

    The dataset is queried for the test tiles, text prompts, and stride.
    Anything the model consumes (e.g. DSM) is read here and threaded
    through the batch dict.
    """
    model.eval()
    stride = stride if stride is not None else dataset.stride_size
    window = dataset.window_size
    num_classes = dataset.num_classes
    drop_last_class = dataset.drop_last_class_in_eval
    text_prompts = dataset.text_prompts
    labels = dataset.labels

    all_preds, all_gts, all_ids = [], [], []

    for img_id, img, dsm, gt_eroded in dataset.eval_iter():
        prob = np.zeros(img.shape[:2] + (num_classes,), dtype=np.float32)
        coords_iter = sliding_window(img, step=stride, window_size=window)

        for coords in grouper(batch_size, coords_iter):
            img_patches = np.stack(
                [img[x:x + w, y:y + h].transpose(2, 0, 1) for x, y, w, h in coords])
            dsm_patches = np.stack(
                [dsm[x:x + w, y:y + h] for x, y, w, h in coords])

            batch = {
                'image': torch.from_numpy(img_patches).cuda(),
                'dsm':   torch.from_numpy(dsm_patches).cuda(),
                'text_prompts': text_prompts,
            }
            logits = model.forward_test(batch).cpu().numpy()

            for out, (x, y, w, h) in zip(logits, coords):
                prob[x:x + w, y:y + h] += out.transpose(1, 2, 0)

        all_preds.append(prob.argmax(axis=-1))
        all_gts.append(gt_eroded)
        all_ids.append(img_id)

    miou = metrics(
        np.concatenate([p.ravel() for p in all_preds]),
        np.concatenate([g.ravel() for g in all_gts]),
        labels, drop_last=drop_last_class, verbose=verbose,
    )

    if return_predictions:
        return miou, all_preds, all_gts, all_ids
    return miou


# ============================================================
#  Training loop
# ============================================================

def train(model: BaseSegmentor,
          optimizer: torch.optim.Optimizer,
          train_loader: DataLoader,
          eval_dataset: Any,
          epochs: int, save_dir: str, save_every: int = 1,
          scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
          log_interval: int = 100,
          iter_logger: Optional[Any] = None) -> float:
    """Generic training loop.

    Args:
        model:        any :class:`BaseSegmentor`.
        train_loader: a DataLoader over a training dataset (returns dict batches).
        eval_dataset: a dataset (may be the same one, in 'test' split) that
                      exposes ``eval_iter()`` and the task metadata.
        save_dir:     where checkpoints are written, under ``<save_dir>/<name>/``.
        save_every:   run eval + checkpoint every N epochs.
        iter_logger:  optional object forwarded verbatim to
                      ``model.on_train_iter_end(iteration, iter_logger)``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    text_prompts = eval_dataset.text_prompts
    dataset_name = getattr(eval_dataset, 'name', 'run')

    best_miou = 0.0
    iteration = 0
    recent_losses: list = []
    EMA_WIN = 100

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch_to_cuda(batch)
            batch['text_prompts'] = text_prompts     # constant per dataset

            optimizer.zero_grad()
            out = model.forward_train(batch)
            loss = out['loss']
            loss.backward()
            optimizer.step()

            model.on_train_iter_end(iteration, iter_logger)

            recent_losses.append(loss.item())
            if len(recent_losses) > EMA_WIN:
                recent_losses.pop(0)

            if iteration % log_interval == 0:
                aux = out.get('aux_losses') or {}
                aux_str = ' '.join(f"{k}={v.item():.4f}" for k, v in aux.items())
                logits = out['logits']
                pred = logits.detach().argmax(dim=1)[0].cpu().numpy()
                gt   = batch['label'][0].cpu().numpy()
                print(f"epoch {epoch}/{epochs} "
                      f"[{batch_idx}/{len(train_loader)}]  "
                      f"loss={loss.item():.4f}  "
                      f"{aux_str}  "
                      f"acc={accuracy(pred, gt):.2f}  "
                      f"loss100={np.mean(recent_losses):.4f}")

            iteration += 1
            del loss, out

        if scheduler is not None:
            scheduler.step()

        if epoch % save_every == 0:
            print(f"[epoch {epoch}] train time = {time.time() - t0:.1f}s")
            miou = evaluate(model, eval_dataset)

            ckpt_dir = save_dir / dataset_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            if epoch == epochs:
                torch.save(model.state_dict(),
                           ckpt_dir / f"epoch{epoch}_{miou:.4f}_last.pth")
            if miou > best_miou:
                best_miou = miou
                path = ckpt_dir / f"epoch{epoch}_{miou:.4f}_best.pth"
                torch.save(model.state_dict(), path)
                print(f"  -> best model saved to {path}")

    print(f"Training finished. Best mIoU: {best_miou:.4f}")
    return best_miou
