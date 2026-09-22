"""Base class every plug-in model inherits from.

The contract is small on purpose:

    forward_train(batch) -> {'loss': scalar, 'logits': [B,C,H,W],
                             'aux_losses': {name: scalar, ...}}
    forward_test (batch) -> logits [B,C,H,W]
    param_groups (base_lr, **extras) -> list passable to torch.optim.*
    on_train_iter_end(iteration, logger) -> None        (optional hook)

``batch`` is a ``dict``, not a fixed positional tuple, so different
models can declare different inputs without changing the trainer.
Common keys populated by the framework:

    'image'         : RGB tensor   [B, 3, H, W]   (always present)
    'label'         : long tensor  [B, H, W]      (training/eval)
    'dsm'           : DSM tensor   [B, H, W]      (multimodal datasets only)
    'text_prompts'  : list[str]    (length = num_classes; injected by trainer)

The segmentation loss is supplied externally via :meth:`setup_loss`
(the trainer builds it from the YAML's ``loss:`` section and passes it
in).  Auxiliary model-internal losses -- like MAPLE's SAFR -- live
inside the model and are summed in ``forward_train``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class BaseSegmentor(nn.Module):
    """Abstract-ish base class for registered segmentation models."""

    DEFAULT_IGNORE_INDEX: int = 255

    def __init__(self) -> None:
        super().__init__()
        # Configured by ``setup_loss`` once the trainer knows which loss
        # the user picked in the YAML.
        self.seg_loss_fn: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    #  Loss configuration
    # ------------------------------------------------------------------

    def setup_loss(self, loss_fn: Optional[nn.Module] = None) -> None:
        """Attach the segmentation loss the trainer built from the YAML.

        Called once after the model is moved to GPU.  ``loss_fn=None``
        falls back to a plain CrossEntropyLoss so the class is usable
        outside the trainer too.

        Subclasses with extra losses (e.g. a deep-supervision auxiliary
        head) may override, but should always call ``super().setup_loss``
        so ``self.seg_loss_fn`` is set.
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.DEFAULT_IGNORE_INDEX)
        self.seg_loss_fn = loss_fn

    # ------------------------------------------------------------------
    #  The model interface used by the trainer
    # ------------------------------------------------------------------

    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Default: pass ``batch['image']`` through ``self.forward``.

        Override this if your model needs different inputs but plain CE
        on logits is fine.  If the loss path itself is exotic, override
        ``forward_train`` instead.
        """
        return self(batch['image'])

    def forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Default training path: predict -> seg loss against ``batch['label']``."""
        if self.seg_loss_fn is None:                    # pragma: no cover
            raise RuntimeError(
                "setup_loss() must be called before training; the trainer "
                "does this automatically.")
        logits = self.predict(batch)
        loss = self.seg_loss_fn(logits, batch['label'])
        return {'loss': loss, 'logits': logits, 'aux_losses': {}}

    @torch.no_grad()
    def forward_test(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Default eval path: return logits only."""
        return self.predict(batch)

    # ------------------------------------------------------------------
    #  Optimizer parameter groups
    # ------------------------------------------------------------------

    def param_groups(self, base_lr: float, **extras: Any) -> List[dict]:
        """Default: one param group with all trainable params at ``base_lr``.

        Override if you want per-component LR scaling (e.g. MAPLE keeps
        the VLP encoder at base_lr * vlp_lr_scale).  ``extras`` carries
        whatever was set under ``optimizer.param_group_extras`` in the YAML.
        """
        return [{
            'params': [p for p in self.parameters() if p.requires_grad],
            'lr': base_lr,
        }]

    # ------------------------------------------------------------------
    #  Optional per-iteration hook (e.g. for SAFR weight logging)
    # ------------------------------------------------------------------

    def on_train_iter_end(self, iteration: int,
                          logger: Optional[Any] = None) -> None:
        """Called by the trainer after each ``optimizer.step()``.  Default no-op."""
        pass

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------

    def param_summary(self) -> Dict[str, int]:
        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'trainable': n_train, 'total': n_total}
