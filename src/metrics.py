"""Confusion-matrix-based evaluation metrics (mF1, mIoU, Kappa)."""
from typing import List

import numpy as np
from sklearn.metrics import confusion_matrix


def _summarise(cm: np.ndarray, label_values: List[str], drop_last: bool) -> dict:
    """Compute OA, per-class accuracy/F1/IoU, kappa from a confusion matrix.

    If `drop_last` is True the final class (typically `clutter` in ISPRS)
    is excluded from mF1 / mIoU per the standard ISPRS protocol.
    """
    n = len(label_values)
    total = cm.sum()
    correct = np.trace(cm)
    oa = 100.0 * correct / total

    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)

    f1 = np.zeros(n)
    for i in range(n):
        denom = cm[i, :].sum() + cm[:, i].sum()
        f1[i] = 2.0 * cm[i, i] / denom if denom > 0 else np.nan

    pa = correct / total
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total * total)
    kappa = (pa - pe) / (1 - pe) if (1 - pe) else 0.0

    iou_denom = cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(cm) / iou_denom

    sl = slice(0, n - 1) if drop_last else slice(0, n)
    return {
        'cm': cm,
        'OA': oa,
        'per_class_acc': per_class_acc,
        'F1': f1,
        'mF1': float(np.nanmean(f1[sl])),
        'IoU': iou,
        'mIoU': float(np.nanmean(iou[sl])),
        'kappa': kappa,
    }


def metrics(predictions: np.ndarray, gts: np.ndarray,
            label_values: List[str], drop_last: bool = True,
            verbose: bool = True) -> float:
    """Evaluate flattened predictions vs ground truth.

    Args:
        drop_last: True (default) for ISPRS — exclude `clutter`.
                   False for Hunan / general use.
    Returns:
        mIoU as a Python float.
    """
    cm = confusion_matrix(gts, predictions, labels=range(len(label_values)))
    s = _summarise(cm, label_values, drop_last)

    if verbose:
        print("Confusion matrix:")
        print(cm)
        print(f"{int(cm.sum())} pixels processed")
        print(f"Overall accuracy: {s['OA']:.2f}")
        print("Per-class accuracy:")
        for name, v in zip(label_values, s['per_class_acc']):
            print(f"  {name}: {v:.4f}")
        print("F1 score:")
        for name, v in zip(label_values, s['F1']):
            print(f"  {name}: {v:.4f}")
        print(f"mF1: {s['mF1']:.4f}   "
              f"mIoU: {s['mIoU']:.4f}   "
              f"Kappa: {s['kappa']:.4f}")
        print("Per-class IoU:", s['IoU'])

    return s['mIoU']
