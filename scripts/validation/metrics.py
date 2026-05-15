"""
metrics.py
==========
Evaluation metrics for binary gully segmentation.

Functions
---------
  pixel_metrics      — IoU, F1, precision, recall, accuracy from arrays
  roc_curve_data     — TPR/FPR over threshold sweep
  pr_curve_data      — Precision/Recall over threshold sweep
  average_precision  — Area under PR curve (AP / mAP)
  confusion_matrix   — 2×2 matrix
  multi_threshold    — Sweep multiple thresholds, return best F1
  print_report       — Human-readable metric table
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

EPS = 1e-6


# ---------------------------------------------------------------------------
# Core pixel-level metrics
# ---------------------------------------------------------------------------

def pixel_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute binary segmentation metrics at a given threshold.

    Args:
        probs     : (N,) or (H, W) float32 probability values in [0, 1].
        targets   : (N,) or (H, W) binary ground truth {0, 1}.
        threshold : Binarisation cutoff.

    Returns:
        Dict with keys: iou, f1, precision, recall, accuracy,
                        tp, fp, fn, tn, support (positive pixels).
    """
    probs = probs.ravel().astype(np.float32)
    targets = targets.ravel().astype(np.float32)

    binary = (probs >= threshold).astype(np.float32)

    tp = float((binary * targets).sum())
    fp = float((binary * (1.0 - targets)).sum())
    fn = float(((1.0 - binary) * targets).sum())
    tn = float(((1.0 - binary) * (1.0 - targets)).sum())

    precision = (tp + EPS) / (tp + fp + EPS)
    recall = (tp + EPS) / (tp + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    iou = (tp + EPS) / (tp + fp + fn + EPS)
    accuracy = (tp + tn + EPS) / (tp + tn + fp + fn + EPS)

    return {
        "threshold": threshold,
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "support": int(tp + fn),
    }


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def roc_curve_data(
    probs: np.ndarray,
    targets: np.ndarray,
    n_thresholds: int = 100,
) -> Dict[str, List[float]]:
    """
    Compute ROC curve (FPR vs TPR) over a sweep of thresholds.

    Returns dict with keys: thresholds, fpr, tpr, auc.
    """
    probs = probs.ravel().astype(np.float32)
    targets = targets.ravel().astype(np.float32)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    fpr_list, tpr_list = [], []
    for thr in thresholds:
        binary = (probs >= thr).astype(np.float32)
        tp = (binary * targets).sum()
        fp = (binary * (1 - targets)).sum()
        fn = ((1 - binary) * targets).sum()
        tn = ((1 - binary) * (1 - targets)).sum()
        tpr = (tp + EPS) / (tp + fn + EPS)
        fpr = (fp + EPS) / (fp + tn + EPS)
        tpr_list.append(float(tpr))
        fpr_list.append(float(fpr))

    # AUC via trapezoidal rule
    auc = float(np.trapz(tpr_list[::-1], fpr_list[::-1]))

    return {
        "thresholds": thresholds.tolist(),
        "fpr": fpr_list,
        "tpr": tpr_list,
        "auc": auc,
    }


# ---------------------------------------------------------------------------
# PR curve
# ---------------------------------------------------------------------------

def pr_curve_data(
    probs: np.ndarray,
    targets: np.ndarray,
    n_thresholds: int = 100,
) -> Dict[str, List[float]]:
    """
    Compute Precision-Recall curve and Average Precision (AP).

    Returns dict with keys: thresholds, precision, recall, ap.
    """
    probs = probs.ravel().astype(np.float32)
    targets = targets.ravel().astype(np.float32)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    prec_list, rec_list = [], []
    for thr in thresholds:
        binary = (probs >= thr).astype(np.float32)
        tp = (binary * targets).sum()
        fp = (binary * (1 - targets)).sum()
        fn = ((1 - binary) * targets).sum()
        prec = (tp + EPS) / (tp + fp + EPS)
        rec = (tp + EPS) / (tp + fn + EPS)
        prec_list.append(float(prec))
        rec_list.append(float(rec))

    # AP: area under PR curve (recall on x-axis)
    ap = float(np.trapz(prec_list[::-1], rec_list[::-1]))

    return {
        "thresholds": thresholds.tolist(),
        "precision": prec_list,
        "recall": rec_list,
        "ap": ap,
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Return a 2×2 confusion matrix:
        [[TN, FP],
         [FN, TP]]
    """
    probs = probs.ravel().astype(np.float32)
    targets = targets.ravel().astype(np.float32)
    binary = (probs >= threshold).astype(np.float32)
    tp = int((binary * targets).sum())
    fp = int((binary * (1 - targets)).sum())
    fn = int(((1 - binary) * targets).sum())
    tn = int(((1 - binary) * (1 - targets)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


# ---------------------------------------------------------------------------
# Best threshold via F1 sweep
# ---------------------------------------------------------------------------

def multi_threshold_sweep(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, object]:
    """
    Sweep thresholds and return per-threshold metrics + the best-F1 result.

    Args:
        probs      : Probability array.
        targets    : Binary ground truth.
        thresholds : List of thresholds to test. Defaults to 0.1…0.9.

    Returns:
        dict with 'results' (list of per-threshold dicts) and
        'best' (dict at threshold maximising F1).
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 1.0, 0.1)]

    results = [pixel_metrics(probs, targets, thr) for thr in thresholds]
    best = max(results, key=lambda r: r["f1"])
    return {"results": results, "best": best}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """Print a formatted metric table to the logger."""
    log.info(f"{'='*50}")
    log.info(f"  {title}")
    log.info(f"{'='*50}")
    for key in ["threshold", "iou", "f1", "precision", "recall", "accuracy"]:
        if key in metrics:
            log.info(f"  {key:<12}: {metrics[key]:.4f}")
    if "auc" in metrics:
        log.info(f"  {'ROC-AUC':<12}: {metrics['auc']:.4f}")
    if "ap" in metrics:
        log.info(f"  {'AP':<12}: {metrics['ap']:.4f}")
    log.info(f"{'='*50}")


def save_metrics_json(metrics: dict, out_path: Path) -> None:
    """Serialise a metrics dict to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    log.info(f"Metrics saved → {out_path}")
