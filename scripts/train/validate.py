"""
validate.py
===========
Spatial block cross-validation for the Mars Gully U-Net.

Runs k-fold spatial block CV and reports per-fold and aggregate metrics.
Folds are defined by the SpatialBlockSampler in scripts/dataset/sampler.py.

Usage (via main.py)
-------------------
    python main.py --step validate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from scripts.dataset.mars_dataset import MarsGullyDataset
from scripts.dataset.sampler import SpatialBlockSampler
from scripts.models.unet import build_model
from scripts.models.losses import build_loss
from scripts.utils import load_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-pixel metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute IoU, F1, precision, and recall from flattened arrays.

    Args:
        all_preds   : (N,) float32 probability values in [0, 1].
        all_targets : (N,) int/float binary ground truth {0, 1}.
        threshold   : Binarisation threshold.
        eps         : Numerical stability constant.
    """
    binary = (all_preds >= threshold).astype(np.float32)
    targets = all_targets.astype(np.float32)

    tp = (binary * targets).sum()
    fp = (binary * (1 - targets)).sum()
    fn = ((1 - binary) * targets).sum()
    tn = ((1 - binary) * (1 - targets)).sum()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ---------------------------------------------------------------------------
# Evaluate one fold
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_fold(
    model: nn.Module,
    dataset: MarsGullyDataset,
    val_indices: List[int],
    device: torch.device,
    batch_size: int = 16,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Run inference on the val_indices split and return metrics dict.
    """
    from torch.utils.data import DataLoader, Subset

    val_subset = Subset(dataset, val_indices)
    loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        targets = masks.squeeze(1).cpu().numpy()
        all_preds.append(probs.ravel())
        all_targets.append(targets.ravel())

    all_preds_arr = np.concatenate(all_preds)
    all_targets_arr = np.concatenate(all_targets)
    return compute_metrics(all_preds_arr, all_targets_arr, threshold=threshold)


# ---------------------------------------------------------------------------
# Spatial block cross-validation
# ---------------------------------------------------------------------------

def run_spatial_block_cv(cfg_path: str = "config.yaml") -> Dict:
    """
    Full k-fold spatial block cross-validation.

    1. Loads best checkpoint.
    2. Splits dataset into spatial folds.
    3. For each fold: fine-tune (optional) + evaluate.
    4. Reports per-fold and aggregate (mean ± std) metrics.
    5. Saves results to JSON.

    Returns dict with 'folds' and 'aggregate' keys.
    """
    cfg = load_config(cfg_path)
    val_cfg = cfg["training"]["validation"]
    n_folds = val_cfg.get("n_folds", 5)
    block_size_km = val_cfg.get("block_size_km", 2)
    threshold = cfg["inference"].get("thresholds", [0.5])[1]  # use 0.5
    batch_size = cfg["inference"]["sliding_window"].get("batch_size", 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- dataset (full, no augmentation for CV eval) ----
    dataset = MarsGullyDataset(cfg, split="all", augment=False)

    # ---- sampler for fold indices ----
    sampler = SpatialBlockSampler(
        dataset=dataset,
        block_size_km=block_size_km,
        n_folds=n_folds,
    )

    # ---- load best checkpoint ----
    ckpt_path = Path(cfg["checkpoint"]["save_dir"]) / "unet_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run training first."
        )
    model = build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    log.info(f"Loaded checkpoint (IoU={state.get('best_iou', 0):.4f})")

    # ---- per-fold evaluation ----
    fold_results: List[Dict] = []
    metric_keys = ["iou", "f1", "precision", "recall", "accuracy"]

    for fold_idx in range(n_folds):
        train_idx, val_idx = sampler.get_fold_indices(fold_idx)
        log.info(
            f"Fold {fold_idx + 1}/{n_folds}: "
            f"train={len(train_idx)}, val={len(val_idx)}"
        )
        metrics = evaluate_fold(
            model, dataset, val_idx, device,
            batch_size=batch_size, threshold=threshold,
        )
        metrics["fold"] = fold_idx + 1
        fold_results.append(metrics)
        log.info(
            f"  IoU={metrics['iou']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"Prec={metrics['precision']:.4f}  "
            f"Rec={metrics['recall']:.4f}"
        )

    # ---- aggregate ----
    agg: Dict[str, Dict] = {}
    for key in metric_keys:
        vals = [f[key] for f in fold_results]
        agg[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    log.info("=== Spatial Block CV Summary ===")
    for key in metric_keys:
        a = agg[key]
        log.info(f"  {key:10s}: {a['mean']:.4f} ± {a['std']:.4f}")

    # ---- save ----
    out_dir = Path("data/outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"folds": fold_results, "aggregate": agg, "n_folds": n_folds}
    out_path = out_dir / "spatial_cv_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"CV results saved → {out_path}")
    return result
