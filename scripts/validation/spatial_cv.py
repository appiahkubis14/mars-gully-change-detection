"""
spatial_cv.py
=============
Spatial block cross-validation runner.

Splits the dataset into 2 km × 2 km geographic blocks and assigns
them to k folds such that no spatial autocorrelation leaks between
the training and validation sets.

Workflow
--------
1. Assign each patch to a 2 km block (via SpatialBlockSampler).
2. For each fold:
   a. Load the best checkpoint.
   b. Run inference on validation patches.
   c. Compute IoU, F1, precision, recall.
3. Aggregate and save results.

Usage
-----
    python main.py --step validate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scripts.dataset.mars_dataset import MarsGullyDataset
from scripts.dataset.sampler import SpatialBlockSampler
from scripts.models.unet import build_model
from scripts.validation.metrics import (
    pixel_metrics,
    roc_curve_data,
    pr_curve_data,
    print_report,
    save_metrics_json,
)
from scripts.utils import load_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference on a fold
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_fold(
    model: torch.nn.Module,
    dataset: MarsGullyDataset,
    indices: List[int],
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a subset of the dataset.

    Returns:
        probs   : (N_pixels,) float32 probability array.
        targets : (N_pixels,) float32 ground-truth array.
    """
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    model.eval()
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().ravel()
        tgts = masks.squeeze(1).cpu().numpy().ravel()
        all_probs.append(probs)
        all_targets.append(tgts)

    return (
        np.concatenate(all_probs).astype(np.float32),
        np.concatenate(all_targets).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Main CV runner
# ---------------------------------------------------------------------------

def run_spatial_cv(cfg_path: str = "config.yaml") -> Dict:
    """
    Full k-fold spatial block cross-validation.

    Returns dict with 'folds' and 'aggregate' keys.
    """
    cfg = load_config(cfg_path)
    val_cfg = cfg["training"]["validation"]
    n_folds = val_cfg.get("n_folds", 5)
    block_size_km = val_cfg.get("block_size_km", 2)
    batch_size = cfg["inference"]["sliding_window"].get("batch_size", 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- dataset (full, no augmentation) ----
    dataset = MarsGullyDataset(cfg, split="all", augment=False)
    sampler = SpatialBlockSampler(
        dataset=dataset,
        block_size_km=block_size_km,
        n_folds=n_folds,
    )

    # ---- load best model ----
    ckpt_dir = Path(cfg["checkpoint"]["save_dir"])
    best_ckpt = ckpt_dir / "unet_best.pth"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {best_ckpt}. Run --step train first.")

    model = build_model(cfg).to(device)
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    log.info(f"Loaded checkpoint (best IoU = {state.get('best_iou', 0):.4f})")

    # ---- per-fold loop ----
    metric_keys = ["iou", "f1", "precision", "recall", "accuracy"]
    fold_results: List[Dict] = []

    for fold_idx in range(n_folds):
        train_idx, val_idx = sampler.get_fold_indices(fold_idx)
        log.info(
            f"Fold {fold_idx + 1}/{n_folds} | "
            f"train={len(train_idx)} val={len(val_idx)}"
        )

        if len(val_idx) == 0:
            log.warning(f"Fold {fold_idx + 1} has no validation samples — skipping.")
            continue

        probs, targets = infer_fold(model, dataset, val_idx, device, batch_size)

        # metrics at 0.5
        m = pixel_metrics(probs, targets, threshold=0.5)
        # ROC + PR
        roc = roc_curve_data(probs, targets, n_thresholds=50)
        pr = pr_curve_data(probs, targets, n_thresholds=50)
        m["roc_auc"] = roc["auc"]
        m["ap"] = pr["ap"]
        m["fold"] = fold_idx + 1

        fold_results.append(m)
        log.info(
            f"  IoU={m['iou']:.4f}  F1={m['f1']:.4f}  "
            f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  "
            f"ROC-AUC={m['roc_auc']:.4f}"
        )

    if not fold_results:
        log.error("No valid folds produced results.")
        return {}

    # ---- aggregate ----
    agg: Dict[str, dict] = {}
    for key in metric_keys + ["roc_auc", "ap"]:
        vals = [f[key] for f in fold_results if key in f]
        if vals:
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    log.info("=== Spatial Block CV — Aggregate ===")
    for key in metric_keys + ["roc_auc", "ap"]:
        if key in agg:
            a = agg[key]
            log.info(f"  {key:<12}: {a['mean']:.4f} ± {a['std']:.4f}")

    # ---- save ----
    out_dir = Path("data/outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "spatial_block_cv",
        "n_folds": n_folds,
        "block_size_km": block_size_km,
        "folds": fold_results,
        "aggregate": agg,
    }
    save_metrics_json(result, out_dir / "spatial_cv.json")
    return result
