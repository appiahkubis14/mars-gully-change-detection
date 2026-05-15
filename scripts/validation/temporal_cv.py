"""
temporal_cv.py
==============
Temporal hold-out validation.

Strategy
--------
  Train on images from 2005–2010  →  Test on images from 2015–2020

This mirrors realistic deployment where the model is trained on
historical imagery and evaluated on future acquisitions it has
never seen — a stronger test of generalisation than random splits.

If date metadata is unavailable in the dataset, the split is
approximated by fraction (first 60 % = train, last 40 % = test).

Usage
-----
    python main.py --step validate --temporal
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scripts.dataset.mars_dataset import MarsGullyDataset
from scripts.models.unet import build_model
from scripts.validation.metrics import (
    pixel_metrics,
    roc_curve_data,
    pr_curve_data,
    multi_threshold_sweep,
    print_report,
    save_metrics_json,
)
from scripts.utils import load_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Date-based split
# ---------------------------------------------------------------------------

def _parse_year(path_str: str) -> Optional[int]:
    """Extract a 4-digit year from a file path string."""
    match = re.search(r"(200[0-9]|201[0-9]|202[0-9])", path_str)
    return int(match.group(1)) if match else None


def temporal_split(
    dataset: MarsGullyDataset,
    train_years: Tuple[int, int] = (2005, 2010),
    test_years: Tuple[int, int] = (2015, 2020),
    fallback_fraction: float = 0.6,
) -> Tuple[List[int], List[int]]:
    """
    Split dataset indices into train and test by acquisition year.

    Args:
        dataset          : MarsGullyDataset with .patch_paths attribute.
        train_years      : Inclusive (start, end) year range for training.
        test_years       : Inclusive (start, end) year range for testing.
        fallback_fraction: Fraction to use as train if date metadata missing.

    Returns:
        train_indices, test_indices
    """
    train_idx, test_idx, unknown_idx = [], [], []

    patch_paths = getattr(dataset, "patch_paths", None)
    if patch_paths is None:
        log.warning("Dataset has no patch_paths attribute; using index-based split.")
        patch_paths = [str(i) for i in range(len(dataset))]

    for i, path in enumerate(patch_paths):
        year = _parse_year(str(path))
        if year is None:
            unknown_idx.append(i)
        elif train_years[0] <= year <= train_years[1]:
            train_idx.append(i)
        elif test_years[0] <= year <= test_years[1]:
            test_idx.append(i)
        # years in between are excluded (gap period)

    if not train_idx and not test_idx:
        # fallback: split by position
        n = len(dataset)
        n_train = int(n * fallback_fraction)
        train_idx = list(range(n_train))
        test_idx = list(range(n_train, n))
        log.warning(
            f"No year metadata found. Falling back to positional split: "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )
    elif unknown_idx:
        # assign unknown to train
        train_idx.extend(unknown_idx)
        log.warning(f"{len(unknown_idx)} patches with unknown dates assigned to train.")

    log.info(
        f"Temporal split: train={len(train_idx)} "
        f"({train_years[0]}–{train_years[1]}), "
        f"test={len(test_idx)} ({test_years[0]}–{test_years[1]})"
    )
    return sorted(train_idx), sorted(test_idx)


# ---------------------------------------------------------------------------
# Inference on subset
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_subset(
    model: torch.nn.Module,
    dataset: MarsGullyDataset,
    indices: List[int],
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model inference on a list of dataset indices."""
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    model.eval()
    all_probs, all_targets = [], []

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
# Main entry point
# ---------------------------------------------------------------------------

def run_temporal_cv(cfg_path: str = "config.yaml") -> Dict:
    """
    Train-on-early, test-on-late temporal hold-out evaluation.

    Returns dict with train/test metrics.
    """
    cfg = load_config(cfg_path)
    val_cfg = cfg["training"]["validation"]
    train_years = tuple(val_cfg.get("temporal_train_years", [2005, 2010]))
    test_years = tuple(val_cfg.get("temporal_test_years", [2015, 2020]))
    batch_size = cfg["inference"]["sliding_window"].get("batch_size", 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- full dataset (no augmentation) ----
    dataset = MarsGullyDataset(cfg, split="all", augment=False)

    train_idx, test_idx = temporal_split(
        dataset,
        train_years=train_years,
        test_years=test_years,
    )

    if not test_idx:
        log.error("No test samples found for temporal CV.")
        return {}

    # ---- load best checkpoint ----
    ckpt_dir = Path(cfg["checkpoint"]["save_dir"])
    best_ckpt = ckpt_dir / "unet_best.pth"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {best_ckpt}. Run --step train first.")

    model = build_model(cfg).to(device)
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    log.info(f"Checkpoint loaded (best IoU={state.get('best_iou', 0):.4f})")

    # ---- evaluate on train split (sanity check) ----
    log.info("Evaluating on TRAIN split (in-sample) …")
    train_probs, train_targets = infer_subset(model, dataset, train_idx, device, batch_size)
    train_metrics = pixel_metrics(train_probs, train_targets, threshold=0.5)
    train_metrics["roc_auc"] = roc_curve_data(train_probs, train_targets, 50)["auc"]
    train_metrics["ap"] = pr_curve_data(train_probs, train_targets, 50)["ap"]
    print_report(train_metrics, title="Train (in-sample) Metrics")

    # ---- evaluate on test split (temporal hold-out) ----
    log.info("Evaluating on TEST split (temporal hold-out) …")
    test_probs, test_targets = infer_subset(model, dataset, test_idx, device, batch_size)
    test_metrics = pixel_metrics(test_probs, test_targets, threshold=0.5)
    test_metrics["roc_auc"] = roc_curve_data(test_probs, test_targets, 50)["auc"]
    test_metrics["ap"] = pr_curve_data(test_probs, test_targets, 50)["ap"]
    print_report(test_metrics, title="Test (temporal hold-out) Metrics")

    # ---- best threshold on test ----
    sweep = multi_threshold_sweep(test_probs, test_targets)
    best = sweep["best"]
    log.info(
        f"Best threshold on test: {best['threshold']:.2f}  "
        f"(F1={best['f1']:.4f}, IoU={best['iou']:.4f})"
    )

    # ---- generalisation gap ----
    gap = {
        key: round(train_metrics.get(key, 0) - test_metrics.get(key, 0), 4)
        for key in ["iou", "f1", "precision", "recall"]
    }
    log.info(f"Generalisation gap (train − test): {gap}")

    result = {
        "method": "temporal_holdout",
        "train_years": list(train_years),
        "test_years": list(test_years),
        "n_train_patches": len(train_idx),
        "n_test_patches": len(test_idx),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "best_threshold": best,
        "generalisation_gap": gap,
        "threshold_sweep": sweep["results"],
    }

    out_dir = Path("data/outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_json(result, out_dir / "temporal_cv.json")
    return result
