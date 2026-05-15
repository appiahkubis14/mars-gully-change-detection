"""
detect_changes.py
=================
Pixel-wise multi-temporal change detection between binary gully masks.

Algorithm
---------
For each consecutive date pair (t, t+1):
  change_map = binary[t+1] & ~binary[t]   → new gully pixels (gain)
  loss_map   = binary[t]   & ~binary[t+1] → healed/buried pixels (loss)
  stable_map = binary[t]   &  binary[t+1] → persistent gully pixels

Outputs
-------
  data/outputs/change_detection/
    ├── change_{date_a}_{date_b}_gain.tif   (uint8)
    ├── change_{date_a}_{date_b}_loss.tif   (uint8)
    ├── change_{date_a}_{date_b}_stable.tif (uint8)
    ├── change_summary.json
    └── change_combined.tif  (3-band: gain/loss/stable)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling

from scripts.utils import load_config, StepCheckpoint
from scripts.inference.threshold import pixel_area_m2

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sort_key(path: Path) -> str:
    """Extract a sortable date string from filename, e.g. '2015'."""
    match = re.search(r"(\d{4})", path.stem)
    return match.group(1) if match else path.stem


def load_binary_map(path: Path) -> Tuple[np.ndarray, dict]:
    """Read a binary GeoTIFF → (H, W) uint8 + profile."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.uint8)
        profile = src.profile
    return arr, profile


def align_to_reference(
    source: np.ndarray,
    src_profile: dict,
    ref_profile: dict,
) -> np.ndarray:
    """
    Reproject `source` to match `ref_profile` spatial extent and resolution.

    Uses nearest-neighbour resampling for binary masks.
    """
    import rasterio.transform as rt
    from rasterio.warp import reproject

    dst_shape = (ref_profile["height"], ref_profile["width"])
    dst_arr = np.zeros(dst_shape, dtype=np.uint8)
    reproject(
        source=source[np.newaxis],
        destination=dst_arr[np.newaxis],
        src_transform=src_profile["transform"],
        src_crs=src_profile.get("crs"),
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile.get("crs"),
        resampling=Resampling.nearest,
    )
    return dst_arr


def compute_change(
    binary_t0: np.ndarray,
    binary_t1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pixel-level change maps between two binary masks.

    Returns:
        gain   : New gully pixels in t1 not present in t0.
        loss   : Pixels present in t0 but absent in t1 (healed/buried).
        stable : Persistent gully pixels in both.
    """
    gain = (binary_t1.astype(bool) & ~binary_t0.astype(bool)).astype(np.uint8)
    loss = (binary_t0.astype(bool) & ~binary_t1.astype(bool)).astype(np.uint8)
    stable = (binary_t0.astype(bool) & binary_t1.astype(bool)).astype(np.uint8)
    return gain, loss, stable


def save_change_band(
    arr: np.ndarray,
    profile: dict,
    output_path: Path,
) -> None:
    """Save a single-band change map as COG GeoTIFF."""
    out_profile = profile.copy()
    out_profile.update(
        dtype="uint8",
        count=1,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=255,
    )
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(arr[np.newaxis])


def save_rgb_change(
    gain: np.ndarray,
    loss: np.ndarray,
    stable: np.ndarray,
    profile: dict,
    output_path: Path,
) -> None:
    """
    Save a 3-band composite change map.
      Band 1: gain   (new gullies)
      Band 2: loss   (healed/buried)
      Band 3: stable (persistent)
    """
    out_profile = profile.copy()
    out_profile.update(
        dtype="uint8",
        count=3,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(gain[np.newaxis],   1)
        dst.write(loss[np.newaxis],   2)
        dst.write(stable[np.newaxis], 3)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_change_detection(
    cfg_path: str = "config.yaml",
    threshold: str = "thr50",
) -> List[dict]:
    """
    Detect changes between all consecutive binary mask pairs.

    Args:
        cfg_path  : Path to config.yaml.
        threshold : Which threshold suffix to use (e.g. 'thr50' for 0.5).

    Returns:
        List of per-pair change dictionaries.
    """
    cfg = load_config(cfg_path)
    binary_dir = Path("data/outputs/binary_maps")
    out_dir = Path("data/outputs/change_detection")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = StepCheckpoint("data/outputs/change_checkpoint.json")

    # ---- collect binary maps matching the threshold ----
    pattern = f"*_{threshold}.tif"
    binary_files = sorted(binary_dir.glob(pattern), key=_sort_key)
    if len(binary_files) < 2:
        log.warning(
            f"Need ≥2 binary maps matching '{pattern}' in {binary_dir}. "
            "Run --step threshold first."
        )
        return []

    log.info(f"Change detection over {len(binary_files)} dates: {[f.stem for f in binary_files]}")

    # ---- load reference profile (first file) ----
    _, ref_profile = load_binary_map(binary_files[0])
    pix_area = pixel_area_m2(ref_profile)

    all_pairs: List[dict] = []

    for i in range(len(binary_files) - 1):
        path_t0 = binary_files[i]
        path_t1 = binary_files[i + 1]
        pair_key = f"{path_t0.stem}__vs__{path_t1.stem}"

        if ckpt.is_done(pair_key):
            log.info(f"[SKIP] {pair_key}")
            continue

        log.info(f"Processing: {path_t0.stem} → {path_t1.stem}")

        mask_t0, profile_t0 = load_binary_map(path_t0)
        mask_t1, profile_t1 = load_binary_map(path_t1)

        # ---- align t1 to t0 if shapes differ ----
        if mask_t0.shape != mask_t1.shape:
            log.warning(
                f"Shape mismatch: t0={mask_t0.shape}, t1={mask_t1.shape}. Reprojecting t1 → t0."
            )
            mask_t1 = align_to_reference(mask_t1, profile_t1, profile_t0)

        gain, loss, stable = compute_change(mask_t0, mask_t1)

        date_a = _sort_key(path_t0)
        date_b = _sort_key(path_t1)
        prefix = f"change_{date_a}_{date_b}"

        save_change_band(gain,   ref_profile, out_dir / f"{prefix}_gain.tif")
        save_change_band(loss,   ref_profile, out_dir / f"{prefix}_loss.tif")
        save_change_band(stable, ref_profile, out_dir / f"{prefix}_stable.tif")
        save_rgb_change(gain, loss, stable, ref_profile, out_dir / f"{prefix}_combined.tif")

        pair_result = {
            "date_t0": date_a,
            "date_t1": date_b,
            "gain_pixels": int(gain.sum()),
            "loss_pixels": int(loss.sum()),
            "stable_pixels": int(stable.sum()),
            "gain_ha": round(gain.sum() * pix_area / 10_000, 4),
            "loss_ha": round(loss.sum() * pix_area / 10_000, 4),
            "stable_ha": round(stable.sum() * pix_area / 10_000, 4),
            "net_change_ha": round((gain.sum() - loss.sum()) * pix_area / 10_000, 4),
        }
        all_pairs.append(pair_result)
        log.info(
            f"  Gain={pair_result['gain_ha']:.2f} ha | "
            f"Loss={pair_result['loss_ha']:.2f} ha | "
            f"Net={pair_result['net_change_ha']:+.2f} ha"
        )
        ckpt.mark_done(pair_key)

    summary_path = out_dir / "change_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_pairs, f, indent=2)
    log.info(f"Change summary → {summary_path}")
    return all_pairs
