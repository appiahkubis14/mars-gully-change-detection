"""
threshold.py
============
Post-process probability maps into binary gully masks.

Steps
-----
1. Apply each threshold in cfg.inference.thresholds.
2. Remove connected components smaller than cfg.inference.min_gully_area_m2.
3. Save binary GeoTIFFs (uint8, 0/1) and summary JSON.

Usage (via main.py)
-------------------
    python main.py --step threshold
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from scipy import ndimage

from scripts.utils import load_config, StepCheckpoint

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def binarise(prob_map: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a threshold and return a uint8 binary array {0, 1}."""
    return (prob_map >= threshold).astype(np.uint8)


def remove_small_components(
    binary: np.ndarray,
    min_pixels: int,
    connectivity: int = 2,
) -> np.ndarray:
    """
    Remove connected components smaller than min_pixels.

    Args:
        binary      : (H, W) uint8 binary array.
        min_pixels  : Minimum component size to keep (in pixels).
        connectivity: 1 = 4-connected, 2 = 8-connected (default).

    Returns:
        Cleaned (H, W) uint8 binary array.
    """
    struct = ndimage.generate_binary_structure(2, connectivity)
    labelled, n_components = ndimage.label(binary, structure=struct)
    if n_components == 0:
        return binary
    sizes = ndimage.sum(binary, labelled, range(1, n_components + 1))
    too_small = np.array(sizes) < min_pixels
    remove_mask = too_small[labelled - 1]
    remove_mask[labelled == 0] = False
    cleaned = binary.copy()
    cleaned[remove_mask] = 0
    return cleaned


def morphological_cleanup(
    binary: np.ndarray,
    closing_size: int = 3,
) -> np.ndarray:
    """
    Optional morphological closing to fill small holes in gully masks.

    Args:
        binary       : (H, W) uint8 binary array.
        closing_size : Structuring element radius for closing.

    Returns:
        (H, W) uint8 cleaned array.
    """
    struct = ndimage.generate_binary_structure(2, 2)
    struct = ndimage.iterate_structure(struct, closing_size)
    closed = ndimage.binary_closing(binary, structure=struct).astype(np.uint8)
    return closed


# ---------------------------------------------------------------------------
# Pixel-to-area conversion
# ---------------------------------------------------------------------------

def pixel_area_m2(profile: dict) -> float:
    """Return the area of one pixel in m² from a rasterio profile."""
    transform = profile.get("transform")
    if transform is None:
        return 1.0
    pixel_width = abs(transform.a)   # degrees or metres
    pixel_height = abs(transform.e)
    # If CRS is geographic (degrees), convert approximately for Mars
    crs = profile.get("crs")
    if crs is not None:
        try:
            if crs.is_geographic:
                # Mars mean radius 3389.5 km
                MARS_RADIUS_M = 3_389_500.0
                import math
                px_m = pixel_width * math.pi / 180.0 * MARS_RADIUS_M
                py_m = pixel_height * math.pi / 180.0 * MARS_RADIUS_M
                return px_m * py_m
        except Exception:
            pass
    return pixel_width * pixel_height


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run_threshold(cfg_path: str = "config.yaml") -> None:
    """
    Threshold all probability maps in data/outputs/probability_maps/ and
    save binary GeoTIFFs to data/outputs/binary_maps/.

    Skips already-processed files via StepCheckpoint.
    """
    cfg = load_config(cfg_path)
    thresholds: List[float] = cfg["inference"].get("thresholds", [0.3, 0.5, 0.7])
    min_area_m2: float = cfg["inference"].get("min_gully_area_m2", 500.0)

    prob_dir = Path("data/outputs/probability_maps")
    binary_dir = Path("data/outputs/binary_maps")
    binary_dir.mkdir(parents=True, exist_ok=True)

    ckpt = StepCheckpoint("data/outputs/threshold_checkpoint.json")
    summary: List[dict] = []

    prob_files = sorted(prob_dir.glob("*_prob.tif"))
    if not prob_files:
        log.warning(f"No probability maps found in {prob_dir}.")
        return

    log.info(f"Processing {len(prob_files)} probability map(s) at thresholds {thresholds}")

    for prob_path in prob_files:
        stem = prob_path.stem.replace("_prob", "")

        with rasterio.open(prob_path) as src:
            prob_map = src.read(1).astype(np.float32)
            profile = src.profile

        pix_area = pixel_area_m2(profile)
        min_pixels = max(1, int(np.ceil(min_area_m2 / pix_area)))

        file_summary = {"stem": stem, "thresholds": {}}

        for thr in thresholds:
            thr_str = f"thr{int(thr * 100):02d}"
            out_name = f"{stem}_{thr_str}.tif"
            out_path = binary_dir / out_name

            if ckpt.is_done(f"thresh_{stem}_{thr_str}"):
                log.info(f"[SKIP] {out_name}")
                continue

            binary = binarise(prob_map, threshold=thr)
            binary = remove_small_components(binary, min_pixels=min_pixels)

            # optional morphological closing (small holes)
            binary = morphological_cleanup(binary, closing_size=2)

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
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(binary[np.newaxis])

            n_pixels = int(binary.sum())
            area_ha = n_pixels * pix_area / 10_000.0
            file_summary["thresholds"][thr_str] = {
                "n_gully_pixels": n_pixels,
                "gully_area_ha": round(area_ha, 4),
                "min_area_m2": min_area_m2,
                "output_file": str(out_path),
            }
            ckpt.mark_done(f"thresh_{stem}_{thr_str}")
            log.info(
                f"  {out_name} | "
                f"gully pixels={n_pixels} | "
                f"area={area_ha:.2f} ha"
            )

        summary.append(file_summary)

    summary_path = binary_dir / "threshold_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Threshold summary → {summary_path}")
