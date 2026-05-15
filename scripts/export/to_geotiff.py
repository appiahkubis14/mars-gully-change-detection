"""
to_geotiff.py
=============
Export probability maps and binary masks as Cloud-Optimized GeoTIFFs (COGs).

A COG is a standard GeoTIFF with:
  • Internal tiling (256×256 or 512×512)
  • Overviews (pyramid levels)
  • DEFLATE compression
  • (Optionally) HTTP range-request compatible layout

Usage (via main.py)
-------------------
    python main.py --step export
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling

log = logging.getLogger(__name__)

OVERVIEW_LEVELS = [2, 4, 8, 16, 32]
BLOCK_SIZE = 256


# ---------------------------------------------------------------------------
# Core COG writer
# ---------------------------------------------------------------------------

def write_cog(
    array: np.ndarray,
    profile: dict,
    output_path: Path,
    compression: str = "DEFLATE",
    overview_resampling: Resampling = Resampling.average,
) -> None:
    """
    Write a numpy array as a Cloud-Optimized GeoTIFF.

    Uses a two-pass approach:
      1. Write to a temporary file with overviews.
      2. Reorder with GDAL's COG driver (or manual copy_with_overviews).

    Args:
        array               : (C, H, W) or (H, W) array.
        profile             : rasterio profile dict.
        output_path         : Destination .tif path.
        compression         : 'DEFLATE' | 'LZW' | 'ZSTD'.
        overview_resampling : Resampling algorithm for overviews.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim == 2:
        array = array[np.newaxis]

    out_profile = profile.copy()
    out_profile.update(
        dtype=str(array.dtype),
        count=array.shape[0],
        compress=compression,
        tiled=True,
        blockxsize=BLOCK_SIZE,
        blockysize=BLOCK_SIZE,
        predictor=2 if np.issubdtype(array.dtype, np.integer) else 3,
        BIGTIFF="IF_SAFER",
    )

    # ---- step 1: write to temp ----
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with rasterio.open(tmp_path, "w", **out_profile) as dst:
            dst.write(array)
            dst.build_overviews(OVERVIEW_LEVELS, overview_resampling)
            dst.update_tags(ns="rio_overview", resampling=overview_resampling.name)

        # ---- step 2: reorder into COG layout (GDAL) ----
        _gdal_cog_copy(tmp_path, output_path, compression)
    finally:
        tmp_path.unlink(missing_ok=True)

    log.info(f"COG saved → {output_path}")


def _gdal_cog_copy(src: Path, dst: Path, compression: str) -> None:
    """
    Use gdal_translate with COG driver to produce a compliant COG.
    Falls back to a plain copy with overviews if GDAL isn't available.
    """
    gdal_translate = shutil.which("gdal_translate")
    if gdal_translate:
        cmd = [
            gdal_translate,
            "-of", "COG",
            "-co", f"COMPRESS={compression}",
            "-co", f"BLOCKSIZE={BLOCK_SIZE}",
            "-co", "COPY_SRC_OVERVIEWS=YES",
            str(src),
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.warning(f"gdal_translate failed: {result.stderr}. Falling back to plain copy.")
            shutil.copy2(src, dst)
    else:
        log.warning("gdal_translate not found; writing plain GeoTIFF with overviews.")
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Batch export helpers
# ---------------------------------------------------------------------------

def export_probability_maps_as_cog(
    prob_dir: Path = Path("data/outputs/probability_maps"),
    out_dir: Optional[Path] = None,
    compression: str = "DEFLATE",
) -> List[Path]:
    """
    Re-export all probability maps as COGs.

    Args:
        prob_dir    : Source directory of probability GeoTIFFs.
        out_dir     : Destination directory (defaults to prob_dir/cog/).
        compression : GDAL compression codec.

    Returns:
        List of output paths.
    """
    if out_dir is None:
        out_dir = prob_dir / "cog"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for prob_path in sorted(prob_dir.glob("*.tif")):
        dst_path = out_dir / prob_path.name
        if dst_path.exists():
            log.info(f"[SKIP] {prob_path.name} already exported as COG.")
            outputs.append(dst_path)
            continue

        with rasterio.open(prob_path) as src:
            arr = src.read()
            profile = src.profile

        write_cog(arr, profile, dst_path, compression=compression)
        outputs.append(dst_path)

    return outputs


def export_binary_maps_as_cog(
    binary_dir: Path = Path("data/outputs/binary_maps"),
    out_dir: Optional[Path] = None,
    compression: str = "DEFLATE",
) -> List[Path]:
    """Re-export all binary masks as COGs."""
    if out_dir is None:
        out_dir = binary_dir / "cog"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for bin_path in sorted(binary_dir.glob("*.tif")):
        dst_path = out_dir / bin_path.name
        if dst_path.exists():
            log.info(f"[SKIP] {bin_path.name}")
            outputs.append(dst_path)
            continue

        with rasterio.open(bin_path) as src:
            arr = src.read()
            profile = src.profile

        write_cog(arr, profile, dst_path, compression=compression)
        outputs.append(dst_path)

    return outputs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_geotiff_export(cfg_path: str = "config.yaml") -> None:
    """Export all GeoTIFFs as COGs using config settings."""
    from scripts.utils import load_config
    cfg = load_config(cfg_path)
    compression = cfg.get("export", {}).get("cog_compression", "DEFLATE")

    log.info("Exporting probability maps as COGs …")
    export_probability_maps_as_cog(compression=compression)

    log.info("Exporting binary maps as COGs …")
    export_binary_maps_as_cog(compression=compression)

    log.info("GeoTIFF COG export complete.")
