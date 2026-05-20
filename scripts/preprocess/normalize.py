"""
Per-band normalization to [0, 1] using percentile clipping.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger, StepCheckpoint

log = get_logger("preprocess.normalize")


def normalize_band(
    arr: np.ndarray,
    clip_low: float = 2.0,
    clip_high: float = 98.0,
    nodata: Optional[float] = None
) -> np.ndarray:
    """
    Normalize a 2D array to [0, 1] using percentile clipping.

    Parameters
    ----------
    arr      : 2D float array
    clip_low : lower percentile for clipping
    clip_high: upper percentile for clipping
    nodata   : pixels with this value are set to 0 and excluded from stats

    Returns
    -------
    float32 array in [0, 1]
    """
    arr = arr.astype(np.float32)

    if nodata is not None:
        mask = arr == nodata
    else:
        mask = ~np.isfinite(arr)

    valid = arr[~mask]
    if valid.size == 0:
        return np.zeros_like(arr)

    lo = np.percentile(valid, clip_low)
    hi = np.percentile(valid, clip_high)

    if hi - lo < 1e-9:
        log.warning("Band has near-zero range — returning zeros")
        out = np.zeros_like(arr)
    else:
        out = np.clip(arr, lo, hi)
        out = (out - lo) / (hi - lo)

    out[mask] = 0.0
    return out.astype(np.float32)


def normalize_raster(
    input_path: Path,
    output_path: Path,
    cfg: Dict,
    checkpoint: Optional[StepCheckpoint] = None,
    force: bool = False
) -> Optional[Path]:
    """Normalize all bands of a GeoTIFF and write result."""
    import rasterio

    norm_cfg = cfg.get("preprocessing", {}).get("normalization", {})
    clip_low, clip_high = norm_cfg.get("clip_percentiles", [2, 98])

    ck_key = f"normalize_{input_path.name}"
    if checkpoint and not force and checkpoint.is_done("normalize", ck_key):
        if output_path.exists():
            log.info(f"[SKIP] Already normalized: {input_path.name}")
            return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(input_path) as src:
            meta = src.meta.copy()
            meta.update({"dtype": "float32", "compress": "deflate", "nodata": 0})
            bands = src.read()  # (C, H, W)

        normalized = np.zeros_like(bands, dtype=np.float32)
        for i in range(bands.shape[0]):
            normalized[i] = normalize_band(
                bands[i], clip_low, clip_high,
                nodata=meta.get("nodata")
            )

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(normalized)

        if checkpoint:
            checkpoint.mark_done("normalize", ck_key, {"output": str(output_path)})
        log.info(f"✓ Normalized: {output_path.name}")
        return output_path

    except Exception as e:
        log.error(f"Normalization failed {input_path.name}: {e}")
        return None


def normalize_all(
    file_map: Dict[str, Path],
    output_dir: Path,
    cfg: Dict,
    checkpoint: StepCheckpoint,
    force: bool = False
) -> Dict[str, Path]:
    results = {}
    for key, src_path in file_map.items():
        if not src_path.exists():
            continue
        out = output_dir / f"{src_path.stem}_norm.tif"
        result = normalize_raster(src_path, out, cfg, checkpoint, force)
        if result:
            results[key] = result
    return results