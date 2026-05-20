"""
Multi-date median compositing for noise reduction.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger, StepCheckpoint

log = get_logger("preprocess.composite")


def median_composite(
    input_paths: List[Path],
    output_path: Path,
    checkpoint: Optional[StepCheckpoint] = None,
    force: bool = False
) -> Optional[Path]:
    """
    Create a pixel-wise median composite from multiple GeoTIFFs.
    All inputs must be coregistered and have the same shape/bands.
    """
    import rasterio

    ck_key = f"composite_{output_path.name}"
    if checkpoint and not force and checkpoint.is_done("composite", ck_key):
        if output_path.exists():
            log.info(f"[SKIP] Composite exists: {output_path.name}")
            return output_path

    valid = [p for p in input_paths if p.exists()]
    if not valid:
        log.warning("No valid inputs for composite")
        return None
    if len(valid) == 1:
        log.info("Only 1 input  -  copying as composite")
        import shutil
        shutil.copy2(valid[0], output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load all bands
        stacks = []
        meta = None
        for p in valid:
            with rasterio.open(p) as src:
                if meta is None:
                    meta = src.meta.copy()
                data = src.read().astype(np.float32)
                stacks.append(data)

        # Stack along new axis: (N_dates, C, H, W)
        # Truncate to min shape
        min_h = min(s.shape[1] for s in stacks)
        min_w = min(s.shape[2] for s in stacks)
        stacks = [s[:, :min_h, :min_w] for s in stacks]

        cube = np.stack(stacks, axis=0)  # (N, C, H, W)
        composite = np.nanmedian(cube, axis=0)  # (C, H, W)

        meta.update({
            "height": composite.shape[1],
            "width": composite.shape[2],
            "dtype": "float32",
            "compress": "deflate"
        })

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(composite.astype(np.float32))

        if checkpoint:
            checkpoint.mark_done("composite", ck_key, {
                "n_inputs": len(valid),
                "output": str(output_path)
            })
        log.info(f"[OK] Composite from {len(valid)} images: {output_path.name}")
        return output_path

    except Exception as e:
        log.error(f"Composite failed: {e}")
        return None


def composite_by_year(
    date_file_map: Dict[str, List[Path]],
    output_dir: Path,
    checkpoint: StepCheckpoint,
    force: bool = False
) -> Dict[str, Path]:
    """
    Create one composite per year from multi-date input groups.

    Parameters
    ----------
    date_file_map : {year_str: [path1, path2, ...]}

    Returns
    -------
    {year_str: composite_path}
    """
    results = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for year, paths in date_file_map.items():
        out = output_dir / f"composite_{year}.tif"
        result = median_composite(paths, out, checkpoint, force)
        if result:
            results[year] = result
    return results


def assign_ctx_geotransform(tif_path: Path, product_id: str = None) -> Path:
    """
    Assign a rough geotransform to a CTX tif that has none.
    Extracts lat/lon from the product_id filename and assigns
    a Simple Cylindrical (Mars) geotransform.
    """
    import re, math
    import rasterio
    from affine import Affine

    pid = product_id or tif_path.stem
    m = re.search(r"(\d+)([NS])(\d+)([EW])", pid.upper())
    if not m:
        return tif_path

    lat = float(m.group(1)) * (-1 if m.group(2)=="S" else 1)
    lon_raw = float(m.group(3))
    lon = lon_raw if m.group(4)=="E" else 360 - lon_raw

    MARS_R = 3_396_190.0
    x_ctr = math.radians(lon) * MARS_R
    y_ctr = math.radians(lat) * MARS_R

    try:
        with rasterio.open(tif_path) as src:
            W, H = src.width, src.height
            profile = src.profile.copy()
            data = src.read()
    except Exception:
        return tif_path

    # CTX pixel is ~6m at native, but tif may be resampled
    # Use 6m/px as a reasonable default
    px = 6.0
    x0 = x_ctr - (W / 2) * px
    y0 = y_ctr + (H / 2) * px  # top-left corner

    transform = Affine(px, 0, x0, 0, -px, y0)
    profile.update(transform=transform)

    out_path = tif_path.with_stem(tif_path.stem + "_geo")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
    return out_path