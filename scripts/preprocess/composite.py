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
        log.info("Only 1 input — copying as composite")
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
        log.info(f"✓ Composite from {len(valid)} images: {output_path.name}")
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
    for year, paths in date_file_map.items():
        out = output_dir / f"composite_{year}.tif"
        result = median_composite(paths, out, checkpoint, force)
        if result:
            results[year] = result
    return results
