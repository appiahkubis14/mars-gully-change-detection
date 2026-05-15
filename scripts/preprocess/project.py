"""
Projection: Reproject all images to a common Mars equirectangular grid.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger, StepCheckpoint, mars_equirectangular_wkt

log = get_logger("preprocess.project")


def reproject_to_mars_grid(
    input_path: Path,
    output_path: Path,
    target_resolution_m: float = 6.0,
    bounds: Optional[List[float]] = None,
    checkpoint: Optional[StepCheckpoint] = None,
    force: bool = False
) -> Optional[Path]:
    """
    Reproject a raster to Mars equirectangular grid at target resolution.

    Parameters
    ----------
    bounds : [min_lon, min_lat, max_lon, max_lat] in degrees
             If None, use the source extent.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.crs import CRS

    ck_key = f"project_{input_path.name}"
    if checkpoint and not force and checkpoint.is_done("project", ck_key):
        if output_path.exists():
            log.info(f"[SKIP] Already projected: {input_path.name}")
            return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dst_crs_wkt = mars_equirectangular_wkt()

    try:
        dst_crs = CRS.from_wkt(dst_crs_wkt)
    except Exception:
        # Fallback: use EPSG:4326-like geographic
        dst_crs = CRS.from_epsg(4326)
        log.warning("Mars WKT unsupported by rasterio — using geographic CRS")

    try:
        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs,
                src.width, src.height,
                *src.bounds,
                resolution=target_resolution_m
            )

            # Optionally clip to bounds
            if bounds:
                from rasterio.transform import from_bounds as tfb
                lon_min, lat_min, lon_max, lat_max = bounds
                # Convert degrees to meters in equirectangular
                R = 3_396_190.0
                x_min = np.radians(lon_min) * R
                x_max = np.radians(lon_max) * R
                y_min = np.radians(lat_min) * R
                y_max = np.radians(lat_max) * R
                width = int((x_max - x_min) / target_resolution_m)
                height = int((y_max - y_min) / target_resolution_m)
                transform = tfb(x_min, y_min, x_max, y_max, width, height)

            meta = src.meta.copy()
            meta.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "dtype": "float32",
                "compress": "deflate",
                "nodata": np.nan
            })

            with rasterio.open(output_path, "w", **meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear
                    )

        if checkpoint:
            checkpoint.mark_done("project", ck_key, {"output": str(output_path)})
        log.info(f"✓ Projected: {output_path.name} ({width}×{height} px, {target_resolution_m}m)")
        return output_path

    except Exception as e:
        log.error(f"Projection failed for {input_path.name}: {e}")
        if checkpoint:
            checkpoint.mark_failed("project", ck_key, str(e))
        return None


def project_all(
    file_map: Dict[str, Path],
    output_dir: Path,
    cfg: Dict,
    checkpoint: StepCheckpoint,
    force: bool = False
) -> Dict[str, Path]:
    """Project all input files to the common Mars grid."""
    resolution = cfg["preprocessing"]["target_resolution_m"]
    results = {}
    for key, src_path in file_map.items():
        if not src_path.exists():
            continue
        out = output_dir / f"{src_path.stem}_projected.tif"
        result = reproject_to_mars_grid(src_path, out, resolution, checkpoint=checkpoint, force=force)
        if result:
            results[key] = result
    return results
