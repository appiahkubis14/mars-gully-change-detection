import warnings
warnings.filterwarnings('ignore', message='.*PROJ.*')
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
        log.warning("Mars WKT unsupported by rasterio  -  using geographic CRS")

    try:
        with rasterio.open(input_path) as src_ds:
            src_crs   = src_ds.crs
            src_width = src_ds.width
            src_height= src_ds.height
            src_bounds= src_ds.bounds
            src_transform = src_ds.transform

        # If no CRS/geotransform (e.g. raw CTX EDR), assign one from bounds
        has_georef = (src_crs is not None and
                      src_transform != rasterio.transform.IDENTITY and
                      src_bounds.left != 0.0)

        if not has_georef and bounds:
            import math
            MARS_R = 3_396_190.0
            lon_min, lat_min, lon_max, lat_max = bounds
            # Convert to metres (Simple Cylindrical)
            x_min = math.radians(lon_min) * MARS_R
            x_max = math.radians(lon_max) * MARS_R
            y_min = math.radians(lat_min) * MARS_R
            y_max = math.radians(lat_max) * MARS_R
            from affine import Affine
            px = (x_max - x_min) / src_width
            py = (y_min - y_max) / src_height   # negative
            src_transform = Affine(px, 0, x_min, 0, py, y_max)
            src_crs = dst_crs   # already in Mars equirectangular
            log.debug(f"Assigned geotransform from bounds for {input_path.name}")
        elif not has_georef:
            log.warning(f"No geotransform and no bounds for {input_path.name} "
                        f"— reprojection will use identity transform")

        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src_crs or dst_crs, dst_crs,
                src_width, src_height,
                src_bounds.left, src_bounds.bottom,
                src_bounds.right, src_bounds.top,
                resolution=target_resolution_m
            ) if has_georef or bounds else (src_transform, src_width, src_height)

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
                "driver": "GTiff",   # always write GeoTIFF, never JP2
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "dtype": "float32",
                "compress": "deflate",
                "nodata": np.nan,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            })
            # Remove JP2-specific options that are invalid for GTiff
            for k in ["QUALITY", "REVERSIBLE", "NBITS"]:
                meta.pop(k, None)

            with rasterio.open(output_path, "w", **meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src_transform,
                        src_crs=src_crs or dst_crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear
                    )

        if checkpoint:
            checkpoint.mark_done("project", ck_key, {"output": str(output_path)})
        log.info(f"[OK] Projected: {output_path.name} ({width}x{height} px, {target_resolution_m}m)")
        return output_path

    except Exception as e:
        log.error(f"Projection failed for {input_path.name}: {e}")
        if checkpoint:
            checkpoint.mark_failed("project", ck_key, str(e))
        return None


def _ensure_geotiff(input_path: Path) -> Path:
    """
    Convert .IMG or .JP2 files to GeoTIFF before reprojection.
    GTiff is the only format rasterio can write back to reliably.
    Returns the .tif path (converting if needed), or input_path unchanged.
    """
    if input_path.suffix.upper() not in (".IMG", ".JP2"):
        return input_path
    tif_path = input_path.with_suffix(".tif")
    if tif_path.exists() and tif_path.stat().st_size > 100_000:
        return tif_path
    try:
        import rasterio, numpy as np
        import warnings as _w
        _w.filterwarnings("ignore")

        with rasterio.open(input_path) as s:
            # Read tile by tile to handle partial JP2 files
            try:
                data = s.read().astype(np.float32)
            except Exception:
                # Partial read: read what we can band by band
                data = np.zeros((s.count, s.height, s.width), dtype=np.float32)
                for bi in range(1, s.count + 1):
                    try:
                        data[bi-1] = s.read(bi).astype(np.float32)
                    except Exception:
                        pass  # leave as zeros if tile is corrupt

            # Normalise each band
            for b in range(data.shape[0]):
                band = data[b]
                valid = band[np.isfinite(band) & (band > 0)]
                if len(valid) > 100:
                    lo, hi = np.percentile(valid, [0.1, 99.9])
                    data[b] = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

            src_has_geo = (s.crs is not None and
                           s.transform.a != 1.0)  # not identity
            profile = {
                "driver": "GTiff", "dtype": "float32",
                "count": data.shape[0],
                "height": data.shape[1], "width": data.shape[2],
                "compress": "deflate", "tiled": True,
                "blockxsize": 256, "blockysize": 256,
            }
            if src_has_geo:
                profile["crs"] = s.crs
                profile["transform"] = s.transform

        with rasterio.open(tif_path, "w", **profile) as d:
            d.write(data)
        sz = tif_path.stat().st_size / 1e6
        log.info(f"[OK] Converted to GeoTIFF: {tif_path.name} ({sz:.1f} MB)")
        return tif_path
    except Exception as e:
        log.warning(f"Conversion failed for {input_path.name}: {e}")
        return input_path


def project_all(
    file_map: Dict[str, Path],
    output_dir: Path,
    cfg: Dict,
    checkpoint: StepCheckpoint,
    force: bool = False
) -> Dict[str, Path]:
    """Project all input files to the common Mars grid."""
    resolution = cfg["preprocessing"]["target_resolution_m"]
    # Build site bounds lookup: {site_key: bounds_list}
    site_bounds = {}
    for sk, sv in cfg.get("study_area", {}).items():
        if isinstance(sv, dict) and "bounds" in sv:
            site_bounds[sk] = sv["bounds"]

    results = {}
    for key, src_path in file_map.items():
        if not src_path.exists():
            log.warning(f"[SKIP] File not found: {src_path}")
            continue
        # Convert PDS3 .IMG to GeoTIFF before reprojecting
        src_path = _ensure_geotiff(src_path)

        # Assign geotransform to CTX tifs that have none (raw EDR conversions)
        if src_path.suffix.lower() == ".tif" and "_ctx" in key:
            try:
                import rasterio as _rio
                with _rio.open(src_path) as _chk:
                    has_geo = (_chk.crs is not None and
                               _chk.transform.a != 1.0)
                if not has_geo:
                    from scripts.preprocess.composite import assign_ctx_geotransform
                    geo_path = assign_ctx_geotransform(src_path, src_path.stem)
                    if geo_path != src_path:
                        src_path = geo_path
                        log.info(f"Assigned geotransform to CTX: {src_path.name}")
            except Exception as _e:
                log.debug(f"CTX geotransform assignment failed: {_e}")
        out = output_dir / f"{key}_projected.tif"
        # Extract site key from file key (e.g. "primary_B01_..." -> "primary")
        site_key = key.split("_")[0]
        bounds = site_bounds.get(site_key)
        result = reproject_to_mars_grid(src_path, out, resolution,
                                        bounds=bounds,
                                        checkpoint=checkpoint, force=force)
        if result:
            results[key] = result
    return results