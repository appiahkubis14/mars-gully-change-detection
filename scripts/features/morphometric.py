"""
Morphometric Feature Computation from MOLA DEM
Computes slope, aspect, curvature (plan + profile), and roughness.
These are critical for gully detection — gullies form on steep slopes.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger

log = get_logger("features.morphometric")


def compute_slope_aspect(
    dem: np.ndarray,
    cell_size_m: float = 463.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (degrees) and aspect (degrees, 0=N) from a DEM.

    Uses 3x3 Horn's method (same as GDAL DEMProcessing).
    """
    # Pad with edge values
    padded = np.pad(dem, 1, mode="edge")

    # Compute partial derivatives using Horn's method
    dz_dx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]) -
        (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8 * cell_size_m)

    dz_dy = (
        (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]) -
        (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
    ) / (8 * cell_size_m)

    # Slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    # Aspect in degrees (0=N, 90=E, 180=S, 270=W)
    aspect = np.degrees(np.arctan2(-dz_dy, dz_dx))
    aspect = np.where(aspect < 0, 90 - aspect, 90 - aspect)
    aspect = np.where(aspect < 0, aspect + 360, aspect)
    aspect = np.where(aspect >= 360, aspect - 360, aspect)

    return slope.astype(np.float32), aspect.astype(np.float32)


def compute_curvature(
    dem: np.ndarray,
    cell_size_m: float = 463.0,
    smoothing_sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute plan curvature (cross-slope) and profile curvature (down-slope).
    Positive plan curvature → converging flow (gully channels).

    Uses Zevenbergen & Thorne (1987) method.
    """
    if smoothing_sigma > 0:
        dem = gaussian_filter(dem.astype(np.float64), sigma=smoothing_sigma)

    cs = cell_size_m
    padded = np.pad(dem, 1, mode="edge")

    # Quadratic surface fit coefficients
    D = (padded[1:-1, 2:] - 2 * padded[1:-1, 1:-1] + padded[1:-1, :-2]) / (2 * cs**2)
    E = (padded[2:, 1:-1] - 2 * padded[1:-1, 1:-1] + padded[:-2, 1:-1]) / (2 * cs**2)
    F = (padded[2:, 2:] - padded[2:, :-2] - padded[:-2, 2:] + padded[:-2, :-2]) / (4 * cs**2)
    G = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * cs)
    H_ = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * cs)

    G2H2 = G**2 + H_**2
    eps = 1e-10

    # Plan curvature
    plan_curv = np.where(
        G2H2 > eps,
        -2 * (D * H_**2 - F * G * H_ + E * G**2) / (G2H2 * np.sqrt(G2H2 + 1)),
        0.0
    )

    # Profile curvature
    prof_curv = np.where(
        G2H2 > eps,
        -2 * (D * G**2 + F * G * H_ + E * H_**2) / (G2H2 * np.sqrt(G2H2 + 1)),
        0.0
    )

    return plan_curv.astype(np.float32), prof_curv.astype(np.float32)


def compute_roughness(
    dem: np.ndarray,
    window: int = 3
) -> np.ndarray:
    """
    Terrain roughness: standard deviation of elevation in a local window.
    High roughness → rugged terrain (potential gully area).
    """
    mean = uniform_filter(dem.astype(np.float64), size=window)
    mean_sq = uniform_filter(dem.astype(np.float64)**2, size=window)
    var = mean_sq - mean**2
    roughness = np.sqrt(np.maximum(var, 0))
    return roughness.astype(np.float32)


def compute_all_morphometrics(
    dem: np.ndarray,
    cell_size_m: float = 463.0,
    smoothing_sigma: float = 1.0,
    features: Optional[list] = None
) -> Dict[str, np.ndarray]:
    """
    Compute all morphometric features from a DEM.

    Parameters
    ----------
    dem        : 2D float32 elevation array (meters)
    cell_size_m: pixel size in meters
    features   : list of feature names to compute; None = all

    Returns
    -------
    dict: {feature_name: 2D float32 array}
    """
    if features is None:
        features = ["slope", "aspect", "curvature_plan", "curvature_profile", "roughness"]

    dem = dem.astype(np.float32)
    # Replace NaN with interpolated values
    nan_mask = ~np.isfinite(dem)
    if nan_mask.any():
        from scipy.interpolate import NearestNDInterpolator
        coords = np.array(np.where(~nan_mask)).T
        vals = dem[~nan_mask]
        interp = NearestNDInterpolator(coords, vals)
        nan_coords = np.array(np.where(nan_mask)).T
        dem[nan_mask] = interp(nan_coords)

    result = {}

    if "slope" in features or "aspect" in features:
        slope, aspect = compute_slope_aspect(dem, cell_size_m)
        if "slope" in features:
            result["slope"] = slope
        if "aspect" in features:
            result["aspect"] = aspect

    if "curvature_plan" in features or "curvature_profile" in features:
        plan, prof = compute_curvature(dem, cell_size_m, smoothing_sigma)
        if "curvature_plan" in features:
            result["curvature_plan"] = plan
        if "curvature_profile" in features:
            result["curvature_profile"] = prof

    if "roughness" in features:
        result["roughness"] = compute_roughness(dem)

    for k, v in result.items():
        log.debug(f"Morphometric {k}: min={v.min():.2f}, max={v.max():.2f}")

    return result


def compute_morphometrics_from_raster(
    dem_path: Path,
    cfg: Dict,
    output_resolution_m: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Load a DEM GeoTIFF and compute morphometric features.
    Optionally resample to output_resolution_m.
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        MAX_DEM_SIZE = 4096  # cap DEM to avoid OOM on projected 20k px DEMs
        with rasterio.open(dem_path) as src:
            h, w = src.height, src.width
            scale = min(1.0, MAX_DEM_SIZE / max(h, w))
            if scale < 1.0:
                out_h = max(64, int(h * scale))
                out_w = max(64, int(w * scale))
                dem = src.read(1, out_shape=(out_h, out_w),
                               resampling=Resampling.bilinear).astype(np.float32)
                cell_size = src.res[0] / scale  # adjusted pixel size
                log.debug(f"DEM downsampled to {out_h}x{out_w} (scale={scale:.3f})")
            else:
                dem = src.read(1).astype(np.float32)
                cell_size = src.res[0]
    except Exception as e:
        log.error(f"Cannot open DEM {dem_path}: {e}")
        return {}

    morph_cfg = cfg.get("feature_engineering", {}).get("morphometric", {})
    if not morph_cfg.get("enabled", True):
        return {}

    features = morph_cfg.get("features", ["slope", "aspect", "curvature_plan", "curvature_profile", "roughness"])
    smoothing = morph_cfg.get("smoothing_sigma", 1.0)

    return compute_all_morphometrics(dem, cell_size, smoothing, features)


def compute_slope(dem: np.ndarray, resolution: float = 463.0) -> np.ndarray:
    """Compute slope in degrees from DEM."""
    slope, _ = compute_slope_aspect(dem, resolution)
    return slope


def compute_aspect(dem: np.ndarray, resolution: float = 463.0) -> np.ndarray:
    """Compute aspect in degrees (0-360) from DEM."""
    _, aspect = compute_slope_aspect(dem, resolution)
    return aspect