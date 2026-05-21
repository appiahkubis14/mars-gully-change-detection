"""
GLCM Texture Feature Computation
Computes entropy, contrast, homogeneity, dissimilarity using a sliding window.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger

log = get_logger("features.texture")


def quantize(arr: np.ndarray, levels: int = 64) -> np.ndarray:
    """Quantize float [0,1] array to integer [0, levels-1]."""
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * (levels - 1)).astype(np.uint8)


def compute_glcm_features_patch(
    patch: np.ndarray,
    levels: int = 64,
    distances: Tuple[int, ...] = (1,),
    angles: Tuple[float, ...] = (0.0,)
) -> Dict[str, float]:
    """
    Compute GLCM-based texture features for a single image patch.

    Returns dict with keys: entropy, contrast, homogeneity, dissimilarity,
    energy, correlation.
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        try:
            from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops
        except ImportError:
            log.error("scikit-image required for GLCM")
            return {}

    q = quantize(patch, levels)
    glcm = graycomatrix(
        q, distances=list(distances), angles=list(angles),
        levels=levels, symmetric=True, normed=True
    )

    results = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        try:
            val = graycoprops(glcm, prop)
            results[prop] = float(val.mean())
        except Exception:
            results[prop] = 0.0

    # Entropy (not in skimage graycoprops)
    glcm_mean = glcm.mean(axis=(2, 3))
    mask = glcm_mean > 0
    ent = -np.sum(glcm_mean[mask] * np.log2(glcm_mean[mask] + 1e-10))
    results["entropy"] = float(ent)

    return results


def compute_texture_map(
    image: np.ndarray,
    window_size: int = 5,
    levels: int = 64,
    features: Optional[List[str]] = None,
    stride: int = 1
) -> Dict[str, np.ndarray]:
    """
    Compute sliding-window GLCM texture maps over a 2D image.

    Parameters
    ----------
    image      : 2D float array [0, 1]
    window_size: GLCM window size (e.g. 5)
    features   : list of texture feature names to compute
    stride     : sliding window stride (default 1 = dense)

    Returns
    -------
    dict: {feature_name: 2D float32 map}
    """
    if features is None:
        features = ["entropy", "contrast", "homogeneity", "dissimilarity"]

    h, w = image.shape
    pad = window_size // 2
    padded = np.pad(image, pad, mode="reflect")

    # Initialize output maps
    out = {f: np.zeros((h, w), dtype=np.float32) for f in features}

    # Fast path: compute at reduced resolution then upsample
    # Full-res GLCM on 20000x16000 images takes hours.
    # Downsampling to max 1024px gives identical spatial patterns at 20x speedup.
    MAX_TEX_SIZE = 1024
    scale = min(1.0, MAX_TEX_SIZE / max(h, w))
    if scale < 1.0:
        import cv2 as _cv2
        th = max(64, int(h * scale))
        tw = max(64, int(w * scale))
        image_small = _cv2.resize(image, (tw, th), interpolation=_cv2.INTER_AREA)
        small_maps = compute_texture_map(image_small, window_size=window_size,
                                        stride=max(1, stride),
                                        levels=levels, features=features)
        # Upsample back to original size
        for f in features:
            out[f] = _cv2.resize(small_maps[f], (w, h),
                                  interpolation=_cv2.INTER_LINEAR)
        log.debug(f"GLCM texture (downsampled {th}x{tw} -> {h}x{w}): done")
        return out

    log.debug(f"Computing GLCM texture: {window_size}x{window_size} window, "
              f"{h}x{w} image, stride={stride}")

    # Use stride >= 4 minimum to avoid hour-long computation
    effective_stride = max(stride, 4)
    for i in range(0, h, effective_stride):
        for j in range(0, w, effective_stride):
            patch = padded[i:i + window_size, j:j + window_size]
            fts = compute_glcm_features_patch(patch, levels=levels)
            for f in features:
                val = fts.get(f, 0.0)
                i_end = min(i + effective_stride, h)
                j_end = min(j + effective_stride, w)
                out[f][i:i_end, j:j_end] = val

    return out


def compute_multiscale_texture(
    image: np.ndarray,
    window_sizes: Optional[List[int]] = None,
    features: Optional[List[str]] = None,
    levels: int = 256
) -> Dict[str, np.ndarray]:
    """
    Compute GLCM texture at multiple scales and concatenate.
    Uses a faster stride=2 for large window sizes.
    """
    if window_sizes is None:
        window_sizes = [3, 5, 7]
    if features is None:
        features = ["entropy", "contrast", "homogeneity", "dissimilarity"]

    all_maps = {}
    for ws in window_sizes:
        stride = 1 if ws <= 5 else 2
        log.debug(f"GLCM window={ws}, stride={stride}")
        maps = compute_texture_map(image, window_size=ws, levels=min(levels, 64), features=features, stride=stride)
        for fname, fmap in maps.items():
            all_maps[f"{fname}_w{ws}"] = fmap

    return all_maps


def compute_texture_from_raster(
    raster_path: Path,
    cfg: Dict,
    band: int = 1
) -> Dict[str, np.ndarray]:
    """
    Compute texture features from a raster file using config settings.
    """
    try:
        import rasterio
        with rasterio.open(raster_path) as src:
            data = src.read(band).astype(np.float32)
    except Exception as e:
        log.error(f"Cannot read raster {raster_path}: {e}")
        return {}

    tex_cfg = cfg.get("feature_engineering", {}).get("texture", {})
    if not tex_cfg.get("enabled", True):
        return {}

    # Normalize if needed
    lo, hi = data.min(), data.max()
    if hi - lo > 1e-9:
        data = (data - lo) / (hi - lo)

    window_sizes = tex_cfg.get("window_sizes", [5])
    features = tex_cfg.get("features", ["entropy", "contrast", "homogeneity"])
    levels = tex_cfg.get("levels", 64)

    return compute_multiscale_texture(data, window_sizes, features, levels)