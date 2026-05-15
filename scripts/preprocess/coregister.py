"""
Image Coregistration
Registers HiRISE (and other) images to the CTX base image.

Methods:
  1. ECC (Enhanced Correlation Coefficient) via OpenCV: best for same-sensor,
     sub-pixel accuracy, translation+rotation only.
  2. SIFT + RANSAC via OpenCV: more robust for cross-sensor, handles scale.

The saved transformation matrix is applied when reprojecting.
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import (
    get_logger, StepCheckpoint, sentinel_exists, write_sentinel,
    minmax_normalize
)

log = get_logger("preprocess.coregister")


def load_as_gray(path: Path, band: int = 1) -> np.ndarray:
    """Load a raster band as float32 grayscale [0, 1]."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read(band).astype(np.float32)
        return minmax_normalize(data)
    except Exception as e:
        log.error(f"Cannot read {path}: {e}")
        raise


def coregister_ecc(
    reference: np.ndarray,
    moving: np.ndarray,
    warp_mode: int = None,
    max_iterations: int = 1000,
    termination_eps: float = 1e-6,
    n_levels: int = 3
) -> Tuple[np.ndarray, float]:
    """
    Coregister `moving` to `reference` using ECC maximization.

    Parameters
    ----------
    reference, moving : np.ndarray  (H, W), float32, [0,1]
    warp_mode : cv2.MOTION_TRANSLATION, MOTION_EUCLIDEAN, MOTION_AFFINE
    n_levels : pyramid levels (multi-scale for large shifts)

    Returns
    -------
    warp_matrix : (2,3) or (3,3) numpy array
    correlation : float  (ECC score, 1.0 = perfect)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV required: pip install opencv-python")

    if warp_mode is None:
        warp_mode = cv2.MOTION_TRANSLATION

    # Downscale for speed + robustness
    ref_work = reference.copy()
    mov_work = moving.copy()

    # Ensure uint8 for OpenCV
    ref_u8 = (ref_work * 255).astype(np.uint8)
    mov_u8 = (mov_work * 255).astype(np.uint8)

    # Match sizes (crop to smaller)
    h = min(ref_u8.shape[0], mov_u8.shape[0])
    w = min(ref_u8.shape[1], mov_u8.shape[1])
    ref_u8 = ref_u8[:h, :w]
    mov_u8 = mov_u8[:h, :w]

    # Initial warp matrix
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_init = np.eye(3, 3, dtype=np.float32)
    else:
        warp_init = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max_iterations,
        termination_eps
    )

    # Multi-scale ECC
    warp_matrix = warp_init.copy()
    scale = 2 ** (n_levels - 1)
    for level in range(n_levels - 1, -1, -1):
        scale = 2 ** level
        ref_small = cv2.resize(
            ref_u8, (w // scale, h // scale),
            interpolation=cv2.INTER_AREA
        )
        mov_small = cv2.resize(
            mov_u8, (w // scale, h // scale),
            interpolation=cv2.INTER_AREA
        )

        if level < n_levels - 1:
            # Scale translation from previous level
            warp_matrix[0, 2] *= 2
            warp_matrix[1, 2] *= 2

        try:
            ecc, warp_matrix = cv2.findTransformECC(
                ref_small.astype(np.float32),
                mov_small.astype(np.float32),
                warp_matrix,
                warp_mode,
                criteria
            )
        except cv2.error as e:
            log.warning(f"ECC failed at level {level}: {e}. Using identity.")
            warp_matrix = warp_init.copy()
            ecc = 0.0

    # Scale back to full resolution
    warp_matrix[0, 2] *= scale
    warp_matrix[1, 2] *= scale

    log.debug(f"ECC coregistration: score={ecc:.4f}, shift=({warp_matrix[0,2]:.2f}, {warp_matrix[1,2]:.2f})")
    return warp_matrix, float(ecc)


def coregister_sift(
    reference: np.ndarray,
    moving: np.ndarray,
    max_features: int = 10000,
    match_ratio: float = 0.75
) -> Tuple[Optional[np.ndarray], int]:
    """
    Coregister using SIFT feature matching + RANSAC homography.
    More robust than ECC for cross-sensor, large offsets.

    Returns
    -------
    homography : (3,3) or None if insufficient matches
    n_inliers : number of RANSAC inliers
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV required")

    ref_u8 = (reference * 255).astype(np.uint8)
    mov_u8 = (moving * 255).astype(np.uint8)

    # Match sizes
    h = min(ref_u8.shape[0], mov_u8.shape[0])
    w = min(ref_u8.shape[1], mov_u8.shape[1])
    ref_u8 = ref_u8[:h, :w]
    mov_u8 = mov_u8[:h, :w]

    # SIFT detector
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = sift.detectAndCompute(ref_u8, None)
    kp2, des2 = sift.detectAndCompute(mov_u8, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        log.warning("Insufficient SIFT keypoints — returning identity")
        return np.eye(3, dtype=np.float32), 0

    # FLANN matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < match_ratio * n.distance]
    log.debug(f"SIFT: {len(good)} good matches from {len(matches)} total")

    if len(good) < 10:
        log.warning("Too few SIFT matches — identity transform")
        return np.eye(3, dtype=np.float32), 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    n_inliers = int(mask.sum()) if mask is not None else 0
    log.debug(f"RANSAC homography: {n_inliers} inliers")

    return H, n_inliers


def apply_warp(
    moving: np.ndarray,
    warp_matrix: np.ndarray,
    output_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Apply a warp matrix to an image."""
    import cv2
    h, w = (output_shape if output_shape else moving.shape[:2])

    if warp_matrix.shape == (2, 3):
        warped = cv2.warpAffine(
            moving.astype(np.float32), warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
    else:
        warped = cv2.warpPerspective(
            moving.astype(np.float32), warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR
        )
    return warped


def coregister_raster(
    reference_path: Path,
    moving_path: Path,
    output_path: Path,
    matrix_path: Optional[Path],
    cfg: Dict,
    checkpoint: StepCheckpoint,
    force: bool = False
) -> Optional[Path]:
    """
    Coregister a raster to the reference and save the result.

    Parameters
    ----------
    reference_path : Path to reference GeoTIFF (CTX)
    moving_path    : Path to image to coregister
    output_path    : Output coregistered GeoTIFF
    matrix_path    : Path to save/load warp matrix JSON
    cfg            : config dict
    checkpoint     : StepCheckpoint
    """
    import rasterio
    from rasterio.transform import from_bounds

    ck_key = f"coreg_{moving_path.name}"
    if not force and checkpoint.is_done("coregister", ck_key):
        if output_path.exists():
            log.info(f"[SKIP] Already coregistered: {moving_path.name}")
            return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Coregistering {moving_path.name} → {reference_path.name}")

    # Load reference and moving as grayscale
    ref_gray = load_as_gray(reference_path, band=1)
    mov_gray = load_as_gray(moving_path, band=1)

    coreg_cfg = cfg.get("preprocessing", {}).get("coregistration", {})
    method = coreg_cfg.get("method", "ecc")
    max_shift = coreg_cfg.get("max_shift_pixels", 50)

    # Load or compute warp matrix
    if matrix_path and matrix_path.exists() and not force:
        with open(matrix_path) as f:
            warp_data = json.load(f)
        warp_matrix = np.array(warp_data["matrix"], dtype=np.float32)
        log.debug(f"Loaded saved warp matrix from {matrix_path}")
    else:
        if method == "ecc":
            import cv2
            warp_matrix, score = coregister_ecc(
                ref_gray, mov_gray,
                warp_mode=cv2.MOTION_TRANSLATION,
                max_iterations=coreg_cfg.get("max_iterations", 1000),
                termination_eps=coreg_cfg.get("termination_eps", 1e-6)
            )
            # Clamp shift to max_shift
            warp_matrix[0, 2] = np.clip(warp_matrix[0, 2], -max_shift, max_shift)
            warp_matrix[1, 2] = np.clip(warp_matrix[1, 2], -max_shift, max_shift)
        elif method == "sift":
            warp_matrix, n_inliers = coregister_sift(ref_gray, mov_gray)
        else:
            log.warning(f"Unknown method '{method}', using identity")
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        if matrix_path:
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            with open(matrix_path, "w") as f:
                json.dump({"matrix": warp_matrix.tolist(), "method": method}, f)

    # Apply warp to all bands and write output
    with rasterio.open(moving_path) as src:
        meta = src.meta.copy()
        data = src.read()  # (C, H, W)

    ref_h, ref_w = ref_gray.shape
    warped_bands = []
    for b in range(data.shape[0]):
        band = data[b].astype(np.float32)
        warped = apply_warp(band, warp_matrix, output_shape=(ref_h, ref_w))
        warped_bands.append(warped)

    warped_data = np.stack(warped_bands, axis=0)

    # Update rasterio metadata
    with rasterio.open(reference_path) as ref_src:
        meta.update({
            "height": ref_h,
            "width": ref_w,
            "transform": ref_src.transform,
            "crs": ref_src.crs,
            "dtype": "float32",
            "compress": "deflate"
        })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(warped_data)

    checkpoint.mark_done("coregister", ck_key, {"output": str(output_path)})
    log.info(f"✓ Coregistered: {output_path}")
    return output_path


def coregister_site(
    site_name: str,
    hirise_paths: Dict[str, Path],
    ctx_path: Path,
    output_dir: Path,
    cfg: Dict,
    checkpoint: StepCheckpoint,
    force: bool = False
) -> Dict[str, Path]:
    """
    Coregister all HiRISE images for a site to the CTX base.
    Returns dict: {obs_id: coregistered_path}
    """
    results = {}
    matrices_dir = output_dir / "warp_matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    for obs_id, hirise_path in hirise_paths.items():
        if not hirise_path.exists():
            log.warning(f"HiRISE path not found: {hirise_path}")
            continue

        out_path = output_dir / f"{obs_id}_coregistered.tif"
        matrix_path = matrices_dir / f"{obs_id}_warp.json"

        result = coregister_raster(
            ctx_path, hirise_path, out_path, matrix_path, cfg, checkpoint, force
        )
        if result:
            results[obs_id] = result

    return results
