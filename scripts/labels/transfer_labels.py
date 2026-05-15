"""
Transfer Label Generation
Adapts Earth mining/landslide masks to serve as initial Mars gully labels.

Strategy:
1. Load Earth binary masks (e.g. from Project 4 mining detection)
2. Apply morphological transformations to match Mars gully shapes:
   - Gullies are elongated, narrow, dendritic
   - Apply skeletonization + dilation to get channel-like shapes
3. Apply random affine transforms (rotate, scale, flip)
4. Paste onto Mars DEM slope map in high-slope areas (>15°)
5. Save as binary masks aligned to HiRISE grid

If no Earth masks available, generate synthetic masks from:
- MOLA slope gradient > 15° → gully candidate zone
- Add synthetic narrow channel masks

Reference:
  Laugier et al. 2021, Remote Sensing (domain adaptation for Mars)
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation, binary_erosion, label as scipy_label,
    rotate as scipy_rotate, zoom as scipy_zoom
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger, StepCheckpoint

log = get_logger("labels.transfer")


def load_earth_mask(mask_path: Path) -> Optional[np.ndarray]:
    """Load a binary Earth mask (GeoTIFF or numpy)."""
    try:
        if mask_path.suffix in (".tif", ".tiff"):
            import rasterio
            with rasterio.open(mask_path) as src:
                data = src.read(1)
            return (data > 0).astype(np.uint8)
        elif mask_path.suffix == ".npy":
            return (np.load(mask_path) > 0).astype(np.uint8)
        else:
            log.warning(f"Unsupported mask format: {mask_path.suffix}")
            return None
    except Exception as e:
        log.error(f"Cannot load Earth mask {mask_path}: {e}")
        return None


def extract_gully_shaped_patches(
    earth_mask: np.ndarray,
    patch_size: int = 512,
    n_patches: int = 50,
    min_fg_fraction: float = 0.02,
    max_fg_fraction: float = 0.40
) -> List[np.ndarray]:
    """
    Extract patches from Earth mask that contain elongated features.
    Filters for patches with appropriate foreground fraction.
    """
    h, w = earth_mask.shape
    patches = []
    attempts = 0
    max_attempts = n_patches * 20

    while len(patches) < n_patches and attempts < max_attempts:
        attempts += 1
        i = np.random.randint(0, max(1, h - patch_size))
        j = np.random.randint(0, max(1, w - patch_size))
        patch = earth_mask[i:i + patch_size, j:j + patch_size]

        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            continue

        fg_frac = patch.mean()
        if min_fg_fraction <= fg_frac <= max_fg_fraction:
            patches.append(patch.copy())

    log.debug(f"Extracted {len(patches)} Earth patches (attempted {attempts})")
    return patches


def morphological_reshape_to_gully(
    patch: np.ndarray,
    elongation: float = 3.0,
    width_pixels: int = 5
) -> np.ndarray:
    """
    Reshape an Earth mask patch to have gully-like morphology:
    - Erode to skeleton
    - Dilate into narrow elongated shapes
    - Apply thinning to simulate channel-like structures
    """
    from scipy.ndimage import label, binary_fill_holes, binary_erosion, binary_dilation

    mask = patch.astype(bool)

    # Erode to thin out blobs
    struct = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(mask, structure=struct, iterations=3)

    # Re-dilate to narrow channels
    dilated = binary_dilation(eroded, structure=struct, iterations=width_pixels)

    # Mask: only keep pixels that are elongated components
    labeled, n_components = label(dilated)
    result = np.zeros_like(dilated, dtype=np.uint8)

    for c in range(1, n_components + 1):
        comp = labeled == c
        # Check elongation using bbox
        rows, cols = np.where(comp)
        if len(rows) < 10:
            continue
        h = rows.max() - rows.min() + 1
        w = cols.max() - cols.min() + 1
        aspect = max(h, w) / (min(h, w) + 1)
        if aspect >= elongation:
            result[comp] = 1

    return result


def generate_synthetic_gully_mask(
    slope_map: np.ndarray,
    aspect_map: np.ndarray,
    slope_threshold: float = 15.0,
    n_gullies: int = 20,
    gully_length_px: int = 50,
    gully_width_px: int = 3,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate a synthetic gully mask from slope and aspect maps.
    Places synthetic narrow channel masks on high-slope terrain.

    This provides initial labels when no Earth data is available.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    h, w = slope_map.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Find high-slope candidate pixels
    candidates = np.argwhere(slope_map > slope_threshold)
    if len(candidates) == 0:
        log.warning("No high-slope pixels found — using random mask")
        candidates = np.argwhere(np.ones((h, w), dtype=bool))

    for _ in range(n_gullies):
        # Pick a random high-slope starting point
        idx = rng.integers(0, len(candidates))
        start_i, start_j = candidates[idx]

        # Determine flow direction from aspect
        if aspect_map is not None:
            asp = aspect_map[start_i, start_j]
            di = -np.cos(np.radians(asp))  # flow direction (downslope)
            dj = np.sin(np.radians(asp))
        else:
            angle = rng.uniform(0, 360)
            di = -np.cos(np.radians(angle))
            dj = np.sin(np.radians(angle))

        # Draw gully channel
        for step in range(gully_length_px):
            pi = int(start_i + di * step)
            pj = int(start_j + dj * step)
            if 0 <= pi < h and 0 <= pj < w:
                for ddi in range(-gully_width_px // 2, gully_width_px // 2 + 1):
                    for ddj in range(-gully_width_px // 2, gully_width_px // 2 + 1):
                        ni, nj = pi + ddi, pj + ddj
                        if 0 <= ni < h and 0 <= nj < w:
                            mask[ni, nj] = 1

    # Add alcoves at gully heads (dilated circles)
    for _ in range(n_gullies):
        idx = rng.integers(0, len(candidates))
        ci, cj = candidates[idx]
        radius = rng.integers(5, 20)
        for di2 in range(-radius, radius + 1):
            for dj2 in range(-radius, radius + 1):
                if di2**2 + dj2**2 <= radius**2:
                    ni, nj = ci + di2, cj + dj2
                    if 0 <= ni < h and 0 <= nj < w:
                        mask[ni, nj] = 1

    log.debug(f"Synthetic mask: {mask.sum()} gully pixels ({mask.mean()*100:.2f}%)")
    return mask


def transfer_labels(
    earth_mask_paths: List[Path],
    target_shape: Tuple[int, int],
    slope_map: Optional[np.ndarray],
    aspect_map: Optional[np.ndarray],
    cfg: Dict,
    n_output_masks: int = 5,
    rng: Optional[np.random.Generator] = None
) -> List[np.ndarray]:
    """
    Generate training labels by transfer from Earth masks or synthetically.

    Returns
    -------
    List of binary masks, each shape = target_shape
    """
    if rng is None:
        rng = np.random.default_rng(42)

    masks = []

    # Try loading Earth masks
    earth_masks = []
    for path in earth_mask_paths:
        m = load_earth_mask(path)
        if m is not None:
            earth_masks.append(m)

    if earth_masks:
        log.info(f"Loaded {len(earth_masks)} Earth masks for transfer")
        for earth_mask in earth_masks:
            patches = extract_gully_shaped_patches(
                earth_mask,
                patch_size=min(target_shape[0], earth_mask.shape[0]),
                n_patches=max(1, n_output_masks // len(earth_masks))
            )
            for patch in patches:
                shaped = morphological_reshape_to_gully(patch)
                if shaped.shape != target_shape:
                    zoom_h = target_shape[0] / shaped.shape[0]
                    zoom_w = target_shape[1] / shaped.shape[1]
                    shaped = scipy_zoom(shaped, (zoom_h, zoom_w), order=0).astype(np.uint8)
                masks.append(shaped)
    else:
        log.info("No Earth masks available — using synthetic label generation")

    # Fill remaining with synthetic masks
    n_synthetic = max(0, n_output_masks - len(masks))
    if n_synthetic > 0 and slope_map is not None:
        log.info(f"Generating {n_synthetic} synthetic gully masks")
        for _ in range(n_synthetic):
            m = generate_synthetic_gully_mask(
                slope_map, aspect_map, rng=rng
            )
            if m.shape != target_shape:
                zoom_h = target_shape[0] / m.shape[0]
                zoom_w = target_shape[1] / m.shape[1]
                m = scipy_zoom(m, (zoom_h, zoom_w), order=0).astype(np.uint8)
            masks.append(m)

    log.info(f"Generated {len(masks)} label masks")
    return masks


def save_label_mask(
    mask: np.ndarray,
    output_path: Path,
    reference_path: Optional[Path] = None
) -> None:
    """Save binary label mask as GeoTIFF."""
    import rasterio
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if reference_path and reference_path.exists():
        with rasterio.open(reference_path) as ref:
            meta = ref.meta.copy()
            meta.update({"count": 1, "dtype": "uint8", "compress": "deflate"})
    else:
        meta = {
            "driver": "GTiff", "count": 1, "dtype": "uint8",
            "height": mask.shape[0], "width": mask.shape[1],
            "compress": "deflate"
        }

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(mask.astype(np.uint8), 1)


def generate_labels_for_site(
    site_name: str,
    target_shape: Tuple[int, int],
    slope_map: Optional[np.ndarray],
    aspect_map: Optional[np.ndarray],
    output_dir: Path,
    cfg: Dict,
    checkpoint: StepCheckpoint,
    earth_mask_dir: Optional[Path] = None,
    reference_raster: Optional[Path] = None,
    force: bool = False
) -> List[Path]:
    """Generate and save label masks for a site."""
    ck_key = f"labels_{site_name}"
    if not force and checkpoint.is_done("labels", ck_key):
        existing = list(output_dir.glob(f"{site_name}_label_*.tif"))
        if existing:
            log.info(f"[SKIP] Labels exist for {site_name}")
            return existing

    output_dir.mkdir(parents=True, exist_ok=True)

    earth_paths = []
    if earth_mask_dir and earth_mask_dir.exists():
        earth_paths = list(earth_mask_dir.glob("*.tif")) + list(earth_mask_dir.glob("*.npy"))

    masks = transfer_labels(
        earth_paths, target_shape,
        slope_map, aspect_map,
        cfg, n_output_masks=5
    )

    saved_paths = []
    for i, mask in enumerate(masks):
        out_path = output_dir / f"{site_name}_label_{i:03d}.tif"
        save_label_mask(mask, out_path, reference_raster)
        saved_paths.append(out_path)

    checkpoint.mark_done("labels", ck_key, {"n_masks": len(saved_paths)})
    log.info(f"✓ Saved {len(saved_paths)} label masks for {site_name}")
    return saved_paths
