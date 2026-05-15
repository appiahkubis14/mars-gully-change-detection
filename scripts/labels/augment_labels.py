"""
Label Augmentation: morphological operations, rotations, flips.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation, binary_erosion,
    rotate as scipy_rotate, zoom as scipy_zoom,
    label as scipy_label
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger

log = get_logger("labels.augment")


def augment_mask(
    image: np.ndarray,
    mask: np.ndarray,
    rotation_angles: Optional[List[int]] = None,
    flip: bool = True,
    scale_range: Optional[Tuple[float, float]] = None,
    dilate_iterations: int = 0,
    erode_iterations: int = 0,
    rng: Optional[np.random.Generator] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate augmented (image, mask) pairs.

    Parameters
    ----------
    image : (C, H, W) or (H, W) float array
    mask  : (H, W) binary mask
    rotation_angles : list of angles in degrees, e.g. [0, 90, 180, 270]
    flip  : whether to add horizontal/vertical flips
    scale_range : (min_scale, max_scale) for random scaling

    Returns
    -------
    images, masks : lists of augmented arrays
    """
    if rng is None:
        rng = np.random.default_rng(42)

    images_out = []
    masks_out = []

    if rotation_angles is None:
        rotation_angles = [0]

    def _rot(img, mask, angle):
        """Rotate image and mask by angle degrees."""
        if img.ndim == 3:
            rotated_img = np.stack([
                scipy_rotate(img[c], angle, reshape=False, mode="nearest")
                for c in range(img.shape[0])
            ], axis=0)
        else:
            rotated_img = scipy_rotate(img, angle, reshape=False, mode="nearest")
        rotated_mask = scipy_rotate(mask.astype(float), angle, reshape=False, mode="constant", cval=0)
        rotated_mask = (rotated_mask > 0.5).astype(np.uint8)
        return rotated_img, rotated_mask

    for angle in rotation_angles:
        img_r, msk_r = _rot(image, mask, angle)

        # Optional morphological augmentation
        if dilate_iterations > 0:
            msk_r = binary_dilation(msk_r, iterations=dilate_iterations).astype(np.uint8)
        if erode_iterations > 0:
            msk_r = binary_erosion(msk_r, iterations=erode_iterations).astype(np.uint8)

        images_out.append(img_r)
        masks_out.append(msk_r)

        if flip:
            # Horizontal flip
            if image.ndim == 3:
                img_fh = img_r[:, :, ::-1].copy()
            else:
                img_fh = img_r[:, ::-1].copy()
            msk_fh = msk_r[:, ::-1].copy()
            images_out.append(img_fh)
            masks_out.append(msk_fh)

            # Vertical flip
            if image.ndim == 3:
                img_fv = img_r[:, ::-1, :].copy()
            else:
                img_fv = img_r[::-1, :].copy()
            msk_fv = msk_r[::-1, :].copy()
            images_out.append(img_fv)
            masks_out.append(msk_fv)

    # Optional scale augmentation
    if scale_range:
        scale = rng.uniform(*scale_range)
        if image.ndim == 3:
            C, H, W = image.shape
            scaled_img = scipy_zoom(image, (1, scale, scale), order=1)
            scaled_mask = scipy_zoom(mask.astype(float), (scale, scale), order=0)
        else:
            scaled_img = scipy_zoom(image, scale, order=1)
            scaled_mask = scipy_zoom(mask.astype(float), scale, order=0)
        scaled_mask = (scaled_mask > 0.5).astype(np.uint8)
        images_out.append(scaled_img)
        masks_out.append(scaled_mask)

    return images_out, masks_out


def extract_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 512,
    stride: int = 256,
    min_fg: float = 0.01
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract overlapping patches from image and mask.
    Only returns patches with at least `min_fg` fraction of foreground.

    Parameters
    ----------
    image : (C, H, W)
    mask  : (H, W)
    """
    C, H, W = image.shape
    img_patches = []
    msk_patches = []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            img_p = image[:, i:i + patch_size, j:j + patch_size]
            msk_p = mask[i:i + patch_size, j:j + patch_size]

            if msk_p.shape != (patch_size, patch_size):
                continue
            if msk_p.mean() >= min_fg:
                img_patches.append(img_p)
                msk_patches.append(msk_p)

    # Add some background patches too (random sample)
    n_bg = max(1, len(img_patches) // 3)
    bg_added = 0
    for i in range(0, H - patch_size + 1, stride * 2):
        for j in range(0, W - patch_size + 1, stride * 2):
            if bg_added >= n_bg:
                break
            img_p = image[:, i:i + patch_size, j:j + patch_size]
            msk_p = mask[i:i + patch_size, j:j + patch_size]
            if msk_p.shape == (patch_size, patch_size) and msk_p.mean() < min_fg:
                img_patches.append(img_p)
                msk_patches.append(msk_p)
                bg_added += 1

    log.debug(f"Extracted {len(img_patches)} patches ({len(img_patches) - bg_added} fg, {bg_added} bg)")
    return img_patches, msk_patches


def save_patches(
    img_patches: List[np.ndarray],
    msk_patches: List[np.ndarray],
    img_dir: Path,
    msk_dir: Path,
    prefix: str = "patch"
) -> int:
    """Save patches to disk as .npy files. Returns count saved."""
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, (img, msk) in enumerate(zip(img_patches, msk_patches)):
        np.save(img_dir / f"{prefix}_{i:06d}.npy", img.astype(np.float32))
        np.save(msk_dir / f"{prefix}_{i:06d}.npy", msk.astype(np.uint8))
        n += 1
    return n
