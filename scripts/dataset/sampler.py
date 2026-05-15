"""
Spatial Block Cross-Validation Sampler
Divides the study area into k×k blocks and assigns them to folds.
This prevents spatial autocorrelation leakage between train/val splits.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger

log = get_logger("dataset.sampler")


class SpatialBlockSampler:
    """
    Splits data into spatial blocks for cross-validation.

    Parameters
    ----------
    n_folds    : number of CV folds
    block_size_km : block size in km
    pixel_size_m  : spatial resolution of input data
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        n_folds: int = 5,
        block_size_km: float = 2.0,
        pixel_size_m: float = 6.0,
        rng: Optional[np.random.Generator] = None
    ):
        self.H = image_height
        self.W = image_width
        self.n_folds = n_folds
        self.block_size_px = int((block_size_km * 1000) / pixel_size_m)
        self.rng = rng or np.random.default_rng(42)

        # Number of blocks in each dimension
        self.n_blocks_h = max(1, self.H // self.block_size_px)
        self.n_blocks_w = max(1, self.W // self.block_size_px)
        self.n_blocks = self.n_blocks_h * self.n_blocks_w

        log.info(
            f"Spatial blocks: {self.n_blocks_h}×{self.n_blocks_w} = {self.n_blocks} blocks "
            f"({self.block_size_px}px each)"
        )

        # Assign blocks to folds randomly
        block_ids = np.arange(self.n_blocks)
        self.rng.shuffle(block_ids)
        self.block_fold = np.array([b % n_folds for b in range(self.n_blocks)])

    def get_fold_mask(self, fold: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get boolean masks for train and validation pixels in a fold.

        Returns
        -------
        train_mask : (H, W) bool
        val_mask   : (H, W) bool
        """
        train_mask = np.zeros((self.H, self.W), dtype=bool)
        val_mask = np.zeros((self.H, self.W), dtype=bool)

        for block_id in range(self.n_blocks):
            block_row = block_id // self.n_blocks_w
            block_col = block_id % self.n_blocks_w

            r_start = block_row * self.block_size_px
            r_end = min(r_start + self.block_size_px, self.H)
            c_start = block_col * self.block_size_px
            c_end = min(c_start + self.block_size_px, self.W)

            if self.block_fold[block_id] == fold:
                val_mask[r_start:r_end, c_start:c_end] = True
            else:
                train_mask[r_start:r_end, c_start:c_end] = True

        return train_mask, val_mask

    def get_patch_fold_assignments(
        self,
        patch_coords: List[Tuple[int, int]],
        patch_size: int
    ) -> np.ndarray:
        """
        Assign each patch to a fold based on its center pixel.

        Parameters
        ----------
        patch_coords : list of (row, col) top-left corners

        Returns
        -------
        fold_ids : array of fold assignments (one per patch)
        """
        fold_ids = np.zeros(len(patch_coords), dtype=int)
        for i, (r, c) in enumerate(patch_coords):
            center_r = r + patch_size // 2
            center_c = c + patch_size // 2
            block_row = min(center_r // self.block_size_px, self.n_blocks_h - 1)
            block_col = min(center_c // self.block_size_px, self.n_blocks_w - 1)
            block_id = block_row * self.n_blocks_w + block_col
            fold_ids[i] = self.block_fold[min(block_id, self.n_blocks - 1)]
        return fold_ids
