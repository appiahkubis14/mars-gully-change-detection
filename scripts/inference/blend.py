"""
blend.py
========
Window functions for seamless sliding-window prediction blending.

The Hanning (raised cosine) window tapers prediction weights toward tile
edges, eliminating boundary artifacts when overlapping tiles are averaged.
"""

from __future__ import annotations

import numpy as np


def hanning_weight_map(tile_size: int) -> np.ndarray:
    """
    Generate a 2-D Hanning window of shape (tile_size, tile_size).

    The window is the outer product of two 1-D Hanning windows, giving
    a smooth taper from 0 at edges to 1 at the centre.  Used to weight
    overlapping tile predictions during accumulation so that edge pixels
    contribute less and seams are invisible in the blended output.

    Args:
        tile_size (int): Side length of the square tile in pixels.

    Returns:
        np.ndarray: float32 weight map, shape (tile_size, tile_size).
    """
    win_1d = np.hanning(tile_size).astype(np.float32)
    win_2d = np.outer(win_1d, win_1d)
    # avoid exact zeros at the border (would leave un-weighted pixels)
    win_2d = np.clip(win_2d, 1e-6, None)
    return win_2d


def linear_weight_map(tile_size: int) -> np.ndarray:
    """
    Generate a 2-D linear (tent) weight map of shape (tile_size, tile_size).

    Linearly increases from 0 at edges to 1 at the centre, providing a
    simpler alternative to the Hanning window.

    Args:
        tile_size (int): Side length of the square tile in pixels.

    Returns:
        np.ndarray: float32 weight map, shape (tile_size, tile_size).
    """
    half = tile_size / 2.0
    coords = np.abs(np.arange(tile_size, dtype=np.float32) - (half - 0.5))
    win_1d = (half - coords) / half
    win_1d = win_1d.clip(1e-6, None)
    win_2d = np.outer(win_1d, win_1d)
    return win_2d.astype(np.float32)


def gaussian_weight_map(tile_size: int, sigma_fraction: float = 0.35) -> np.ndarray:
    """
    Generate a 2-D Gaussian weight map of shape (tile_size, tile_size).

    Args:
        tile_size      (int):   Side length in pixels.
        sigma_fraction (float): Sigma as a fraction of tile_size.

    Returns:
        np.ndarray: float32 weight map.
    """
    sigma = tile_size * sigma_fraction
    coords = np.arange(tile_size, dtype=np.float32) - (tile_size - 1) / 2.0
    gauss_1d = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    win_2d = np.outer(gauss_1d, gauss_1d)
    win_2d = win_2d.clip(1e-6, None)
    return win_2d.astype(np.float32)


def get_weight_map(method: str, tile_size: int) -> np.ndarray:
    """
    Factory: return the requested weight map for a given tile size.

    Args:
        method    (str): 'hanning' | 'linear' | 'gaussian'.
        tile_size (int): Tile side length in pixels.

    Returns:
        np.ndarray: float32 weight map of shape (tile_size, tile_size).

    Raises:
        ValueError: If method is not recognised.
    """
    method = method.lower()
    if method == "hanning":
        return hanning_weight_map(tile_size)
    elif method in ("linear", "tent"):
        return linear_weight_map(tile_size)
    elif method == "gaussian":
        return gaussian_weight_map(tile_size)
    else:
        raise ValueError(
            f"Unknown blending method: {method!r}. "
            "Choose 'hanning', 'linear', or 'gaussian'."
        )
