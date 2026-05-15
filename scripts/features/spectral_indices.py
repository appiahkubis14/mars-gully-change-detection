"""
Martian Spectral Indices
Computed from HiRISE (RED, BG, IR bands) and CRISM (multispectral).

Indices:
  RSI   - Reduced Slope Index: highlights bare rocky slopes (gully channels)
  DCI   - Dust Cover Index: fresh vs dust-covered surfaces
  SWI   - Short-Wave Index: CRISM carbonate/phyllosilicate indicator
  NDSI  - Normalized Difference Snow/Ice Index: CO2 frost detection
  REDNESS - Iron oxide proxy

Reference:
  Pelkey et al. 2007, JGR Planets (CRISM spectral parameters)
  McEwen et al. 2010, Science (HiRISE gully detection)
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger

log = get_logger("features.spectral")

EPS = 1e-8


def rsi(red: np.ndarray, blue: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Reduced Slope Index.
    Highlights bare rocky slopes characteristic of gully channels.
    High values → steep, rocky, low-dust surfaces (likely gully walls).
    RSI = (RED - BLUE) / (RED + BLUE + eps)
    """
    red = red.astype(np.float32)
    blue = blue.astype(np.float32)
    return ((red - blue) / (red + blue + eps)).astype(np.float32)


def dci(nir: np.ndarray, red: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Dust Cover Index.
    Distinguishes fresh (low dust) from heavily dust-mantled surfaces.
    Negative values → fresh, dust-free surface (active gullies).
    DCI = (NIR - RED) / (NIR + RED + eps)
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return ((nir - red) / (nir + red + eps)).astype(np.float32)


def swi(b2400: np.ndarray, b2300: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Short-Wave Index (CRISM only).
    Carbonate/phyllosilicate absorption feature at ~2300 nm.
    High values → carbonate or altered material.
    SWI = (B2400 - B2300) / (B2400 + B2300 + eps)
    """
    b2400 = b2400.astype(np.float32)
    b2300 = b2300.astype(np.float32)
    return ((b2400 - b2300) / (b2400 + b2300 + eps)).astype(np.float32)


def ndsi(green: np.ndarray, red: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Normalized Difference Snow/Ice Index (adapted for Mars CO2 frost).
    High values → CO2 frost or water ice (triggers gully activity).
    NDSI = (GREEN - RED) / (GREEN + RED + eps)
    """
    green = green.astype(np.float32)
    red = red.astype(np.float32)
    return ((green - red) / (green + red + eps)).astype(np.float32)


def redness_index(red: np.ndarray, blue: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Redness index: proxy for iron oxide (dust, oxidized soil).
    High values → heavily oxidized/dusty surface (inactive).
    REDNESS = RED / (BLUE + eps)
    """
    red = red.astype(np.float32)
    blue = blue.astype(np.float32)
    return (red / (blue + eps)).astype(np.float32)


def band_ratio(b1: np.ndarray, b2: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Generic band ratio b1/b2."""
    return (b1.astype(np.float32) / (b2.astype(np.float32) + eps))


def compute_all_indices(
    band_dict: Dict[str, np.ndarray],
    cfg: Dict
) -> Dict[str, np.ndarray]:
    """
    Compute all configured spectral indices from a band dictionary.

    Parameters
    ----------
    band_dict : dict mapping band names to 2D arrays
      Expected keys: "RED", "BLUE", "GREEN", "NIR", "B2300", "B2400"
      Missing keys are handled gracefully (index skipped with warning).
    cfg : config dict

    Returns
    -------
    dict mapping index name → 2D float32 array
    """
    indices = {}
    idx_cfg = cfg.get("feature_engineering", {}).get("spectral_indices", [])

    for idx_def in idx_cfg:
        name = idx_def["name"]
        requires = idx_def.get("requires", [])
        missing = [r for r in requires if r not in band_dict]

        if missing:
            log.debug(f"Skipping {name}: missing bands {missing}")
            continue

        try:
            if name == "RSI":
                indices[name] = rsi(band_dict["RED"], band_dict["BLUE"])
            elif name == "DCI":
                indices[name] = dci(band_dict["NIR"], band_dict["RED"])
            elif name == "SWI":
                indices[name] = swi(band_dict["B2400"], band_dict["B2300"])
            elif name == "NDSI":
                indices[name] = ndsi(band_dict["GREEN"], band_dict["RED"])
            elif name == "REDNESS":
                indices[name] = redness_index(band_dict["RED"], band_dict["BLUE"])
            else:
                log.warning(f"Unknown index {name}")
                continue
            log.debug(f"Computed {name}: min={indices[name].min():.3f}, max={indices[name].max():.3f}")
        except Exception as e:
            log.error(f"Failed to compute {name}: {e}")

    return indices


def hirise_to_band_dict(hirise_array: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Map HiRISE multi-band array (C, H, W) to named band dict.
    HiRISE band order: [0]=RED (panchromatic), [1]=BG (blue-green), [2]=IR
    """
    d = {}
    if hirise_array.shape[0] >= 1:
        d["RED"] = hirise_array[0]
    if hirise_array.shape[0] >= 2:
        d["BLUE"] = hirise_array[1]   # BG channel
        d["GREEN"] = hirise_array[1]  # approximate
    if hirise_array.shape[0] >= 3:
        d["NIR"] = hirise_array[2]    # IR channel
    return d


def load_hirise_bands(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load HiRISE GeoTIFF and return named band dict."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)  # (C, H, W)
        return hirise_to_band_dict(data)
    except Exception as e:
        log.error(f"Cannot load HiRISE bands from {path}: {e}")
        return None


def compute_indices_from_raster(
    raster_path: Path,
    cfg: Dict,
    sensor: str = "hirise"
) -> Dict[str, np.ndarray]:
    """Compute spectral indices from a raster file."""
    try:
        import rasterio
        with rasterio.open(raster_path) as src:
            data = src.read().astype(np.float32)
    except Exception as e:
        log.error(f"Cannot open raster {raster_path}: {e}")
        return {}

    if sensor == "hirise":
        band_dict = hirise_to_band_dict(data)
    else:
        # Generic: use band indices 0, 1, 2
        band_dict = {}
        if data.shape[0] >= 1:
            band_dict["RED"] = data[0]
        if data.shape[0] >= 2:
            band_dict["GREEN"] = data[1]
        if data.shape[0] >= 3:
            band_dict["BLUE"] = data[2]
        if data.shape[0] >= 4:
            band_dict["NIR"] = data[3]

    return compute_all_indices(band_dict, cfg)
