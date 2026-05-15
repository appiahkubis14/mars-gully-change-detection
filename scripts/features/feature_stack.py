"""
Feature Stack Builder
Combines HiRISE bands, spectral indices, texture, and morphometrics
into a single (C, H, W) numpy array ready for model training.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger, StepCheckpoint
from scripts.features.spectral_indices import compute_all_indices, hirise_to_band_dict
from scripts.features.texture import compute_multiscale_texture
from scripts.features.morphometric import compute_morphometrics_from_raster

log = get_logger("features.stack")


def resize_to_shape(
    arr: np.ndarray,
    target_h: int,
    target_w: int,
    order: int = 1
) -> np.ndarray:
    """Resize 2D array to target shape using zoom."""
    from scipy.ndimage import zoom
    if arr.shape == (target_h, target_w):
        return arr
    zoom_h = target_h / arr.shape[0]
    zoom_w = target_w / arr.shape[1]
    return zoom(arr, (zoom_h, zoom_w), order=order).astype(np.float32)


def normalize_feature(arr: np.ndarray) -> np.ndarray:
    """Normalize a feature map to [0, 1]."""
    arr = arr.astype(np.float32)
    lo = np.nanpercentile(arr, 2)
    hi = np.nanpercentile(arr, 98)
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    out = np.clip(arr, lo, hi)
    return ((out - lo) / (hi - lo)).astype(np.float32)


def build_feature_stack(
    hirise_path: Optional[Path],
    ctx_path: Optional[Path],
    mola_path: Optional[Path],
    cfg: Dict,
    target_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a (C, H, W) feature stack from available data sources.

    Returns
    -------
    stack : np.ndarray (C, H, W), float32, [0, 1]
    names : list of feature names (one per channel)
    """
    import rasterio

    features = []
    names = []

    # ── 1. HiRISE bands ──────────────────────────────────────────────────────
    if hirise_path and hirise_path.exists():
        with rasterio.open(hirise_path) as src:
            hirise_data = src.read().astype(np.float32)  # (C, H, W)
            if target_shape is None:
                target_shape = (src.height, src.width)

        for i in range(hirise_data.shape[0]):
            band = resize_to_shape(hirise_data[i], *target_shape)
            features.append(normalize_feature(band))
            names.append(f"hirise_b{i+1}")

        # Spectral indices from HiRISE
        band_dict = hirise_to_band_dict(hirise_data)
        indices = compute_all_indices(band_dict, cfg)
        for idx_name, idx_arr in indices.items():
            arr = resize_to_shape(idx_arr, *target_shape)
            features.append(normalize_feature(arr))
            names.append(f"idx_{idx_name.lower()}")

        # Texture from HiRISE RED band
        tex_cfg = cfg.get("feature_engineering", {}).get("texture", {})
        if tex_cfg.get("enabled", True):
            red_norm = normalize_feature(hirise_data[0])
            red_small = resize_to_shape(red_norm, *target_shape)
            tex_maps = compute_multiscale_texture(
                red_small,
                window_sizes=tex_cfg.get("window_sizes", [5]),
                features=tex_cfg.get("features", ["entropy", "contrast"]),
                levels=tex_cfg.get("levels", 64)
            )
            for tex_name, tex_arr in tex_maps.items():
                arr = resize_to_shape(tex_arr, *target_shape)
                features.append(normalize_feature(arr))
                names.append(f"tex_{tex_name}")
    else:
        log.warning("HiRISE data not available for feature stack")

    # ── 2. CTX band ──────────────────────────────────────────────────────────
    if ctx_path and ctx_path.exists():
        with rasterio.open(ctx_path) as src:
            ctx_data = src.read(1).astype(np.float32)
            if target_shape is None:
                target_shape = ctx_data.shape

        ctx_rs = resize_to_shape(ctx_data, *target_shape)
        features.append(normalize_feature(ctx_rs))
        names.append("ctx_pan")
    else:
        log.warning("CTX data not available for feature stack")

    # ── 3. MOLA morphometrics ────────────────────────────────────────────────
    if mola_path and mola_path.exists():
        morph = compute_morphometrics_from_raster(mola_path, cfg)
        for feat_name, feat_arr in morph.items():
            if target_shape:
                feat_arr = resize_to_shape(feat_arr, *target_shape)
            features.append(normalize_feature(feat_arr))
            names.append(f"dem_{feat_name}")
    else:
        log.warning("MOLA data not available for feature stack")

    if not features:
        raise ValueError("No features could be computed — check input files")

    stack = np.stack(features, axis=0)  # (C, H, W)
    log.info(f"Feature stack: {stack.shape} ({len(names)} channels)")
    log.debug(f"Channels: {names}")

    return stack, names


def save_feature_stack(
    stack: np.ndarray,
    names: List[str],
    output_path: Path,
    metadata: Optional[Dict] = None
) -> None:
    """Save feature stack as .npz with channel names."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"stack": stack, "names": np.array(names)}
    if metadata:
        for k, v in metadata.items():
            save_dict[k] = v
    np.savez_compressed(output_path, **save_dict)
    log.info(f"Saved feature stack: {output_path} ({stack.shape})")


def save_feature_stack_geotiff(
    stack: np.ndarray,
    names: List[str],
    output_path: Path,
    reference_path: Path
) -> None:
    """Save feature stack as multi-band GeoTIFF (for QGIS inspection)."""
    import rasterio

    with rasterio.open(reference_path) as ref:
        transform = ref.transform
        crs = ref.crs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=stack.shape[1],
        width=stack.shape[2],
        count=stack.shape[0],
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="deflate"
    ) as dst:
        dst.write(stack)
        dst.update_tags(channels=",".join(names))
    log.info(f"Saved feature GeoTIFF: {output_path}")


def build_site_features(
    site_name: str,
    hirise_paths: Dict[str, Path],
    ctx_path: Optional[Path],
    mola_path: Optional[Path],
    output_dir: Path,
    cfg: Dict,
    checkpoint: StepCheckpoint,
    force: bool = False
) -> Dict[str, Path]:
    """
    Build and save feature stacks for all HiRISE dates at a site.
    Returns {obs_id: npz_path}
    """
    results = {}

    for obs_id, hirise_path in hirise_paths.items():
        ck_key = f"features_{obs_id}"
        out_npz = output_dir / f"{obs_id}_features.npz"

        if not force and checkpoint.is_done("features", ck_key):
            if out_npz.exists():
                log.info(f"[SKIP] Features exist: {obs_id}")
                results[obs_id] = out_npz
                continue

        log.info(f"Building features for {obs_id}")
        try:
            stack, names = build_feature_stack(hirise_path, ctx_path, mola_path, cfg)
            save_feature_stack(stack, names, out_npz, metadata={"obs_id": obs_id, "site": site_name})

            # Also save GeoTIFF if HiRISE reference available
            if hirise_path.exists():
                tif_out = output_dir / f"{obs_id}_features.tif"
                save_feature_stack_geotiff(stack, names, tif_out, hirise_path)

            checkpoint.mark_done("features", ck_key, {
                "shape": list(stack.shape),
                "channels": names,
                "npz": str(out_npz)
            })
            results[obs_id] = out_npz

        except Exception as e:
            log.error(f"Feature engineering failed for {obs_id}: {e}")
            checkpoint.mark_failed("features", ck_key, str(e))

    return results
