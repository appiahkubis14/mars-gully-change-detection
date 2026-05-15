"""
predict.py
==========
Sliding-window inference for the Mars Gully U-Net.

Runs the trained model over arbitrarily large GeoTIFF feature stacks
using overlapping tiles (default 512×512, stride 256) blended with a
Hanning window to produce seamless probability maps.

Usage (via main.py)
-------------------
    python main.py --step infer
    python main.py --step infer --site gasa --date 2015
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import torch

from scripts.models.unet import build_model
from scripts.inference.blend import hanning_weight_map
from scripts.utils import load_config, StepCheckpoint

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------

def load_trained_model(
    cfg: dict,
    ckpt_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Load U-Net from the best checkpoint (or explicit path)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt_path is None:
        ckpt_path = str(Path(cfg["checkpoint"]["save_dir"]) / "unet_best.pth")
    state = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    log.info(f"Model loaded from {ckpt_path}  (best IoU={state.get('best_iou', 0):.4f})")
    return model


# ---------------------------------------------------------------------------
# Feature stack reader
# ---------------------------------------------------------------------------

def read_feature_stack(feature_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Read a feature stack GeoTIFF → (C, H, W) float32 array + rasterio profile.

    NaNs are replaced with 0.
    """
    with rasterio.open(feature_path) as src:
        arr = src.read().astype(np.float32)   # (C, H, W)
        profile = src.profile
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr, profile


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    feature_stack: np.ndarray,
    tile_size: int = 512,
    stride: int = 256,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    n_channels: int = 8,
) -> np.ndarray:
    """
    Run the model over a (C, H, W) feature stack using overlapping tiles.

    Returns a (H, W) probability map in [0, 1].

    Args:
        model          : Trained U-Net (eval mode).
        feature_stack  : (C, H, W) float32 array, C = n_channels.
        tile_size      : Spatial size of each tile (pixels).
        stride         : Step between tile starts; overlap = tile_size - stride.
        batch_size     : Number of tiles to process in one forward pass.
        device         : Torch device.
        n_channels     : Expected input channels; stack is sliced/padded to match.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C, H, W = feature_stack.shape

    # ---- fix channel count ----
    if C < n_channels:
        pad = np.zeros((n_channels - C, H, W), dtype=np.float32)
        feature_stack = np.concatenate([feature_stack, pad], axis=0)
    elif C > n_channels:
        feature_stack = feature_stack[:n_channels]

    # ---- accumulation buffers ----
    prob_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)
    window = hanning_weight_map(tile_size)  # (tile_size, tile_size)

    # ---- tile coordinates ----
    row_starts = list(range(0, max(H - tile_size, 0) + 1, stride))
    col_starts = list(range(0, max(W - tile_size, 0) + 1, stride))
    if row_starts and row_starts[-1] + tile_size < H:
        row_starts.append(H - tile_size)
    if col_starts and col_starts[-1] + tile_size < W:
        col_starts.append(W - tile_size)

    coords = [
        (r, c)
        for r in row_starts
        for c in col_starts
    ]
    log.info(
        f"Sliding window: {len(coords)} tiles | "
        f"tile={tile_size} stride={stride} | image={H}×{W}"
    )

    # ---- batch inference ----
    for batch_start in range(0, len(coords), batch_size):
        batch_coords = coords[batch_start: batch_start + batch_size]
        tiles: List[np.ndarray] = []

        for (r, c) in batch_coords:
            r_end = min(r + tile_size, H)
            c_end = min(c + tile_size, W)
            tile = feature_stack[:, r:r_end, c:c_end]

            # pad if the tile is smaller than tile_size (edge tiles)
            ph = tile_size - tile.shape[1]
            pw = tile_size - tile.shape[2]
            if ph > 0 or pw > 0:
                tile = np.pad(tile, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            tiles.append(tile)

        batch_tensor = torch.from_numpy(np.stack(tiles)).to(device)
        with torch.cuda.amp.autocast():
            logits = model(batch_tensor)            # (B, 1, T, T)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B, T, T)

        # ---- accumulate ----
        for i, (r, c) in enumerate(batch_coords):
            r_end = min(r + tile_size, H)
            c_end = min(c + tile_size, W)
            h_crop = r_end - r
            w_crop = c_end - c
            prob_map[r:r_end, c:c_end] += probs[i, :h_crop, :w_crop] * window[:h_crop, :w_crop]
            weight_map[r:r_end, c:c_end] += window[:h_crop, :w_crop]

    # ---- normalise by accumulated weights ----
    weight_map = np.where(weight_map == 0, 1.0, weight_map)
    prob_map /= weight_map
    return prob_map.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# GeoTIFF writer
# ---------------------------------------------------------------------------

def save_probability_map(
    prob_map: np.ndarray,
    profile: dict,
    output_path: Path,
) -> None:
    """Save a (H, W) float32 probability map as a single-band GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        predictor=3,
    )
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(prob_map[np.newaxis].astype(np.float32))
    log.info(f"Probability map saved → {output_path}")


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def run_inference(cfg_path: str = "config.yaml") -> None:
    """
    Run inference on all available feature stacks for all dates and sites.

    Skips already-processed files (checkpoint resume).
    """
    cfg = load_config(cfg_path)
    inf_cfg = cfg["inference"]
    tile_size = inf_cfg["sliding_window"]["tile_size"]
    stride = inf_cfg["sliding_window"]["stride"]
    batch_size = inf_cfg["sliding_window"]["batch_size"]
    n_channels = cfg["training"]["in_channels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(cfg, device=device)

    feature_dir = Path("data/processed/feature_stacks")
    output_dir = Path("data/outputs/probability_maps")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_files = sorted(feature_dir.glob("*.tif"))
    if not feature_files:
        log.warning(f"No feature stack GeoTIFFs found in {feature_dir}. Run --step features first.")
        return

    log.info(f"Found {len(feature_files)} feature stack(s) to process.")

    ckpt = StepCheckpoint("data/outputs/inference_checkpoint.json")

    for feat_path in feature_files:
        stem = feat_path.stem
        out_path = output_dir / f"{stem}_prob.tif"

        if ckpt.is_done(f"infer_{stem}"):
            log.info(f"[SKIP] {stem} already processed.")
            continue

        log.info(f"Processing: {feat_path.name}")
        feature_stack, profile = read_feature_stack(feat_path)

        prob_map = sliding_window_predict(
            model=model,
            feature_stack=feature_stack,
            tile_size=tile_size,
            stride=stride,
            batch_size=batch_size,
            device=device,
            n_channels=n_channels,
        )

        save_probability_map(prob_map, profile, out_path)
        ckpt.mark_done(f"infer_{stem}")

    log.info("Inference complete.")
