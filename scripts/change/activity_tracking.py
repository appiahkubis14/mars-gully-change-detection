"""
activity_tracking.py
====================
Track individual gully instances across multiple dates using connected
components and IoU-based matching.

Algorithm
---------
For each date's binary mask:
  1. Label connected components (ndimage.label).
  2. Match components to the previous date's components via IoU.
  3. Assign persistent IDs; unmatched new components get new IDs.
  4. Compute per-component area, centroid, expansion rate (Kalman).

Outputs
-------
  data/outputs/change_detection/
    ├── activity_tracks.json
    └── active_sites.geojson   (active gullies across all dates)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from scripts.change.kalman_smooth import KalmanAreaTracker
from scripts.utils import load_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connected-component labelling
# ---------------------------------------------------------------------------

def get_components(
    binary: np.ndarray,
    min_pixels: int = 5,
) -> Tuple[np.ndarray, int, List[dict]]:
    """
    Label connected components and return component metadata.

    Args:
        binary     : (H, W) uint8 binary mask.
        min_pixels : Minimum size to keep (avoids noise).

    Returns:
        labelled  : (H, W) int32 label array (0 = background).
        n_comp    : Number of valid components.
        comp_info : List of dicts with 'id', 'area_px', 'centroid'.
    """
    struct = ndimage.generate_binary_structure(2, 2)
    labelled_raw, n_raw = ndimage.label(binary, structure=struct)
    sizes = ndimage.sum(binary, labelled_raw, range(1, n_raw + 1))

    labelled = np.zeros_like(labelled_raw)
    comp_info = []
    new_id = 1
    for raw_id, size in enumerate(sizes, start=1):
        if size >= min_pixels:
            mask_c = labelled_raw == raw_id
            labelled[mask_c] = new_id
            cy, cx = ndimage.center_of_mass(mask_c)
            comp_info.append({
                "id": new_id,
                "area_px": int(size),
                "centroid_row": float(cy),
                "centroid_col": float(cx),
            })
            new_id += 1

    return labelled, len(comp_info), comp_info


# ---------------------------------------------------------------------------
# IoU-based component matching
# ---------------------------------------------------------------------------

def match_components(
    labelled_prev: np.ndarray,
    labelled_curr: np.ndarray,
    iou_threshold: float = 0.3,
) -> Dict[int, Optional[int]]:
    """
    Match components in `labelled_curr` to those in `labelled_prev` by IoU.

    Args:
        labelled_prev  : (H, W) int32 label array at t-1.
        labelled_curr  : (H, W) int32 label array at t.
        iou_threshold  : Minimum IoU to declare a match.

    Returns:
        curr_to_prev : dict mapping curr_id → prev_id (or None if new).
    """
    prev_ids = np.unique(labelled_prev[labelled_prev > 0])
    curr_ids = np.unique(labelled_curr[labelled_curr > 0])

    curr_to_prev: Dict[int, Optional[int]] = {}

    for curr_id in curr_ids:
        mask_curr = labelled_curr == curr_id
        best_iou = 0.0
        best_prev = None

        for prev_id in prev_ids:
            mask_prev = labelled_prev == prev_id
            intersection = (mask_curr & mask_prev).sum()
            if intersection == 0:
                continue
            union = (mask_curr | mask_prev).sum()
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_prev = int(prev_id)

        curr_to_prev[int(curr_id)] = best_prev if best_iou >= iou_threshold else None

    return curr_to_prev


# ---------------------------------------------------------------------------
# Activity tracker across all dates
# ---------------------------------------------------------------------------

class GullyActivityTracker:
    """
    Tracks gully instances across multiple dates.

    Usage::

        tracker = GullyActivityTracker(pixel_area_m2=36.0)  # 6m CTX
        for date, binary in zip(dates, binaries):
            tracker.ingest(date, binary)
        records = tracker.get_tracks()
    """

    def __init__(
        self,
        pixel_area_m2: float = 36.0,
        iou_threshold: float = 0.3,
        min_pixels: int = 5,
    ):
        self.pixel_area_m2 = pixel_area_m2
        self.iou_threshold = iou_threshold
        self.min_pixels = min_pixels

        self._next_global_id = 1
        self._prev_labelled: Optional[np.ndarray] = None
        self._prev_local_to_global: Dict[int, int] = {}
        self._tracks: Dict[int, dict] = {}  # global_id → track info

    def _area_ha(self, n_pixels: int) -> float:
        return round(n_pixels * self.pixel_area_m2 / 10_000.0, 6)

    def ingest(self, date: float, binary: np.ndarray) -> None:
        """Process one date's binary mask."""
        labelled, n_comp, comp_info = get_components(
            binary, min_pixels=self.min_pixels
        )

        if self._prev_labelled is None:
            # First date — assign fresh global IDs to all components
            local_to_global: Dict[int, int] = {}
            for c in comp_info:
                gid = self._next_global_id
                self._next_global_id += 1
                local_to_global[c["id"]] = gid
                self._tracks[gid] = {
                    "global_id": gid,
                    "first_seen": date,
                    "last_seen": date,
                    "dates": [date],
                    "areas_ha": [self._area_ha(c["area_px"])],
                    "centroids": [(c["centroid_row"], c["centroid_col"])],
                    "kalman": KalmanAreaTracker(),
                }
                self._tracks[gid]["kalman"].update(date, self._area_ha(c["area_px"]))
        else:
            curr_to_prev = match_components(
                self._prev_labelled, labelled, iou_threshold=self.iou_threshold
            )
            local_to_global: Dict[int, int] = {}

            for c in comp_info:
                lid = c["id"]
                prev_lid = curr_to_prev.get(lid)
                if prev_lid is not None and prev_lid in self._prev_local_to_global:
                    # matched to an existing track
                    gid = self._prev_local_to_global[prev_lid]
                else:
                    # new gully instance
                    gid = self._next_global_id
                    self._next_global_id += 1
                    self._tracks[gid] = {
                        "global_id": gid,
                        "first_seen": date,
                        "last_seen": date,
                        "dates": [],
                        "areas_ha": [],
                        "centroids": [],
                        "kalman": KalmanAreaTracker(),
                    }

                local_to_global[lid] = gid
                track = self._tracks[gid]
                area = self._area_ha(c["area_px"])
                track["last_seen"] = date
                track["dates"].append(date)
                track["areas_ha"].append(area)
                track["centroids"].append((c["centroid_row"], c["centroid_col"]))
                track["kalman"].update(date, area)

        self._prev_labelled = labelled
        self._prev_local_to_global = local_to_global
        log.debug(f"Date {date}: {n_comp} components, {len(self._tracks)} total tracks.")

    def get_tracks(self) -> List[dict]:
        """Return serialisable track records (without KalmanAreaTracker objects)."""
        records = []
        for gid, t in self._tracks.items():
            kal_summary = t["kalman"].summary()
            records.append({
                "global_id": gid,
                "first_seen": t["first_seen"],
                "last_seen": t["last_seen"],
                "n_observations": len(t["dates"]),
                "dates": t["dates"],
                "areas_ha": t["areas_ha"],
                "smoothed_areas_ha": kal_summary["smoothed_areas"],
                "velocities_ha_year": kal_summary["velocities"],
                "total_gain_ha": kal_summary["total_gain_ha"],
                "mean_expansion_ha_year": kal_summary["mean_expansion_ha_year"],
                "centroids": t["centroids"],
            })
        return sorted(records, key=lambda r: r["global_id"])


# ---------------------------------------------------------------------------
# GeoJSON export helper
# ---------------------------------------------------------------------------

def tracks_to_geojson(
    tracks: List[dict],
    profile: dict,
) -> dict:
    """
    Convert track centroids to a GeoJSON FeatureCollection.

    Each feature is the centroid of the gully's last known position.
    """
    transform = profile.get("transform")
    features = []

    for track in tracks:
        if not track["centroids"]:
            continue
        last_row, last_col = track["centroids"][-1]
        if transform is not None:
            lon, lat = transform * (last_col, last_row)
        else:
            lon, lat = float(last_col), float(last_row)

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "gully_id": track["global_id"],
                "first_seen": track["first_seen"],
                "last_seen": track["last_seen"],
                "n_obs": track["n_observations"],
                "area_ha_latest": track["areas_ha"][-1] if track["areas_ha"] else 0,
                "area_ha_smoothed_latest": (
                    track["smoothed_areas_ha"][-1]
                    if track["smoothed_areas_ha"] else 0
                ),
                "expansion_ha_year": track["mean_expansion_ha_year"],
                "total_gain_ha": track["total_gain_ha"],
            },
        })

    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def run_activity_tracking(cfg_path: str = "config.yaml") -> None:
    """
    Track gully instances across all binary mask dates and export GeoJSON.
    """
    import re
    import rasterio

    from scripts.utils import load_config
    from scripts.inference.threshold import pixel_area_m2

    cfg = load_config(cfg_path)
    binary_dir = Path("data/outputs/binary_maps")
    out_dir = Path("data/outputs/change_detection")
    out_dir.mkdir(parents=True, exist_ok=True)

    thr_str = "thr50"
    binary_files = sorted(binary_dir.glob(f"*_{thr_str}.tif"))
    if len(binary_files) < 2:
        log.warning("Need >= 2 binary maps for tracking.")
        return

    with rasterio.open(binary_files[0]) as src:
        profile = src.profile
    pix_area = pixel_area_m2(profile)

    tracker = GullyActivityTracker(
        pixel_area_m2=pix_area,
        iou_threshold=cfg.get("change_detection", {}).get("min_activity_confidence", 0.3),
    )

    # Load all binary maps, aligning to the first map's shape
    ref_shape = None
    for bf in binary_files:
        match = re.search(r"(\d+)", bf.stem)
        date = float(match.group(1)) if match else 0.0
        with rasterio.open(bf) as ds:
            binary = ds.read(1).astype(np.uint8)
        # Align shape to reference (first map)
        if ref_shape is None:
            ref_shape = binary.shape
        elif binary.shape != ref_shape:
            import cv2 as _cv2
            binary = _cv2.resize(
                binary, (ref_shape[1], ref_shape[0]),
                interpolation=_cv2.INTER_NEAREST
            ).astype(np.uint8)
        tracker.ingest(date, binary)

    tracks = tracker.get_tracks()
    tracks_path = out_dir / "activity_tracks.json"
    with open(tracks_path, "w") as f:
        json.dump(tracks, f, indent=2)
    log.info(f"Activity tracks -> {tracks_path}  ({len(tracks)} gully instances)")

    geojson = tracks_to_geojson(tracks, profile)
    geojson_path = out_dir / "active_sites.geojson"
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)
    log.info(f"Active sites GeoJSON -> {geojson_path}")