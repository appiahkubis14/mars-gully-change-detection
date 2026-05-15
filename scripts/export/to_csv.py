"""
to_csv.py
=========
Generate CSV summary reports from pipeline outputs.

Reports
-------
1. per_site_summary.csv
   Columns: site_id, date, gully_area_ha, n_polygons, prob_mean, prob_max

2. change_report.csv
   Columns: date_t0, date_t1, gain_ha, loss_ha, net_change_ha, kalman_velocity_ha_year

3. activity_report.csv
   Per tracked gully instance: first_seen, last_seen, total_gain_ha, expansion_rate_ha_year
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"CSV → {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Per-site summary
# ---------------------------------------------------------------------------

def generate_per_site_summary(cfg: dict) -> Path:
    """
    Aggregate probability map and binary mask stats per date and write CSV.
    """
    import rasterio

    prob_dir = Path("data/outputs/probability_maps")
    binary_dir = Path("data/outputs/binary_maps")
    out_path = Path("data/outputs/reports/per_site_summary.csv")

    # ---- collect probability map stats ----
    prob_stats: Dict[str, dict] = {}
    for prob_path in sorted(prob_dir.glob("*_prob.tif")):
        with rasterio.open(prob_path) as src:
            arr = src.read(1).astype(np.float32)
        stem = prob_path.stem.replace("_prob", "")
        valid = arr[arr >= 0]
        prob_stats[stem] = {
            "prob_mean": float(valid.mean()) if len(valid) else 0.0,
            "prob_max": float(valid.max()) if len(valid) else 0.0,
        }

    # ---- collect binary mask stats ----
    rows: List[dict] = []
    for bin_path in sorted(binary_dir.glob("*_thr50.tif")):
        with rasterio.open(bin_path) as src:
            arr = src.read(1).astype(np.uint8)
            transform = src.transform
            crs = src.crs

        # pixel area in m²
        try:
            from scripts.inference.threshold import pixel_area_m2
            pix_m2 = pixel_area_m2(src.profile)
        except Exception:
            pix_m2 = abs(transform.a * transform.e)

        n_pixels = int(arr.sum())
        area_ha = n_pixels * pix_m2 / 10_000.0

        # count polygons from GeoJSON if available
        geojson_path = (
            Path("data/outputs/change_detection")
            / (bin_path.stem + ".geojson")
        )
        n_polygons = 0
        if geojson_path.exists():
            with open(geojson_path) as f:
                gj = json.load(f)
            n_polygons = len(gj.get("features", []))

        stem = bin_path.stem.replace("_thr50", "")
        ps = prob_stats.get(stem, {})

        rows.append({
            "site_id": stem,
            "n_gully_pixels": n_pixels,
            "gully_area_ha": round(area_ha, 4),
            "n_polygons": n_polygons,
            "prob_mean": round(ps.get("prob_mean", 0.0), 4),
            "prob_max": round(ps.get("prob_max", 0.0), 4),
        })

    fields = ["site_id", "n_gully_pixels", "gully_area_ha", "n_polygons", "prob_mean", "prob_max"]
    _write_csv(out_path, fields, rows)
    return out_path


# ---------------------------------------------------------------------------
# Change report
# ---------------------------------------------------------------------------

def generate_change_report() -> Path:
    """Read change_summary.json and write change_report.csv."""
    summary_path = Path("data/outputs/change_detection/change_summary.json")
    out_path = Path("data/outputs/reports/change_report.csv")

    if not summary_path.exists():
        log.warning(f"No change_summary.json at {summary_path}. Run --step change first.")
        return out_path

    with open(summary_path) as f:
        pairs = json.load(f)

    fields = [
        "date_t0", "date_t1",
        "gain_ha", "loss_ha", "stable_ha", "net_change_ha",
        "kalman_velocity_ha_year",
    ]
    rows = []
    for p in pairs:
        rows.append({
            "date_t0": p.get("date_t0", ""),
            "date_t1": p.get("date_t1", ""),
            "gain_ha": p.get("gain_ha", 0),
            "loss_ha": p.get("loss_ha", 0),
            "stable_ha": p.get("stable_ha", 0),
            "net_change_ha": p.get("net_change_ha", 0),
            "kalman_velocity_ha_year": p.get("kalman_velocity_ha_year", ""),
        })

    _write_csv(out_path, fields, rows)
    return out_path


# ---------------------------------------------------------------------------
# Activity report
# ---------------------------------------------------------------------------

def generate_activity_report() -> Path:
    """Read activity_tracks.json and write activity_report.csv."""
    tracks_path = Path("data/outputs/change_detection/activity_tracks.json")
    out_path = Path("data/outputs/reports/activity_report.csv")

    if not tracks_path.exists():
        log.warning(f"No activity_tracks.json. Run --step change first.")
        return out_path

    with open(tracks_path) as f:
        tracks = json.load(f)

    fields = [
        "gully_id", "first_seen", "last_seen", "n_observations",
        "area_ha_latest", "area_ha_smoothed_latest",
        "total_gain_ha", "mean_expansion_ha_year",
    ]
    rows = []
    for t in tracks:
        rows.append({
            "gully_id": t.get("global_id", ""),
            "first_seen": t.get("first_seen", ""),
            "last_seen": t.get("last_seen", ""),
            "n_observations": t.get("n_observations", 0),
            "area_ha_latest": (
                t["areas_ha"][-1] if t.get("areas_ha") else 0
            ),
            "area_ha_smoothed_latest": (
                t["smoothed_areas_ha"][-1] if t.get("smoothed_areas_ha") else 0
            ),
            "total_gain_ha": t.get("total_gain_ha", 0),
            "mean_expansion_ha_year": t.get("mean_expansion_ha_year", 0),
        })

    _write_csv(out_path, fields, rows)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_csv_export(cfg_path: str = "config.yaml") -> None:
    from scripts.utils import load_config
    cfg = load_config(cfg_path)

    Path("data/outputs/reports").mkdir(parents=True, exist_ok=True)
    generate_per_site_summary(cfg)
    generate_change_report()
    generate_activity_report()
    log.info("CSV export complete.")
