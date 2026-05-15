"""
to_geojson.py
=============
Convert binary gully masks to GeoJSON polygon feature collections.

Uses rasterio.features.shapes to extract polygon contours from
binary rasters, then writes a GeoJSON FeatureCollection per file.

Usage
-----
    python main.py --step export   (called automatically)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def binary_to_polygons(
    binary: np.ndarray,
    transform,
    crs=None,
    min_area_m2: float = 500.0,
    simplify_tolerance: float = 0.0,
) -> List[dict]:
    """
    Convert a binary (H, W) uint8 raster to a list of GeoJSON-compatible
    polygon dicts.

    Args:
        binary            : (H, W) uint8 array {0, 1}.
        transform         : rasterio Affine transform.
        crs               : rasterio CRS (stored as metadata).
        min_area_m2       : Minimum polygon area to keep.
        simplify_tolerance: Shapely simplify tolerance (0 = no simplify).

    Returns:
        List of GeoJSON Feature dicts.
    """
    binary = (binary > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    features = []
    for geom_dict, val in shapes(binary, mask=binary, transform=transform):
        if val == 0:
            continue
        geom = shape(geom_dict)
        if simplify_tolerance > 0:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)
        area = geom.area  # in CRS units (degrees² or m²)

        # Convert area if CRS is geographic (degrees → m² on Mars)
        if crs is not None:
            try:
                if hasattr(crs, 'is_geographic') and crs.is_geographic:
                    import math
                    MARS_RADIUS_M = 3_389_500.0
                    # Approximate: degrees² × (π/180)² × R²
                    area = area * (math.pi / 180.0) ** 2 * MARS_RADIUS_M ** 2
            except Exception:
                pass

        if area < min_area_m2:
            continue

        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "area_m2": round(area, 2),
                "area_ha": round(area / 10_000, 4),
            },
        })

    return features


# ---------------------------------------------------------------------------
# Per-file export
# ---------------------------------------------------------------------------

def export_binary_to_geojson(
    binary_path: Path,
    out_path: Path,
    min_area_m2: float = 500.0,
    simplify_tolerance: float = 0.0,
) -> int:
    """
    Export one binary GeoTIFF to GeoJSON.

    Returns:
        Number of polygon features written.
    """
    with rasterio.open(binary_path) as src:
        binary = src.read(1).astype(np.uint8)
        transform = src.transform
        crs = src.crs

    features = binary_to_polygons(
        binary, transform, crs,
        min_area_m2=min_area_m2,
        simplify_tolerance=simplify_tolerance,
    )

    crs_str = crs.to_string() if crs else "Unknown"
    collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs_str}},
        "features": features,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(collection, f, separators=(",", ":"))

    log.info(f"GeoJSON → {out_path}  ({len(features)} polygons)")
    return len(features)


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------

def run_geojson_export(cfg_path: str = "config.yaml") -> None:
    """
    Export all binary masks to GeoJSON.
    """
    from scripts.utils import load_config
    cfg = load_config(cfg_path)

    min_area = cfg["inference"].get("min_gully_area_m2", 500.0)
    binary_dir = Path("data/outputs/binary_maps")
    out_dir = Path("data/outputs/change_detection")
    out_dir.mkdir(parents=True, exist_ok=True)

    for bin_path in sorted(binary_dir.glob("*.tif")):
        out_path = out_dir / (bin_path.stem + ".geojson")
        if out_path.exists():
            log.info(f"[SKIP] {out_path.name}")
            continue
        export_binary_to_geojson(bin_path, out_path, min_area_m2=min_area)

    # also export the active_sites GeoJSON from activity tracking
    active_sites = Path("data/outputs/change_detection/active_sites.geojson")
    if active_sites.exists():
        log.info(f"Active sites GeoJSON already exists: {active_sites}")

    log.info("GeoJSON export complete.")
