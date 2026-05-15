"""
stac_catalog.py
===============
Build a STAC 1.0 catalog from the pipeline outputs.

Structure
---------
data/outputs/stac/
  catalog.json           ← root Catalog
  items/
    {stem}_item.json     ← one Item per probability map / binary mask

References
----------
  https://stacspec.org/en/about/stac-spec/
  pystac: https://pystac.readthedocs.io/
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import rasterio
from rasterio.crs import CRS

log = logging.getLogger(__name__)

try:
    import pystac
    from pystac import Catalog, Item, Asset, Extent, SpatialExtent, TemporalExtent
    from pystac.extensions.eo import EOExtension
    HAS_PYSTAC = True
except ImportError:
    HAS_PYSTAC = False
    log.warning("pystac not installed; falling back to raw JSON STAC export.")


# ---------------------------------------------------------------------------
# Bounding-box helper
# ---------------------------------------------------------------------------

def get_bbox_and_footprint(tif_path: Path):
    """Extract bbox [west, south, east, north] and GeoJSON footprint from a GeoTIFF."""
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs

    if crs is not None:
        try:
            from rasterio.warp import transform_bounds
            west, south, east, north = transform_bounds(
                crs, CRS.from_epsg(4326), *bounds
            )
        except Exception:
            west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
    else:
        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top

    bbox = [west, south, east, north]
    footprint = {
        "type": "Polygon",
        "coordinates": [[
            [west, south], [east, south],
            [east, north], [west, north],
            [west, south],
        ]],
    }
    return bbox, footprint


def parse_date(stem: str) -> Optional[str]:
    """Extract a 4-digit year from a filename stem and return ISO datetime string."""
    match = re.search(r"(\d{4})", stem)
    if match:
        year = int(match.group(1))
        return datetime(year, 7, 1, tzinfo=timezone.utc).isoformat()
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Raw JSON fallback
# ---------------------------------------------------------------------------

def _build_raw_json_stac(
    prob_files: List[Path],
    binary_files: List[Path],
    stac_dir: Path,
    provider: str,
    stac_version: str,
) -> None:
    """Build a STAC catalog as raw JSON files (no pystac dependency)."""
    items_dir = stac_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    item_links = []

    # Group by stem (remove _prob / _thr suffix)
    stems = set()
    for f in prob_files:
        stems.add(f.stem.replace("_prob", ""))
    for f in binary_files:
        for thr in ["_thr30", "_thr50", "_thr70"]:
            stems.add(f.stem.replace(thr, ""))

    for stem in sorted(stems):
        item_id = stem.replace(" ", "_")
        item_path = items_dir / f"{item_id}_item.json"

        # find matching files
        prob_path = next(
            (p for p in prob_files if stem in p.stem), None
        )
        if prob_path is None:
            continue

        bbox, footprint = get_bbox_and_footprint(prob_path)
        dt_str = parse_date(stem)

        assets: dict = {
            "probability_map": {
                "href": f"../../probability_maps/{prob_path.name}",
                "type": "image/tiff; application=geotiff",
                "title": "Gully Probability Map",
                "roles": ["data"],
            }
        }
        for thr_suffix, thr_label in [("_thr30", "0.3"), ("_thr50", "0.5"), ("_thr70", "0.7")]:
            bin_match = next(
                (p for p in binary_files if stem in p.stem and thr_suffix in p.stem), None
            )
            if bin_match:
                assets[f"binary_{thr_label}"] = {
                    "href": f"../../binary_maps/{bin_match.name}",
                    "type": "image/tiff; application=geotiff",
                    "title": f"Binary Mask (threshold={thr_label})",
                    "roles": ["data"],
                }

        item = {
            "type": "Feature",
            "stac_version": stac_version,
            "id": item_id,
            "geometry": footprint,
            "bbox": bbox,
            "properties": {
                "datetime": dt_str,
                "title": f"Mars Gully Detection – {stem}",
                "provider": provider,
                "platform": "MRO",
                "instruments": ["HiRISE", "CTX"],
                "mission": "Mars Reconnaissance Orbiter",
            },
            "links": [{"rel": "root", "href": "../catalog.json"}],
            "assets": assets,
        }

        with open(item_path, "w") as f:
            json.dump(item, f, indent=2)

        item_links.append({
            "rel": "item",
            "href": f"items/{item_id}_item.json",
            "type": "application/json",
        })

    catalog = {
        "type": "Catalog",
        "id": "mars-gully-digital-twin",
        "stac_version": stac_version,
        "description": (
            "STAC Catalog for the Mars Gully Digital Twin pipeline. "
            "Contains gully probability maps and binary masks derived from "
            "HiRISE/CTX imagery using a U-Net with attention gates."
        ),
        "title": "Mars Gully Digital Twin",
        "links": item_links,
        "providers": [{"name": provider, "roles": ["producer", "licensor"]}],
    }

    with open(stac_dir / "catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)

    log.info(
        f"Raw JSON STAC catalog → {stac_dir / 'catalog.json'} "
        f"({len(item_links)} items)"
    )


# ---------------------------------------------------------------------------
# pystac-based builder
# ---------------------------------------------------------------------------

def _build_pystac_catalog(
    prob_files: List[Path],
    binary_files: List[Path],
    stac_dir: Path,
    provider: str,
    stac_version: str,
) -> None:
    """Build a fully compliant STAC catalog using pystac."""
    catalog = Catalog(
        id="mars-gully-digital-twin",
        description=(
            "Mars Gully Digital Twin – STAC Catalog of probability maps "
            "and binary gully masks produced by a multi-sensor deep learning pipeline."
        ),
        title="Mars Gully Digital Twin",
    )

    for prob_path in sorted(prob_files):
        stem = prob_path.stem.replace("_prob", "")
        item_id = stem.replace(" ", "_")

        bbox, footprint = get_bbox_and_footprint(prob_path)
        dt_str = parse_date(stem)
        dt = datetime.fromisoformat(dt_str)

        item = Item(
            id=item_id,
            geometry=footprint,
            bbox=bbox,
            datetime=dt,
            properties={
                "title": f"Mars Gully Detection – {stem}",
                "provider": provider,
                "platform": "MRO",
                "instruments": ["HiRISE", "CTX"],
            },
        )

        item.add_asset(
            "probability_map",
            Asset(
                href=str(prob_path.resolve()),
                media_type="image/tiff; application=geotiff",
                title="Gully Probability Map",
                roles=["data"],
            ),
        )

        for thr_suffix, thr_label in [("_thr30", "0.3"), ("_thr50", "0.5"), ("_thr70", "0.7")]:
            bin_match = next(
                (p for p in binary_files if stem in p.stem and thr_suffix in p.stem), None
            )
            if bin_match:
                item.add_asset(
                    f"binary_{thr_label}",
                    Asset(
                        href=str(bin_match.resolve()),
                        media_type="image/tiff; application=geotiff",
                        title=f"Binary Mask (threshold={thr_label})",
                        roles=["data"],
                    ),
                )

        catalog.add_item(item)

    stac_dir.mkdir(parents=True, exist_ok=True)
    catalog.normalize_hrefs(str(stac_dir))
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    log.info(f"pystac STAC catalog → {stac_dir / 'catalog.json'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_stac_export(cfg_path: str = "config.yaml") -> None:
    """Build the STAC catalog from all available outputs."""
    from scripts.utils import load_config
    cfg = load_config(cfg_path)

    export_cfg = cfg.get("export", {})
    stac_version = export_cfg.get("stac_version", "1.0.0")
    provider = export_cfg.get(
        "stac_provider", "Copernicus Master's in Digital Earth"
    )

    prob_dir = Path("data/outputs/probability_maps")
    binary_dir = Path("data/outputs/binary_maps")
    stac_dir = Path("data/outputs/stac")

    prob_files = sorted(prob_dir.glob("*_prob.tif"))
    binary_files = sorted(binary_dir.glob("*.tif"))

    if not prob_files:
        log.warning("No probability maps found; STAC catalog will be empty.")

    if HAS_PYSTAC:
        _build_pystac_catalog(prob_files, binary_files, stac_dir, provider, stac_version)
    else:
        _build_raw_json_stac(prob_files, binary_files, stac_dir, provider, stac_version)

    log.info("STAC export complete.")
