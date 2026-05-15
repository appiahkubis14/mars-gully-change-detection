"""
folium_map.py
=============
Generate an interactive HTML Folium dashboard for the Mars Gully Digital Twin.

Layers
------
1. Mars basemap (THEMIS IR Day / MOLA shaded relief / CTX Mosaic)
2. Gully probability map (latest date, colormap blue→red, opacity 0.6)
3. Binary gully mask (thr=0.5, orange overlay)
4. Change detection map (gain=green, stable=yellow)
5. Gully polygon layer (clickable popups with area, date info)
6. Active gully sites (activity tracking markers)

Embedded time-series Plotly HTML is loaded from time_series_plot.py.

Output
------
data/outputs/dashboard/mars_gully_dashboard.html

Usage
-----
    python main.py --step dashboard
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import folium
    from folium import plugins as folium_plugins
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    log.warning("folium not installed. Install with: pip install folium")

try:
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------

def _prob_colormap():
    """blue→red linear colormap for probability maps (folium.LinearColormap)."""
    if not HAS_FOLIUM:
        return None
    from branca.colormap import linear
    return linear.RdYlBu_11.to_step(10).scale(0, 1)


def _prob_array_to_png(
    prob: np.ndarray,
    colormap: str = "RdYlBu_r",
    opacity: float = 0.65,
) -> bytes:
    """
    Convert a (H, W) probability array to a RGBA PNG bytes object.
    Pixels with prob < 0.05 are made transparent.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from PIL import Image

    cmap = cm.get_cmap(colormap)
    rgba = (cmap(prob) * 255).astype(np.uint8)
    # make low-probability pixels transparent
    alpha = (prob >= 0.05).astype(np.uint8) * int(opacity * 255)
    rgba[:, :, 3] = alpha

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GeoTIFF → Folium ImageOverlay helper
# ---------------------------------------------------------------------------

def _raster_to_image_overlay(
    tif_path: Path,
    layer_name: str,
    colormap: str = "RdYlBu_r",
    opacity: float = 0.65,
    threshold: Optional[float] = None,
) -> Optional[object]:
    """
    Load a GeoTIFF, convert to PNG, and return a Folium ImageOverlay.

    Args:
        tif_path   : Path to GeoTIFF.
        layer_name : Folium layer name.
        colormap   : Matplotlib colormap name.
        opacity    : Layer opacity (0–1).
        threshold  : If set, binarise at this value (for binary masks).

    Returns:
        folium.raster_layers.ImageOverlay or None on failure.
    """
    if not (HAS_FOLIUM and HAS_RASTERIO):
        return None
    try:
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
            bounds = src.bounds
            crs = src.crs

        # project bounds to WGS84
        try:
            west, south, east, north = transform_bounds(
                crs, CRS.from_epsg(4326), *bounds
            )
        except Exception:
            west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top

        arr = np.nan_to_num(arr, nan=0.0)
        if threshold is not None:
            arr = (arr >= threshold).astype(np.float32)
        else:
            # normalise to [0, 1]
            lo, hi = np.percentile(arr[arr > 0], [2, 98]) if arr.max() > 0 else (0, 1)
            arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)

        png_bytes = _prob_array_to_png(arr, colormap=colormap, opacity=opacity)

        import base64
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        img_url = f"data:image/png;base64,{b64}"

        overlay = folium.raster_layers.ImageOverlay(
            image=img_url,
            bounds=[[south, west], [north, east]],
            name=layer_name,
            opacity=1.0,   # opacity already embedded in PNG alpha
            show=True,
        )
        return overlay

    except Exception as exc:
        log.warning(f"Could not create overlay for {tif_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def build_dashboard(cfg_path: str = "config.yaml") -> Path:
    """
    Build the interactive HTML Folium dashboard.

    Returns:
        Path to the output HTML file.
    """
    from scripts.utils import load_config
    from scripts.dashboard.mars_basemap import get_basemap_tilelayer

    cfg = load_config(cfg_path)
    dash_cfg = cfg.get("dashboard", {})
    overlay_opacity = dash_cfg.get("overlay_opacity", 0.7)
    out_dir = Path("data/outputs/dashboard")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mars_gully_dashboard.html"

    if not HAS_FOLIUM:
        log.error("folium is not installed. Dashboard cannot be generated.")
        return out_path

    # ---- study area centre ----
    study = cfg.get("study_area", {}).get("primary", {})
    center_lat = study.get("center_lat", -35.7)
    center_lon = study.get("center_lon", 129.5)

    # ---- create map ----
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles=None,                  # we add basemaps manually
        control_scale=True,
    )

    # ---- basemap layers ----
    for bm_name in ["THEMIS IR Day", "MOLA Color", "CTX Mosaic"]:
        try:
            tl = get_basemap_tilelayer(bm_name, show=(bm_name == "THEMIS IR Day"))
            if tl:
                tl.add_to(fmap)
        except Exception as exc:
            log.warning(f"Could not add basemap {bm_name}: {exc}")

    # ---- probability map overlay (latest date) ----
    prob_dir = Path("data/outputs/probability_maps")
    prob_files = sorted(prob_dir.glob("*_prob.tif"))
    if prob_files:
        latest_prob = prob_files[-1]
        overlay = _raster_to_image_overlay(
            latest_prob,
            layer_name=f"Gully Probability ({latest_prob.stem})",
            colormap="RdYlBu_r",
            opacity=overlay_opacity,
        )
        if overlay:
            overlay.add_to(fmap)

    # ---- binary mask overlay (thr=0.5, latest) ----
    binary_dir = Path("data/outputs/binary_maps")
    binary_files = sorted(binary_dir.glob("*_thr50.tif"))
    if binary_files:
        latest_bin = binary_files[-1]
        bin_overlay = _raster_to_image_overlay(
            latest_bin,
            layer_name=f"Binary Mask thr=0.5 ({latest_bin.stem})",
            colormap="Oranges",
            opacity=0.5,
            threshold=0.5,
        )
        if bin_overlay:
            bin_overlay.add_to(fmap)

    # ---- change detection overlay ----
    change_dir = Path("data/outputs/change_detection")
    gain_files = sorted(change_dir.glob("*_gain.tif"))
    if gain_files:
        gain_overlay = _raster_to_image_overlay(
            gain_files[-1],
            layer_name=f"Gully Gain ({gain_files[-1].stem})",
            colormap="Greens",
            opacity=0.6,
            threshold=0.5,
        )
        if gain_overlay:
            gain_overlay.add_to(fmap)

    # ---- polygon layer from GeoJSON ----
    geojson_files = sorted(change_dir.glob("*_thr50.geojson"))
    if geojson_files:
        latest_gj = geojson_files[-1]
        try:
            with open(latest_gj) as f:
                gj_data = json.load(f)

            folium.GeoJson(
                gj_data,
                name=f"Gully Polygons ({latest_gj.stem})",
                style_function=lambda x: {
                    "fillColor": "#ff4400",
                    "color": "#cc2200",
                    "weight": 1,
                    "fillOpacity": 0.4,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["area_ha"],
                    aliases=["Area (ha):"],
                    localize=True,
                ),
                popup=folium.GeoJsonPopup(
                    fields=["area_m2", "area_ha"],
                    aliases=["Area (m²)", "Area (ha)"],
                ),
            ).add_to(fmap)
        except Exception as exc:
            log.warning(f"Could not add polygon layer: {exc}")

    # ---- active sites markers ----
    active_sites_path = change_dir / "active_sites.geojson"
    if active_sites_path.exists():
        try:
            with open(active_sites_path) as f:
                sites = json.load(f)

            for feat in sites.get("features", []):
                coords = feat["geometry"]["coordinates"]
                props = feat.get("properties", {})
                lon, lat = coords[0], coords[1]
                popup_html = (
                    f"<b>Gully ID:</b> {props.get('gully_id', '?')}<br>"
                    f"<b>First seen:</b> {props.get('first_seen', '?')}<br>"
                    f"<b>Last seen:</b> {props.get('last_seen', '?')}<br>"
                    f"<b>Area (latest):</b> {props.get('area_ha_latest', 0):.2f} ha<br>"
                    f"<b>Expansion rate:</b> {props.get('expansion_ha_year', 0):.3f} ha/yr<br>"
                    f"<b>Total gain:</b> {props.get('total_gain_ha', 0):.2f} ha<br>"
                )
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color="#e74c3c",
                    fill=True,
                    fill_color="#e74c3c",
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=280),
                    tooltip=f"Gully #{props.get('gully_id', '?')}",
                ).add_to(fmap)
        except Exception as exc:
            log.warning(f"Could not add active sites: {exc}")

    # ---- study area site markers ----
    for site_key, site_cfg in cfg.get("study_area", {}).items():
        if isinstance(site_cfg, dict) and "center_lat" in site_cfg:
            folium.Marker(
                location=[site_cfg["center_lat"], site_cfg["center_lon"]],
                popup=site_cfg.get("name", site_key),
                icon=folium.Icon(color="blue", icon="info-sign"),
                tooltip=site_cfg.get("name", site_key),
            ).add_to(fmap)

    # ---- embed time-series iframe ----
    ts_path = out_dir / "time_series.html"
    if ts_path.exists():
        ts_html = f"""
        <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                    background:white;padding:10px;border-radius:8px;
                    box-shadow:2px 2px 6px rgba(0,0,0,0.3);max-width:420px;">
          <b style="font-size:13px;">Gully Area Time Series</b>
          <iframe src="time_series.html" width="400" height="280"
                  style="border:none;margin-top:6px;display:block;"></iframe>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(ts_html))

    # ---- colorbar ----
    try:
        import branca.colormap as bcm
        cmap = bcm.LinearColormap(
            colors=["#313695", "#74add1", "#fdae61", "#f46d43", "#a50026"],
            vmin=0, vmax=1,
            caption="Gully Probability",
        )
        cmap.add_to(fmap)
    except Exception:
        pass

    # ---- layer control ----
    folium.LayerControl(collapsed=False).add_to(fmap)

    # ---- title box ----
    title_html = """
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                z-index:9999;background:rgba(255,255,255,0.92);padding:10px 20px;
                border-radius:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.3);
                font-family:Arial;text-align:center;">
      <h3 style="margin:0;color:#c0392b;">🔴 Mars Gully Digital Twin</h3>
      <p style="margin:2px 0;font-size:12px;color:#555;">
        Multi-sensor deep learning | Gasa · Palikir · Russell craters
      </p>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))

    # ---- save ----
    fmap.save(str(out_path))
    log.info(f"Dashboard saved → {out_path}")
    return out_path
