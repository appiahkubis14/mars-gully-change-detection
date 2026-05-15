"""
mars_basemap.py
===============
Mars basemap tile definitions for use with Folium.

Available basemaps
------------------
1. THEMIS IR Day         — 100 m/pixel thermal infrared (USGS)
2. MOLA Shaded Relief    — colour-coded hillshaded DEM (USGS)
3. HiRISE Greyscale      — very high resolution (site-specific)
4. CTX Mosaic            — 6 m/pixel (Murray Lab / USGS)

All tiles are served via the USGS/NASA planetary tile servers or
public TileLayer services; no authentication required.

Usage
-----
    from scripts.dashboard.mars_basemap import get_basemap_tilelayer
    tl = get_basemap_tilelayer("THEMIS IR Day")
    tl.add_to(folium_map)
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

# ---- optional import (only used for type hints) ----
try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ---------------------------------------------------------------------------
# Basemap definitions
# ---------------------------------------------------------------------------

# Each entry: name → dict with url_template, attribution, max_zoom, min_zoom
BASEMAPS: Dict[str, dict] = {
    "THEMIS IR Day": {
        "url": (
            "https://planetarymaps.usgs.gov/cgi-bin/mapserv?"
            "map=/maps/mars/mars_simp_cyl.map"
            "&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
            "&LAYERS=THEMIS_Day_IR_Mosaic"
            "&BBOX={bbox-epsg-3857}&WIDTH=256&HEIGHT=256"
            "&SRS=EPSG:3857&STYLES=&FORMAT=image/jpeg&TRANSPARENT=FALSE"
        ),
        "attribution": (
            "THEMIS IR Day Mosaic – Christensen et al. (2004), "
            "USGS Astrogeology Science Center"
        ),
        "max_zoom": 10,
        "min_zoom": 1,
        "is_wms": True,
        "wms_layers": "THEMIS_Day_IR_Mosaic",
        "wms_url": "https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/mars/mars_simp_cyl.map",
    },
    "MOLA Color": {
        "url": (
            "https://planetarymaps.usgs.gov/cgi-bin/mapserv?"
            "map=/maps/mars/mars_simp_cyl.map"
            "&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
            "&LAYERS=MOLA_Color"
            "&BBOX={bbox-epsg-3857}&WIDTH=256&HEIGHT=256"
            "&SRS=EPSG:3857&STYLES=&FORMAT=image/jpeg&TRANSPARENT=FALSE"
        ),
        "attribution": (
            "MOLA Color DEM – Smith et al. (2001), "
            "USGS Astrogeology Science Center"
        ),
        "max_zoom": 8,
        "min_zoom": 1,
        "is_wms": True,
        "wms_layers": "MOLA_Color",
        "wms_url": "https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/mars/mars_simp_cyl.map",
    },
    "MOLA Shaded Relief": {
        "url": (
            "https://planetarymaps.usgs.gov/cgi-bin/mapserv?"
            "map=/maps/mars/mars_simp_cyl.map"
            "&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
            "&LAYERS=MOLA_Shade"
            "&BBOX={bbox-epsg-3857}&WIDTH=256&HEIGHT=256"
            "&SRS=EPSG:3857&STYLES=&FORMAT=image/jpeg&TRANSPARENT=FALSE"
        ),
        "attribution": (
            "MOLA Shaded Relief – Smith et al. (2001), "
            "USGS Astrogeology Science Center"
        ),
        "max_zoom": 8,
        "min_zoom": 1,
        "is_wms": True,
        "wms_layers": "MOLA_Shade",
        "wms_url": "https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/mars/mars_simp_cyl.map",
    },
    "CTX Mosaic": {
        "url": (
            "https://murray-lab.caltech.edu/CTX/tiles/beta01/"
            "MurrayLab_CTX_V01_beta01_{z}/{x}/{y}.jpg"
        ),
        "attribution": (
            "CTX Mosaic – Murray Lab, Caltech "
            "(https://murray-lab.caltech.edu/CTX/)"
        ),
        "max_zoom": 12,
        "min_zoom": 1,
        "is_wms": False,
    },
    "Viking MDIM": {
        "url": (
            "https://planetarymaps.usgs.gov/cgi-bin/mapserv?"
            "map=/maps/mars/mars_simp_cyl.map"
            "&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
            "&LAYERS=MDIM21"
            "&BBOX={bbox-epsg-3857}&WIDTH=256&HEIGHT=256"
            "&SRS=EPSG:3857&STYLES=&FORMAT=image/jpeg&TRANSPARENT=FALSE"
        ),
        "attribution": "Viking MDIM 2.1 – USGS Astrogeology Science Center",
        "max_zoom": 9,
        "min_zoom": 1,
        "is_wms": True,
        "wms_layers": "MDIM21",
        "wms_url": "https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/mars/mars_simp_cyl.map",
    },
}


# ---------------------------------------------------------------------------
# Folium layer factory
# ---------------------------------------------------------------------------

def get_basemap_tilelayer(
    name: str = "THEMIS IR Day",
    show: bool = True,
) -> Optional[object]:
    """
    Return a Folium TileLayer or WmsTileLayer for the named Mars basemap.

    Args:
        name : Key in BASEMAPS dict.
        show : Whether the layer is visible by default.

    Returns:
        folium.TileLayer or folium.raster_layers.WmsTileLayer, or None.

    Raises:
        ImportError : If folium is not installed.
        KeyError    : If name is not in BASEMAPS.
    """
    if not HAS_FOLIUM:
        raise ImportError("folium is required. Install it: pip install folium")

    if name not in BASEMAPS:
        raise KeyError(
            f"Unknown basemap: {name!r}. "
            f"Choose from: {list(BASEMAPS.keys())}"
        )

    bm = BASEMAPS[name]

    if bm["is_wms"]:
        return folium.raster_layers.WmsTileLayer(
            url=bm["wms_url"],
            layers=bm["wms_layers"],
            name=name,
            fmt="image/jpeg",
            transparent=False,
            attribution=bm["attribution"],
            show=show,
        )
    else:
        return folium.TileLayer(
            tiles=bm["url"],
            attr=bm["attribution"],
            name=name,
            max_zoom=bm["max_zoom"],
            min_zoom=bm["min_zoom"],
            show=show,
        )


def list_basemaps() -> list:
    """Return a list of available basemap names."""
    return list(BASEMAPS.keys())
