"""
Manual Data Download Guide
Prints step-by-step instructions for data that cannot be fully automated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger

log = get_logger("guide.manual")

HIRISE_GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║              HIRISE MANUAL DOWNLOAD GUIDE                        ║
╠══════════════════════════════════════════════════════════════════╣
║ The automated downloader handles most HiRISE downloads.          ║
║ If it fails (network, authentication), use the manual method:    ║
╠══════════════════════════════════════════════════════════════════╣

1. Go to: https://www.uahirise.org/hiwish/
   OR: https://hirise-pds.lpl.arizona.edu/PDS/EDR/

2. For Gasa Crater gullies, download these observations:
   - ESP_012821_1440_RED.JP2   (2009)
   - ESP_016625_1440_RED.JP2   (2010)
   - ESP_023957_1440_RED.JP2   (2012)
   - ESP_034579_1440_RED.JP2   (2015)
   - ESP_055527_1440_RED.JP2   (2018)

3. For Palikir Crater:
   - ESP_011428_1445_RED.JP2
   - ESP_022351_1445_RED.JP2
   - ESP_034303_1445_RED.JP2

4. For Russell Crater:
   - ESP_016042_1255_RED.JP2
   - ESP_034313_1255_RED.JP2
   - ESP_047959_1255_RED.JP2

5. Place files in: data/raw/hirise/{OBSERVATION_ID}/

   Example: data/raw/hirise/ESP_012821_1440/ESP_012821_1440_RED.JP2

6. Direct URL pattern:
   https://hirise-pds.lpl.arizona.edu/PDS/EDR/{PHASE}/ORB_{NNNN00}_{NNNN99}/{OBS_ID}/{OBS_ID}_RED.JP2

   Example:
   https://hirise-pds.lpl.arizona.edu/PDS/EDR/ESP/ORB_012800_012899/ESP_012821_1440/ESP_012821_1440_RED.JP2

7. Alternative: HiRISE Image Viewer (HiView)
   - Download: https://www.uahirise.org/hiview/
   - Export .JP2 files from the viewer

╚══════════════════════════════════════════════════════════════════╝
"""

CTX_GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║                 CTX MANUAL DOWNLOAD GUIDE                        ║
╠══════════════════════════════════════════════════════════════════╣

1. Use NASA ODE (Orbital Data Explorer):
   https://ode.rsl.wustl.edu/mars/

2. Select:
   - Instrument: MRO/CTX
   - Product Type: EDR
   - Draw bounding box over study site

3. Click "Search Products" and download .IMG files

4. Place in: data/raw/ctx/{OBSERVATION_ID}.IMG

5. Run conversion:
   gdal_translate -of GTiff data/raw/ctx/{OBS}.IMG data/raw/ctx/{OBS}.tif

6. Alternatively use JMARS (free NASA software):
   https://jmars.asu.edu/
   Layer → Add Layer → CTX Image Gallery
   Export visible area as GeoTIFF

╚══════════════════════════════════════════════════════════════════╝
"""

CRISM_GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║                CRISM MANUAL DOWNLOAD GUIDE                       ║
╠══════════════════════════════════════════════════════════════════╣
║ CRISM data requires special handling (large files, many bands)   ║
╠══════════════════════════════════════════════════════════════════╣

1. Go to: https://crism-map.jhuapl.edu/
   (CRISM Analysis Tool - CAT-Web)

2. OR use ODE:
   https://ode.rsl.wustl.edu/mars/
   Select MRO/CRISM, Product Type: TRDR (Targeted Reduced Data Record)

3. For Gasa Crater area (lon 129.5, lat -35.7):
   Search within bounds [128.5, -36.5, 130.5, -34.9]

4. Download the _IF*.IMG files (I/F, atmospherically corrected)
   Avoid the raw DN files (_RA*.IMG)

5. Place in: data/raw/crism/{PRODUCT_ID}/
   e.g. data/raw/crism/FRT0000A37A_07_IF183S_TRR3.IMG

6. Convert to GeoTIFF:
   gdal_translate -of GTiff FRT0000A37A_07_IF183S_TRR3.IMG crism_gasa.tif

7. Key CRISM wavelengths for Mars gullies:
   - ~530 nm (Fe oxidation)
   - ~1000 nm (Fe2+/Fe3+)
   - ~1400 nm (hydration)
   - ~2300 nm (carbonate/phyllosilicate)

NOTE: CRISM is optional. The pipeline runs without it using only
      HiRISE and CTX bands.

╚══════════════════════════════════════════════════════════════════╝
"""

MOLA_GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║                MOLA MANUAL DOWNLOAD GUIDE                        ║
╠══════════════════════════════════════════════════════════════════╣

1. Direct download (automated):
   https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/data/megt90n000eb.img

2. If automated download fails:
   a. Go to: https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/data/
   b. Download: megt90n000eb.img (~2 MB, 4ppd global DEM)
   c. Download: megt90n000eb.lbl (label file)
   d. Place in: data/raw/mola/

3. Higher resolution option (463 m = 4ppd, globally):
   wget https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/data/megt90n000eb.img

4. Pre-projected GeoTIFF alternative from USGS:
   https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif
   (~450 MB, ready to use)

5. Place in: data/raw/mola/
   The pipeline will subset to each study area automatically.

╚══════════════════════════════════════════════════════════════════╝
"""

THEMIS_GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║              THEMIS MANUAL DOWNLOAD GUIDE                        ║
╠══════════════════════════════════════════════════════════════════╣

THEMIS IR Day mosaic is used as a basemap (100 m/pixel).

1. Global THEMIS IR Day mosaic (~4 GB):
   https://astrogeology.usgs.gov/search/map/Mars/Odyssey/THEMIS-IR-Mosaic-ASU/Mars_MO_THEMIS-IR-Day_mosaic_global_100m_v12

2. Smaller local tiles via JMARS:
   https://jmars.asu.edu/
   - THEMIS IR Layer → Export visible area

3. Web tile source for Folium basemap:
   URL Template: https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/{z}/{x}/{y}.jpg
   (No download needed - used directly as tile URL in dashboard)

4. For local processing (optional):
   Place THEMIS GeoTIFF in: data/raw/themis/themis_ir_day_global.tif

╚══════════════════════════════════════════════════════════════════╝
"""

LABELS_GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║           MANUAL LABELLING GUIDE (QGIS)                          ║
╠══════════════════════════════════════════════════════════════════╣

For high-quality training labels, digitize gullies manually in QGIS:

SETUP:
1. Install QGIS: https://qgis.org/
2. Open QGIS → New Project
3. Set CRS: ESRI:104971 (Mars equirectangular)
   OR use geographic (lat/lon) with Mars sphere

LOAD DATA:
4. Layer → Add Layer → Add Raster Layer
   Select: data/raw/hirise/{obs_id}/{obs_id}_RED.JP2
5. Render as grayscale, stretch to min/max

CREATE LABEL LAYER:
6. Layer → Create Layer → New GeoPackage Layer
   - File: data/raw/labels/gully_labels_{site}.gpkg
   - Geometry type: Polygon
   - Attributes: gully_id (int), confidence (int 1-3), date_seen (string)

DIGITIZE GULLIES:
7. Enable editing (pencil icon)
8. Use "Add Polygon Feature" tool
9. Trace gully outline carefully:
   - Follow the eroded channel and alcove
   - Include apron deposit at toe
   - 1 polygon per gully system (not per channel)
10. For each polygon, set:
    - confidence: 1=possible, 2=probable, 3=certain
    - date_seen: "2009" or "2010" etc.

EXPORT:
11. Right-click layer → Export → Save Features As
    Format: GeoJSON
    File: data/raw/labels/gully_labels_{site}.geojson
    CRS: Same as project

RASTERIZE:
12. In QGIS: Raster → Conversion → Rasterize
    Input: gully_labels.geojson
    Burn value: 1
    Output: data/processed/masks/{site}_labels.tif
    Resolution: 6 m (match CTX)

OR use the pipeline command:
   python main.py --step labels --method manual --labels-path data/raw/labels/

EXISTING LABELLED DATASETS:
- Dundas et al. 2019 Gully Catalog (contact authors for shapefile)
- HiRISE Change Detection DB: https://www.uahirise.org/edr/

╚══════════════════════════════════════════════════════════════════╝
"""


def print_all_guides():
    """Print all manual download guides."""
    guides = [
        ("HiRISE", HIRISE_GUIDE),
        ("CTX", CTX_GUIDE),
        ("CRISM", CRISM_GUIDE),
        ("MOLA", MOLA_GUIDE),
        ("THEMIS", THEMIS_GUIDE),
        ("Labels", LABELS_GUIDE),
    ]
    for name, guide in guides:
        print(guide)


def print_guide(name: str):
    """Print guide for a specific dataset."""
    guides = {
        "hirise": HIRISE_GUIDE,
        "ctx": CTX_GUIDE,
        "crism": CRISM_GUIDE,
        "mola": MOLA_GUIDE,
        "themis": THEMIS_GUIDE,
        "labels": LABELS_GUIDE,
    }
    g = guides.get(name.lower())
    if g:
        print(g)
    else:
        print(f"No guide for '{name}'. Options: {list(guides.keys())}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manual data download instructions")
    parser.add_argument("--dataset", default="all",
                        help="Dataset to show guide for (hirise/ctx/crism/mola/themis/labels/all)")
    args = parser.parse_args()

    if args.dataset == "all":
        print_all_guides()
    else:
        print_guide(args.dataset)
