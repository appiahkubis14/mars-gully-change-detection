"""
MOLA (Mars Orbiter Laser Altimeter) DEM Downloader
Downloads the global MOLA MEGDR (Mission Experiment Gridded Data Record)
at 1/128 degree per pixel (~463 m) resolution, then subsets to study areas.

Source: https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/data/
Global file: megt90n000fb.img (float32, 32-bit, 1440 x 720 pixels)
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger, StepCheckpoint, ensure_dir

log = get_logger("downloader.mola")

MOLA_BASE = "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/data/"
# 128 ppd (pixels per degree) global DEM
MOLA_FILE = "megt90n000fb.img"
MOLA_LBL = "megt90n000fb.lbl"
MOLA_128PPD_COLS = 46080  # 360 * 128
MOLA_128PPD_ROWS = 23040  # 180 * 128
MOLA_NODATA = -32768

# Simpler 4 ppd file for quick download (~9 MB)
MOLA_4PPD_FILE = "megt90n000eb.img"
MOLA_4PPD_COLS = 1440
MOLA_4PPD_ROWS = 720


def download_mola_global(
    output_dir: Path,
    checkpoint: StepCheckpoint,
    resolution: str = "4ppd",
    force: bool = False
) -> Optional[Path]:
    """
    Download the global MOLA DEM.

    Parameters
    ----------
    resolution : "4ppd" (463m) or "128ppd" (very large ~3 GB)

    Returns
    -------
    Path to downloaded .img file.
    """
    ck_key = f"mola_global_{resolution}"
    if not force and checkpoint.is_done("mola_download", ck_key):
        log.info("[SKIP] MOLA global DEM already downloaded")
        img_file = MOLA_FILE if resolution == "128ppd" else MOLA_4PPD_FILE
        p = output_dir / img_file
        if p.exists():
            return p

    output_dir.mkdir(parents=True, exist_ok=True)

    if resolution == "128ppd":
        filename = MOLA_FILE
        lbl_filename = MOLA_LBL
        min_size_mb = 1000.0
    else:
        filename = MOLA_4PPD_FILE
        lbl_filename = MOLA_4PPD_FILE.replace(".img", ".lbl")
        min_size_mb = 1.0

    img_url = MOLA_BASE + filename
    lbl_url = MOLA_BASE + lbl_filename

    log.info(f"Downloading MOLA global DEM ({resolution}): {img_url}")

    img_path = output_dir / filename
    if not img_path.exists():
        ok = _download_with_progress(img_url, img_path)
        if not ok:
            log.error("MOLA DEM download failed")
            # Try alternate source (USGS)
            usgs_url = f"https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
            log.info(f"Trying USGS alternate: {usgs_url}")
            img_path = output_dir / "mola_global_463m.tif"
            ok = _download_with_progress(usgs_url, img_path)
            if not ok:
                checkpoint.mark_failed("mola_download", ck_key, "All sources failed")
                return None

    # Download label file
    lbl_path = output_dir / lbl_filename
    if not lbl_path.exists() and not filename.endswith(".tif"):
        _download_with_progress(lbl_url, lbl_path)

    checkpoint.mark_done("mola_download", ck_key, {"path": str(img_path)})
    log.info(f"✓ MOLA DEM ready: {img_path}")
    return img_path


def _download_with_progress(url: str, dest: Path, retries: int = 3) -> bool:
    """Download file with progress bar and resume support."""
    import time
    existing = dest.stat().st_size if dest.exists() else 0

    for attempt in range(1, retries + 1):
        try:
            session = requests.Session()
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            resp = session.get(url, headers=headers, stream=True, timeout=120)
            if resp.status_code == 416:
                log.debug("File already complete (416)")
                return True
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            mode = "ab" if existing else "wb"
            with open(dest, mode) as f, tqdm(
                total=total, initial=existing,
                unit="B", unit_scale=True,
                desc=dest.name[:50], leave=True
            ) as pbar:
                for chunk in resp.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            log.warning(f"Download attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(10 * attempt)
    return False


def mola_img_to_geotiff(img_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Convert MOLA PDS .img to GeoTIFF using GDAL.
    The PDS file is a raw binary with a PDS label.
    """
    if img_path.suffix == ".tif":
        return img_path

    tif_path = output_dir / (img_path.stem + ".tif")
    if tif_path.exists():
        return tif_path

    # Try gdal_translate with PDS driver
    try:
        result = subprocess.run(
            ["gdal_translate",
             "-of", "GTiff",
             "-co", "COMPRESS=DEFLATE",
             "-co", "TILED=YES",
             str(img_path), str(tif_path)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            log.info(f"✓ MOLA GeoTIFF: {tif_path}")
            return tif_path
        else:
            log.warning(f"gdal_translate failed: {result.stderr[:200]}")
    except Exception as e:
        log.warning(f"GDAL conversion failed: {e}")

    # Fallback: read as raw binary and write GeoTIFF manually
    return _mola_raw_to_geotiff(img_path, tif_path)


def _mola_raw_to_geotiff(img_path: Path, tif_path: Path) -> Optional[Path]:
    """
    Read MOLA 4ppd raw binary and write GeoTIFF with proper georeferencing.
    MOLA 4ppd: int16, big-endian, 1440 cols × 720 rows, global coverage.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        data = np.fromfile(str(img_path), dtype=">i2").reshape(
            MOLA_4PPD_ROWS, MOLA_4PPD_COLS
        ).astype(np.float32)
        data[data == MOLA_NODATA] = np.nan

        # Mars equirectangular transform
        transform = from_bounds(-180, -90, 180, 90, MOLA_4PPD_COLS, MOLA_4PPD_ROWS)
        crs_wkt = (
            'GEOGCS["GCS_Mars_2000",'
            'DATUM["D_Mars_2000",'
            'SPHEROID["Mars_2000_IAU_IAG",3396190.0,169.89444722361179]],'
            'PRIMEM["Reference_Meridian",0.0],'
            'UNIT["Degree",0.0174532925199433]]'
        )

        with rasterio.open(
            tif_path, "w",
            driver="GTiff",
            height=MOLA_4PPD_ROWS,
            width=MOLA_4PPD_COLS,
            count=1,
            dtype=np.float32,
            crs=CRS.from_wkt(crs_wkt),
            transform=transform,
            compress="deflate"
        ) as dst:
            dst.write(data, 1)
            dst.update_tags(
                description="MOLA MEGDR 4ppd global DEM",
                units="meters",
                nodata=str(np.nan)
            )

        log.info(f"✓ MOLA GeoTIFF (manual): {tif_path}")
        return tif_path

    except Exception as e:
        log.error(f"Manual MOLA conversion failed: {e}")
        return None


def subset_mola(
    global_tif: Path,
    bounds: List[float],
    output_path: Path
) -> Optional[Path]:
    """
    Extract a bounding-box subset from the global MOLA DEM.

    Parameters
    ----------
    bounds : [min_lon, min_lat, max_lon, max_lat]
    """
    if output_path.exists():
        log.debug(f"MOLA subset exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bounds

    try:
        result = subprocess.run(
            ["gdal_translate",
             "-projwin", str(lon_min), str(lat_max), str(lon_max), str(lat_min),
             "-of", "GTiff",
             "-co", "COMPRESS=DEFLATE",
             str(global_tif), str(output_path)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            log.info(f"✓ MOLA subset: {output_path}")
            return output_path
        log.error(f"gdal_translate subset failed: {result.stderr[:200]}")
        return None
    except Exception as e:
        log.error(f"MOLA subset error: {e}")
        return None


def download_mola_all(
    cfg: Dict,
    output_dir: Path,
    checkpoint: StepCheckpoint,
    force: bool = False
) -> Dict[str, Optional[Path]]:
    """Download MOLA global DEM and extract subsets for all study sites."""
    if not cfg["data_sources"]["mola"]["enabled"]:
        log.info("MOLA disabled — skipping")
        return {}

    output_dir = Path(output_dir)

    # Download global DEM (4ppd ≈ 463m, manageable size)
    global_img = download_mola_global(output_dir, checkpoint, resolution="4ppd", force=force)
    if global_img is None:
        log.error("MOLA global DEM not available")
        return {}

    # Convert to GeoTIFF if needed
    global_tif = mola_img_to_geotiff(global_img, output_dir)
    if global_tif is None:
        log.error("MOLA GeoTIFF conversion failed")
        return {}

    # Extract subsets
    results = {"global": global_tif}
    for area_key in ["primary", "secondary", "tertiary"]:
        area = cfg["study_area"].get(area_key)
        if not area:
            continue
        site_name = area["name"].replace(" ", "_")
        bounds = area["bounds"]
        subset_path = output_dir / "subsets" / f"{site_name}_mola.tif"
        subset = subset_mola(global_tif, bounds, subset_path)
        results[site_name] = subset

    return results


if __name__ == "__main__":
    import yaml
    from scripts.utils import setup_logging
    setup_logging("INFO")

    cfg_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ck = StepCheckpoint("data/models")
    results = download_mola_all(cfg, Path("data/raw/mola"), ck)
    print(results)
