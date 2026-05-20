import os
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

    # Check if the USGS GeoTIFF was already downloaded successfully
    usgs_tif = output_dir / "mola_global_463m.tif"
    if usgs_tif.exists() and usgs_tif.stat().st_size > 100_000_000:
        log.info(f"[OK] MOLA GeoTIFF already present: {usgs_tif}")
        checkpoint.mark_done("mola_download", ck_key, {"path": str(usgs_tif)})
        return usgs_tif

    img_path = output_dir / filename
    if not img_path.exists():
        ok = _download_with_progress(img_url, img_path)
        if not ok:
            log.error("MOLA DEM download failed")
            # Try alternate source (USGS GeoTIFF — preferred for Windows)
            usgs_url = "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
            log.info(f"Trying USGS alternate: {usgs_url}")
            img_path = usgs_tif
            ok = _download_with_progress(usgs_url, img_path)
            if not ok:
                checkpoint.mark_failed("mola_download", ck_key, "All sources failed")
                return None

    # Download label file
    lbl_path = output_dir / lbl_filename
    if not lbl_path.exists() and not filename.endswith(".tif"):
        _download_with_progress(lbl_url, lbl_path)

    checkpoint.mark_done("mola_download", ck_key, {"path": str(img_path)})
    log.info(f"[OK] MOLA DEM ready: {img_path}")
    return img_path


def _download_with_progress(url: str, dest: Path, retries: int = 10) -> bool:
    """
    Robust download with byte-range resume.
    Handles ConnectionResetError (WinError 10054) from NASA/USGS servers
    on large files (MOLA GeoTIFF is 2.13 GB).
    """
    import time, random

    for attempt in range(1, retries + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        try:
            session = requests.Session()
            session.headers.update({
                "Accept-Encoding": "identity",  # required for Range resume
                "Connection": "keep-alive",
            })
            headers = {"Range": f"bytes={existing}-"}
            resp = session.get(url, headers=headers, stream=True, timeout=(30, 300))
            if resp.status_code == 416:
                log.debug("File already complete (416)")
                return True
            if resp.status_code not in (200, 206):
                log.warning(f"HTTP {resp.status_code} on attempt {attempt}")
                raise requests.HTTPError(response=resp)
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0)) + existing
            mode = "ab" if existing else "wb"
            with open(dest, mode) as f, tqdm(
                total=total, initial=existing,
                unit="B", unit_scale=True,
                desc=dest.name[:50], leave=True
            ) as pbar:
                for chunk in resp.iter_content(256 * 1024):  # 256 KB chunks
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            mb = dest.stat().st_size / 1e6 if dest.exists() else 0
            log.warning(f"Download attempt {attempt}/{retries} failed at {mb:.1f} MB: {type(e).__name__}: {e}")
            if attempt < retries:
                wait = 15 * attempt + random.uniform(0, 10)
                log.info(f"Resuming in {wait:.0f}s (file preserved at {mb:.1f} MB)...")
                time.sleep(wait)
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
            log.info(f"[OK] MOLA GeoTIFF: {tif_path}")
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

        log.info(f"[OK] MOLA GeoTIFF (manual): {tif_path}")
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

    bounds : [min_lon, min_lat, max_lon, max_lat]

    Tries gdal_translate first; falls back to pure rasterio windowed read
    so it works on Windows without a GDAL binary on PATH.
    """
    if output_path.exists():
        log.debug(f"MOLA subset exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bounds

    # --- Try gdal_translate first ---
    import shutil
    gdal_bin = shutil.which("gdal_translate")
    if gdal_bin:
        try:
            result = subprocess.run(
                [gdal_bin,
                 "-projwin", str(lon_min), str(lat_max), str(lon_max), str(lat_min),
                 "-of", "GTiff", "-co", "COMPRESS=DEFLATE",
                 str(global_tif), str(output_path)],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                log.info(f"[OK] MOLA subset (gdal): {output_path}")
                return output_path
            log.warning(f"gdal_translate failed: {result.stderr[:200]}")
        except Exception as e:
            log.warning(f"gdal_translate error: {e}")

    # --- Fallback: rasterio windowed read (no GDAL binary required) ---
    log.info("Using rasterio for MOLA subset (gdal_translate not found)")
    return _subset_mola_rasterio(global_tif, bounds, output_path)


def _subset_mola_rasterio(
    global_tif: Path,
    bounds: List[float],
    output_path: Path
) -> Optional[Path]:
    """
    PROJ-INDEPENDENT MOLA subset using pure affine pixel arithmetic.

    The USGS MOLA GeoTIFF (mola_global_463m.tif) is in a Simple Cylindrical
    projection with units of METRES (not degrees). The affine transform is:
        T.c = 0.0         (x origin, metres)
        T.f = 5335046.0   (y origin = lat 90N in metres)
        T.a = 463.094     (metres per pixel, longitude direction)
        T.e = -463.094    (metres per pixel, latitude direction, negative)

    Bounds from config.yaml are in DEGREES (lon/lat). We convert to metres
    using the Mars mean radius (3,396,190 m) before computing pixel indices.

    No PROJ, no pyproj, no window_from_bounds -- pure arithmetic only.
    """
    import math
    import rasterio
    import rasterio.windows
    import numpy as np

    MARS_R = 3_396_190.0  # Mars mean radius, metres

    def lon_to_m(lon_deg: float) -> float:
        return lon_deg * math.pi / 180.0 * MARS_R

    def lat_to_m(lat_deg: float) -> float:
        return lat_deg * math.pi / 180.0 * MARS_R

    try:
        lon_min, lat_min, lon_max, lat_max = bounds

        # Convert degree bounds to metres
        x_min = lon_to_m(lon_min)
        x_max = lon_to_m(lon_max)
        y_min = lat_to_m(lat_min)   # more negative (further south)
        y_max = lat_to_m(lat_max)   # less negative (further north)

        with rasterio.open(global_tif) as src:
            T = src.transform
            W = src.width
            H = src.height

            # Affine coefficients (in metres)
            origin_x = T.c   # x-coordinate of left edge of col=0
            origin_y = T.f   # y-coordinate of top edge of row=0
            px_x = T.a       # metres per pixel in x (positive)
            px_y = T.e       # metres per pixel in y (negative)

            log.debug(
                f"MOLA transform: origin=({origin_x:.1f},{origin_y:.1f}) m "
                f"px=({px_x:.3f},{px_y:.3f}) m/px  size={W}x{H}"
            )
            log.debug(
                f"Bounds in metres: x=[{x_min:.0f},{x_max:.0f}] "
                f"y=[{y_min:.0f},{y_max:.0f}]"
            )

            # Pixel column: col = (x - origin_x) / px_x
            col_min = int((x_min - origin_x) / px_x)
            col_max = int((x_max - origin_x) / px_x) + 1

            # Pixel row: row = (origin_y - y) / |px_y|
            # (y decreases as row index increases)
            row_min = int((origin_y - y_max) / abs(px_y))
            row_max = int((origin_y - y_min) / abs(px_y)) + 1

            # Clamp to raster bounds
            col_min = max(0, min(col_min, W - 1))
            col_max = max(col_min + 1, min(col_max, W))
            row_min = max(0, min(row_min, H - 1))
            row_max = max(row_min + 1, min(row_max, H))

            width  = col_max - col_min
            height = row_max - row_min

            log.debug(
                f"Pixel window: cols={col_min}-{col_max} "
                f"rows={row_min}-{row_max} size={width}x{height}"
            )

            if width <= 0 or height <= 0:
                log.error(
                    f"Empty window after clamping. Bounds (deg) {bounds} -> "
                    f"metres x=[{x_min:.0f},{x_max:.0f}] y=[{y_min:.0f},{y_max:.0f}]. "
                    f"Raster x=[{origin_x:.0f},{origin_x+W*px_x:.0f}] "
                    f"y=[{origin_y+H*px_y:.0f},{origin_y:.0f}]"
                )
                return None

            win  = rasterio.windows.Window(col_min, row_min, width, height)
            data = src.read(1, window=win)

            # New affine transform for the subset window
            new_origin_x = origin_x + col_min * px_x
            new_origin_y = origin_y + row_min * px_y  # px_y is negative

            from affine import Affine
            new_transform = Affine(px_x, 0.0, new_origin_x,
                                   0.0, px_y, new_origin_y)

            profile = src.profile.copy()

            # Block size must be multiple of 16 for tiled GeoTIFF
            def _blk(n):
                b = min(256, n)
                return max(16, (b // 16) * 16)

            use_tiling = (width >= 16 and height >= 16)
            profile.update(
                width=width, height=height,
                transform=new_transform,
                compress="deflate",
                count=1, driver="GTiff",
            )
            if use_tiling:
                profile.update(tiled=True,
                               blockxsize=_blk(width),
                               blockysize=_blk(height))
            else:
                for k in ("tiled", "blockxsize", "blockysize"):
                    profile.pop(k, None)

            # Drop CRS to avoid PROJ validation error during write
            crs_backup = profile.pop("crs", None)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)
            if crs_backup is not None:
                try:
                    dst.crs = crs_backup
                except Exception:
                    pass

        # Convert window back to degrees for the log message
        lon_out_min = (new_origin_x) / MARS_R * 180.0 / math.pi
        lon_out_max = (new_origin_x + width * px_x) / MARS_R * 180.0 / math.pi
        lat_out_max = (new_origin_y) / MARS_R * 180.0 / math.pi
        lat_out_min = (new_origin_y + height * px_y) / MARS_R * 180.0 / math.pi

        log.info(
            f"[OK] MOLA subset: {output_path.name} "
            f"[{width}x{height}px | "
            f"{lon_out_min:.2f}-{lon_out_max:.2f}E | "
            f"{lat_out_min:.2f}-{lat_out_max:.2f}]"
        )
        return output_path

    except Exception as e:
        log.error(f"MOLA rasterio subset failed: {e}", exc_info=True)
        return None


def download_mola_all(
    cfg: Dict,
    output_dir: Path,
    checkpoint: StepCheckpoint,
    force: bool = False
) -> Dict[str, Optional[Path]]:
    """Download MOLA global DEM and extract subsets for all study sites."""
    if not cfg["data_sources"]["mola"]["enabled"]:
        log.info("MOLA disabled  -  skipping")
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