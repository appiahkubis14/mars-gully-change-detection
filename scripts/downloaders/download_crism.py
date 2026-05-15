"""
CRISM (Compact Reconnaissance Imaging Spectrometer for Mars) Downloader
Downloads targeted TRDR (Targeted Reduced Data Record) products from NASA PDS.

CRISM data is more complex:
  - TRDR products: I/F spectra, wavelength calibrated, atmospherically corrected
  - Files: *_IF*.IMG (I/F) and *_DE*.IMG (detector)
  - Wavelengths: 362–3920 nm (544 channels); VNIR + IR detectors

PDS URL: https://pds-geosciences.wustl.edu/mro/mro-m-crism-3-rdr-targeted-v1/
"""

import re
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import requests
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger, StepCheckpoint, ensure_dir

log = get_logger("downloader.crism")

PDS_BASE = "https://pds-geosciences.wustl.edu/mro/mro-m-crism-3-rdr-targeted-v1/"
ODE_API = "https://ode.rsl.wustl.edu/mars/product/product_files"
CHUNK_SIZE = 1024 * 1024


def search_crism_by_bbox(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    session: requests.Session,
    max_results: int = 5
) -> List[Dict]:
    """
    Search ODE REST API for CRISM TRDR products within a bounding box.
    Returns list of product metadata dicts.
    """
    params = {
        "ihid": "MRO",
        "iid": "CRISM",
        "pt": "TRDR",
        "westlon": lon_min,
        "eastlon": lon_max,
        "minlat": lat_min,
        "maxlat": lat_max,
        "output": "JSON",
        "results": max_results
    }
    search_url = "https://ode.rsl.wustl.edu/mars/product/product_meta_data"
    try:
        resp = session.get(search_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        products = (
            data.get("ODEResults", {})
                .get("Products", {})
                .get("Product", [])
        )
        if isinstance(products, dict):
            products = [products]
        log.info(f"Found {len(products)} CRISM products in bbox")
        return products
    except Exception as e:
        log.warning(f"CRISM ODE search failed: {e}")
        return []


def get_crism_product_urls(product_id: str, session: requests.Session) -> List[str]:
    """Get file URLs for a CRISM product ID."""
    params = {
        "ihid": "MRO",
        "iid": "CRISM",
        "pt": "TRDR",
        "product_id": product_id,
        "output": "JSON"
    }
    try:
        resp = session.get(ODE_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        products = (
            data.get("ODEResults", {})
                .get("Products", {})
                .get("Product", [])
        )
        if isinstance(products, dict):
            products = [products]
        urls = []
        for p in products:
            pfiles = p.get("Product_files", {}).get("Product_file", [])
            if isinstance(pfiles, dict):
                pfiles = [pfiles]
            for pf in pfiles:
                url = pf.get("URL", "")
                if url.endswith((".IMG", ".HDR", ".LBL")):
                    urls.append(url)
        return urls
    except Exception as e:
        log.warning(f"Cannot get CRISM URLs for {product_id}: {e}")
        return []


def crism_img_to_geotiff(img_path: Path, out_path: Path) -> bool:
    """Convert CRISM .IMG (ENVI-format) to GeoTIFF using GDAL."""
    # CRISM .IMG files are ENVI-format with .HDR sidecar
    try:
        result = subprocess.run(
            ["gdal_translate", "-of", "GTiff",
             "-co", "COMPRESS=DEFLATE",
             "-co", "TILED=YES",
             str(img_path), str(out_path)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            log.error(f"gdal_translate error: {result.stderr[:300]}")
            return False
        log.info(f"✓ CRISM converted: {out_path.name}")
        return True
    except FileNotFoundError:
        log.error("gdal_translate not found")
        return False


def _download_file(url: str, dest: Path, session: requests.Session) -> bool:
    existing = dest.stat().st_size if dest.exists() else 0
    for attempt in range(1, 4):
        try:
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            resp = session.get(url, headers=headers, stream=True, timeout=120)
            if resp.status_code == 416:
                return True
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            mode = "ab" if existing else "wb"
            with open(dest, mode) as f, tqdm(
                total=total, initial=existing, unit="B",
                unit_scale=True, desc=dest.name[:40], leave=False
            ) as pbar:
                for chunk in resp.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            log.warning(f"Attempt {attempt} failed: {e}")
            if attempt < 3:
                time.sleep(10 * attempt)
    return False


def download_crism_for_site(
    site_name: str,
    bounds: List[float],
    output_dir: Path,
    checkpoint: StepCheckpoint,
    session: requests.Session,
    max_products: int = 3,
    force: bool = False
) -> List[Path]:
    """
    Search and download CRISM TRDR products for a study site.

    Parameters
    ----------
    bounds : [min_lon, min_lat, max_lon, max_lat]

    Returns list of GeoTIFF paths.
    """
    ck_key = f"{site_name}_crism"
    if not force and checkpoint.is_done("crism_download", ck_key):
        log.info(f"[SKIP] CRISM for {site_name} already downloaded")
        return list((output_dir / site_name).glob("*.tif"))

    site_dir = output_dir / site_name
    site_dir.mkdir(parents=True, exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = bounds
    products = search_crism_by_bbox(
        lon_min, lat_min, lon_max, lat_max, session, max_products
    )

    if not products:
        log.warning(f"No CRISM products found for {site_name}")
        # Write empty sentinel so we don't keep searching
        checkpoint.mark_done("crism_download", ck_key, {"products": 0})
        return []

    result_paths = []
    for product in products[:max_products]:
        pid = product.get("pdsid", product.get("ProductId", "unknown"))
        log.info(f"Downloading CRISM product: {pid}")

        urls = get_crism_product_urls(pid, session)
        if not urls:
            log.warning(f"No download URLs for {pid}")
            continue

        # Download .IMG and .HDR files
        downloaded = []
        for url in urls:
            filename = url.split("/")[-1]
            dest = site_dir / filename
            if not dest.exists():
                ok = _download_file(url, dest, session)
                if ok:
                    downloaded.append(dest)
            else:
                downloaded.append(dest)

        # Convert .IMG to GeoTIFF
        for dl in downloaded:
            if dl.suffix == ".IMG":
                tif_path = dl.with_suffix(".tif")
                if not tif_path.exists():
                    crism_img_to_geotiff(dl, tif_path)
                if tif_path.exists():
                    result_paths.append(tif_path)

    checkpoint.mark_done(
        "crism_download", ck_key,
        {"products": len(result_paths)}
    )
    return result_paths


def download_crism_all(
    cfg: Dict,
    output_dir: Path,
    checkpoint: StepCheckpoint,
    force: bool = False
) -> Dict[str, List[Path]]:
    """Download CRISM products for all configured study sites."""
    if not cfg["data_sources"]["crism"]["enabled"]:
        log.info("CRISM disabled in config — skipping")
        return {}

    session = requests.Session()
    session.headers["User-Agent"] = "MarsGullyDigitalTwin/1.0"
    output_dir = Path(output_dir)
    results = {}

    for area_key in ["primary", "secondary", "tertiary"]:
        area = cfg["study_area"].get(area_key)
        if not area:
            continue
        site_name = area["name"].replace(" ", "_")
        bounds = area["bounds"]
        log.info(f"=== CRISM for {site_name} ===")
        paths = download_crism_for_site(
            site_name, bounds, output_dir, checkpoint, session, force=force
        )
        results[site_name] = paths

    return results


if __name__ == "__main__":
    import yaml
    from scripts.utils import setup_logging
    setup_logging("INFO")

    cfg_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ck = StepCheckpoint("data/models")
    results = download_crism_all(cfg, Path("data/raw/crism"), ck)
    print(f"CRISM results: {results}")
