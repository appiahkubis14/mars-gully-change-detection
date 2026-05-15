"""
HRSC (High Resolution Stereo Camera, Mars Express) Downloader
Downloads from ESA PSA (Planetary Science Archive).

HRSC is optional (disabled by default in config) but provides:
- Stereo-derived DTM at 50-200 m/pixel (much better than MOLA)
- 5-channel colour images (stereo nadir, 4 photometry channels)

ESA PSA REST API: https://psa.esa.int/psa/api/v1.0/
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger, StepCheckpoint

log = get_logger("downloader.hrsc")

PSA_API = "https://psa.esa.int/psa/api/v1.0/"
CHUNK_SIZE = 1024 * 1024


def search_hrsc_products(
    bounds: List[float],
    session: requests.Session,
    product_type: str = "HRND",
    max_results: int = 5
) -> List[Dict]:
    """
    Search ESA PSA for HRSC nadir images within a bounding box.
    product_type:
      "HRND" = nadir image
      "HRDTM" = stereo DTM
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    params = {
        "mission": "MEX",
        "instrument": "HRSC",
        "product_type": product_type,
        "min_longitude": lon_min,
        "max_longitude": lon_max,
        "min_latitude": lat_min,
        "max_latitude": lat_max,
        "format": "json",
        "page_size": max_results
    }
    try:
        resp = session.get(f"{PSA_API}products", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        log.warning(f"ESA PSA search failed: {e}")
        return []


def get_hrsc_download_url(product_id: str, session: requests.Session) -> Optional[str]:
    """Get direct download URL for an HRSC product."""
    try:
        resp = session.get(f"{PSA_API}products/{product_id}/files", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for f in data.get("files", []):
            url = f.get("download_url", "")
            if url.endswith((".IMG", ".TIF", ".img", ".tif")):
                return url
    except Exception as e:
        log.debug(f"Cannot get HRSC URL: {e}")
    return None


def download_hrsc_all(
    cfg: Dict,
    output_dir: Path,
    checkpoint: StepCheckpoint,
    force: bool = False
) -> Dict:
    """Download HRSC data if enabled."""
    if not cfg["data_sources"]["hrsc"]["enabled"]:
        log.info("HRSC disabled in config — skipping")
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
        ck_key = f"{site_name}_hrsc"

        if not force and checkpoint.is_done("hrsc_download", ck_key):
            log.info(f"[SKIP] HRSC for {site_name} done")
            continue

        log.info(f"Searching HRSC for {site_name}")
        products = search_hrsc_products(bounds, session)
        if not products:
            log.warning(f"No HRSC products for {site_name}")
            continue

        site_dir = output_dir / site_name
        site_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []

        for p in products[:3]:
            pid = p.get("id", "")
            url = get_hrsc_download_url(pid, session)
            if url:
                dest = site_dir / url.split("/")[-1]
                if not dest.exists():
                    ok = _download(url, dest, session)
                    if ok:
                        downloaded.append(dest)
                else:
                    downloaded.append(dest)

        results[site_name] = downloaded
        checkpoint.mark_done("hrsc_download", ck_key, {"files": len(downloaded)})

    return results


def _download(url: str, dest: Path, session: requests.Session) -> bool:
    try:
        resp = session.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=dest.name[:40], leave=False
        ) as pbar:
            for chunk in resp.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        log.error(f"HRSC download failed: {e}")
        return False
