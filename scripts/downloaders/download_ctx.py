"""
CTX (Context Camera, MRO) Image Downloader
Downloads .IMG files from NASA PDS and converts to GeoTIFF.

CTX PDS URL:
  https://pds-imaging.jpl.nasa.gov/data/mro/ctx/
  → mrox_NNNN/data/{volume}/{filename}.IMG

CTX filenames encode: {instrument}_{orbit}_{lat_lon}_{phase}.IMG
Example: B02_010415_1440_XI_36S230W.IMG
"""

import os
import re
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import (
    get_logger, ensure_dir, StepCheckpoint, file_size_mb, write_sentinel
)

log = get_logger("downloader.ctx")

PDS_BASE = "https://pds-imaging.jpl.nasa.gov/data/mro/ctx/"
ODE_BASE = "https://ode.rsl.wustl.edu/mars/datafile/standard_meta_data"
CHUNK_SIZE = 1024 * 1024


def ctx_obs_to_url(obs_id: str) -> Optional[str]:
    """
    Resolve a CTX observation ID to its PDS download URL via ODE REST API.
    The ODE (PDS Orbital Data Explorer) provides product URL lookup.

    Parameters
    ----------
    obs_id : str
        CTX observation ID, e.g. "B02_010415_1440_XI_36S230W"

    Returns
    -------
    str or None
    """
    # Try ODE product search
    params = {
        "ihid": "MRO",
        "iid": "CTX",
        "pt": "EDR",
        "product_id": obs_id,
        "output": "JSON",
        "pretty": "false"
    }
    ode_url = "https://ode.rsl.wustl.edu/mars/product/product_files"
    try:
        resp = requests.get(ode_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        products = data.get("ODEResults", {}).get("Products", {}).get("Product", [])
        if isinstance(products, dict):
            products = [products]
        for product in products:
            files = product.get("Product_files", {}).get("Product_file", [])
            if isinstance(files, dict):
                files = [files]
            for pf in files:
                url = pf.get("URL", "")
                if url.endswith(".IMG"):
                    return url
    except Exception as e:
        log.debug(f"ODE lookup failed for {obs_id}: {e}")

    # Fallback: try ODE product search endpoint (alternate URL format)
    try:
        alt_url = "https://ode.rsl.wustl.edu/mars/product/product_files"
        params2 = {
            "ihid": "MRO", "iid": "CTX", "pt": "EDR",
            "product_id": obs_id + "*",   # wildcard match
            "output": "JSON", "pretty": "false"
        }
        resp2 = requests.get(alt_url, params=params2, timeout=30)
        resp2.raise_for_status()
        data2 = resp2.json()
        products2 = data2.get("ODEResults", {}).get("Products", {}).get("Product", [])
        if isinstance(products2, dict):
            products2 = [products2]
        for product in products2:
            files = product.get("Product_files", {}).get("Product_file", [])
            if isinstance(files, dict):
                files = [files]
            for pf in files:
                url = pf.get("URL", "")
                if url.endswith(".IMG") or url.endswith(".img"):
                    log.debug(f"ODE alt found URL: {url}")
                    return url
    except Exception as e:
        log.debug(f"ODE alt lookup failed: {e}")

    log.warning(
        f"Could not resolve CTX URL for {obs_id}. "
        "Use scripts/downloaders/manual_data_guide.py for manual download instructions."
    )
    return None


def img_to_geotiff(img_path: Path, out_path: Path) -> bool:
    """
    Convert PDS .IMG to GeoTIFF using gdal_translate.
    Returns True on success.
    """
    try:
        result = subprocess.run(
            ["gdal_translate", "-of", "GTiff", str(img_path), str(out_path)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            log.error(f"gdal_translate failed: {result.stderr}")
            return False
        log.debug(f"Converted {img_path.name} -> {out_path.name}")
        return True
    except FileNotFoundError:
        log.error("gdal_translate not found  -  install GDAL")
        return False
    except subprocess.TimeoutExpired:
        log.error(f"gdal_translate timeout for {img_path}")
        return False


def download_ctx_observation(
    obs_id: str,
    output_dir: Path,
    session: requests.Session,
    checkpoint: StepCheckpoint,
    convert_to_tif: bool = True,
    force: bool = False
) -> Optional[Path]:
    """
    Download a single CTX observation.

    Returns
    -------
    Path to the downloaded/converted file, or None on failure.
    """
    ck_key = obs_id
    if not force and checkpoint.is_done("ctx_download", ck_key):
        # Return existing tif or img path
        for ext in [".tif", ".IMG"]:
            p = output_dir / f"{obs_id}{ext}"
            if p.exists():
                log.info(f"[SKIP] {obs_id} already downloaded")
                return p
        # checkpoint exists but file missing — re-download
        checkpoint.clear("ctx_download")

    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = output_dir / f"{obs_id}.IMG"
    tif_path = output_dir / f"{obs_id}.tif"

    # Download .IMG if not present
    if not img_path.exists():
        url = ctx_obs_to_url(obs_id)
        if url is None:
            log.error(f"Cannot resolve URL for CTX {obs_id}")
            checkpoint.mark_failed("ctx_download", ck_key, "URL resolution failed")
            return None

        log.info(f"Downloading CTX {obs_id} from {url}")
        ok = _download_file(url, img_path, session)
        if not ok:
            checkpoint.mark_failed("ctx_download", ck_key, "Download failed")
            return None
    else:
        log.debug(f"IMG already exists: {img_path}")

    # Convert to GeoTIFF
    if convert_to_tif and not tif_path.exists():
        ok = img_to_geotiff(img_path, tif_path)
        if not ok:
            checkpoint.mark_failed("ctx_download", ck_key, "GDAL conversion failed")
            return None
        # Remove .IMG to save space
        img_path.unlink(missing_ok=True)
        result_path = tif_path
    elif convert_to_tif and tif_path.exists():
        result_path = tif_path
    else:
        result_path = img_path

    checkpoint.mark_done("ctx_download", ck_key, {"path": str(result_path)})
    log.info(f"[OK] CTX {obs_id} ready: {result_path}")
    return result_path


def _download_file(
    url: str, dest: Path, session: requests.Session = None, retries: int = 8
) -> bool:
    """Robust download with byte-range resume for CTX .IMG files."""
    import random
    for attempt in range(1, retries + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        try:
            s = requests.Session()
            s.headers.update({"Accept-Encoding": "identity", "Connection": "keep-alive"})
            headers = {"Range": f"bytes={existing}-"}
            resp = s.get(url, headers=headers, stream=True, timeout=(30, 300))
            if resp.status_code == 416:
                return True
            if resp.status_code not in (200, 206):
                raise requests.HTTPError(response=resp)
            total = int(resp.headers.get("Content-Length", 0)) + existing
            mode = "ab" if existing else "wb"
            with open(dest, mode) as f, tqdm(
                total=total, initial=existing, unit="B",
                unit_scale=True, desc=dest.name[:45], leave=True
            ) as pbar:
                for chunk in resp.iter_content(256 * 1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            mb = dest.stat().st_size / 1e6 if dest.exists() else 0
            log.warning(f"Attempt {attempt}/{retries} failed at {mb:.1f} MB: {type(e).__name__}: {e}")
            if attempt < retries:
                wait = 15 * attempt + random.uniform(0, 10)
                log.info(f"Resuming in {wait:.0f}s...")
                time.sleep(wait)
    return False


def download_ctx_all(
    cfg: Dict,
    output_dir: Path,
    checkpoint: StepCheckpoint,
    force: bool = False
) -> Dict[str, Dict[str, Optional[Path]]]:
    """Download all configured CTX observations."""
    if not cfg["data_sources"]["ctx"]["enabled"]:
        log.info("CTX disabled in config  -  skipping")
        return {}

    session = requests.Session()
    session.headers["User-Agent"] = "MarsGullyDigitalTwin/1.0"

    known = cfg["data_sources"]["ctx"].get("known_images", {})
    results = {}

    for site, obs_ids in known.items():
        log.info(f"=== CTX downloads for site: {site} ===")
        results[site] = {}
        for obs_id in obs_ids:
            path = download_ctx_observation(
                obs_id, Path(output_dir), session, checkpoint, force=force
            )
            results[site][obs_id] = path

    return results


if __name__ == "__main__":
    import yaml
    cfg_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    from scripts.utils import setup_logging
    setup_logging("INFO", "mars_pipeline.log")

    ck = StepCheckpoint("data/models")
    results = download_ctx_all(cfg, Path("data/raw/ctx"), ck)
    print(results)