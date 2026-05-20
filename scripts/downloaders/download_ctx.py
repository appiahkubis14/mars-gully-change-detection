"""
CTX (MRO Context Camera) Downloader
=====================================
Downloads CTX EDR .IMG files from the PDS Imaging Node.

VERIFIED URL STRUCTURE (from live directory listing):
  Base: https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/{volume}/data/
  File: {PRODUCT_ID}.IMG

KEY INSIGHT from pds-imaging.jpl.nasa.gov/volumes/mro.html:
  CTX volumes are CUMULATIVE, not orbit-based. Volume numbers grow
  continuously across the mission. The volume containing a product is
  found by searching the cumulative index file, which maps product IDs
  to their volumes. We download this index once and cache it.

  Cumulative index (tab-separated):
  https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/{latest_vol}/index/cumindex.tab

ALTERNATIVE: The PDS Image Atlas search returns direct product URLs:
  https://pds-imaging.jpl.nasa.gov/api/search?q=PRODUCT_ID&target=MARS&i=CTX

Windows note: all log messages use ASCII only.
"""

import os
import re
import sys
import csv
import time
import random
import logging
import io
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger, StepCheckpoint, file_size_mb

log = get_logger("downloader.ctx")

PDS_CTX_BASE   = "https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/"
PDS_IMAGING    = "https://pds-imaging.jpl.nasa.gov"
ODE_SEARCH     = "https://ode.rsl.wustl.edu/mars/lroproductsearch/products/mrproducts"
ATLAS_SEARCH   = "https://pds-imaging.jpl.nasa.gov/api/search"

# Latest known CTX volume (updated to release 76)
LATEST_CTX_VOL = "mrox_5508"

# Local cache for the cumulative index
_CUMINDEX_CACHE: Dict[str, str] = {}   # product_id -> volume_id


# ---------------------------------------------------------------------------
# Cumulative index lookup (most reliable method)
# ---------------------------------------------------------------------------

def _load_cumindex(cache_dir: Path) -> Dict[str, str]:
    """
    Download and parse the CTX cumulative index (maps product_id -> volume).
    The index is a fixed-width .TAB file (~100 MB, cached locally).
    Returns dict: PRODUCT_ID.upper() -> volume_id (e.g. 'mrox_0634')
    """
    global _CUMINDEX_CACHE
    if _CUMINDEX_CACHE:
        return _CUMINDEX_CACHE

    cache_file = cache_dir / "ctx_cumindex.csv"

    if not cache_file.exists():
        # Download the cumulative index from the latest volume
        url = f"{PDS_CTX_BASE}{LATEST_CTX_VOL}/index/cumindex.tab"
        log.info(f"Downloading CTX cumulative index (~100 MB, one-time): {url}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            r = requests.get(url, stream=True, timeout=(30, 300),
                             headers={"Accept-Encoding": "identity"})
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(cache_file, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc="ctx_cumindex.tab", leave=True
            ) as pbar:
                for chunk in r.iter_content(256 * 1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        pbar.update(len(chunk))
            log.info(f"[OK] CTX cumindex cached: {cache_file}")
        except Exception as e:
            log.warning(f"Could not download CTX cumindex: {e}")
            return {}

    # Parse: fixed-width or comma-separated PDS3 table
    # Columns of interest: VOLUME_ID (col 0), FILE_SPECIFICATION_NAME (col 1),
    # PRODUCT_ID (col ~11), varies by release. We search for PRODUCT_ID string.
    log.info("Parsing CTX cumulative index...")
    result = {}
    try:
        with open(cache_file, "r", encoding="latin-1", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # The .TAB format is comma-separated with quoted strings
                # VOLUME_ID is first field, FILE_SPECIFICATION_NAME is second
                # e.g.: "MROX_0634","DATA/B02_010415_1440_XI_36S230W.IMG",...
                parts = [p.strip().strip('"') for p in line.split(",")]
                if len(parts) < 2:
                    continue
                vol = parts[0].lower()
                file_path = parts[1]
                # Extract product ID from file path (filename without .IMG)
                fname = file_path.split("/")[-1].split("\\")[-1]
                pid = fname.replace(".IMG", "").replace(".img", "").upper()
                if pid:
                    result[pid] = vol
    except Exception as e:
        log.warning(f"Error parsing cumindex: {e}")

    _CUMINDEX_CACHE = result
    log.info(f"CTX cumindex loaded: {len(result):,} products indexed")
    return result


def _lookup_via_cumindex(product_id: str, cache_dir: Path) -> Optional[str]:
    """Look up a CTX product's volume via the cumulative index."""
    idx = _load_cumindex(cache_dir)
    vol = idx.get(product_id.upper())
    if vol:
        url = f"{PDS_CTX_BASE}{vol}/data/{product_id}.IMG"
        log.debug(f"Cumindex resolved: {product_id} -> {url}")
        return url
    return None


# ---------------------------------------------------------------------------
# ODE REST API lookup
# ---------------------------------------------------------------------------

def _ode_lookup(product_id: str) -> Optional[str]:
    """Query the ODE REST API for a CTX product URL."""
    for endpoint in [
        "https://ode.rsl.wustl.edu/mars/lroproductsearch/products/mrproducts",
        "https://ode.rsl.wustl.edu/mars/lroproductsearch/products",
    ]:
        try:
            params = {
                "target": "Mars", "ihid": "MRO", "iid": "CTX",
                "pt": "EDR", "product_id": product_id,
                "output": "JSON", "limit": 5,
            }
            r = requests.get(endpoint, params=params, timeout=30)
            if r.status_code != 200:
                continue
            data = r.json()
            products = data.get("ODEResults", {}).get("Products", {})
            if not products:
                continue
            plist = products.get("Product", [])
            if isinstance(plist, dict):
                plist = [plist]
            for prod in plist:
                files = prod.get("Product_files", {}).get("Product_file", [])
                if isinstance(files, dict):
                    files = [files]
                for pf in files:
                    url = pf.get("URL", "")
                    if url.lower().endswith(".img"):
                        log.debug(f"ODE resolved: {product_id} -> {url}")
                        return url
        except Exception as e:
            log.debug(f"ODE endpoint {endpoint} failed: {e}")
    return None


# ---------------------------------------------------------------------------
# PDS Image Atlas search
# ---------------------------------------------------------------------------

def _atlas_lookup(product_id: str) -> Optional[str]:
    """Query the PDS Image Atlas search API."""
    try:
        params = {
            "q": product_id,
            "target": "MARS",
            "i": "CTX",
            "output": "JSON",
        }
        r = requests.get(ATLAS_SEARCH, params=params, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            pid = src.get("PRODUCT_ID", "")
            if pid.upper() == product_id.upper():
                url = src.get("PRODUCT_FILE_URL", "")
                if url:
                    return url
    except Exception as e:
        log.debug(f"Atlas lookup failed for {product_id}: {e}")
    return None


# ---------------------------------------------------------------------------
# Direct HEAD-probe fallback (wide volume range)
# ---------------------------------------------------------------------------

def _head_probe(product_id: str, session: requests.Session) -> Optional[str]:
    """
    Probe a wide range of volumes with HEAD requests.
    CTX orbit counter in product ID is NOT the MRO orbit.
    We probe 50 volumes in the approximate range.
    """
    # Extract the sequential counter from the product ID
    m = re.match(r"[A-Z]\d+_(\d+)_", product_id)
    if not m:
        return None
    counter = int(m.group(1))
    # Empirical: counter/15 ≈ volume number (very rough)
    approx = max(1, counter // 15)
    lo = max(1, approx - 25)
    hi = approx + 50

    log.debug(f"HEAD-probing volumes mrox_{lo:04d} to mrox_{hi:04d} for {product_id}")
    for vol_num in range(lo, hi):
        vol = f"mrox_{vol_num:04d}"
        url = f"{PDS_CTX_BASE}{vol}/data/{product_id}.IMG"
        try:
            r = session.head(url, timeout=8, allow_redirects=True)
            if r.status_code == 200:
                log.debug(f"HEAD found: {url}")
                return url
        except requests.RequestException:
            continue
    return None


# ---------------------------------------------------------------------------
# Master resolve function
# ---------------------------------------------------------------------------

def resolve_ctx_url(product_id: str, session: requests.Session,
                    cache_dir: Path = Path("data/raw/ctx/.cache")) -> Optional[str]:
    """
    Find the download URL for a CTX product using multiple strategies:
    1. Cumulative index file (most reliable, downloaded once)
    2. ODE REST API
    3. PDS Atlas search
    4. HEAD-probe volume range (slowest fallback)
    """
    # 1. Cumulative index
    url = _lookup_via_cumindex(product_id, cache_dir)
    if url:
        # Verify it actually exists
        try:
            r = session.head(url, timeout=10, allow_redirects=True)
            if r.status_code == 200:
                return url
            log.debug(f"Cumindex URL returned {r.status_code}, trying other methods")
        except Exception:
            pass

    # 2. ODE API
    url = _ode_lookup(product_id)
    if url:
        return url

    # 3. Atlas
    url = _atlas_lookup(product_id)
    if url:
        return url

    # 4. HEAD probe
    url = _head_probe(product_id, session)
    if url:
        return url

    log.warning(
        f"Could not resolve CTX URL for {product_id}. "
        f"Search manually: https://pds-imaging.jpl.nasa.gov/search/?q={product_id}"
    )
    return None


# ---------------------------------------------------------------------------
# Segmented download (same pattern as HiRISE)
# ---------------------------------------------------------------------------

def _get_size(url: str) -> int:
    try:
        r = requests.head(url, timeout=15, headers={"Accept-Encoding": "identity"},
                          allow_redirects=True)
        return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def _fetch_segment(url: str, dest: Path, start: int, end: int,
                   retries: int = 6) -> bool:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url,
                             headers={"Range": f"bytes={start}-{end}",
                                      "Accept-Encoding": "identity"},
                             stream=True, timeout=(20, 120))
            if r.status_code not in (200, 206):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            with open(dest, "ab") as f:
                for chunk in r.iter_content(256 * 1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            return True
        except Exception as e:
            wait = 5 * attempt + random.uniform(0, 5)
            log.debug(f"Segment attempt {attempt}: {e}. Retry in {wait:.0f}s")
            time.sleep(wait)
    return False


def _download_file(url: str, dest: Path, retries: int = 8) -> bool:
    """Segmented download with resume for CTX .IMG files."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _get_size(url)
    existing = dest.stat().st_size if dest.exists() else 0

    if total and existing >= total:
        log.info(f"[SKIP] {dest.name} already complete ({existing/1e6:.0f} MB)")
        return True

    seg_mb = 20
    seg = seg_mb * 1024 * 1024
    segments = [(i, min(i + seg - 1, total - 1)) for i in range(0, total or seg, seg)]

    log.info(f"Downloading {dest.name} ({total/1e6:.0f} MB) "
             f"in {len(segments)} x {seg_mb} MB segments")

    with tqdm(total=total or 0, initial=existing, unit="B",
              unit_scale=True, desc=dest.name[:45], leave=True) as pbar:
        current = existing
        for i, (s, e) in enumerate(segments):
            if total and e < existing:
                continue
            actual_s = max(s, existing)
            if actual_s > e:
                continue
            ok = _fetch_segment(url, dest, actual_s, e, retries)
            if not ok:
                log.error(f"[FAIL] CTX segment {i+1}/{len(segments)}")
                return False
            new_size = dest.stat().st_size
            pbar.update(new_size - current)
            current = new_size
            time.sleep(0.2)

    mb = file_size_mb(dest)
    log.info(f"[OK] {dest.name} ({mb:.1f} MB)")
    return True




# ---------------------------------------------------------------------------
# CTX IMG -> GeoTIFF conversion
# ---------------------------------------------------------------------------

def convert_img_to_geotiff(img_path: Path, output_path: Path = None) -> Optional[Path]:
    """
    Convert a CTX PDS3 .IMG file to a GeoTIFF.

    CTX .IMG files are raw 8-bit or 16-bit PDS3 images with embedded
    labels. GDAL can read them directly via the PDS3 driver.
    We extract the image data and write a GeoTIFF with basic metadata.

    The output has no map projection (CTX EDRs are not map-projected);
    projection is applied in the preprocess step using the spacecraft
    pointing metadata from the .LBL file.

    Args:
        img_path    : Path to the .IMG file
        output_path : Destination .tif path (default: same dir, .tif extension)

    Returns:
        Path to the output GeoTIFF, or None on failure.
    """
    if output_path is None:
        output_path = img_path.with_suffix(".tif")

    if output_path.exists() and output_path.stat().st_size > 10_000:
        log.debug(f"[SKIP] Already converted: {output_path.name}")
        return output_path

    try:
        import rasterio
        import numpy as np

        # GDAL PDS3 driver reads .IMG directly
        with rasterio.open(img_path) as src:
            data = src.read()           # (bands, H, W)
            meta = src.meta.copy()
            tags = src.tags()

        # CTX is single-band, but may read as multi-band from PDS label
        if data.ndim == 3 and data.shape[0] > 1:
            data = data[0:1]            # keep first band only

        # Normalise to float32 [0, 1] using 0.1-99.9th percentile
        valid = data[data > 0]
        if len(valid) > 0:
            lo, hi = np.percentile(valid, [0.1, 99.9])
            data_f = np.clip((data.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
        else:
            data_f = data.astype(np.float32)

        profile = {
            "driver":   "GTiff",
            "dtype":    "float32",
            "count":    1,
            "height":   data_f.shape[1],
            "width":    data_f.shape[2],
            "compress": "deflate",
            "tiled":    True,
            "blockxsize": 256,
            "blockysize": 256,
            "predictor": 3,
            # No CRS/transform: raw EDR has no map projection
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data_f)
            # Preserve PDS label metadata as GeoTIFF tags
            if tags:
                dst.update_tags(**{k: str(v) for k, v in list(tags.items())[:20]})

        log.info(f"[OK] CTX IMG -> GeoTIFF: {output_path.name} "
                 f"[{data_f.shape[2]}x{data_f.shape[1]}px]")
        return output_path

    except Exception as e:
        log.error(f"CTX IMG conversion failed for {img_path.name}: {e}")
        return None


def convert_all_ctx_imgs(ctx_dir: Path) -> Dict[str, Path]:
    """Convert all downloaded CTX .IMG files to GeoTIFFs."""
    results = {}
    for img_path in sorted(ctx_dir.rglob("*.IMG")):
        tif_path = convert_img_to_geotiff(img_path)
        if tif_path:
            results[img_path.stem] = tif_path
    return results

# ---------------------------------------------------------------------------
# Batch download
# ---------------------------------------------------------------------------

def download_ctx_all(
    cfg: dict,
    output_dir: Path,
    checkpoint: StepCheckpoint,
    force: bool = False,
) -> Dict[str, Dict]:
    """Download all configured CTX observations."""
    if not cfg["data_sources"]["ctx"]["enabled"]:
        log.info("CTX disabled in config -- skipping")
        return {}

    output_dir = Path(output_dir)
    cache_dir = output_dir / ".cache"
    session = requests.Session()
    session.headers["User-Agent"] = "MarsGullyDigitalTwin/1.0 (academic research)"
    results = {}

    known = cfg["data_sources"]["ctx"].get("known_images", {})
    for site, product_ids in known.items():
        log.info(f"=== CTX downloads for site: {site} ===")
        results[site] = {}
        for pid in product_ids:
            if not force and checkpoint.is_done("ctx_download", pid):
                log.info(f"[SKIP] {pid}")
                continue
            url = resolve_ctx_url(pid, session, cache_dir)
            if not url:
                log.error(f"Cannot resolve URL for CTX {pid}")
                continue
            dest = output_dir / site / f"{pid}.IMG"
            log.info(f"Downloading CTX {pid}")
            ok = _download_file(url, dest)
            if ok:
                # Convert .IMG to GeoTIFF immediately after download
                tif_path = convert_img_to_geotiff(dest)
                results[site][pid] = tif_path or dest
                checkpoint.mark_done("ctx_download", pid,
                                     {"path": str(tif_path or dest),
                                      "img": str(dest)})
    return results


# ---------------------------------------------------------------------------
# Compat shims for tests
# ---------------------------------------------------------------------------

def parse_ode_response(resp) -> list:
    try:
        data = resp.json()
        products = data.get("ODEResults", {}).get("Products", {})
        if not products:
            return []
        plist = products.get("Product", [])
        return [plist] if isinstance(plist, dict) else plist
    except Exception:
        return []


def build_ode_query_url(target: str = "Mars", bbox: tuple = None) -> str:
    url = f"{ODE_SEARCH}?target={target}&ihid=MRO&iid=CTX&pt=EDR&output=JSON"
    if bbox:
        url += f"&minlon={bbox[1]}&minlat={bbox[0]}&maxlon={bbox[3]}&maxlat={bbox[2]}"
    return url