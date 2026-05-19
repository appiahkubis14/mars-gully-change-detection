"""
HiRISE (MRO) Image Downloader
==============================
Downloads processed HiRISE RDR products from the PDS archive.

Verified URL structure (from uahirise.org product pages):
  Browse/preview:  https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/ESP/{ORB_BLOCK}/{OBS_ID}/
  FULL DOWNLOAD:   https://hirise-pds.lpl.arizona.edu/download/PDS/RDR/ESP/{ORB_BLOCK}/{OBS_ID}/

Products in priority order (all are usable GeoTIFFs or JP2s):
  {OBS_ID}_RED.JP2     -- panchromatic mosaic map-projected (300-700 MB)
  {OBS_ID}_COLOR.JP2   -- IRB color mosaic map-projected (140-400 MB)
  {OBS_ID}_MRGB.JP2    -- merged RGB map-projected, in EXTRAS tree (150-300 MB)
  {OBS_ID}_MIRB.JP2    -- merged IRB map-projected, in EXTRAS tree

Windows note: all log messages use ASCII only (no Unicode symbols).
"""

import re
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger, StepCheckpoint, file_size_mb

log = get_logger("downloader.hirise")

# The /download/ prefix is required for full JP2 downloads
PDS_DOWNLOAD_RDR  = "https://hirise-pds.lpl.arizona.edu/download/PDS/RDR/"
PDS_DOWNLOAD_EXTRAS = "https://hirise-pds.lpl.arizona.edu/download/PDS/EXTRAS/RDR/"
# The non-download path serves directory listings (HTML) but NOT full files
PDS_LISTING_RDR   = "https://hirise-pds.lpl.arizona.edu/PDS/RDR/"
PDS_LISTING_EXTRAS = "https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/"

CHUNK_SIZE = 256 * 1024   # 256 KB — smaller chunks survive PDS connection resets


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _orbit_block(orbit: int) -> str:
    lo = (orbit // 100) * 100
    return f"ORB_{lo:06d}_{lo + 99:06d}"


def _parse_obs(obs_id: str) -> Tuple[str, int]:
    """Return (phase, orbit_number) from an observation ID."""
    parts = obs_id.split("_")
    return parts[0], int(parts[1])


def build_download_candidates(obs_id: str) -> List[Tuple[str, str]]:
    """
    Return candidate (download_url, filename) pairs in priority order.

    Tries both the RDR tree and the EXTRAS/RDR tree, which has additional
    merged-color products (MRGB, MIRB).
    """
    phase, orbit = _parse_obs(obs_id)
    block = _orbit_block(orbit)
    path = f"{phase}/{block}/{obs_id}/"

    candidates = []
    # Primary RDR products (smaller, map-projected)
    for suffix in ["RED.JP2", "COLOR.JP2"]:
        fname = f"{obs_id}_{suffix}"
        candidates.append((f"{PDS_DOWNLOAD_RDR}{path}{fname}", fname))
    # EXTRAS/RDR products (merged color, slightly smaller)
    for suffix in ["MRGB.JP2", "MIRB.JP2", "RED.QLOOK.JP2", "COLOR.QLOOK.JP2"]:
        fname = f"{obs_id}_{suffix}"
        candidates.append((f"{PDS_DOWNLOAD_EXTRAS}{path}{fname}", fname))
    return candidates


def build_listing_url(obs_id: str, extras: bool = False) -> str:
    """Build the HTML directory listing URL (no /download/ prefix)."""
    phase, orbit = _parse_obs(obs_id)
    block = _orbit_block(orbit)
    base = PDS_LISTING_EXTRAS if extras else PDS_LISTING_RDR
    return f"{base}{phase}/{block}/{obs_id}/"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _head_check(url: str, session: requests.Session, timeout: int = 15) -> bool:
    """Return True if url responds HTTP 200."""
    try:
        r = session.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _scrape_listing(url: str, session: requests.Session) -> List[str]:
    """Scrape an HTML directory listing, return JP2/IMG file hrefs."""
    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200:
            return []
        pattern = re.compile(r'href="([^"]+\.(JP2|IMG|LBL))"', re.IGNORECASE)
        return [urljoin(url, m.group(1)) for m in pattern.finditer(r.text)]
    except requests.RequestException:
        return []


def find_hirise_download_urls(obs_id: str, session: requests.Session) -> List[str]:
    """
    Locate downloadable JP2 files for an observation.

    Strategy:
    1. HEAD-probe the known /download/ URLs in priority order
       (most reliable -- no directory listing needed)
    2. Scrape listing pages as fallback to discover filenames
    """
    log.debug(f"Probing download URLs for {obs_id}")
    found = []

    # Strategy 1: direct HEAD probes on /download/ URLs
    for url, fname in build_download_candidates(obs_id):
        if _head_check(url, session):
            log.debug(f"Found via HEAD: {fname}")
            found.append(url)
            # We want at most one RED and one COLOR/MRGB
            if len(found) >= 2:
                break

    if found:
        return found

    # Strategy 2: scrape directory listings, fix URLs to use /download/
    for extras in [False, True]:
        listing_url = build_listing_url(obs_id, extras=extras)
        raw_urls = _scrape_listing(listing_url, session)
        if raw_urls:
            log.debug(f"Directory listing OK ({listing_url}): {len(raw_urls)} entries")
            # Rewrite to /download/ equivalents and verify
            for raw in raw_urls:
                if not raw.lower().endswith(".jp2"):
                    continue
                # Replace the listing base with the download base
                if extras:
                    dl = raw.replace(PDS_LISTING_EXTRAS, PDS_DOWNLOAD_EXTRAS)
                else:
                    dl = raw.replace(PDS_LISTING_RDR, PDS_DOWNLOAD_RDR)
                if _head_check(dl, session):
                    found.append(dl)
                if len(found) >= 2:
                    break
        if found:
            return found

    log.warning(
        f"No files found for {obs_id}. "
        f"Check https://www.uahirise.org/{obs_id} to verify the observation exists. "
        f"Run: python scripts/downloaders/manual_data_guide.py for manual instructions."
    )
    return []


# ---------------------------------------------------------------------------
# File download with resume
# ---------------------------------------------------------------------------

def _get_remote_size(url: str) -> int:
    """Return total file size in bytes via HEAD request."""
    try:
        r = requests.head(url, timeout=20, allow_redirects=True,
                          headers={"Accept-Encoding": "identity"})
        return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def _download_segment(url: str, dest_path: Path,
                      start: int, end: int, retries: int = 6) -> bool:
    """
    Download exactly the byte range [start, end] and append to dest_path.

    Each call is a fresh short HTTP request (~30s at 500 kB/s for 15 MB).
    This defeats the HiRISE PDS server which resets connections after ~40s.
    """
    import random
    for attempt in range(1, retries + 1):
        try:
            s = requests.Session()
            s.headers.update({"Accept-Encoding": "identity",
                               "User-Agent": "MarsGullyDigitalTwin/1.0"})
            r = s.get(url, headers={"Range": f"bytes={start}-{end}"},
                      stream=True, timeout=(20, 120))
            if r.status_code not in (200, 206):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            with open(dest_path, "ab") as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            return True
        except Exception as e:
            wait = 5 * attempt + random.uniform(0, 5)
            log.debug(f"Segment {start//1_000_000:.0f}-{end//1_000_000:.0f} MB "
                      f"attempt {attempt}: {type(e).__name__}. Retry in {wait:.0f}s")
            time.sleep(wait)
    return False


def download_file(
    url: str,
    dest_path: Path,
    session: requests.Session = None,
    min_size_mb: float = 1.0,
    segment_mb: int = 15,
    retries: int = 6,
    retry_delay: float = 5.0,
) -> bool:
    """
    Segmented download: fetches the file in fixed 15 MB byte-range segments.

    WHY SEGMENTED?
    The HiRISE PDS server resets TCP connections after ~20-40 seconds,
    regardless of chunk size or keep-alive settings (server-side policy).
    Each 15 MB segment at 500 kB/s completes in ~30s -- under the limit.
    Failed segments are retried independently. Already-written segments
    are skipped automatically on resume.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    seg_bytes = segment_mb * 1024 * 1024

    total = _get_remote_size(url)
    if total == 0:
        log.warning(f"Cannot get file size for {dest_path.name}; using streaming fallback")
        return _streaming_fallback(url, dest_path, min_size_mb, retries)

    existing = dest_path.stat().st_size if dest_path.exists() else 0
    if existing >= total:
        log.info(f"[SKIP] {dest_path.name} already complete ({existing/1_000_000:.1f} MB)")
        return True

    # Build segment list
    segments = []
    pos = 0
    while pos < total:
        segments.append((pos, min(pos + seg_bytes - 1, total - 1)))
        pos += seg_bytes

    first_needed = max(0, existing // seg_bytes)
    log.info(
        f"Downloading {dest_path.name} ({total/1_000_000:.1f} MB) "
        f"in {len(segments)} x {segment_mb} MB segments "
        f"(starting at segment {first_needed + 1}/{len(segments)})"
    )

    with tqdm(total=total, initial=existing, unit="B", unit_scale=True,
              desc=dest_path.name[:45], leave=True) as pbar:
        current = existing
        for i, (seg_start, seg_end) in enumerate(segments):
            # Fully written?
            if seg_end < existing:
                continue
            # Partially written?
            actual_start = max(seg_start, existing)
            if actual_start > seg_end:
                continue

            ok = _download_segment(url, dest_path, actual_start, seg_end, retries)
            if not ok:
                log.error(f"[FAIL] Segment {i+1}/{len(segments)} could not be downloaded.")
                log.error("Re-run the download command to resume automatically.")
                return False

            new_size = dest_path.stat().st_size
            pbar.update(new_size - current)
            current = new_size
            time.sleep(0.3)  # brief pause between segments

    mb = file_size_mb(dest_path)
    if mb < min_size_mb:
        log.warning(f"Final file small ({mb:.1f} MB): {dest_path.name}")
        return False

    log.info(f"[OK] {dest_path.name} complete ({mb:.1f} MB)")
    return True


def _streaming_fallback(url: str, dest_path: Path,
                        min_size_mb: float, retries: int) -> bool:
    """Streaming download without segmentation (last resort)."""
    import random
    for attempt in range(1, retries + 1):
        existing = dest_path.stat().st_size if dest_path.exists() else 0
        try:
            r = requests.get(url,
                             headers={"Range": f"bytes={existing}-",
                                      "Accept-Encoding": "identity"},
                             stream=True, timeout=(20, 120))
            if r.status_code == 416:
                return True
            r.raise_for_status()
            with open(dest_path, "ab") as f:
                for chunk in r.iter_content(256 * 1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            if file_size_mb(dest_path) >= min_size_mb:
                return True
        except Exception as e:
            mb = dest_path.stat().st_size / 1_000_000 if dest_path.exists() else 0
            log.warning(f"Fallback attempt {attempt}: {e} (at {mb:.1f} MB)")
            time.sleep(15 * attempt + random.uniform(0, 5))
    return False


# ---------------------------------------------------------------------------
# Observation-level download
# ---------------------------------------------------------------------------

def download_observation(
    obs_id: str,
    output_dir: Path,
    session: requests.Session,
    checkpoint: StepCheckpoint,
    force: bool = False,
) -> Dict[str, Path]:
    """
    Download HiRISE JP2 products for one observation ID.
    Returns dict of {product_key: local_path}.
    """
    ck_key = obs_id
    if not force and checkpoint.is_done("hirise_download", ck_key):
        log.info(f"[SKIP] {obs_id} already downloaded")
        obs_dir = output_dir / obs_id
        result = {}
        for suffix, key in [("RED.JP2", "RED"), ("COLOR.JP2", "COLOR"),
                             ("MRGB.JP2", "MRGB"), ("MIRB.JP2", "MIRB")]:
            p = obs_dir / f"{obs_id}_{suffix}"
            if p.exists():
                result[key] = p
        if result:
            return result

    log.info(f"Downloading HiRISE observation: {obs_id}")
    obs_dir = output_dir / obs_id
    obs_dir.mkdir(parents=True, exist_ok=True)

    urls = find_hirise_download_urls(obs_id, session)
    if not urls:
        checkpoint.mark_failed("hirise_download", ck_key, "No files found")
        return {}

    result = {}
    for url in urls:
        fname = url.split("/")[-1]
        dest = obs_dir / fname
        ok = download_file(url, dest, session)
        if ok:
            for key in ["MRGB", "MIRB", "COLOR", "RED"]:
                if f"_{key}" in fname or f"_{key}." in fname:
                    result.setdefault(key, dest)
                    break

    if result:
        checkpoint.mark_done(
            "hirise_download", ck_key,
            {"files": [str(v) for v in result.values()]}
        )
    return result


# ---------------------------------------------------------------------------
# Batch download (all sites from config)
# ---------------------------------------------------------------------------

def download_hirise_all(
    cfg: dict,
    output_dir: Path,
    checkpoint: StepCheckpoint,
    force: bool = False,
) -> Dict[str, Dict]:
    """Download all configured HiRISE observations for all study sites."""
    if not cfg["data_sources"]["hirise"]["enabled"]:
        log.info("HiRISE disabled in config -- skipping")
        return {}

    output_dir = Path(output_dir)
    session = requests.Session()
    session.headers["User-Agent"] = "MarsGullyDigitalTwin/1.0 (academic research)"

    known = cfg["data_sources"]["hirise"].get("known_images", {})
    results = {}

    for site, obs_ids in known.items():
        log.info(f"=== HiRISE downloads for site: {site} ===")
        results[site] = {}
        for obs_id in obs_ids:
            paths = download_observation(
                obs_id, output_dir, session, checkpoint, force=force
            )
            results[site][obs_id] = paths

    return results


# ---------------------------------------------------------------------------
# Compatibility shims
# ---------------------------------------------------------------------------

def obs_id_to_url(obs_id: str) -> str:
    """Return a download directory URL for an observation (compat shim)."""
    phase, orbit = _parse_obs(obs_id)
    block = _orbit_block(orbit)
    return f"{PDS_DOWNLOAD_RDR}{phase}/{block}/{obs_id}/"


def download_hirise_file(url: str, dest: Path,
                         session: requests.Session = None) -> bool:
    """Download a single file (compat shim for tests)."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return True
    s = session or requests.Session()
    return download_file(url, dest, s)


if __name__ == "__main__":
    import yaml
    from scripts.utils import setup_logging
    setup_logging("INFO", "mars_pipeline.log")
    cfg_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ck = StepCheckpoint("data/models")
    results = download_hirise_all(cfg, Path("data/raw/hirise"), ck)
    print(f"Downloaded {sum(len(v) for v in results.values())} observations")