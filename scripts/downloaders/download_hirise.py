"""
HiRISE (MRO) Image Downloader
Fetches .JP2 images from NASA PDS via the HiRISE PDS archive.

HiRISE PDS URL structure:
  https://hirise-pds.lpl.arizona.edu/PDS/EDR/{ORBIT_PREFIX}/{OBSERVATION_ID}/
  e.g. https://hirise-pds.lpl.arizona.edu/PDS/EDR/ESP/ORB_012800_012899/ESP_012821_1440/

Each observation has:
  - {OBS_ID}_RED.JP2       (grayscale panchromatic, 0.25m)
  - {OBS_ID}_RED.LBL       (PDS label)
  - {OBS_ID}_COLOR.JP2     (color, 0.5m - not always available)
"""

import os
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
from scripts.utils import (
    get_logger, ensure_dir, sentinel_exists, write_sentinel,
    file_size_mb, StepCheckpoint
)

log = get_logger("downloader.hirise")

PDS_BASE = "https://hirise-pds.lpl.arizona.edu/PDS/EDR/"
CHUNK_SIZE = 1024 * 1024  # 1 MB


def obs_id_to_url(obs_id: str) -> str:
    """
    Convert HiRISE observation ID to PDS directory URL.

    Example:
      ESP_012821_1440 →
      https://hirise-pds.lpl.arizona.edu/PDS/EDR/ESP/ORB_012800_012899/ESP_012821_1440/
    """
    parts = obs_id.split("_")
    phase = parts[0]                      # ESP, PSP, TRA, etc.
    orbit = int(parts[1])
    # orbit block: round down to nearest 100
    block_low = (orbit // 100) * 100
    block_high = block_low + 99
    orbit_dir = f"ORB_{block_low:06d}_{block_high:06d}"
    return f"{PDS_BASE}{phase}/{orbit_dir}/{obs_id}/"


def list_observation_files(obs_id: str, session: requests.Session) -> List[str]:
    """
    Scrape the PDS directory listing for an observation ID.
    Returns list of file URLs (.JP2, .LBL).
    """
    url = obs_id_to_url(obs_id)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Cannot list {url}: {e}")
        return []

    # Parse href links from HTML listing
    pattern = re.compile(r'href="([^"]+\.(JP2|LBL|IMG|TIF))"', re.IGNORECASE)
    files = [urljoin(url, m.group(1)) for m in pattern.finditer(resp.text)]
    return files


def download_file(
    url: str,
    dest_path: Path,
    session: requests.Session,
    min_size_mb: float = 0.1,
    retries: int = 3,
    retry_delay: float = 5.0
) -> bool:
    """
    Download a single file with resume support and progress bar.
    Returns True on success.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: check existing file size
    existing_bytes = dest_path.stat().st_size if dest_path.exists() else 0

    for attempt in range(1, retries + 1):
        try:
            headers = {}
            if existing_bytes > 0:
                headers["Range"] = f"bytes={existing_bytes}-"

            resp = session.get(url, headers=headers, stream=True, timeout=60)

            # 416 = range not satisfiable → file already complete
            if resp.status_code == 416:
                log.debug(f"File already complete: {dest_path.name}")
                return True

            resp.raise_for_status()

            total = int(resp.headers.get("Content-Length", 0))
            mode = "ab" if existing_bytes > 0 else "wb"

            with open(dest_path, mode) as f, tqdm(
                total=total,
                initial=existing_bytes,
                unit="B",
                unit_scale=True,
                desc=dest_path.name,
                leave=False
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Verify minimum size
            actual_mb = file_size_mb(dest_path)
            if actual_mb < min_size_mb:
                log.warning(
                    f"File suspiciously small ({actual_mb:.2f} MB): {dest_path.name}"
                )
                return False

            log.info(f"✓ Downloaded {dest_path.name} ({actual_mb:.1f} MB)")
            return True

        except requests.RequestException as e:
            log.warning(f"Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                time.sleep(retry_delay * attempt)
            else:
                log.error(f"✗ All retries exhausted for {url}")
                return False

    return False


def download_observation(
    obs_id: str,
    output_dir: Path,
    session: requests.Session,
    checkpoint: StepCheckpoint,
    bands: List[str] = None,
    force: bool = False
) -> Dict[str, Path]:
    """
    Download all files for a HiRISE observation ID.

    Parameters
    ----------
    obs_id : str
        e.g. "ESP_012821_1440"
    output_dir : Path
        Root directory for HiRISE data (data/raw/hirise/)
    session : requests.Session
    checkpoint : StepCheckpoint
    bands : list of str
        Which band suffixes to download: ["RED", "COLOR", "BG", "IR", "UV"]
        Default: ["RED", "COLOR"]
    force : bool
        Re-download even if checkpoint says done.

    Returns
    -------
    dict mapping band → local Path
    """
    if bands is None:
        bands = ["RED", "COLOR"]

    ck_key = obs_id
    if not force and checkpoint.is_done("hirise_download", ck_key):
        log.info(f"[SKIP] {obs_id} already downloaded (checkpoint)")
        # Return existing paths
        result = {}
        for band in bands:
            p = output_dir / obs_id / f"{obs_id}_{band}.JP2"
            if p.exists():
                result[band] = p
        return result

    obs_dir = output_dir / obs_id
    obs_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Downloading HiRISE observation: {obs_id}")
    files = list_observation_files(obs_id, session)

    if not files:
        log.error(f"No files found for {obs_id}. Check PDS availability.")
        checkpoint.mark_failed("hirise_download", ck_key, "No files listed")
        return {}

    result = {}
    for file_url in files:
        filename = file_url.split("/")[-1]

        # Filter by requested bands
        if not any(f"_{band}." in filename for band in bands):
            continue

        dest = obs_dir / filename
        ok = download_file(file_url, dest, session)

        if ok and filename.endswith(".JP2"):
            # Determine band name
            for band in bands:
                if f"_{band}.JP2" in filename:
                    result[band] = dest
                    break

    if result:
        checkpoint.mark_done("hirise_download", ck_key, {"files": list(str(v) for v in result.values())})

    return result


def download_hirise_all(
    cfg: Dict,
    output_dir: Path,
    checkpoint: StepCheckpoint,
    force: bool = False
) -> Dict[str, Dict[str, Path]]:
    """
    Download all configured HiRISE observations for all study sites.

    Returns nested dict: {site_name: {obs_id: {band: path}}}
    """
    if not cfg["data_sources"]["hirise"]["enabled"]:
        log.info("HiRISE disabled in config — skipping")
        return {}

    output_dir = Path(output_dir)
    session = requests.Session()
    session.headers.update({"User-Agent": "MarsGullyDigitalTwin/1.0"})

    known = cfg["data_sources"]["hirise"].get("known_images", {})
    results = {}

    for site, obs_ids in known.items():
        log.info(f"=== HiRISE downloads for site: {site} ===")
        results[site] = {}
        for obs_id in obs_ids:
            paths = download_observation(
                obs_id,
                output_dir,
                session,
                checkpoint,
                bands=["RED", "COLOR"],
                force=force
            )
            results[site][obs_id] = paths

    return results


def construct_label_url(obs_id: str) -> str:
    """Return URL for the PDS .LBL metadata file."""
    base = obs_id_to_url(obs_id)
    return f"{base}{obs_id}_RED.LBL"


def fetch_observation_metadata(obs_id: str, session: requests.Session) -> Dict:
    """Fetch and parse the PDS label for an observation."""
    url = construct_label_url(obs_id)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"Cannot fetch label {url}: {e}")
        return {}

    metadata = {}
    for line in resp.text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            metadata[k.strip()] = v.strip().strip('"')

    return metadata


if __name__ == "__main__":
    import sys
    import yaml

    cfg_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path("data/raw/hirise")
    ck = StepCheckpoint("data/models")

    from scripts.utils import setup_logging
    setup_logging("INFO", "mars_pipeline.log")

    results = download_hirise_all(cfg, out_dir, ck)
    print(f"Downloaded {sum(len(v) for v in results.values())} observations")
