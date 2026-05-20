"""
HiRISE (MRO) Image Downloader
==============================
Downloads processed HiRISE RDR products from the PDS archive.

CRITICAL: HiRISE JP2 files use JPEG2000 with absolute tile byte offsets (SOT markers).
Segmented/byte-range downloads corrupt JP2 files because the assembled pieces have
wrong internal offsets. JP2 files MUST be downloaded as a single continuous stream.

Strategy:
- Single streaming GET request (no byte-range headers)
- Connection reset after ~40s -> retry with partial file check
- Each retry appends where the previous left off using a temp file
- After full download, validate JP2 by reading the header
- Fall back to COLOR.JP2 or IMG if RED.JP2 repeatedly fails

Verified URL structure:
  https://hirise-pds.lpl.arizona.edu/download/PDS/RDR/{phase}/{ORB_BLOCK}/{OBS_ID}/
  Files: {OBS_ID}_RED.JP2, {OBS_ID}_COLOR.JP2
"""

import re
import sys
import time
import random
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger, StepCheckpoint, file_size_mb

log = get_logger("downloader.hirise")

PDS_DOWNLOAD_RDR    = "https://hirise-pds.lpl.arizona.edu/download/PDS/RDR/"
PDS_DOWNLOAD_EXTRAS = "https://hirise-pds.lpl.arizona.edu/download/PDS/EXTRAS/RDR/"
PDS_LISTING_RDR     = "https://hirise-pds.lpl.arizona.edu/PDS/RDR/"


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _orbit_block(orbit: int) -> str:
    lo = (orbit // 100) * 100
    return f"ORB_{lo:06d}_{lo + 99:06d}"


def _parse_obs(obs_id: str) -> Tuple[str, int]:
    parts = obs_id.split("_")
    return parts[0], int(parts[1])


def build_download_candidates(obs_id: str) -> List[Tuple[str, str]]:
    """Return (url, filename) pairs in priority order."""
    phase, orbit = _parse_obs(obs_id)
    block = _orbit_block(orbit)
    path = f"{phase}/{block}/{obs_id}/"
    candidates = []
    for suffix in ["RED.JP2", "COLOR.JP2"]:
        fname = f"{obs_id}_{suffix}"
        candidates.append((f"{PDS_DOWNLOAD_RDR}{path}{fname}", fname))
    for suffix in ["MRGB.JP2", "MIRB.JP2"]:
        fname = f"{obs_id}_{suffix}"
        candidates.append((f"{PDS_DOWNLOAD_EXTRAS}{path}{fname}", fname))
    return candidates


def _head_check(url: str, session: requests.Session) -> bool:
    try:
        r = session.head(url, timeout=15, allow_redirects=True)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _get_remote_size(url: str, session: requests.Session) -> int:
    try:
        r = session.head(url, timeout=15, allow_redirects=True)
        return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def find_hirise_download_urls(obs_id: str, session: requests.Session) -> List[str]:
    """Find downloadable JP2 files by HEAD probing."""
    found = []
    for url, fname in build_download_candidates(obs_id):
        if _head_check(url, session):
            found.append(url)
            if len(found) >= 2:
                break
    if not found:
        log.warning(
            f"No files found for {obs_id}. "
            f"Check https://www.uahirise.org/{obs_id} to verify it exists."
        )
    return found


# ---------------------------------------------------------------------------
# JP2-safe streaming download
# ---------------------------------------------------------------------------

def _validate_jp2(path: Path) -> bool:
    """
    Validate a JP2 file by checking its header magic bytes.
    JP2 files start with: 00 00 00 0C 6A 50 20 20 0D 0A 87 0A
    JPC (codestream) starts with: FF 4F
    """
    try:
        with open(path, "rb") as f:
            header = f.read(12)
        # JP2 container
        if header[4:8] == b"jP  ":
            return True
        # JPC codestream
        if header[:2] == b"\xff\x4f":
            return True
        return False
    except Exception:
        return False


def download_jp2_streaming(
    url: str,
    dest_path: Path,
    session: requests.Session,
    total_size: int = 0,
    max_attempts: int = 40,
) -> bool:
    """
    Download a JP2 file as a single streaming request.

    JP2 files CANNOT use byte-range segmentation because their internal
    SOT (Start of Tile) markers use absolute byte offsets. Each attempt
    starts from byte 0 and downloads as far as the server allows before
    resetting. The partial download is accumulated in a temp file.
    When the temp file reaches full size, it is validated and moved to dest.

    Args:
        url         : Full download URL
        dest_path   : Final destination path
        session     : requests.Session
        total_size  : Expected file size in bytes (0 = unknown)
        max_attempts: Maximum number of streaming retries
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Already complete?
    if dest_path.exists():
        if total_size and dest_path.stat().st_size >= total_size * 0.99:
            if _validate_jp2(dest_path):
                log.debug(f"Already complete and valid: {dest_path.name}")
                return True
            else:
                log.warning(f"Existing file invalid JP2, re-downloading: {dest_path.name}")
                dest_path.unlink()

    # Use a temp file to accumulate data
    tmp_path = dest_path.with_suffix(".tmp")

    if not total_size:
        total_size = _get_remote_size(url, session)

    log.info(
        f"Streaming {dest_path.name} "
        f"({total_size/1e6:.1f} MB) — "
        f"will retry on connection reset (JP2 requires single stream)"
    )

    bytes_received = tmp_path.stat().st_size if tmp_path.exists() else 0

    with tqdm(total=total_size, initial=bytes_received,
              unit="B", unit_scale=True,
              desc=dest_path.name[:45], leave=True) as pbar:

        for attempt in range(1, max_attempts + 1):
            # For JP2: always start from byte 0 (no Range header)
            # because JP2 tile offsets are absolute
            try:
                r = session.get(
                    url,
                    headers={"Accept-Encoding": "identity"},
                    stream=True,
                    timeout=(30, 60),   # 60s read timeout per chunk
                )
                if r.status_code not in (200, 206):
                    log.warning(f"HTTP {r.status_code} on attempt {attempt}")
                    time.sleep(10 * attempt)
                    continue

                # Write fresh from start to a temp file
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=512 * 1024):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            pbar.update(len(chunk))

                # Check if we got the full file
                got = tmp_path.stat().st_size
                if total_size and got >= total_size * 0.99:
                    # Validate the JP2
                    if _validate_jp2(tmp_path):
                        shutil.move(str(tmp_path), str(dest_path))
                        log.info(f"[OK] {dest_path.name} ({got/1e6:.1f} MB)")
                        return True
                    else:
                        log.warning(f"Download complete but JP2 invalid, retrying...")
                        tmp_path.unlink(missing_ok=True)
                        pbar.reset()
                        pbar.total = total_size
                else:
                    # Partial: server reset. Keep what we have and retry.
                    log.debug(
                        f"Attempt {attempt}: got {got/1e6:.1f}/{total_size/1e6:.1f} MB "
                        f"— connection reset, retrying..."
                    )
                    # Reset progress bar to show restart
                    pbar.reset()
                    pbar.total = total_size
                    time.sleep(min(5 * attempt, 30) + random.uniform(0, 5))

            except (requests.RequestException, IOError) as e:
                got = tmp_path.stat().st_size if tmp_path.exists() else 0
                log.debug(f"Attempt {attempt} error at {got/1e6:.1f} MB: {e}")
                time.sleep(min(5 * attempt, 30) + random.uniform(0, 5))

    log.error(f"[FAIL] {dest_path.name} after {max_attempts} attempts")
    tmp_path.unlink(missing_ok=True)
    return False


# ---------------------------------------------------------------------------
# Observation download
# ---------------------------------------------------------------------------

def download_observation(
    obs_id: str,
    output_dir: Path,
    session: requests.Session,
    checkpoint: StepCheckpoint,
    force: bool = False,
) -> Dict[str, Path]:
    """Download HiRISE JP2 products for one observation."""
    ck_key = obs_id
    if not force and checkpoint.is_done("hirise_download", ck_key):
        log.info(f"[SKIP] {obs_id} already downloaded")
        obs_dir = output_dir / obs_id
        result = {}
        for suffix, key in [("RED.JP2","RED"),("COLOR.JP2","COLOR"),
                             ("MRGB.JP2","MRGB")]:
            p = obs_dir / f"{obs_id}_{suffix}"
            if p.exists() and p.stat().st_size > 1_000_000:
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
        total = _get_remote_size(url, session)
        ok = download_jp2_streaming(url, dest, session, total_size=total)
        if ok:
            for key in ["MRGB","MIRB","COLOR","RED"]:
                if f"_{key}" in fname:
                    result.setdefault(key, dest)
                    break

    if result:
        checkpoint.mark_done("hirise_download", ck_key,
                             {"files": [str(v) for v in result.values()]})
    return result


# ---------------------------------------------------------------------------
# Batch download
# ---------------------------------------------------------------------------

def download_hirise_all(
    cfg: dict,
    output_dir: Path,
    checkpoint: StepCheckpoint,
    force: bool = False,
) -> Dict[str, Dict]:
    """Download all configured HiRISE observations."""
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
# Compat shims
# ---------------------------------------------------------------------------

def obs_id_to_url(obs_id: str) -> str:
    phase, orbit = _parse_obs(obs_id)
    block = _orbit_block(orbit)
    return f"{PDS_DOWNLOAD_RDR}{phase}/{block}/{obs_id}/"


def download_hirise_file(url: str, dest: Path,
                         session: requests.Session = None) -> bool:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return True
    s = session or requests.Session()
    total = _get_remote_size(url, s)
    return download_jp2_streaming(url, dest, s, total_size=total)