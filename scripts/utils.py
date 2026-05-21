import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*PROJ.*")
_warnings.filterwarnings("ignore", message=".*geotransform.*")
_warnings.filterwarnings("ignore", message=".*NotGeoreferencedWarning.*")
_warnings.filterwarnings("ignore", category=UserWarning)

"""
Shared utilities: logging, checkpointing, CRS handling, file I/O
"""
import os
import sys
import json
import time
import logging
import hashlib
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml


# ─── Logging ──────────────────────────────────────────────────────────────────


def _fix_proj_env() -> None:
    """
    Isolate rasterio from conflicting PROJ installations (e.g. PostgreSQL).
    Must be called before any rasterio/pyproj import.
    Sets PROJ_DATA and PROJ_LIB to rasterio's bundled proj database.
    """
    import os
    try:
        import importlib.util
        spec = importlib.util.find_spec("rasterio")
        if spec and spec.origin:
            rasterio_dir = Path(spec.origin).parent
            proj_dir = rasterio_dir / "proj"
            if not proj_dir.exists():
                # Try pyproj bundle
                spec2 = importlib.util.find_spec("pyproj")
                if spec2 and spec2.origin:
                    proj_dir = Path(spec2.origin).parent / "proj_dir" / "share" / "proj"
            if proj_dir.exists():
                os.environ["PROJ_DATA"] = str(proj_dir)
                os.environ["PROJ_LIB"]  = str(proj_dir)
    except Exception:
        pass


# Fix PROJ at module import time (before any rasterio usage)
_fix_proj_env()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    name: str = "mars_gully"
) -> logging.Logger:
    """Configure root logger with console + optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console — force UTF-8 on Windows to prevent cp1252 UnicodeEncodeError.
    import io
    if sys.platform == "win32":
        try:
            # Wrap stdout in a UTF-8 writer that replaces unencodable chars
            utf8_stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        except AttributeError:
            utf8_stdout = sys.stdout   # fallback for IDEs / redirected streams
    else:
        utf8_stdout = sys.stdout
    ch = logging.StreamHandler(utf8_stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the mars_gully namespace."""
    return logging.getLogger(f"mars_gully.{name}")


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(config_path: Union[str, Path]) -> Dict:
    """Load YAML config, resolving env var overrides."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config(cfg: Dict, path: Union[str, Path]) -> None:
    """Save config dict to YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


# ─── Checkpointing ────────────────────────────────────────────────────────────

class StepCheckpoint:
    """
    Lightweight file-based checkpoint system.
    Records which pipeline steps have completed successfully.
    """

    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.dir / "checkpoint_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {}

    def _save_registry(self) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)

    def is_done(self, step: str, key: str = "") -> bool:
        """Check if a step (with optional key) is already complete."""
        entry = self.registry.get(f"{step}:{key}")
        return entry is not None and entry.get("status") == "done"

    def mark_done(self, step: str, key: str = "", metadata: Optional[Dict] = None) -> None:
        """Mark a step as successfully completed."""
        self.registry[f"{step}:{key}"] = {
            "status": "done",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        self._save_registry()

    def mark_failed(self, step: str, key: str = "", error: str = "") -> None:
        self.registry[f"{step}:{key}"] = {
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "error": error
        }
        self._save_registry()

    def clear(self, step: Optional[str] = None) -> None:
        """Clear checkpoints for a step or all steps."""
        if step is None:
            self.registry = {}
        else:
            keys = [k for k in self.registry if k.startswith(f"{step}:")]
            for k in keys:
                del self.registry[k]
        self._save_registry()


# ─── File utilities ───────────────────────────────────────────────────────────

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory and all parents; return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_md5(path: Union[str, Path], chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size_mb(path: Union[str, Path]) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def safe_copy(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """Copy file only if it does not already exist at destination."""
    dst = Path(dst)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# ─── CRS / Spatial helpers ────────────────────────────────────────────────────

def mars_equirectangular_wkt() -> str:
    """
    Return WKT for Mars equirectangular projection (IAU 49900 / ESRI 104971).
    Mars radius: 3396190 m (equatorial, IAU 2000).
    """
    return (
        'PROJCS["Mars_Equirectangular",'
        'GEOGCS["GCS_Mars_2000_Sphere",'
        'DATUM["D_Mars_2000_Sphere",'
        'SPHEROID["Mars_2000_Sphere_IAU_IAG",3396190.0,0.0]],'
        'PRIMEM["Reference_Meridian",0.0],'
        'UNIT["Degree",0.0174532925199433]],'
        'PROJECTION["Equidistant_Cylindrical"],'
        'PARAMETER["False_Easting",0.0],'
        'PARAMETER["False_Northing",0.0],'
        'PARAMETER["Central_Meridian",0.0],'
        'PARAMETER["Standard_Parallel_1",0.0],'
        'UNIT["Meter",1.0]]'
    )


def lonlat_to_meters(lon: float, lat: float) -> Tuple[float, float]:
    """Convert Mars lon/lat (degrees) to equirectangular meters."""
    R = 3_396_190.0
    x = np.radians(lon) * R
    y = np.radians(lat) * R
    return x, y


def bounds_to_meters(
    bounds: List[float]
) -> Tuple[float, float, float, float]:
    """Convert [min_lon, min_lat, max_lon, max_lat] to meter bounds."""
    x_min, y_min = lonlat_to_meters(bounds[0], bounds[1])
    x_max, y_max = lonlat_to_meters(bounds[2], bounds[3])
    return x_min, y_min, x_max, y_max


# ─── Array helpers ────────────────────────────────────────────────────────────

def percentile_clip(
    arr: np.ndarray,
    low: float = 2.0,
    high: float = 98.0,
    axis: Optional[Tuple] = None
) -> np.ndarray:
    """Clip array between low/high percentiles; return float32."""
    arr = arr.astype(np.float32)
    lo = np.nanpercentile(arr, low, axis=axis, keepdims=True)
    hi = np.nanpercentile(arr, high, axis=axis, keepdims=True)
    return np.clip(arr, lo, hi)


def minmax_normalize(
    arr: np.ndarray,
    low: float = 2.0,
    high: float = 98.0
) -> np.ndarray:
    """Percentile-clip then rescale to [0, 1]."""
    arr = percentile_clip(arr, low, high)
    lo = arr.min()
    hi = arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def pad_to_multiple(
    arr: np.ndarray,
    multiple: int = 32,
    mode: str = "reflect"
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pad H, W dimensions to be divisible by `multiple`."""
    h, w = arr.shape[-2], arr.shape[-1]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    pad_h = (0, ph)
    pad_w = (0, pw)
    if arr.ndim == 2:
        padded = np.pad(arr, (pad_h, pad_w), mode=mode)
    elif arr.ndim == 3:
        padded = np.pad(arr, ((0, 0), pad_h, pad_w), mode=mode)
    else:
        padded = np.pad(arr, ((0, 0), (0, 0), pad_h, pad_w), mode=mode)
    return padded, (0, ph, 0, pw)


# ─── Timing ───────────────────────────────────────────────────────────────────

class Timer:
    """Context manager and explicit start/stop timer."""

    def __init__(self, name: str = ""):
        self.name = name
        self._start = None
        self.elapsed = 0.0

    def start(self):
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        logger = get_logger("timer")
        label = f"[{self.name}] " if self.name else ""
        logger.debug(f"{label}Elapsed: {self.elapsed:.2f}s")


# ─── Progress helpers ─────────────────────────────────────────────────────────

def human_size(n_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


# ─── Sentinel files ───────────────────────────────────────────────────────────

def write_sentinel(path: Union[str, Path], metadata: Optional[Dict] = None) -> None:
    """Write a small JSON sentinel file indicating step completion."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"done": True, "timestamp": datetime.utcnow().isoformat()}
    if metadata:
        payload.update(metadata)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def sentinel_exists(path: Union[str, Path]) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        return data.get("done", False)
    except Exception:
        return False