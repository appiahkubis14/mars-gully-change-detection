"""
kalman_smooth.py
================
Kalman filter for smoothing Mars gully area time series.

Reduces false positives from sensor artefacts and atmospheric effects
(dust hazes on Mars) by applying a constant-velocity Kalman filter to
the per-epoch gully area measurements.

Also provides batch smoothing for multiple tracked gully instances.

Usage
-----
    from scripts.change.kalman_smooth import smooth_area_series, KalmanAreaTracker

    # Smooth a single site's area over time
    dates = [2009.0, 2010.0, 2012.0, 2015.0, 2018.0]
    areas = [1.2, 1.3, 1.8, 2.1, 2.4]   # ha
    smoothed = smooth_area_series(dates, areas)

    # Or use the class-based tracker for streaming updates
    tracker = KalmanAreaTracker()
    for date, area in zip(dates, areas):
        tracker.update(date, area)
    rates = tracker.expansion_rates()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scalar Kalman filter implementation (no external filterpy required)
# ---------------------------------------------------------------------------

class ScalarKalmanFilter:
    """
    1-D constant-velocity Kalman filter for a scalar measurement (area).

    State vector: [area, velocity]  (units: ha, ha/year)

    Args:
        process_noise_area (float): Process noise for area (Q[0,0]).
        process_noise_vel  (float): Process noise for velocity (Q[1,1]).
        measurement_noise  (float): Observation noise variance (R).
        init_area          (float): Initial area estimate.
        init_velocity      (float): Initial velocity estimate.
    """

    def __init__(
        self,
        process_noise_area: float = 0.01,
        process_noise_vel: float = 0.005,
        measurement_noise: float = 0.1,
        init_area: float = 0.0,
        init_velocity: float = 0.0,
    ):
        # State: [area, velocity]
        self.x = np.array([[init_area], [init_velocity]], dtype=np.float64)

        # State covariance
        self.P = np.eye(2, dtype=np.float64) * 1.0

        # Process noise covariance Q
        self.Q = np.diag([process_noise_area, process_noise_vel])

        # Measurement noise covariance R (scalar → 1×1)
        self.R = np.array([[measurement_noise]])

        # Measurement matrix H: we observe area only
        self.H = np.array([[1.0, 0.0]])

    def predict(self, dt: float) -> None:
        """
        Predict next state given elapsed time dt (years).
        State transition: area += velocity * dt
        """
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement: float) -> float:
        """
        Update state with a new area measurement.

        Returns the corrected area estimate.
        """
        z = np.array([[measurement]])
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)    # Kalman gain
        self.x = self.x + K @ y
        I_KH = np.eye(2) - K @ self.H
        self.P = I_KH @ self.P
        return float(self.x[0, 0])

    @property
    def area(self) -> float:
        return float(self.x[0, 0])

    @property
    def velocity(self) -> float:
        return float(self.x[1, 0])


# ---------------------------------------------------------------------------
# Batch smoothing
# ---------------------------------------------------------------------------

def smooth_area_series(
    times: List[float],
    areas: List[float],
    process_noise_area: float = 0.01,
    process_noise_vel: float = 0.005,
    measurement_noise: float = 0.1,
) -> Dict[str, List[float]]:
    """
    Smooth a time series of area measurements with a Kalman filter.

    Args:
        times              : List of decimal-year timestamps.
        areas              : Corresponding area measurements in ha.
        process_noise_area : Q[0,0].
        process_noise_vel  : Q[1,1].
        measurement_noise  : R[0,0].

    Returns:
        dict with keys:
          'times'           : original timestamps
          'raw_areas'       : original area values
          'smoothed_areas'  : Kalman-smoothed estimates
          'velocities'      : instantaneous velocity (ha/year)
    """
    if len(times) != len(areas) or len(times) < 1:
        raise ValueError("times and areas must be non-empty and equal length.")

    kf = ScalarKalmanFilter(
        process_noise_area=process_noise_area,
        process_noise_vel=process_noise_vel,
        measurement_noise=measurement_noise,
        init_area=areas[0],
        init_velocity=0.0,
    )

    smoothed: List[float] = []
    velocities: List[float] = []

    for i, (t, a) in enumerate(zip(times, areas)):
        if i > 0:
            dt = max(times[i] - times[i - 1], 0.01)
            kf.predict(dt)
        s = kf.update(a)
        smoothed.append(round(s, 6))
        velocities.append(round(kf.velocity, 6))

    return {
        "times": list(times),
        "raw_areas": list(areas),
        "smoothed_areas": smoothed,
        "velocities": velocities,
    }


# ---------------------------------------------------------------------------
# Tracker class
# ---------------------------------------------------------------------------

class KalmanAreaTracker:
    """
    Streaming Kalman filter tracker for a single gully's area over time.

    Usage::

        tracker = KalmanAreaTracker()
        tracker.update(2009.0, 1.2)
        tracker.update(2010.0, 1.35)
        tracker.update(2012.0, 1.8)
        rates = tracker.expansion_rates()  # ha/year per interval
    """

    def __init__(
        self,
        process_noise_area: float = 0.01,
        process_noise_vel: float = 0.005,
        measurement_noise: float = 0.1,
    ):
        self._kf: Optional[ScalarKalmanFilter] = None
        self._times: List[float] = []
        self._raw: List[float] = []
        self._smoothed: List[float] = []
        self._velocities: List[float] = []
        self._process_noise_area = process_noise_area
        self._process_noise_vel = process_noise_vel
        self._measurement_noise = measurement_noise

    def update(self, time: float, area: float) -> float:
        """
        Ingest a new observation (time in decimal years, area in ha).
        Returns the Kalman-smoothed area estimate.
        """
        if self._kf is None:
            self._kf = ScalarKalmanFilter(
                process_noise_area=self._process_noise_area,
                process_noise_vel=self._process_noise_vel,
                measurement_noise=self._measurement_noise,
                init_area=area,
                init_velocity=0.0,
            )
            smoothed = area
        else:
            dt = max(time - self._times[-1], 0.01)
            self._kf.predict(dt)
            smoothed = self._kf.update(area)

        self._times.append(time)
        self._raw.append(area)
        self._smoothed.append(round(smoothed, 6))
        self._velocities.append(round(self._kf.velocity, 6))
        return smoothed

    def expansion_rates(self) -> List[float]:
        """Ha/year expansion rate at each observation."""
        return list(self._velocities)

    def summary(self) -> Dict:
        return {
            "times": self._times,
            "raw_areas": self._raw,
            "smoothed_areas": self._smoothed,
            "velocities": self._velocities,
            "total_gain_ha": round(
                max(self._smoothed) - self._smoothed[0], 4
            ) if self._smoothed else 0.0,
            "mean_expansion_ha_year": round(
                float(np.mean(self._velocities)), 4
            ) if self._velocities else 0.0,
        }


# ---------------------------------------------------------------------------
# High-level pipeline entry point
# ---------------------------------------------------------------------------

def run_kalman_smoothing(cfg_path: str = "config.yaml") -> None:
    """
    Read change_summary.json, compute cumulative area per site, smooth,
    and save kalman_smoothed.json.
    """
    from scripts.utils import load_config
    cfg = load_config(cfg_path)

    change_dir = Path("data/outputs/change_detection")
    summary_path = change_dir / "change_summary.json"
    if not summary_path.exists():
        log.warning(f"No change_summary.json found at {summary_path}. Run --step change first.")
        return

    with open(summary_path) as f:
        pairs = json.load(f)

    if not pairs:
        log.warning("change_summary.json is empty.")
        return

    # ---- build cumulative area series ----
    # Start from first date, accumulate net changes
    dates: List[float] = []
    areas: List[float] = []
    cumulative = 0.0

    for pair in pairs:
        if not dates:
            dates.append(float(pair["date_t0"]))
            areas.append(cumulative)
        cumulative += pair["net_change_ha"]
        dates.append(float(pair["date_t1"]))
        areas.append(max(cumulative, 0.0))   # area can't be negative

    smoothed_result = smooth_area_series(dates, areas)
    out_path = change_dir / "kalman_smoothed.json"
    with open(out_path, "w") as f:
        json.dump(smoothed_result, f, indent=2)
    log.info(f"Kalman-smoothed area series → {out_path}")

    # ---- per-pair expansion rate ----
    for i, pair in enumerate(pairs):
        if i < len(smoothed_result["velocities"]) - 1:
            pair["kalman_velocity_ha_year"] = smoothed_result["velocities"][i + 1]

    with open(summary_path, "w") as f:
        json.dump(pairs, f, indent=2)
    log.info("change_summary.json updated with Kalman velocities.")
