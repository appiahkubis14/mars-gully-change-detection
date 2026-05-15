"""
time_series_plot.py
===================
Generate Plotly time-series plots of gully activity for the dashboard.

Outputs
-------
  data/outputs/dashboard/
    ├── time_series.html        ← main embedded widget
    ├── area_time_series.html   ← full-page area plot
    └── change_bar_chart.html   ← gain/loss bar chart

Usage
-----
    python main.py --step dashboard   (called from folium_map.py)
    from scripts.dashboard.time_series_plot import generate_all_plots
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    log.warning("plotly not installed. Time-series plots will be skipped.")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_kalman(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_change_summary(path: Path) -> Optional[list]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_activity_tracks(path: Path) -> Optional[list]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def build_area_time_series(
    kalman_data: dict,
    change_data: Optional[list] = None,
) -> "go.Figure":
    """
    Line plot: raw area vs Kalman-smoothed area over time.
    Adds shaded uncertainty band (±1 std from measurement noise ≈0.1 ha).
    """
    times = kalman_data["times"]
    raw = kalman_data["raw_areas"]
    smoothed = kalman_data["smoothed_areas"]
    velocities = kalman_data.get("velocities", [])

    sigma = 0.1  # approximate uncertainty (ha)

    fig = sp.make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Gully Area Over Time", "Expansion Velocity (ha/yr)"),
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35],
    )

    # ---- raw area ----
    fig.add_trace(
        go.Scatter(
            x=times, y=raw,
            mode="markers",
            name="Observed Area",
            marker=dict(color="#e74c3c", size=8, symbol="circle"),
        ),
        row=1, col=1,
    )

    # ---- smoothed + band ----
    upper = [s + sigma for s in smoothed]
    lower = [max(s - sigma, 0) for s in smoothed]

    fig.add_trace(
        go.Scatter(
            x=times + times[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(52,152,219,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±1σ band",
            showlegend=True,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=times, y=smoothed,
            mode="lines+markers",
            name="Kalman Smoothed",
            line=dict(color="#2980b9", width=2),
            marker=dict(color="#2980b9", size=6),
        ),
        row=1, col=1,
    )

    # ---- velocity bar ----
    if velocities:
        vel_colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in velocities]
        fig.add_trace(
            go.Bar(
                x=times, y=velocities,
                name="Expansion Rate",
                marker_color=vel_colors,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="grey", row=2, col=1)

    fig.update_layout(
        title="Mars Gully Area Time Series",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.05),
        height=500,
        margin=dict(l=50, r=20, t=50, b=20),
    )
    fig.update_yaxes(title_text="Area (ha)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (ha/yr)", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)

    return fig


def build_change_bar_chart(change_data: list) -> "go.Figure":
    """
    Stacked bar chart of gain / loss per date pair.
    """
    date_labels = [
        f"{p['date_t0']}→{p['date_t1']}" for p in change_data
    ]
    gains = [p.get("gain_ha", 0) for p in change_data]
    losses = [-p.get("loss_ha", 0) for p in change_data]   # negative for below-axis
    net = [p.get("net_change_ha", 0) for p in change_data]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=date_labels, y=gains,
        name="New Gully Area (gain)",
        marker_color="#27ae60",
    ))
    fig.add_trace(go.Bar(
        x=date_labels, y=losses,
        name="Lost Gully Area",
        marker_color="#e74c3c",
    ))
    fig.add_trace(go.Scatter(
        x=date_labels, y=net,
        name="Net Change",
        mode="lines+markers",
        line=dict(color="#2c3e50", width=2, dash="dot"),
    ))
    fig.add_hline(y=0, line_color="grey", line_dash="dash")

    fig.update_layout(
        barmode="overlay",
        title="Gully Area Change per Date Pair",
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=50, r=20, t=50, b=60),
    )
    fig.update_yaxes(title_text="Area Change (ha)")
    fig.update_xaxes(title_text="Date Interval")

    return fig


def build_per_gully_plot(tracks: list, top_n: int = 10) -> "go.Figure":
    """
    Multi-line plot of the top N largest gully instances over time.
    """
    # sort by total gain, take top_n
    sorted_tracks = sorted(
        tracks, key=lambda t: t.get("total_gain_ha", 0), reverse=True
    )[:top_n]

    fig = go.Figure()
    colors = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
    ]
    for i, track in enumerate(sorted_tracks):
        times = track.get("dates", [])
        smoothed = track.get("smoothed_areas_ha", track.get("areas_ha", []))
        gid = track.get("global_id", i)
        c = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=times, y=smoothed,
            mode="lines+markers",
            name=f"Gully #{gid}",
            line=dict(color=c, width=1.5),
            marker=dict(color=c, size=5),
        ))

    fig.update_layout(
        title=f"Top {len(sorted_tracks)} Gully Instances (Kalman-smoothed area)",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=50, r=20, t=50, b=60),
    )
    fig.update_yaxes(title_text="Area (ha)")
    fig.update_xaxes(title_text="Year")
    return fig


# ---------------------------------------------------------------------------
# Compact embedded widget
# ---------------------------------------------------------------------------

def build_compact_widget(
    kalman_data: Optional[dict],
    change_data: Optional[list],
) -> "go.Figure":
    """
    A compact 2-subplot figure suitable for embedding in the Folium iframe.
    """
    if kalman_data is None and change_data is None:
        fig = go.Figure()
        fig.update_layout(title="No data available yet", height=250)
        return fig

    n_cols = (1 if kalman_data is None else 1) + (1 if change_data else 0)
    n_cols = max(n_cols, 1)

    if kalman_data and change_data:
        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Area", "Change"))
    else:
        fig = go.Figure()

    if kalman_data:
        times = kalman_data["times"]
        smoothed = kalman_data["smoothed_areas"]
        raw = kalman_data["raw_areas"]
        kwargs = dict(row=1, col=1) if change_data else {}
        fig.add_trace(go.Scatter(x=times, y=raw, mode="markers", name="Observed",
                                 marker=dict(color="#e74c3c", size=6)), **kwargs)
        fig.add_trace(go.Scatter(x=times, y=smoothed, mode="lines", name="Smoothed",
                                 line=dict(color="#2980b9", width=2)), **kwargs)

    if change_data:
        date_labels = [f"{p['date_t0']}→{p['date_t1']}" for p in change_data]
        gains = [p.get("gain_ha", 0) for p in change_data]
        kwargs2 = dict(row=1, col=2) if kalman_data else {}
        fig.add_trace(go.Bar(x=date_labels, y=gains, name="Gain (ha)",
                             marker_color="#27ae60"), **kwargs2)

    fig.update_layout(
        template="plotly_white",
        height=260,
        showlegend=False,
        margin=dict(l=30, r=10, t=30, b=40),
        title="Gully Time Series",
    )
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_plots(cfg_path: str = "config.yaml") -> None:
    """Generate all time-series plots and save to the dashboard directory."""
    if not HAS_PLOTLY:
        log.error("plotly not installed; skipping time-series plots.")
        return

    out_dir = Path("data/outputs/dashboard")
    out_dir.mkdir(parents=True, exist_ok=True)

    change_dir = Path("data/outputs/change_detection")
    kalman_data = _load_kalman(change_dir / "kalman_smoothed.json")
    change_data = _load_change_summary(change_dir / "change_summary.json")
    tracks = _load_activity_tracks(change_dir / "activity_tracks.json")

    # ---- compact embedded widget ----
    widget = build_compact_widget(kalman_data, change_data)
    widget.write_html(str(out_dir / "time_series.html"), include_plotlyjs="cdn", full_html=True)
    log.info(f"Embedded time-series widget → {out_dir / 'time_series.html'}")

    # ---- full area time series ----
    if kalman_data:
        fig_area = build_area_time_series(kalman_data, change_data)
        fig_area.write_html(
            str(out_dir / "area_time_series.html"),
            include_plotlyjs="cdn", full_html=True
        )
        log.info(f"Full area time series → {out_dir / 'area_time_series.html'}")

    # ---- change bar chart ----
    if change_data:
        fig_bar = build_change_bar_chart(change_data)
        fig_bar.write_html(
            str(out_dir / "change_bar_chart.html"),
            include_plotlyjs="cdn", full_html=True
        )
        log.info(f"Change bar chart → {out_dir / 'change_bar_chart.html'}")

    # ---- per-gully ----
    if tracks:
        fig_gully = build_per_gully_plot(tracks)
        fig_gully.write_html(
            str(out_dir / "per_gully_tracks.html"),
            include_plotlyjs="cdn", full_html=True
        )
        log.info(f"Per-gully tracks → {out_dir / 'per_gully_tracks.html'}")
