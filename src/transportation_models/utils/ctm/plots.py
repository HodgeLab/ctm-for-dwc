"""Matplotlib renderers for CTM corridors and simulation results.

Two groups of plotters live here:

**Corridor / cell layout** (used by :mod:`scripts.build_ctm_from_osm`
and :mod:`scripts.regenerate_cell_artifacts`):

* :func:`plot_corridor_map`  -- lat/lon overlay of the mainline + ramps.
* :func:`plot_cell_layout`   -- strip plot of cells along postmile, with
                                ramp markers + optional Caltrans secondary
                                axis.

Neither function needs the live osmnx graph: ``plot_corridor_map`` reads
its upstream/downstream end-marker coordinates from
``corridor.mainline_segments[0].geometry.coords[0]`` etc., so a
reconstructed :class:`Corridor` (from cached GeoJSON via
``cells.corridor_from_artifacts``) plots identically to the freshly
extracted one.

**Simulation result** (used by :mod:`scripts.run_ctm_ctmsim_demo` and
:mod:`scripts.simulate_ctm_corridor`):

* :func:`downsample_to_plot_period` / :func:`per_period_totals`
  reduce sim-step arrays to a coarser plotting cadence.
* :func:`plot_flow_density_contour`  -- space-time heatmap of a
  per-cell quantity (flow, density, ...).
* :func:`plot_aggregate_metrics`     -- 2x2 grid of VHT/VMT/delay/ploss
  per plotting period.
* :func:`plot_per_cell_metrics`      -- 2x2 bar chart of the same four
  metrics summed over the horizon, per cell.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from .osm import Corridor

# Distinct colors per lane count so the corridor map reads at a glance.
_LANE_COLORS = {
    2: "#1f77b4", 3: "#2ca02c", 4: "#ff7f0e",
    5: "#d62728", 6: "#9467bd", 7: "#8c564b",
}
_DEFAULT_LANE_COLOR = "#7f7f7f"


def _lane_color(lanes: int) -> str:
    return _LANE_COLORS.get(int(lanes), _DEFAULT_LANE_COLOR)


def plot_corridor_map(corridor: Corridor, out_path: Path) -> None:
    """Overlay mainline (colored by lane count) + ramp gores on a lat/lon map.

    Reads everything it needs from ``corridor`` -- no osmnx graph required,
    so this works against both freshly-extracted and reconstructed-from-
    artifacts corridors.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    # Mainline as LineCollections per lane count (single legend entry each).
    by_lanes: dict[int, list[list[tuple[float, float]]]] = {}
    for seg in corridor.mainline_segments:
        by_lanes.setdefault(seg.lanes, []).append(list(seg.geometry.coords))
    for lanes, lines in sorted(by_lanes.items()):
        lc = LineCollection(
            lines, colors=_lane_color(lanes), linewidths=2.2,
            label=f"mainline · {lanes} lanes",
        )
        ax.add_collection(lc)

    # Ramp gores -- one marker style per kind.
    on_xy = np.array(
        [(r.geometry.x, r.geometry.y) for r in corridor.ramp_junctions if r.kind == "on"]
    )
    off_xy = np.array(
        [(r.geometry.x, r.geometry.y) for r in corridor.ramp_junctions if r.kind == "off"]
    )
    if on_xy.size:
        ax.scatter(on_xy[:, 0], on_xy[:, 1], marker="^", s=60,
                   edgecolor="black", facecolor="#2ca02c", linewidths=0.6,
                   label=f"on-ramp ({len(on_xy)})", zorder=5)
    if off_xy.size:
        ax.scatter(off_xy[:, 0], off_xy[:, 1], marker="v", s=60,
                   edgecolor="black", facecolor="#d62728", linewidths=0.6,
                   label=f"off-ramp ({len(off_xy)})", zorder=5)

    # Corridor endpoints, pulled from the first / last segment's geometry.
    up_xy = corridor.mainline_segments[0].geometry.coords[0]
    dn_xy = corridor.mainline_segments[-1].geometry.coords[-1]
    ax.scatter([up_xy[0]], [up_xy[1]], marker="o", s=110, color="white",
               edgecolor="black", linewidths=1.5, label="upstream end", zorder=6)
    ax.scatter([dn_xy[0]], [dn_xy[1]], marker="s", s=110, color="black",
               edgecolor="black", linewidths=1.5, label="downstream end", zorder=6)

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(
        f"{corridor.ref} {corridor.direction}  ·  "
        f"{corridor.n_mainline} mainline segments, "
        f"{len(corridor.on_ramp_postmiles)} on / "
        f"{len(corridor.off_ramp_postmiles)} off ramps  ·  "
        f"{corridor.total_length:.2f} mi"
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.autoscale_view()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_cell_layout(
    corridor: Corridor, cells_df: pd.DataFrame, out_path: Path,
) -> None:
    """Strip plot: cells along postmile, colored by lane count, ramps on top."""
    fig, ax = plt.subplots(figsize=(13, 3))

    for cell in cells_df.itertuples(index=False):
        ax.barh(
            0, width=cell.length, left=cell.pm_start,
            color=_lane_color(cell.lanes), edgecolor="white",
            height=0.5, linewidth=0.4,
        )
    # On-ramps: triangle above the strip at pm_start of cells with on_ramp=True.
    on_pms = cells_df.loc[cells_df["on_ramp"], "pm_start"].to_list()
    off_pms = cells_df.loc[cells_df["off_ramp"], "pm_end"].to_list()
    if on_pms:
        ax.scatter(on_pms, [0.4] * len(on_pms), marker="v", color="#2ca02c",
                   edgecolor="black", linewidths=0.5, s=55, zorder=4,
                   label=f"on-ramps ({len(on_pms)})")
    if off_pms:
        ax.scatter(off_pms, [-0.4] * len(off_pms), marker="^", color="#d62728",
                   edgecolor="black", linewidths=0.5, s=55, zorder=4,
                   label=f"off-ramps ({len(off_pms)})")

    # Lane-count legend.
    lane_handles = []
    for lanes in sorted(cells_df["lanes"].unique()):
        lane_handles.append(
            plt.Rectangle((0, 0), 1, 1, color=_lane_color(int(lanes)),
                          label=f"{lanes} lanes")
        )
    ax.legend(
        handles=lane_handles
        + [h for h in ax.get_legend_handles_labels()[0]
           if not isinstance(h, plt.Rectangle)],
        loc="upper center", bbox_to_anchor=(0.5, -0.25),
        ncol=min(8, len(lane_handles) + 2), fontsize=8, frameon=False,
    )

    ax.set_xlim(corridor.pm_start - 0.1, corridor.pm_end + 0.1)
    ax.set_ylim(-0.9, 0.9)
    ax.set_xlabel("postmile from corridor upstream end [mi]")
    ax.set_yticks([])
    ax.set_title(
        f"{corridor.ref} {corridor.direction}  ·  {len(cells_df)} cells  ·  "
        f"upstream → downstream"
    )
    ax.grid(axis="x", alpha=0.3)

    # Secondary x-axis showing Caltrans postmiles when available. They run in
    # the route's own direction, so for a westbound corridor they decrease
    # along travel -- which is exactly what we want to communicate.
    if "caltrans_pm_start" in cells_df.columns:
        pm_start_local = corridor.pm_start
        pm_end_local = corridor.pm_end
        cal_start = cells_df["caltrans_pm_start"].iloc[0]
        cal_end = cells_df["caltrans_pm_end"].iloc[-1]
        slope = (cal_end - cal_start) / (pm_end_local - pm_start_local)

        def local_to_cal(x):
            return cal_start + (x - pm_start_local) * slope

        def cal_to_local(c):
            return pm_start_local + (c - cal_start) / slope

        secax = ax.secondary_xaxis("top", functions=(local_to_cal, cal_to_local))
        secax.set_xlabel("Caltrans postmile [mi]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---- Simulation-result helpers --------------------------------------------


def downsample_to_plot_period(
    arr: np.ndarray, *, state: bool, steps_per_period: int, n_samples: int,
) -> np.ndarray:
    """Pick per-plotting-period columns of a ``(N, T+1)`` state or ``(N, T)`` flow array.

    State arrays sample at step boundaries ``0, P, 2P, ..., n_samples*P``
    (length ``n_samples + 1``); flow arrays sample at the start of each
    plotting period (length ``n_samples``).
    """
    if state:
        cols = np.arange(n_samples + 1) * steps_per_period
    else:
        cols = np.arange(n_samples) * steps_per_period
    return arr[:, cols]


def per_period_totals(
    per_step: np.ndarray, *, steps_per_period: int, n_samples: int,
) -> np.ndarray:
    """Sum a ``(T,)`` per-sim-step series into per-plotting-period totals ``(n_samples,)``."""
    return per_step[: n_samples * steps_per_period].reshape(
        n_samples, steps_per_period,
    ).sum(axis=1)


def plot_flow_density_contour(
    arr_KN: np.ndarray, *,
    title: str, cbar_label: str, cmap: str,
    horizon_h: float, y_label: str = "time [h]",
    out_path: Path,
) -> None:
    """Render a ``(K, N)`` space-time array as a heatmap.

    Time runs up the y-axis from 0 to ``horizon_h``; cell index runs
    along the x-axis. The ``y_label`` is overridable so callers can
    distinguish "time of day" from "time from sim start".
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    n_cells = arr_KN.shape[1]
    im = ax.imshow(
        arr_KN, aspect="auto", origin="lower", cmap=cmap,
        extent=[-0.5, n_cells - 0.5, 0.0, horizon_h],
    )
    ax.set_xlabel("cell index (upstream -> downstream)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


_METRIC_SERIES = [
    ("VHT [veh*h]",              "tab:blue"),
    ("VMT [veh*mi]",             "tab:green"),
    ("delay [veh*h]",            "tab:red"),
    ("productivity loss [mi*h]", "tab:purple"),
]


def plot_aggregate_metrics(
    time_h: np.ndarray,
    vht: np.ndarray, vmt: np.ndarray, delay: np.ndarray, ploss: np.ndarray,
    *,
    period_label: str, x_label: str = "time [h]",
    out_path: Path,
) -> None:
    """Stack VHT/VMT/delay/ploss per plotting period in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    series = list(zip((vht, vmt, delay, ploss), _METRIC_SERIES))
    for ax, (y, (label, color)) in zip(axes.flat, series):
        ax.plot(time_h, y, color=color, lw=1.4)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel(x_label)
    fig.suptitle(f"Aggregate performance metrics per {period_label} period")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_cell_metrics(metrics, *, title: str, out_path: Path) -> None:
    """Horizon-total VHT/VMT/delay/ploss broken down by cell."""
    vht = metrics.vht_per_cell.sum(axis=1)
    vmt = metrics.vmt_per_cell.sum(axis=1)
    delay = metrics.delay_per_cell.sum(axis=1)
    ploss = metrics.productivity_loss_per_cell.sum(axis=1)
    cells = np.arange(vht.size)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    series = list(zip((vht, vmt, delay, ploss), _METRIC_SERIES))
    for ax, (y, (label, color)) in zip(axes.flat, series):
        ax.bar(cells, y, color=color)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")
    for ax in axes[-1, :]:
        ax.set_xlabel("cell index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
