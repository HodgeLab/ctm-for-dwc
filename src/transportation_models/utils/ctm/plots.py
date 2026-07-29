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
* :func:`plot_geh_heatmap`           -- direct/tiebreak cell x hour
  heatmap of the flow GEH statistic.
* :func:`plot_qq_pooled`             -- corridor-pooled two-sample
  sim-vs-observed QQ for one quantity.
* :func:`plot_qq_percell`            -- per-direct/tiebreak-cell faceted
  two-sample QQ grid.

**Cross-study comparison** (used by
:mod:`scripts.compare_ctm_case_studies`):

* :func:`plot_metric_vs_distance`    -- one corridor metric vs corridor
  length, a line per ramp-fill strategy.
* :func:`plot_qq_cross_study`        -- overlaid corridor-pooled
  sim-vs-observed QQ curves, color = distance, line style = strategy.
"""

from __future__ import annotations

import math
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
    horizon_h: float, pm_edges: np.ndarray,
    x_label: str = "distance from upstream end [mi]",
    y_label: str = "time [h]",
    vmax: float | None = None,
    out_path: Path,
) -> None:
    """Render a ``(K, N)`` space-time array as a heatmap.

    Time runs up the y-axis from 0 to ``horizon_h``. ``pm_edges`` is
    distance along the corridor at each cell boundary, length
    ``n_cells + 1`` (e.g. ``[pm_start_0, pm_end_0, pm_end_1, ...,
    pm_end_{N-1}]``), so cells with non-uniform length render at their
    actual widths rather than being normalized to a uniform grid. The
    x_label / y_label defaults work for distance from the corridor
    upstream end and arbitrary time origins; pass overrides for e.g.
    absolute Caltrans postmile or "time of day". ``vmax`` caps the color
    scale (values above it saturate at the top color, flagged on the
    colorbar); ``None`` autoscales.
    """
    pm_edges = np.asarray(pm_edges, dtype=float)
    n_samples, n_cells = arr_KN.shape
    if pm_edges.shape != (n_cells + 1,):
        raise ValueError(
            f"pm_edges must have shape ({n_cells + 1},) (= n_cells + 1); "
            f"got {pm_edges.shape}."
        )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    y_edges = np.linspace(0.0, horizon_h, n_samples + 1)
    mesh = ax.pcolormesh(
        pm_edges, y_edges, arr_KN, cmap=cmap, shading="flat", vmax=vmax,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.colorbar(
        mesh, ax=ax, label=cbar_label,
        extend="max" if vmax is not None else "neither",
    )
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


def plot_geh_heatmap(
    hourly_geh: np.ndarray, *,
    row_labels: list[str], hour_starts: pd.DatetimeIndex,
    title: str, out_path: Path,
    threshold: float = 5.0, vmax: float = 10.0,
) -> None:
    """Heatmap of flow GEH per direct/tiebreak cell (rows) x hour (cols).

    ``hourly_geh`` is ``(n_rows, n_hours)``; NaN cells (incomplete observed
    hour) render blank. Colors run green (good, GEH <= ``threshold``) to
    red (poor), clipped at ``vmax``. ``row_labels`` annotate each cell;
    ``hour_starts`` labels the x-axis at the start of each hourly bin.
    """
    n_rows, n_hours = hourly_geh.shape
    fig, ax = plt.subplots(figsize=(min(14, 2 + 0.5 * n_hours), 1.5 + 0.4 * n_rows))
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("lightgrey")
    masked = np.ma.masked_invalid(hourly_geh)
    mesh = ax.imshow(
        masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(range(n_hours))
    ax.set_xticklabels(
        [t.strftime("%H:%M") for t in hour_starts], rotation=90, fontsize=8,
    )
    ax.set_xlabel("hour start")
    ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax, label="GEH", extend="max")
    cbar.ax.axhline(threshold, color="black", lw=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---- QQ error-distribution plots ------------------------------------------


_QQ_UNITS = {"flow": "veh/h", "density": "veh/mi"}


def _two_sample_qq(ax, *, sim: np.ndarray, obs: np.ndarray, unit: str) -> None:
    """Two-sample QQ on ``ax``: sorted observed (x) vs sorted sim (y), 45-deg line."""
    if sim.size < 2:
        ax.text(0.5, 0.5, "insufficient samples", ha="center", va="center",
                transform=ax.transAxes)
    else:
        sim_sorted = np.sort(sim)
        obs_sorted = np.sort(obs)
        ax.scatter(obs_sorted, sim_sorted, s=12, color="tab:blue", alpha=0.7)
        lo = float(min(obs_sorted[0], sim_sorted[0]))
        hi = float(max(obs_sorted[-1], sim_sorted[-1]))
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--")
    ax.set_xlabel(f"observed quantiles [{unit}]")
    ax.set_ylabel(f"sim quantiles [{unit}]")
    ax.grid(alpha=0.3)


def plot_qq_pooled(qq, *, quantity: str, out_path: Path) -> None:
    """Corridor-pooled QQ diagnostic for ``quantity`` ('flow' or 'density').

    Two-sample QQ of the simulated vs observed marginal distribution
    (sorted observed on x, sorted sim on y, with the 45-deg line). Points
    on the line => the model reproduces the marginal distribution of the
    quantity.

    ``qq`` is a :class:`utils.ctm.validation.QQResult`.
    """
    unit = _QQ_UNITS[quantity]
    sim, obs = qq.pooled(quantity)
    fig, ax = plt.subplots(figsize=(6, 5))

    _two_sample_qq(ax, sim=sim, obs=obs, unit=unit)
    ax.set_title(
        f"Sim vs observed {quantity} QQ -- corridor-pooled "
        f"({sim.size} samples)"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_qq_percell(
    qq, *, quantity: str, out_path: Path,
    min_samples: int = 2,
) -> None:
    """Faceted per-VDS-cell two-sample QQ grid.

    One panel per direct/tiebreak VDS cell with at least ``min_samples``
    paired samples (sim vs observed marginal QQ); cells with fewer are
    skipped. If no cell qualifies, a placeholder figure is written so the
    promised path still exists.

    ``qq`` is a :class:`utils.ctm.validation.QQResult`.
    """
    unit = _QQ_UNITS[quantity]
    cells = [
        c for c in qq.per_cell
        if getattr(c, f"{quantity}_sim").size >= min_samples
    ]
    if not cells:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "no cells with sufficient samples",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    n = len(cells)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 3.1 * nrows), squeeze=False,
    )
    for ax, c in zip(axes.flat, cells):
        sim = getattr(c, f"{quantity}_sim")
        obs = getattr(c, f"{quantity}_obs")
        _two_sample_qq(ax, sim=sim, obs=obs, unit=unit)
        ax.set_title(f"cell {c.cell} (VDS {c.vds_id})", fontsize=9)
    for ax in axes.flat[n:]:
        ax.set_axis_off()

    fig.suptitle(
        f"{quantity.capitalize()} sim vs observed QQ by direct/tiebreak cell"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---- Cross-study comparison plots -----------------------------------------


def plot_metric_vs_distance(
    summary: pd.DataFrame, *,
    metric: str, ylabel: str, title: str, out_path: Path,
    strategy_colors: dict[str, str],
) -> None:
    """One validation metric vs corridor length, a line per ramp strategy.

    ``summary`` has one row per study with ``distance_mi``,
    ``ramp_strategy`` and the metric column. Each ramp strategy becomes a
    colored line (from ``strategy_colors``) over the studied lengths,
    sorted by distance. NaN metric values (e.g. GEH for a sim that doesn't
    cover whole hours) render as gaps.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy in sorted(summary["ramp_strategy"].unique()):
        sub = summary[summary["ramp_strategy"] == strategy].sort_values(
            "distance_mi"
        )
        ax.plot(
            sub["distance_mi"], sub[metric],
            marker="o", color=strategy_colors[strategy], label=strategy,
        )
    ax.set_xlabel("corridor length [mi]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(title="ramp fill strategy", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_qq_cross_study(
    curves: list[dict], *,
    quantity: str, out_path: Path,
    distance_colors: dict[float, str], strategy_styles: dict[str, str],
) -> None:
    """Overlaid corridor-pooled sim-vs-observed QQ curves across studies.

    Each entry in ``curves`` is a dict with ``distance`` (miles),
    ``strategy`` and the pooled paired ``sim`` / ``obs`` arrays for
    ``quantity``. Each study is drawn as a two-sample QQ curve (sorted
    observed on x, sorted sim on y) with its color set by distance and its
    line style by ramp strategy, over a shared y=x reference. Two legends
    decode the color (distance) and style (strategy) channels.
    """
    unit = _QQ_UNITS[quantity]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    bounds: list[float] = []
    obs_max = 0.0
    for c in curves:
        sim = np.sort(c["sim"])
        obs = np.sort(c["obs"])
        if sim.size < 2:
            continue
        ax.plot(
            obs, sim,
            color=distance_colors[c["distance"]],
            ls=strategy_styles[c["strategy"]], lw=1.6, alpha=0.85,
        )
        bounds += [float(obs[0]), float(obs[-1]), float(sim[0]), float(sim[-1])]
        obs_max = max(obs_max, float(obs[-1]))

    if bounds:
        # Cap the axes at 150% of the largest observed quantile so a few
        # extreme sim outliers can't blow up the shared scale.
        lo = min(bounds)
        hi = 1.5 * obs_max
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls=":", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel(f"observed quantiles [{unit}]")
    ax.set_ylabel(f"sim quantiles [{unit}]")
    ax.set_title(f"Sim vs observed {quantity} QQ -- corridor-pooled")
    ax.grid(alpha=0.3)

    # Two legends: color decodes distance, line style decodes strategy.
    color_handles = [
        plt.Line2D([], [], color=col, lw=2.0, label=f"{dist:g} mi")
        for dist, col in sorted(distance_colors.items())
    ]
    style_handles = [
        plt.Line2D([], [], color="black", ls=style, lw=1.6, label=strategy)
        for strategy, style in strategy_styles.items()
    ]
    leg1 = ax.legend(
        handles=color_handles, title="distance", fontsize=8,
        loc="upper left",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=style_handles, title="ramp fill strategy", fontsize=8,
        loc="lower right",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
