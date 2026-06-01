"""Matplotlib renderers for a CTM :class:`Corridor` and its cell layout.

Two PNGs we use across the Step-1 build script
(:mod:`scripts.build_ctm_from_osm`) and the cell-artifact regenerator
(:mod:`scripts.regenerate_cell_artifacts`):

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
