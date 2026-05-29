"""Steps 1+2 end-to-end demo: OSM (osmnx) -> Corridor -> CTM cell table.

Loads a freeway network from OSM (either a cached graphml file or a fresh
osmnx download), runs the corridor extraction in
:mod:`transportation_models.utils.ctm.osm`, lays out cells with
:func:`transportation_models.utils.ctm.cells.cells_from_corridor`, and writes:

* ``cells.csv``            -- the freeway-schema cell table (Step 4 fills in
                              the FD parameters before this can be passed to
                              ``io.freeway_from_dataframe``). Includes
                              ``caltrans_pm_start``/``caltrans_pm_end`` when
                              ``--postmiles`` is supplied.
* ``corridor_map.png``     -- folium-free matplotlib overlay of mainline +
                              ramps, colored by lane count
* ``cell_layout.png``      -- strip plot of cells along the corridor with
                              ramp markers (secondary axis shows Caltrans
                              PM when available)

Run from the repo root::

    # Cached graphml (reproducible, doesn't hit Overpass):
    python scripts/build_ctm_from_osm.py \\
        --graphml tests/fixtures/osm/i210_bbox.graphml \\
        --ref "I 210" --direction W

    # Same, but anchored to Caltrans postmiles (auto-downloads the SHN
    # Postmiles Tenth dataset for Route 210 only, ~900 KB):
    python scripts/build_ctm_from_osm.py \\
        --graphml tests/fixtures/osm/i210_bbox.graphml \\
        --ref "I 210" --direction W \\
        --postmiles auto

    # Or point at a pre-downloaded SHN file (see
    # `transportation_models.utils.ctm.caltrans` for the source):
    python scripts/build_ctm_from_osm.py \\
        --graphml tests/fixtures/osm/i210_bbox.graphml \\
        --ref "I 210" --direction W \\
        --postmiles data/caltrans/shn_postmiles_tenth.geojson

    # Fresh osmnx download (network required):
    python scripts/build_ctm_from_osm.py \\
        --bbox -118.155,34.13,-117.85,34.16 \\
        --ref "I 210" --direction W

    # Or pass an osmnx-recognizable place string:
    python scripts/build_ctm_from_osm.py \\
        --place "Pasadena, California, USA" \\
        --ref "I 210" --direction W
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from matplotlib.collections import LineCollection

from transportation_models.utils.ctm.caltrans import (
    download_caltrans_postmiles,
    extract_route_number,
)
from transportation_models.utils.ctm.cells import cells_from_corridor
from transportation_models.utils.ctm.osm import Corridor, corridor_from_graph
from transportation_models.utils.ctm.vds import (
    assign_vds_to_cells,
    download_pems_station_metadata,
    load_pems_station_metadata,
    project_vds_to_corridor,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "scripts/output/ctm_corridor"


# ---- Argument resolvers ---------------------------------------------------


def _resolve_postmiles_arg(value: str | None, ref: str):
    """Translate the ``--postmiles`` CLI value into what corridor_from_graph wants.

    * ``None``    -> ``None`` (no Caltrans annotation).
    * ``"auto"``  -> filtered download for the route parsed from ``--ref``
                     (uses the in-repo cache if it already exists).
    * any other   -> treated as a filesystem path that ``load_postmiles``
                     reads directly.
    """
    if value is None:
        return None
    if value == "auto":
        route = extract_route_number(ref)
        path = download_caltrans_postmiles(routes=[route], progress=True)
        return path
    return Path(value).expanduser().resolve()


def _resolve_pems_metadata_arg(value: str | None, district: int | None):
    """Translate ``--pems-metadata`` into a file path (or None).

    * ``None``      -> ``None`` (Step 2 skipped, cells get no vds_* columns).
    * ``"auto"``    -> :func:`download_pems_station_metadata` for the requested
                       district (latest available year). Needs ``--district``
                       and PeMS credentials in the env.
    * any other     -> filesystem path to a PeMS ``station_meta`` text file.
    """
    if value is None:
        return None
    if value == "auto":
        if district is None:
            raise SystemExit(
                "--pems-metadata auto requires --district (Caltrans district id)"
            )
        return download_pems_station_metadata(district=district, progress=True)
    return Path(value).expanduser().resolve()


# ---- Graph loaders --------------------------------------------------------


def load_graph(args: argparse.Namespace):
    """Load an osmnx MultiDiGraph from whichever source was specified.

    Always applies ``simplify_graph`` so the corridor extraction sees one
    edge per important node-to-important-node span.
    """
    if args.graphml is not None:
        g = ox.load_graphml(args.graphml)
        print(f"loaded graphml: {g.number_of_nodes()} nodes, "
              f"{g.number_of_edges()} edges  ({args.graphml})")
        return g

    fetch_kwargs = dict(
        custom_filter='["highway"~"motorway|motorway_link"]',
        simplify=False,
        retain_all=True,
        truncate_by_edge=True,
    )
    if args.bbox is not None:
        west, south, east, north = (float(v) for v in args.bbox.split(","))
        # osmnx 2.x: graph_from_bbox((west, south, east, north), ...)
        g = ox.graph_from_bbox((west, south, east, north), **fetch_kwargs)
        print(f"downloaded bbox ({west}, {south}, {east}, {north})")
    elif args.place is not None:
        g = ox.graph_from_place(args.place, **fetch_kwargs)
        print(f"downloaded place {args.place!r}")
    else:
        raise SystemExit("must supply one of --graphml, --bbox, --place")
    print(f"  raw: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    g = ox.simplify_graph(g)
    print(f"  simplified: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    return g


# ---- Plotting -------------------------------------------------------------


# Distinct colors per lane count so the corridor map reads at a glance.
_LANE_COLORS = {
    2: "#1f77b4", 3: "#2ca02c", 4: "#ff7f0e",
    5: "#d62728", 6: "#9467bd", 7: "#8c564b",
}
_DEFAULT_LANE_COLOR = "#7f7f7f"


def _lane_color(lanes: int) -> str:
    return _LANE_COLORS.get(int(lanes), _DEFAULT_LANE_COLOR)


def plot_corridor_map(graph, corridor: Corridor, out_path: Path) -> None:
    """Overlay mainline (colored by lane count) + ramp gores on a lat/lon map."""
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

    # Corridor endpoints.
    up = graph.nodes[corridor.mainline_segments[0].u]
    dn = graph.nodes[corridor.mainline_segments[-1].v]
    ax.scatter([up["x"]], [up["y"]], marker="o", s=110, color="white",
               edgecolor="black", linewidths=1.5, label="upstream end", zorder=6)
    ax.scatter([dn["x"]], [dn["y"]], marker="s", s=110, color="black",
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


def plot_cell_layout(corridor: Corridor, cells_df, out_path: Path) -> None:
    """Strip plot: cells along postmile, colored by lane count, ramps on top."""
    fig, ax = plt.subplots(figsize=(13, 3))

    cells = list(cells_df.itertuples(index=False))
    for cell in cells:
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


# ---- Console summary ------------------------------------------------------


def print_summary(corridor: Corridor, cells_df) -> None:
    print()
    print(f"=== Corridor: {corridor.ref} {corridor.direction} ===")
    print(f"  mainline segments : {corridor.n_mainline}")
    print(f"  on-ramps          : {len(corridor.on_ramp_postmiles)}")
    print(f"  off-ramps         : {len(corridor.off_ramp_postmiles)}")
    print(f"  total length      : {corridor.total_length:.3f} mi")
    print(f"  lane-count breaks : {len(corridor.lane_change_postmiles())}")
    print()
    print(f"=== Cell layout ({len(cells_df)} cells) ===")
    print(f"  pm range          : {cells_df.pm_start.iloc[0]:.3f} -> "
          f"{cells_df.pm_end.iloc[-1]:.3f} mi")
    print(f"  mean cell length  : {cells_df.length.mean():.3f} mi")
    print(f"  min / max length  : {cells_df.length.min():.3f} / "
          f"{cells_df.length.max():.3f} mi")
    print(f"  cells with on-ramp: {int(cells_df.on_ramp.sum())}")
    print(f"  cells with off    : {int(cells_df.off_ramp.sum())}")
    print(f"  lanes histogram   : "
          + ", ".join(f"{int(n)} lanes -> {c}"
                       for n, c in cells_df.lanes.value_counts().sort_index().items()))
    if "caltrans_pm_start" in cells_df.columns:
        cal_start = cells_df["caltrans_pm_start"].iloc[0]
        cal_end = cells_df["caltrans_pm_end"].iloc[-1]
        print(
            f"  Caltrans PM range : {cal_start:.3f} -> {cal_end:.3f} "
            f"(span {abs(cal_end - cal_start):.3f} mi)"
        )
    if "vds_source" in cells_df.columns:
        from collections import Counter
        srcs = Counter(cells_df["vds_source"])
        print(f"  VDS sources       : "
              + ", ".join(f"{k}={v}" for k, v in sorted(srcs.items())))
        multi = int((cells_df["vds_n_candidates"] >= 2).sum())
        if multi:
            print(f"  multi-VDS cells   : {multi} "
                  f"(tiebreaker resolved them)")


# ---- main -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--graphml", type=Path, help="cached osmnx graphml")
    src.add_argument("--bbox", help="osmnx bbox 'west,south,east,north'")
    src.add_argument("--place", help="osmnx place string")
    parser.add_argument("--ref", required=True, help="OSM ref, e.g. 'I 210'")
    parser.add_argument("--direction", required=True,
                        help="N/S/E/W or numeric bearing in degrees")
    parser.add_argument("--pm-range", default=None,
                        help="optional 'pm_start,pm_end' crop in cumulative mi")
    parser.add_argument(
        "--postmiles",
        default=None,
        help=(
            "Caltrans SHN postmile source. Either 'auto' (downloads the "
            "single-route subset of the SHN Postmiles Tenth dataset using "
            "the route parsed from --ref, ~1 MB), or a filesystem path to "
            "a pre-downloaded postmile shapefile / geojson. Omitting this "
            "leaves Caltrans PMs unset and the cells.csv only carries "
            "cumulative-length postmiles."
        ),
    )
    parser.add_argument(
        "--pems-metadata",
        default=None,
        help=(
            "PeMS station_meta source. Either 'auto' (downloads the latest "
            "snapshot for --district via the PeMS clearinghouse; needs "
            "PEMS_USERNAME / PEMS_PASSWORD in the environment), or a "
            "filesystem path to a pre-downloaded station_meta.txt file. "
            "Omitting this skips Step 2 (cells.csv has no vds_* columns)."
        ),
    )
    parser.add_argument(
        "--district", type=int, default=None,
        help=(
            "Caltrans district for --pems-metadata auto (e.g. 4 for the Bay "
            "Area / I-880, 7 for LA / I-210). Ignored if --pems-metadata is "
            "an explicit path or omitted."
        ),
    )
    parser.add_argument(
        "--vds-tiebreaker", choices=["midpoint", "lowest_id"], default="midpoint",
        help="strategy when multiple VDSs land in one cell (default: midpoint)",
    )
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"output directory (default: {DEFAULT_OUT}/<ref>_<dir>)")
    args = parser.parse_args()

    direction: str | float = args.direction
    try:
        direction = float(args.direction)
    except ValueError:
        pass

    pm_range = None
    if args.pm_range is not None:
        lo, hi = (float(v) for v in args.pm_range.split(","))
        pm_range = (lo, hi)

    # Resolve --postmiles into something corridor_from_graph accepts (Path or
    # None). 'auto' triggers a route-filtered Caltrans download; an explicit
    # path is passed through.
    postmiles_arg = _resolve_postmiles_arg(args.postmiles, args.ref)

    slug = f"{args.ref.replace(' ', '_')}_{args.direction}"
    out_dir = (args.out_dir or DEFAULT_OUT / slug).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = load_graph(args)
    corridor = corridor_from_graph(
        graph,
        ref=args.ref,
        direction=direction,
        pm_range=pm_range,
        postmiles=postmiles_arg,
    )
    cells_df = cells_from_corridor(corridor)

    # Step 2 (optional): load PeMS station metadata, project VDSs onto the
    # corridor, and assign them to cells with the Dervisoglu et al. 2014
    # downstream-assignment fallback for cells that lack their own detector.
    pems_path = _resolve_pems_metadata_arg(args.pems_metadata, args.district)
    if pems_path is not None:
        vds_gdf = load_pems_station_metadata(
            pems_path,
            freeway=args.ref,
            direction=args.direction if isinstance(direction, str) else args.direction,
        )
        projected = project_vds_to_corridor(vds_gdf, corridor)
        cells_df = assign_vds_to_cells(
            cells_df, projected, tiebreaker=args.vds_tiebreaker,
        )

    cells_csv = out_dir / "cells.csv"
    cells_df.to_csv(cells_csv, index=False)

    plot_corridor_map(graph, corridor, out_dir / "corridor_map.png")
    plot_cell_layout(corridor, cells_df, out_dir / "cell_layout.png")

    print_summary(corridor, cells_df)
    print()
    print(f"Wrote {out_dir}/")
    for name in ("cells.csv", "corridor_map.png", "cell_layout.png"):
        print(f"  {name}")


if __name__ == "__main__":
    main()
