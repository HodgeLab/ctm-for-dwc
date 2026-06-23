"""Steps 1+2 end-to-end demo: OSM (osmnx) -> Corridor -> CTM cell table.

Loads a freeway network from OSM (either a cached graphml file or a fresh
osmnx download), runs the corridor extraction in
:mod:`transportation_models.utils.ctm.osm`, lays out cells with
:func:`transportation_models.utils.ctm.cells.cells_from_corridor`, and writes:

* ``cells.csv``            -- the freeway-schema cell table (Step 4 fills in
                              the FD parameters before this can be passed to
                              ``io.freeway_from_dataframe``). Includes
                              ``caltrans_pm_start``/``caltrans_pm_end`` when
                              ``--postmiles`` is supplied, and
                              ``vds_*`` / ``on_ramp_vds_id`` /
                              ``off_ramp_vds_id`` when ``--pems-metadata``
                              is supplied.
* ``cells.geojson``        -- one LineString per cell with all of cells.csv's
                              columns as properties. Drop into QGIS / Google
                              Earth to overlay the cell layout on satellite
                              imagery.
* ``ramps.geojson``        -- one POINT per ramp gore. Same QGIS workflow.
* ``mainline.geojson``     -- one LineString per :class:`MainlineSegment`
                              (u, v, pm_start, pm_end, lanes). Together with
                              ``ramps.geojson`` + ``corridor.json`` this lets
                              ``scripts/regenerate_cell_artifacts.py`` rebuild
                              the cell GeoJSON + PNGs after manual edits to
                              ``cells.csv`` (Step 5) without needing the
                              original osmnx graph.
* ``corridor.json``        -- corridor metadata (``ref``, ``direction``,
                              ``target_bearing``, ``crs``) and the optional
                              Caltrans postmile lookup, completing the
                              round-trip sidecar set.
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

    # Fresh osmnx download (network required).
    # Note the `=`: a leading `-` in the value confuses argparse, so bind with `=`.
    python scripts/build_ctm_from_osm.py \\
        --bbox=-118.155,34.13,-117.85,34.16 \\
        --ref "I 210" --direction W

    # Or pass an osmnx-recognizable place string:
    python scripts/build_ctm_from_osm.py \\
        --place "Pasadena, California, USA" \\
        --ref "I 210" --direction W
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import osmnx as ox

from transportation_models.utils.ctm.caltrans import (
    download_caltrans_postmiles,
    extract_route_number,
)
from transportation_models.utils.ctm.cells import (
    cells_from_corridor,
    cells_to_geodataframe,
    corridor_to_artifacts,
    flag_cell_length_warnings,
)
from transportation_models.utils.ctm.osm import Corridor, corridor_from_graph
from transportation_models.utils.ctm.plots import (
    plot_cell_layout,
    plot_corridor_map,
)
from transportation_models.utils.ctm.vds import (
    assign_ramp_vds_to_cells,
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
    if "length_warning" in cells_df.columns:
        too_short = int((cells_df["length_warning"] == "too_short").sum())
        too_long = int((cells_df["length_warning"] == "too_long").sum())
        if too_short or too_long:
            print(f"  length warnings   : too_short={too_short} (< 0.2 mi), "
                  f"too_long={too_long} (> 1.0 mi)")
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
        if "vds_lanes" in cells_df.columns:
            both = cells_df[cells_df["vds_lanes"].notna()]
            mismatch = int((both["lanes"] != both["vds_lanes"]).sum())
            if mismatch:
                print(f"  lane mismatches   : {mismatch} cells where OSM "
                      f"`lanes` != PeMS `vds_lanes` (PeMS is usually the "
                      f"authoritative source)")
        if "on_ramp_vds_id" in cells_df.columns:
            n_on_resolved = int(cells_df["on_ramp_vds_id"].notna().sum())
            n_on_total = int(cells_df["on_ramp"].sum())
            n_off_resolved = int(cells_df["off_ramp_vds_id"].notna().sum())
            n_off_total = int(cells_df["off_ramp"].sum())
            print(f"  ramp VDSs matched : on-ramp {n_on_resolved}/"
                  f"{n_on_total}, off-ramp {n_off_resolved}/{n_off_total}")


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
    parser.add_argument(
        "--pm-range", default=None,
        help=(
            "optional 'pm_start,pm_end' crop in *cumulative* mi from the "
            "corridor's upstream end (NOT Caltrans postmiles). Whole "
            "mainline segments are kept iff both endpoints lie inside the "
            "range; segments that straddle a bound are dropped entirely. "
            "Use a wide range (e.g. 2,6) rather than a tight one to avoid "
            "excluding every segment."
        ),
    )
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
        "--bearing-tolerance", type=float, default=60.0,
        help=(
            "max degrees an edge's bearing may deviate from --direction to "
            "count as mainline (default: 60). Raise it (e.g. 90) for corridors "
            "that curve sharply near a terminus, like I-880 swinging WNW into "
            "the I-80 junction; too low silently truncates the corridor there."
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
        print(f"pm-range crop requested: {lo:.3f}-{hi:.3f} cumulative mi",
              file=sys.stderr)

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
        bearing_tolerance=args.bearing_tolerance,
        pm_range=pm_range,
        postmiles=postmiles_arg,
    )
    cells_df = cells_from_corridor(corridor)
    # Annotate with `length_warning` so downstream consumers can flag cells
    # that may strain the CTM dynamics (too short -> Δt floored by CFL,
    # too long -> averages over too much heterogeneity).
    cells_df = flag_cell_length_warnings(cells_df)

    # Step 2 (optional): load PeMS station metadata, project VDSs onto the
    # corridor, and assign them to cells with the Dervisoglu et al. 2014
    # downstream-assignment fallback for cells that lack their own detector.
    pems_path = _resolve_pems_metadata_arg(args.pems_metadata, args.district)
    if pems_path is not None:
        mainline_vds = load_pems_station_metadata(
            pems_path, freeway=args.ref, direction=args.direction,
        )
        projected = project_vds_to_corridor(mainline_vds, corridor)
        cells_df = assign_vds_to_cells(
            cells_df, projected, tiebreaker=args.vds_tiebreaker,
        )
        # Ramp-VDS pass (separate from the mainline assignment): pull the
        # on-ramp (Type=OR) and off-ramp (Type=FR) detectors and snap each
        # corridor ramp gore to its nearest matching ramp VDS, so cells.csv
        # surfaces both the mainline detector and the ramp detectors for
        # validation.
        ramp_vds = load_pems_station_metadata(
            pems_path, freeway=args.ref, direction=args.direction,
            mainline_only=False,
        )
        ramp_vds = ramp_vds[ramp_vds["type"].astype(str).str.upper().isin({"OR", "FR"})]
        cells_df = assign_ramp_vds_to_cells(cells_df, corridor, ramp_vds)

    cells_csv = out_dir / "cells.csv"
    cells_df.to_csv(cells_csv, index=False)

    # GeoJSON sidecars: one LineString per cell + one Point per ramp gore.
    # These let users load the cell layout into QGIS / Google Earth on top
    # of a basemap for visual validation against satellite imagery.
    cells_gdf = cells_to_geodataframe(cells_df, corridor)
    cells_gdf.to_file(out_dir / "cells.geojson", driver="GeoJSON")
    # Round-trip sidecars (mainline.geojson + ramps.geojson + corridor.json)
    # so `scripts/regenerate_cell_artifacts.py` can rebuild the cell layout
    # and the PNGs after a user hand-edits cells.csv, without needing the
    # original osmnx graph.
    corridor_to_artifacts(corridor, out_dir)

    plot_corridor_map(corridor, out_dir / "corridor_map.png")
    plot_cell_layout(corridor, cells_df, out_dir / "cell_layout.png")

    print_summary(corridor, cells_df)
    print()
    print(f"Wrote {out_dir}/")
    for name in (
        "cells.csv", "cells.geojson",
        "mainline.geojson", "ramps.geojson", "corridor.json",
        "corridor_map.png", "cell_layout.png",
    ):
        if (out_dir / name).exists():
            print(f"  {name}")


if __name__ == "__main__":
    main()
