"""List the OSM `ref` codes present in a region so users can pick a `--ref` for
:mod:`scripts.build_ctm_from_osm`.

Run from the repo root::

    # From a cached graphml:
    python scripts/list_osm_refs.py --graphml tests/fixtures/osm/i210_bbox.graphml

    # From a fresh bbox download (Oakland, looking for I-880 etc.).
    # Note the `=`: a leading `-` in the value confuses argparse, so bind with `=`.
    python scripts/list_osm_refs.py --bbox=-122.35,37.70,-122.20,37.84

    # From a place string:
    python scripts/list_osm_refs.py --place "Oakland, California, USA"

Prints a sorted table::

    ref         n_edges  length_mi  top_names           bbox
    --------  ---------  ---------  ------------------  -----------------------
    I 880            48       17.4  Nimitz Freeway      -122.30..-122.20 37.71..
    I 580            36       11.1  MacArthur Freeway   ...
    ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import osmnx as ox

from ctm_for_dwc.utils.ctm.osm import summarize_refs


def load_graph(args) -> "ox.MultiDiGraph":   # noqa: F821 - lazy hint
    fetch_kwargs = dict(
        custom_filter='["highway"~"motorway|motorway_link"]',
        simplify=False, retain_all=True, truncate_by_edge=True,
    )
    if args.graphml is not None:
        return ox.load_graphml(args.graphml)
    if args.bbox is not None:
        w, s, e, n = (float(v) for v in args.bbox.split(","))
        g = ox.graph_from_bbox((w, s, e, n), **fetch_kwargs)
        return ox.simplify_graph(g)
    if args.place is not None:
        g = ox.graph_from_place(args.place, **fetch_kwargs)
        return ox.simplify_graph(g)
    raise SystemExit("must supply --graphml, --bbox, or --place")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--graphml", type=Path)
    src.add_argument("--bbox", help="'west,south,east,north' in WGS84 degrees")
    src.add_argument("--place", help="osmnx place string")
    parser.add_argument(
        "--include-ramps", action="store_true",
        help="count motorway_link edges too (default: motorway only)",
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="show only the top N refs by length (default: show all)",
    )
    args = parser.parse_args()

    graph = load_graph(args)
    types = ("motorway", "motorway_link") if args.include_ramps else ("motorway",)
    df = summarize_refs(graph, highway_types=types)

    if df.empty:
        print(f"(no refs found in {len(graph)} nodes / {graph.number_of_edges()} edges)")
        return

    if args.top is not None:
        df = df.head(args.top)

    # Pretty-print with a compact bbox column.
    width = max(len(r) for r in df["ref"]) + 1
    print(f"{'ref':<{width}}  {'edges':>5}  {'mi':>7}  "
          f"{'top names':<32}  bbox  (lon W..E, lat S..N)")
    print("-" * (width + 5 + 7 + 32 + 25 + 6))
    for _, row in df.iterrows():
        bbox = (f"{row.lon_min:.3f}..{row.lon_max:.3f}  "
                f"{row.lat_min:.3f}..{row.lat_max:.3f}")
        names = row.top_names[:32] if row.top_names else "—"
        print(f"{row.ref:<{width}}  {row.n_edges:>5}  {row.length_mi:>7.2f}  "
              f"{names:<32}  {bbox}")


if __name__ == "__main__":
    main()
