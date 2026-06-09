"""Step 1: extract a CTM :class:`Corridor` from an osmnx ``MultiDiGraph``.

This module owns the spatial side of Step 1 in ``docs/ctm_module.md``. The
:class:`Corridor` dataclass is the typed handoff between the OSM-driven
extraction here and the cell-layout logic in ``cells.py`` (Layer 2).

The expected upstream pipeline is::

    import osmnx as ox
    g = ox.graph_from_bbox(
        (north, south, east, west),
        custom_filter='["highway"~"motorway|motorway_link"]',
    )
    g = ox.simplify_graph(g)
    corridor = corridor_from_graph(g, ref="I 210", direction="W")

osmnx's download / simplification helpers stay in the caller so users can
pick whichever fetch method fits (``graph_from_place``, ``graph_from_xml``,
``load_graphml``, …). :func:`corridor_from_graph` does only the work osmnx
can't: pick one direction of a divided freeway out of a bidirectional graph,
order the mainline upstream→downstream, and attach ramp gore points.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, Point

_METERS_PER_MILE = 1609.344

# Compass labels -> bearing in degrees (0 = N, 90 = E, ...).
_COMPASS_BEARINGS: dict[str, float] = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}


# ---- Public dataclasses ---------------------------------------------------


@dataclass(frozen=True)
class MainlineSegment:
    """One mainline edge of the corridor, between two important OSM nodes.

    After osmnx simplification each edge spans a contiguous stretch of the
    freeway between cell-boundary candidates (ramp gores, lane-count changes,
    or other degree>2 nodes). ``pm_start``/``pm_end`` are cumulative miles
    from the corridor's upstream end.
    """

    u: int                  # OSM node id at the upstream end
    v: int                  # OSM node id at the downstream end
    geometry: LineString    # WGS84 (lon, lat) running u -> v
    length: float           # mi, cumulative-length-compatible
    lanes: int              # mainline lane count
    pm_start: float         # mi
    pm_end: float           # mi


@dataclass(frozen=True)
class RampJunction:
    """A point where an on- or off-ramp meets the mainline.

    ``postmile`` is the corridor postmile at ``mainline_node`` (the gore
    point's mainline endpoint). On-ramps merge *into* the mainline at this
    node; off-ramps split *off* the mainline at this node.
    """

    kind: Literal["on", "off"]
    postmile: float
    mainline_node: int
    ramp_terminus_node: int
    geometry: Point         # the gore point's (lon, lat) on the mainline
    lanes: int              # ramp lane count
    name: Optional[str]     # OSM ref/name of the ramp, if tagged


@dataclass
class Corridor:
    """One direction of a freeway extracted from an osmnx graph.

    Mainline segments are ordered upstream → downstream; ramp junctions are
    sorted by postmile. ``crs`` is the CRS of the input graph (osmnx default
    is EPSG:4326) and applies to all geometries here.

    ``caltrans_postmiles`` is populated when ``corridor_from_graph`` is
    called with a postmile shapefile (see :mod:`.caltrans`); otherwise it's
    ``None``. The mapping is ``osm_node_id → Caltrans postmile`` and may run
    counter to the corridor's travel direction (e.g. westbound corridors
    have monotonically *decreasing* Caltrans PMs along travel).
    """

    ref: str
    direction: str
    target_bearing: float           # degrees, 0 = N
    mainline_segments: list[MainlineSegment]
    ramp_junctions: list[RampJunction]
    crs: str = "EPSG:4326"
    caltrans_postmiles: Optional[dict[int, float]] = None

    @property
    def n_mainline(self) -> int:
        return len(self.mainline_segments)

    @property
    def pm_start(self) -> float:
        return self.mainline_segments[0].pm_start

    @property
    def pm_end(self) -> float:
        return self.mainline_segments[-1].pm_end

    @property
    def total_length(self) -> float:
        """Corridor length [mi]."""
        return self.pm_end - self.pm_start

    @property
    def on_ramp_postmiles(self) -> list[float]:
        return [r.postmile for r in self.ramp_junctions if r.kind == "on"]

    @property
    def off_ramp_postmiles(self) -> list[float]:
        return [r.postmile for r in self.ramp_junctions if r.kind == "off"]

    def lane_change_postmiles(self) -> list[float]:
        """Postmiles where the mainline lane count changes between segments."""
        out: list[float] = []
        for prev, cur in zip(self.mainline_segments, self.mainline_segments[1:]):
            if prev.lanes != cur.lanes:
                out.append(cur.pm_start)
        return out


# ---- Entry point ----------------------------------------------------------


def corridor_from_graph(
    graph: nx.MultiDiGraph,
    *,
    ref: str,
    direction: str | float,
    pm_range: Optional[tuple[float, float]] = None,
    postmiles=None,
    bearing_tolerance: float = 60.0,
) -> Corridor:
    """Pull a single-direction freeway corridor out of an osmnx graph.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Output of any ``osmnx.graph_from_*`` (or ``load_graphml``) call,
        ideally after ``osmnx.simplify_graph``. Must be in EPSG:4326 (the
        osmnx default); node attributes ``x``/``y`` are required. Edges
        should carry ``highway`` and ``ref`` from OSM.
    ref : str
        OSM ``ref`` value identifying the corridor, e.g. ``"I 210"``.
        Matches edges whose ``ref`` is exactly this string or contains it
        as one entry of a ``;``-separated list / Python list.
    direction : {"N","S","E","W"} or float
        Direction of travel along the corridor, used to pick one carriageway
        of a divided freeway out of the bidirectional graph. Compass letters
        resolve to 0/90/180/270; a numeric value is treated as a bearing in
        degrees clockwise from north.
    pm_range : (float, float), optional
        Inclusive postmile bounds (miles from corridor start) to keep. If
        omitted the full extracted corridor is returned.
    postmiles : str, Path, or GeoDataFrame, optional
        Caltrans SHN postmile dataset (path to shapefile/geopackage/geojson,
        or a pre-loaded GeoDataFrame with ``Route``/``PM``/POINT columns).
        When supplied, every mainline node is annotated with a Caltrans PM
        in :attr:`Corridor.caltrans_postmiles`; the corridor's existing
        cumulative-length ``pm_start``/``pm_end`` are not changed.
    bearing_tolerance : float, default 60.0
        Mainline edges whose bearing differs from ``direction`` by more than
        this many degrees are dropped. The default tolerates curved
        freeways; tighten it on long straight segments.

    Returns
    -------
    Corridor
        Mainline ordered upstream-to-downstream by cumulative length, with
        on- and off-ramp junctions attached at the OSM node where they meet
        the mainline.

    Raises
    ------
    ValueError
        If no mainline edges match ``ref`` and ``direction`` within
        ``bearing_tolerance``, or if the filtered mainline subgraph is not a
        single directed path.
    """
    target_bearing = _resolve_target_bearing(direction)

    # 1. Collect mainline edges (highway == motorway) matching ref + bearing.
    mainline_edges: list[tuple[int, int, int, dict]] = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        if not _is_mainline(data) or not _matches_ref(data, ref):
            continue
        if not _bearing_close(_edge_bearing(graph, u, v, data),
                              target_bearing, bearing_tolerance):
            continue
        mainline_edges.append((u, v, key, data))
    if not mainline_edges:
        raise ValueError(
            f"No mainline edges match ref={ref!r}, direction={direction!r} "
            f"(tolerance {bearing_tolerance}°)"
        )

    # 2. Build the mainline-only subgraph and take its largest weakly-
    #    connected component (drops orphan motorway pieces from neighboring
    #    interchanges that share the ref tag).
    sub = nx.MultiDiGraph()
    sub.graph.update(graph.graph)
    for u, v, key, data in mainline_edges:
        if u not in sub:
            sub.add_node(u, **graph.nodes[u])
        if v not in sub:
            sub.add_node(v, **graph.nodes[v])
        sub.add_edge(u, v, key=key, **data)
    components = sorted(nx.weakly_connected_components(sub), key=len, reverse=True)
    main_nodes = set(components[0])
    sub = sub.subgraph(main_nodes).copy()

    # 3. Order the mainline as a directed path upstream -> downstream. Greedy
    #    walk preferring the dominant carriageway by lanes (then length, then
    #    bearing closeness) so parallel HOV/auxiliary ways sharing the same
    #    ref as the trunk don't trip us up.
    ordered_nodes = _walk_directed_path(sub, target_bearing=target_bearing)

    # 4. Build MainlineSegment list with cumulative-length postmiles. Lengths
    #    use osmnx-tagged ``length`` (meters) when present, else are computed
    #    from the geometry via osmnx's great-circle helper.
    segments: list[MainlineSegment] = []
    pm = 0.0
    for u, v in zip(ordered_nodes, ordered_nodes[1:]):
        # MultiDiGraph: there can be multiple parallel edges; pick the one with
        # bearing closest to target (already filtered to be within tolerance).
        u_attrs, v_attrs = graph.nodes[u], graph.nodes[v]
        key = _pick_best_parallel_edge(sub, u, v, target_bearing)
        data = sub[u][v][key]
        geom = _edge_geometry(data, u_attrs, v_attrs)
        length_m = _edge_length_m(data, geom)
        length_mi = length_m / _METERS_PER_MILE
        lanes = _parse_lanes(data.get("lanes"))
        segments.append(MainlineSegment(
            u=u, v=v, geometry=geom, length=length_mi, lanes=lanes,
            pm_start=pm, pm_end=pm + length_mi,
        ))
        pm += length_mi

    # 5. Identify ramps: motorway_link edges with exactly one endpoint on the
    #    mainline path. The mainline endpoint's postmile is the gore postmile.
    pm_at: dict[int, float] = {ordered_nodes[0]: 0.0}
    for seg in segments:
        pm_at[seg.v] = seg.pm_end
    ramps: list[RampJunction] = []
    for u, v, _key, data in graph.edges(keys=True, data=True):
        if not _is_ramp(data):
            continue
        u_on_main = u in main_nodes
        v_on_main = v in main_nodes
        if u_on_main == v_on_main:
            # Both or neither on mainline -- not a gore we can resolve.
            continue
        if v_on_main:           # ramp enters mainline at v
            kind: Literal["on", "off"] = "on"
            main_node, term_node = v, u
        else:                   # ramp leaves mainline at u
            kind = "off"
            main_node, term_node = u, v
        gore_xy = graph.nodes[main_node]
        ramps.append(RampJunction(
            kind=kind,
            postmile=pm_at[main_node],
            mainline_node=main_node,
            ramp_terminus_node=term_node,
            geometry=Point(gore_xy["x"], gore_xy["y"]),
            lanes=_parse_lanes(data.get("lanes")) or 1,
            # Prefer the ramp's ``name`` ("Hill Street Off-ramp") over its
            # ``ref``, which on a freeway ramp is typically the freeway's own
            # number and is uninformative for ramp identification.
            name=_first_str(data.get("name")) or _first_str(data.get("ref")),
        ))
    ramps.sort(key=lambda r: (r.postmile, r.kind))

    corridor = Corridor(
        ref=ref,
        direction=str(direction),
        target_bearing=target_bearing,
        mainline_segments=segments,
        ramp_junctions=ramps,
        crs=str(graph.graph.get("crs", "EPSG:4326")),
    )

    if pm_range is not None:
        corridor = _crop_pm_range(corridor, pm_range)

    if postmiles is not None:
        # Import locally to avoid pulling geopandas into the import graph for
        # callers that don't use Caltrans postmiles.
        from .caltrans import caltrans_postmiles_for_corridor, load_postmiles
        postmiles_gdf = load_postmiles(postmiles, target_crs=corridor.crs)
        corridor.caltrans_postmiles = caltrans_postmiles_for_corridor(
            corridor, postmiles_gdf, graph,
        )
    return corridor


# ---- Ref discovery --------------------------------------------------------


def summarize_refs(
    graph: nx.MultiDiGraph,
    *,
    highway_types: Iterable[str] = ("motorway",),
) -> pd.DataFrame:
    """Inventory the OSM ``ref`` values in ``graph`` to help users pick one.

    Useful when you've fetched an osmnx graph for an unfamiliar area
    (Oakland, say) and want to know which ``--ref`` strings
    :func:`corridor_from_graph` will accept. By default we count only
    ``highway=motorway`` edges (freeway mainlines); pass ``highway_types``
    to include ramps or surface roads.

    Returns one row per ``ref`` with:

    ============  ============================================================
    column        meaning
    ============  ============================================================
    ref           the OSM ``ref`` string (e.g. ``"I 880"``)
    n_edges       how many edges of the requested highway type(s) carry it
    length_mi     total OSM-tagged length of those edges, in miles
    top_names     up to 3 most-common ``name`` tags on those edges, comma-
                  separated (the human-readable description -- e.g.
                  ``"Nimitz Freeway"`` for I 880)
    lon_min       westernmost lon of the ref's bbox
    lon_max       easternmost lon
    lat_min       southernmost lat
    lat_max       northernmost lat
    ============  ============================================================

    Sorted by ``length_mi`` descending so the longest freeway in the area
    shows up first.
    """
    target_types = set(highway_types)
    by_ref: dict[str, dict] = defaultdict(lambda: {
        "n_edges": 0, "length_m": 0.0,
        "names": Counter(),
        "lon_min": float("inf"), "lon_max": float("-inf"),
        "lat_min": float("inf"), "lat_max": float("-inf"),
    })
    for u, v, _k, data in graph.edges(keys=True, data=True):
        hw = data.get("highway")
        edge_types = set(hw) if isinstance(hw, list) else {hw} if hw is not None else set()
        if not edge_types & target_types:
            continue
        for ref in _iter_refs(data.get("ref")):
            entry = by_ref[ref]
            entry["n_edges"] += 1
            entry["length_m"] += float(data.get("length") or 0.0)
            name = _first_str(data.get("name"))
            if name:
                entry["names"][name] += 1
            for n in (u, v):
                xy = graph.nodes[n]
                x, y = float(xy["x"]), float(xy["y"])
                entry["lon_min"] = min(entry["lon_min"], x)
                entry["lon_max"] = max(entry["lon_max"], x)
                entry["lat_min"] = min(entry["lat_min"], y)
                entry["lat_max"] = max(entry["lat_max"], y)

    rows: list[dict] = []
    for ref, e in by_ref.items():
        rows.append({
            "ref": ref,
            "n_edges": e["n_edges"],
            "length_mi": e["length_m"] / _METERS_PER_MILE,
            "top_names": ", ".join(name for name, _ in e["names"].most_common(3)),
            "lon_min": e["lon_min"], "lon_max": e["lon_max"],
            "lat_min": e["lat_min"], "lat_max": e["lat_max"],
        })
    df = pd.DataFrame(rows, columns=[
        "ref", "n_edges", "length_mi", "top_names",
        "lon_min", "lon_max", "lat_min", "lat_max",
    ])
    return df.sort_values("length_mi", ascending=False, ignore_index=True)


def _iter_refs(value) -> Iterable[str]:
    """Yield each ref string from an OSM ``ref`` tag (list or ``;``-joined)."""
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        candidates = (str(v) for v in value)
    else:
        candidates = str(value).split(";")
    for c in candidates:
        c = c.strip()
        if c:
            yield c


# ---- Helpers --------------------------------------------------------------


def _resolve_target_bearing(direction: str | float) -> float:
    if isinstance(direction, str):
        key = direction.strip().upper()
        if key not in _COMPASS_BEARINGS:
            raise ValueError(
                f"direction must be one of {sorted(_COMPASS_BEARINGS)} or a "
                f"bearing in degrees; got {direction!r}"
            )
        return _COMPASS_BEARINGS[key]
    return float(direction) % 360.0


def _first_str(value) -> Optional[str]:
    """Return the first scalar string in a possibly-list-valued OSM tag."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for v in value:
            if v is not None:
                return str(v)
        return None
    return str(value)


def _matches_highway(data: dict, target: str) -> bool:
    """Tag-aware equality for ``highway`` which osmnx may store as a list."""
    val = data.get("highway")
    if val is None:
        return False
    if isinstance(val, (list, tuple)):
        return target in val
    return val == target


def _is_mainline(data: dict) -> bool:
    return _matches_highway(data, "motorway")


def _is_ramp(data: dict) -> bool:
    return _matches_highway(data, "motorway_link")


def _matches_ref(data: dict, ref: str) -> bool:
    """Match ``ref`` against OSM's ``ref`` which may be list-valued or ``;``-joined."""
    val = data.get("ref")
    if val is None:
        return False
    candidates: Iterable[str]
    if isinstance(val, (list, tuple)):
        candidates = (str(v) for v in val)
    else:
        candidates = str(val).split(";")
    return ref in {c.strip() for c in candidates}


def _parse_lanes(value) -> int:
    """Parse OSM ``lanes`` (str / int / list-of-those) into a single int.

    For list values we take the mode (or first element on tie) so a merged
    edge whose underlying ways have inconsistent lane counts still reduces
    to a representative integer. Returns 0 when the tag is missing or
    unparseable -- callers can decide whether that's tolerable.
    """
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        ints: list[int] = []
        for v in value:
            try:
                ints.append(int(str(v).split(";")[0]))
            except ValueError:
                continue
        if not ints:
            return 0
        # mode via Counter (ties pick the smallest, matching pandas .mode().iloc[0]).
        from collections import Counter
        counts = Counter(ints)
        most = max(counts.values())
        return min(n for n, c in counts.items() if c == most)
    try:
        return int(str(value).split(";")[0])
    except ValueError:
        return 0


def _edge_geometry(data: dict, u_attrs: dict, v_attrs: dict) -> LineString:
    """Return the edge's LineString geometry, synthesizing from node xy if absent."""
    geom = data.get("geometry")
    if isinstance(geom, LineString):
        return geom
    return LineString([(u_attrs["x"], u_attrs["y"]), (v_attrs["x"], v_attrs["y"])])


def _edge_length_m(data: dict, geom: LineString) -> float:
    """Length in meters: prefer osmnx's ``length`` tag, else great-circle of geom."""
    length = data.get("length")
    if length is not None:
        return float(length)
    # great-circle sum across the LineString vertices via osmnx's bearing module
    # (which uses haversine internally; cheap and avoids reprojecting).
    coords = list(geom.coords)
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        total += _haversine_m(lat1, lon1, lat2, lon2)
    return total


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 lat/lon points, in meters."""
    r = 6_371_009.0   # mean Earth radius (m), matches osmnx
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def _edge_bearing(graph: nx.MultiDiGraph, u: int, v: int, data: dict) -> float:
    """Edge bearing from node u to node v (degrees clockwise from north)."""
    bearing = data.get("bearing")
    if bearing is not None:
        return float(bearing)
    u_attrs, v_attrs = graph.nodes[u], graph.nodes[v]
    return float(ox.bearing.calculate_bearing(
        u_attrs["y"], u_attrs["x"], v_attrs["y"], v_attrs["x"],
    ))


def _bearing_close(actual: float, target: float, tolerance: float) -> bool:
    diff = abs((actual - target + 180.0) % 360.0 - 180.0)
    return diff <= tolerance


def _pick_best_parallel_edge(
    sub: nx.MultiDiGraph, u: int, v: int, target_bearing: float
) -> int:
    """Among parallel edges u->v in ``sub``, the one closest to ``target_bearing``."""
    keys = list(sub[u][v])
    if len(keys) == 1:
        return keys[0]
    best_key = keys[0]
    best_diff = float("inf")
    for k in keys:
        b = _edge_bearing(sub, u, v, sub[u][v][k])
        diff = abs((b - target_bearing + 180.0) % 360.0 - 180.0)
        if diff < best_diff:
            best_diff = diff
            best_key = k
    return best_key


def _walk_directed_path(
    sub: nx.MultiDiGraph, *, target_bearing: float,
) -> list[int]:
    """Return the nodes of ``sub`` in upstream→downstream order.

    When the mainline subgraph has multiple sources, multiple sinks, or
    internal branches -- typically caused by OSM-tagged parallel
    carriageways (HOV / express / auxiliary lanes that share the trunk's
    ref) -- the walker picks the dominant route by edge score:

    1. **lanes** -- more lanes wins (HOV stubs are usually 1-2 lanes).
    2. **length** -- when lanes tie, longer continuation wins.
    3. **bearing closeness** to ``target_bearing`` -- final tiebreak so a
       trunk staying on-axis beats a hooking ramp.

    The walk starts at the source with the best outgoing score and follows
    successors greedily until reaching a sink, ignoring lower-scoring
    parallel branches.

    Raises ``ValueError`` only if the subgraph is empty or every node is
    cyclic (no sources at all).
    """
    def in_n(n: int) -> int:
        return len(set(sub.predecessors(n)))

    def out_n(n: int) -> int:
        return len(set(sub.successors(n)))

    sources = [n for n in sub.nodes if in_n(n) == 0]
    if not sources:
        raise ValueError(
            "Mainline subgraph has no source node (likely a cycle); cannot order"
        )

    def best_outgoing(n: int) -> tuple:
        best = (-1, -1.0, -360.0)
        for succ in sub.successors(n):
            for _k, data in sub[n][succ].items():
                s = _edge_walk_score(data, sub, n, succ, target_bearing)
                if s > best:
                    best = s
        return best

    # Start at the source whose outgoing edge looks most like the trunk.
    start = max(sources, key=best_outgoing)

    nodes: list[int] = [start]
    seen = {start}
    cur = start
    while True:
        # Successor candidates: skip those already visited so a parallel-then-
        # rejoining auxiliary path doesn't lure us into an infinite loop.
        candidates = [n for n in sub.successors(cur) if n not in seen]
        if not candidates:
            break  # reached a sink (or all successors already visited)

        def succ_score(succ: int) -> tuple:
            return max(
                _edge_walk_score(data, sub, cur, succ, target_bearing)
                for _k, data in sub[cur][succ].items()
            )

        cur = max(candidates, key=succ_score)
        nodes.append(cur)
        seen.add(cur)
    return nodes


def _edge_walk_score(
    data: dict, sub: nx.MultiDiGraph, u: int, v: int, target_bearing: float,
) -> tuple[int, float, float]:
    """Tuple-ordered score for picking an edge during the corridor walk.

    Returns ``(lanes, length_m, -bearing_delta_deg)`` so ``max(...)`` picks
    the most-lane, longest, on-bearing edge in that order of preference.
    """
    lanes = _parse_lanes(data.get("lanes"))
    length = float(data.get("length") or 0.0)
    bearing = _edge_bearing(sub, u, v, data)
    delta = abs((bearing - target_bearing + 180.0) % 360.0 - 180.0)
    return (lanes, length, -delta)


def _crop_pm_range(corridor: Corridor, pm_range: tuple[float, float]) -> Corridor:
    """Restrict ``corridor`` to mainline whose endpoints lie in ``pm_range``.

    Cells are kept whole (not split mid-segment); a segment is included iff
    both endpoints fall within the range. Ramps inside the kept span are
    carried over; postmiles are *not* re-zeroed -- they stay in the original
    coordinate so callers can join against external data.
    """
    lo, hi = sorted(pm_range)
    kept_segs = [s for s in corridor.mainline_segments
                 if s.pm_start >= lo and s.pm_end <= hi]
    if not kept_segs:
        raise ValueError(
            f"pm_range={pm_range} excludes all mainline segments "
            f"(corridor spans {corridor.pm_start:.3f}-{corridor.pm_end:.3f})"
        )
    pm_min = kept_segs[0].pm_start
    pm_max = kept_segs[-1].pm_end
    kept_ramps = [r for r in corridor.ramp_junctions if pm_min <= r.postmile <= pm_max]
    return Corridor(
        ref=corridor.ref,
        direction=corridor.direction,
        target_bearing=corridor.target_bearing,
        mainline_segments=kept_segs,
        ramp_junctions=kept_ramps,
        crs=corridor.crs,
    )
