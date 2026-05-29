"""Caltrans State Highway Network postmile lookups for a :class:`Corridor`.

Caltrans publishes a SHN postmile dataset whose primary form is a GeoDataFrame
of POINT geometries -- one per posted postmile marker -- with at least these
columns:

* ``Route`` (int)   -- the state-numbered route (e.g. 210, 134, 101).
* ``PM`` (float)    -- the postmile value at this marker.

Caltrans postmiles run in a route-specific direction (typically increasing
from the southern / western terminus), so for a westbound or southbound
corridor they will *decrease* along the direction of travel. The lookup
helpers here preserve that sign convention: callers downstream (e.g. the
golden tests that compare against the dissertation's PM 24-39 span on
I-210W) can compare directly.

The integration with :func:`corridor_from_graph` is purely additive: when
``postmiles`` is supplied, each mainline node gets a Caltrans postmile
attached to :attr:`Corridor.caltrans_postmiles`; the corridor's existing
cumulative-length ``pm_start``/``pm_end`` are unchanged so downstream code
that doesn't care about Caltrans coordinates keeps working as-is.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Union

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point

from .osm import Corridor, MainlineSegment, _haversine_m, _METERS_PER_MILE

PostmileSource = Union[str, Path, gpd.GeoDataFrame]


def extract_route_number(ref: str) -> int:
    """Parse the numeric route from an OSM ``ref`` like ``"I 210"`` or ``"CA 134"``.

    Raises ``ValueError`` if no digit run is present.
    """
    m = re.search(r"\d+", str(ref))
    if not m:
        raise ValueError(f"could not parse a route number from ref={ref!r}")
    return int(m.group(0))


def load_postmiles(source: PostmileSource, *, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Load a SHN postmile GeoDataFrame and reproject to ``target_crs``.

    Accepts either a path to a shapefile / geopackage / geojson, or a
    pre-loaded GeoDataFrame. Output is in ``target_crs`` (defaults to
    EPSG:4326, the osmnx default for graphs).
    """
    if isinstance(source, gpd.GeoDataFrame):
        gdf = source
    else:
        gdf = gpd.read_file(source)
    if gdf.crs is None:
        raise ValueError("postmile GeoDataFrame must declare a CRS")
    if str(gdf.crs).lower() != target_crs.lower():
        gdf = gdf.to_crs(target_crs)
    for col in ("Route", "PM"):
        if col not in gdf.columns:
            raise ValueError(f"postmile GeoDataFrame missing required column {col!r}")
    return gdf


def caltrans_postmiles_for_corridor(
    corridor: Corridor,
    postmiles: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
) -> dict[int, float]:
    """Assign a Caltrans postmile to every mainline node in ``corridor``.

    For each mainline node we find the nearest postmile *marker* of the
    corridor's route, project both onto the mainline polyline, and report
    ``marker_PM ± (distance_marker_to_node_along_mainline / 1609.344 mi)``.
    The sign of the adjustment is set so that postmiles flow monotonically
    along the corridor's travel direction.

    The lookup is O(nodes × markers) -- fine at corridor scale (tens of
    nodes, low hundreds of markers per route).
    """
    route = extract_route_number(corridor.ref)
    route_markers = postmiles.loc[postmiles["Route"] == route].copy()
    if route_markers.empty:
        raise ValueError(
            f"no postmile markers for Route {route} (corridor ref={corridor.ref!r})"
        )

    # Pull marker xy and PM out as plain arrays for vectorized nearest-neighbor.
    marker_xy = np.column_stack(
        [route_markers.geometry.x.to_numpy(), route_markers.geometry.y.to_numpy()]
    )
    marker_pm = route_markers["PM"].to_numpy(dtype=float)

    nodes_in_order = _mainline_nodes_in_order(corridor)
    pm_at: dict[int, float] = {}

    # Cumulative length (in mi) for each mainline node, indexed by position in
    # ``nodes_in_order``. Used to compute "along the corridor" distance between
    # any two mainline nodes.
    cum_mi = _cumulative_lengths(corridor)
    node_idx = {n: i for i, n in enumerate(nodes_in_order)}

    # Sample the corridor's polyline at each mainline node for projection.
    for n in nodes_in_order:
        node_xy = graph.nodes[n]
        nx_, ny_ = node_xy["x"], node_xy["y"]
        # Nearest marker (planar distance in degrees is fine for argmin on the
        # small scale of one route corridor).
        dx = marker_xy[:, 0] - nx_
        dy = marker_xy[:, 1] - ny_
        j = int(np.argmin(dx * dx + dy * dy))
        mx, my = float(marker_xy[j, 0]), float(marker_xy[j, 1])
        base_pm = float(marker_pm[j])

        # Find the corridor's nearest-along-mainline node to the marker and use
        # *that* node's cumulative length as the marker's "where on the
        # corridor it lives" reference. The signed difference between this
        # node and the marker's reference is the postmile adjustment.
        ref_node_idx = _nearest_mainline_idx_to_xy(graph, nodes_in_order, mx, my)
        delta_mi_along = cum_mi[node_idx[n]] - cum_mi[ref_node_idx]
        sign = _route_sign(corridor)
        pm_at[n] = base_pm + sign * delta_mi_along

    return pm_at


# ---- Internals ------------------------------------------------------------


def _mainline_nodes_in_order(corridor: Corridor) -> list[int]:
    """Mainline node IDs ordered upstream → downstream."""
    if not corridor.mainline_segments:
        return []
    seq = [corridor.mainline_segments[0].u]
    seq += [seg.v for seg in corridor.mainline_segments]
    return seq


def _cumulative_lengths(corridor: Corridor) -> list[float]:
    """Cumulative mi from upstream end, parallel to ``_mainline_nodes_in_order``."""
    cum = [0.0]
    for seg in corridor.mainline_segments:
        cum.append(cum[-1] + seg.length)
    return cum


def _nearest_mainline_idx_to_xy(
    graph: nx.MultiDiGraph, nodes_in_order: Iterable[int], x: float, y: float
) -> int:
    """Index (into ``nodes_in_order``) of the mainline node nearest to (x, y)."""
    best = 0
    best_d = float("inf")
    for i, n in enumerate(nodes_in_order):
        nx_, ny_ = graph.nodes[n]["x"], graph.nodes[n]["y"]
        d = (nx_ - x) ** 2 + (ny_ - y) ** 2
        if d < best_d:
            best_d = d
            best = i
    return best


def _route_sign(corridor: Corridor) -> int:
    """+1 if Caltrans PM increases along travel; -1 if it decreases.

    Caltrans postmiles run south-to-north or west-to-east along each route,
    so for north- or east-running corridors the corridor's travel direction
    agrees with PM (+1); for south or west, it opposes (-1). Picks the sign
    from the corridor's target bearing (``[0, 180)`` ⇒ N/E component ⇒ +1).
    Routes with non-standard PM direction (CA 1 along the bending coast,
    e.g.) would need an explicit override -- not yet supported.
    """
    return 1 if 0.0 <= corridor.target_bearing < 180.0 else -1
