"""Tests for the Caltrans postmile integration in
``utils/ctm/caltrans.py`` and the parameters it adds to
``corridor_from_graph`` / ``cells_from_corridor``.

The unit tests build a synthetic in-memory ``MultiDiGraph`` (so we don't need
osmnx file IO) plus a synthetic postmile GeoDataFrame (so we don't ship a
Caltrans shapefile) and assert that the projected postmiles match what we
analytically expect. The I-210 integration test pulls a real osmnx fixture
and a synthetic Caltrans-style GDF anchored to the dissertation's two named
landmarks (Vernon Ave at PM 38.97, SR-134 split at PM 24.94) and checks that
the resulting Caltrans PMs straddle that span.
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pytest
from pathlib import Path
from shapely.geometry import LineString, Point

from transportation_models.utils.ctm.caltrans import (
    extract_route_number,
    load_postmiles,
)
from transportation_models.utils.ctm.cells import cells_from_corridor
from transportation_models.utils.ctm.osm import corridor_from_graph


REPO = Path(__file__).resolve().parents[1]
GRAPHML = REPO / "tests/fixtures/osm/i210_bbox.graphml"


# ---- Helpers --------------------------------------------------------------


def _wgs84_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:4326"
    return g


def _add_node(g, n, lon, lat):
    g.add_node(n, x=lon, y=lat)


def _add_edge(g, u, v, **attrs):
    u_x, u_y = g.nodes[u]["x"], g.nodes[u]["y"]
    v_x, v_y = g.nodes[v]["x"], g.nodes[v]["y"]
    attrs.setdefault("geometry", LineString([(u_x, u_y), (v_x, v_y)]))
    attrs.setdefault("oneway", True)
    attrs.setdefault("osmid", u * 1000 + v)
    g.add_edge(u, v, **attrs)


def _eastbound_graph() -> nx.MultiDiGraph:
    """A straight 4-node eastbound 'I 210' along latitude 34.05, lon increasing."""
    g = _wgs84_graph()
    _add_node(g, 1, -118.40, 34.05)
    _add_node(g, 2, -118.39, 34.05)
    _add_node(g, 3, -118.38, 34.05)
    _add_node(g, 4, -118.37, 34.05)
    _add_edge(g, 1, 2, highway="motorway", ref="I 210", lanes="4")
    _add_edge(g, 2, 3, highway="motorway", ref="I 210", lanes="4")
    _add_edge(g, 3, 4, highway="motorway", ref="I 210", lanes="4")
    return g


def _westbound_graph() -> nx.MultiDiGraph:
    """Same as above but going west: lon decreases along travel."""
    g = _wgs84_graph()
    _add_node(g, 1, -118.37, 34.05)
    _add_node(g, 2, -118.38, 34.05)
    _add_node(g, 3, -118.39, 34.05)
    _add_node(g, 4, -118.40, 34.05)
    _add_edge(g, 1, 2, highway="motorway", ref="I 210", lanes="4")
    _add_edge(g, 2, 3, highway="motorway", ref="I 210", lanes="4")
    _add_edge(g, 3, 4, highway="motorway", ref="I 210", lanes="4")
    return g


def _synthetic_postmiles(node_pms: dict[tuple[float, float], tuple[int, float]]) -> gpd.GeoDataFrame:
    """Build a GDF with one POINT per (lon, lat) -> (Route, PM)."""
    geoms, routes, pms = [], [], []
    for (lon, lat), (route, pm) in node_pms.items():
        geoms.append(Point(lon, lat))
        routes.append(route)
        pms.append(pm)
    return gpd.GeoDataFrame(
        {"Route": routes, "PM": pms, "geometry": geoms},
        crs="EPSG:4326",
    )


# ---- extract_route_number -------------------------------------------------


def test_extract_route_number_handles_common_prefixes():
    assert extract_route_number("I 210") == 210
    assert extract_route_number("CA 134") == 134
    assert extract_route_number("US 101") == 101
    assert extract_route_number("I-880") == 880


def test_extract_route_number_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_route_number("just a name")


# ---- load_postmiles -------------------------------------------------------


def test_load_postmiles_accepts_geodataframe_directly():
    gdf = _synthetic_postmiles({(-118.40, 34.05): (210, 38.0)})
    out = load_postmiles(gdf)
    assert out is not None
    assert "Route" in out.columns and "PM" in out.columns


def test_load_postmiles_reprojects_to_target_crs():
    # Build the GDF in EPSG:3857 (Web Mercator); ask for EPSG:4326.
    gdf = _synthetic_postmiles({(-118.40, 34.05): (210, 38.0)})
    gdf = gdf.to_crs("EPSG:3857")
    out = load_postmiles(gdf, target_crs="EPSG:4326")
    assert str(out.crs).endswith(":4326")
    pt = out.geometry.iloc[0]
    assert pytest.approx(pt.x, abs=1e-4) == -118.40
    assert pytest.approx(pt.y, abs=1e-4) == 34.05


def test_load_postmiles_requires_route_and_pm_columns():
    gdf = gpd.GeoDataFrame(
        {"Route": [210], "geometry": [Point(-118.40, 34.05)]}, crs="EPSG:4326"
    )
    with pytest.raises(ValueError, match="PM"):
        load_postmiles(gdf)


# ---- caltrans_postmiles_for_corridor (via corridor_from_graph) -----------


def test_eastbound_corridor_postmiles_increase_along_travel():
    """For an eastbound corridor, Caltrans PM should increase along travel."""
    g = _eastbound_graph()
    # Anchor PM 38.0 at node 1's location; markers also at nodes 2..4 so the
    # nearest-marker lookup is exact and the adjustment term is zero.
    postmiles = _synthetic_postmiles({
        (-118.40, 34.05): (210, 38.0),
        (-118.39, 34.05): (210, 38.6),
        (-118.38, 34.05): (210, 39.2),
        (-118.37, 34.05): (210, 39.8),
    })
    corridor = corridor_from_graph(
        g, ref="I 210", direction="E", postmiles=postmiles,
    )
    assert corridor.caltrans_postmiles is not None
    pms_in_order = [corridor.caltrans_postmiles[n] for n in [1, 2, 3, 4]]
    assert pms_in_order == [pytest.approx(38.0), pytest.approx(38.6),
                            pytest.approx(39.2), pytest.approx(39.8)]
    # Monotonically increasing along travel.
    assert all(b > a for a, b in zip(pms_in_order, pms_in_order[1:]))


def test_westbound_corridor_postmiles_decrease_along_travel():
    """For westbound, Caltrans PM should DECREASE along travel."""
    g = _westbound_graph()
    # Anchor PM 38.0 at node 1's location (upstream), with PM decreasing
    # westward (mile markers placed at each node).
    postmiles = _synthetic_postmiles({
        (-118.37, 34.05): (210, 38.0),     # node 1, upstream
        (-118.38, 34.05): (210, 37.4),
        (-118.39, 34.05): (210, 36.8),
        (-118.40, 34.05): (210, 36.2),     # node 4, downstream
    })
    corridor = corridor_from_graph(
        g, ref="I 210", direction="W", postmiles=postmiles,
    )
    pms_in_order = [corridor.caltrans_postmiles[n] for n in [1, 2, 3, 4]]
    assert pms_in_order == [pytest.approx(38.0), pytest.approx(37.4),
                            pytest.approx(36.8), pytest.approx(36.2)]
    assert all(b < a for a, b in zip(pms_in_order, pms_in_order[1:]))


def test_postmile_adjustment_kicks_in_for_distant_marker():
    """A node halfway between two markers should land halfway between their PMs."""
    g = _eastbound_graph()
    # Only two markers, at nodes 1 and 4 (the endpoints). Nodes 2 and 3 are
    # interior and rely on the distance-along-mainline adjustment.
    postmiles = _synthetic_postmiles({
        (-118.40, 34.05): (210, 38.0),
        (-118.37, 34.05): (210, 39.8),
    })
    corridor = corridor_from_graph(
        g, ref="I 210", direction="E", postmiles=postmiles,
    )
    # Edge length per node-pair at this latitude is ~0.6 mi, so 4-node total
    # is ~1.8 mi -> PM span is 1.8. Node 2 should be ~1/3 of the way (PM~38.6),
    # node 3 ~2/3 (PM~39.2). Allow a generous tolerance because at lat 34.05
    # the geodesic per 0.01° lon isn't exactly 0.6 mi.
    pms = [corridor.caltrans_postmiles[n] for n in [1, 2, 3, 4]]
    assert pms[0] == pytest.approx(38.0)
    assert pms[-1] == pytest.approx(39.8)
    # Interior PMs should be monotonic between the two anchors.
    assert pms[0] < pms[1] < pms[2] < pms[3]


def test_no_postmile_argument_leaves_caltrans_field_none():
    g = _eastbound_graph()
    corridor = corridor_from_graph(g, ref="I 210", direction="E")
    assert corridor.caltrans_postmiles is None


def test_missing_route_in_postmile_gdf_raises():
    g = _eastbound_graph()
    # Postmile GDF has only Route 5 (I-5), not the requested Route 210.
    postmiles = _synthetic_postmiles({(-118.40, 34.05): (5, 720.0)})
    with pytest.raises(ValueError, match="no postmile markers for Route 210"):
        corridor_from_graph(g, ref="I 210", direction="E", postmiles=postmiles)


# ---- cells_from_corridor pass-through ------------------------------------


def test_cells_get_caltrans_pm_columns_when_postmiles_provided():
    g = _eastbound_graph()
    postmiles = _synthetic_postmiles({
        (-118.40, 34.05): (210, 38.0),
        (-118.37, 34.05): (210, 39.8),
    })
    corridor = corridor_from_graph(g, ref="I 210", direction="E", postmiles=postmiles)
    cells = cells_from_corridor(corridor)
    assert "caltrans_pm_start" in cells.columns
    assert "caltrans_pm_end" in cells.columns
    # First cell starts at the upstream anchor's PM.
    assert cells.iloc[0]["caltrans_pm_start"] == pytest.approx(38.0)
    # Last cell ends at the downstream anchor's PM.
    assert cells.iloc[-1]["caltrans_pm_end"] == pytest.approx(39.8)


def test_cells_without_postmiles_skip_caltrans_columns():
    g = _eastbound_graph()
    corridor = corridor_from_graph(g, ref="I 210", direction="E")
    cells = cells_from_corridor(corridor)
    assert "caltrans_pm_start" not in cells.columns
    assert "caltrans_pm_end" not in cells.columns


# ---- Integration on the I-210 fixture ------------------------------------


@pytest.fixture(scope="module")
def i210_graph():
    if not GRAPHML.exists():
        pytest.skip(f"I-210 OSM fixture not found at {GRAPHML}")
    return ox.load_graphml(GRAPHML)


def _i210_dissertation_postmiles() -> gpd.GeoDataFrame:
    """Two-point Caltrans-style fixture anchoring the dissertation's named landmarks.

    Vernon Ave / I-210 interchange (Glendora): PM 38.97 at approx (-117.864, 34.137)
    I-210 / SR-134 split (Pasadena):           PM 24.94 at approx (-118.149, 34.150)
    """
    return _synthetic_postmiles({
        (-117.864, 34.137): (210, 38.97),
        (-118.149, 34.150): (210, 24.94),
    })


def test_i210w_caltrans_postmiles_span_dissertation_range(i210_graph):
    corridor = corridor_from_graph(
        i210_graph, ref="I 210", direction="W",
        postmiles=_i210_dissertation_postmiles(),
    )
    assert corridor.caltrans_postmiles is not None
    pms = list(corridor.caltrans_postmiles.values())
    # WB I-210 -> PMs should decrease along travel; range should cover
    # roughly the dissertation's PM 24-39 span.
    assert max(pms) > 35.0, f"max PM {max(pms):.2f} unexpectedly low"
    assert min(pms) < 28.0, f"min PM {min(pms):.2f} unexpectedly high"
    # Travel direction (upstream -> downstream) -> PMs strictly decreasing.
    seq = [corridor.caltrans_postmiles[corridor.mainline_segments[0].u]] + [
        corridor.caltrans_postmiles[seg.v] for seg in corridor.mainline_segments
    ]
    decreasing = sum(1 for a, b in zip(seq, seq[1:]) if b < a)
    assert decreasing / max(1, len(seq) - 1) > 0.9, (
        f"only {decreasing}/{len(seq)-1} consecutive node pairs have decreasing PM"
    )


def test_i210w_caltrans_pm_total_close_to_dissertation(i210_graph):
    """Sum of Caltrans PM deltas along the corridor should match CTMSIM's 14.03 mi."""
    corridor = corridor_from_graph(
        i210_graph, ref="I 210", direction="W",
        postmiles=_i210_dissertation_postmiles(),
    )
    nodes = [corridor.mainline_segments[0].u] + [
        seg.v for seg in corridor.mainline_segments
    ]
    pms = np.array([corridor.caltrans_postmiles[n] for n in nodes])
    total_pm_delta = abs(pms[0] - pms[-1])
    # Dissertation reports 14.03 mi from Vernon to SR-134. Our bbox crops
    # slightly tighter; allow generous tolerance.
    assert 11.0 < total_pm_delta < 15.5, (
        f"Caltrans PM total delta {total_pm_delta:.3f} mi outside expected range"
    )
