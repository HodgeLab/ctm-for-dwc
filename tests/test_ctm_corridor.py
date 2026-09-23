"""Tests for ``utils/ctm/osm.py``: extracting a CTM Corridor from an osmnx graph.

The tests build small in-memory ``MultiDiGraph`` instances that look like
what ``osmnx.simplify_graph`` would emit -- WGS84 node coordinates, OSM
``highway``/``ref``/``lanes`` tags on the edges -- so the test surface
stays focused on :func:`corridor_from_graph`'s own logic (ref + bearing
filter, single-path ordering, postmile assignment, ramp attachment) and
doesn't depend on Overpass.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from ctm_for_dwc.utils.ctm.osm import (
    Corridor,
    MainlineSegment,
    RampJunction,
    corridor_from_graph,
)


# ---- Fixture builder ------------------------------------------------------


def _add_node(g: nx.MultiDiGraph, n: int, lon: float, lat: float) -> None:
    g.add_node(n, x=lon, y=lat)


def _add_edge(
    g: nx.MultiDiGraph,
    u: int,
    v: int,
    *,
    highway: str,
    ref: Optional[str] = None,
    lanes: Optional[int] = None,
    name: Optional[str] = None,
) -> None:
    """Add a single-direction edge with the geometry implied by node xy."""
    u_x, u_y = g.nodes[u]["x"], g.nodes[u]["y"]
    v_x, v_y = g.nodes[v]["x"], g.nodes[v]["y"]
    attrs: dict = {
        "highway": highway,
        "oneway": True,
        "geometry": LineString([(u_x, u_y), (v_x, v_y)]),
        "osmid": (u * 1000 + v),
    }
    if ref is not None:
        attrs["ref"] = ref
    if lanes is not None:
        attrs["lanes"] = str(lanes)
    if name is not None:
        attrs["name"] = name
    g.add_edge(u, v, **attrs)


def _empty_wgs84_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:4326"
    return g


def _build_westbound_freeway() -> nx.MultiDiGraph:
    """A 4-segment westbound 'I 210' with one on-ramp, one off-ramp, one lane drop.

    Layout (along a constant latitude, longitudes decreasing -> westward):
        1 --motorway--> 2 --motorway--> 3 --motorway--> 4 --motorway--> 5
        10 --motorway_link--> 2          on-ramp at PM ~0.3
        3 --motorway_link--> 20          off-ramp at PM ~0.6
        4 has fewer lanes than 1..3 (lane drop)
    Plus a *northbound* mainline edge that shares ref="I 210" -- it must be
    filtered out by the direction filter so we know the corridor selection
    works on a real bidirectional graph.
    """
    g = _empty_wgs84_graph()
    # mainline nodes along latitude 34.05, 0.01° lon apart (~0.6 mi each).
    _add_node(g, 1, -118.40, 34.05)
    _add_node(g, 2, -118.41, 34.05)
    _add_node(g, 3, -118.42, 34.05)
    _add_node(g, 4, -118.43, 34.05)
    _add_node(g, 5, -118.44, 34.05)
    # ramp termini -- placed off-corridor so their bearings differ from mainline.
    _add_node(g, 10, -118.405, 34.04)   # on-ramp terminus (south of mainline)
    _add_node(g, 20, -118.425, 34.06)   # off-ramp terminus (north of mainline)
    # opposite-direction mainline node -- shares ref but bearing ~90° (eastbound)
    _add_node(g, 100, -118.40, 34.06)
    _add_node(g, 101, -118.39, 34.06)

    # westbound mainline (bearing ~270°)
    _add_edge(g, 1, 2, highway="motorway", ref="I 210", lanes=4)
    _add_edge(g, 2, 3, highway="motorway", ref="I 210", lanes=4)
    _add_edge(g, 3, 4, highway="motorway", ref="I 210", lanes=4)
    _add_edge(g, 4, 5, highway="motorway", ref="I 210", lanes=3)   # lane drop
    # ramps
    _add_edge(g, 10, 2, highway="motorway_link", ref="I 210", lanes=1, name="Hill On")
    _add_edge(g, 3, 20, highway="motorway_link", ref="I 210", lanes=1, name="Allen Off")
    # eastbound mainline that shares ref -- bearing ~90°, should be filtered out
    _add_edge(g, 100, 101, highway="motorway", ref="I 210", lanes=4)
    return g


# ---- Tests ----------------------------------------------------------------


def test_basic_westbound_corridor_structure():
    g = _build_westbound_freeway()
    corridor = corridor_from_graph(g, ref="I 210", direction="W")

    assert isinstance(corridor, Corridor)
    assert corridor.ref == "I 210"
    assert corridor.direction == "W"
    assert corridor.target_bearing == 270.0
    # 4 mainline segments: 1->2, 2->3, 3->4, 4->5.
    assert corridor.n_mainline == 4
    # And only those: the eastbound 100->101 motorway must be filtered out.
    main_nodes = [seg.u for seg in corridor.mainline_segments] + [
        corridor.mainline_segments[-1].v
    ]
    assert main_nodes == [1, 2, 3, 4, 5]


def test_segments_are_ordered_and_postmiles_monotonic():
    corridor = corridor_from_graph(
        _build_westbound_freeway(), ref="I 210", direction="W"
    )
    pm_ends = [seg.pm_end for seg in corridor.mainline_segments]
    assert pm_ends == sorted(pm_ends)
    # pm_start of segment k+1 must equal pm_end of segment k.
    for prev, cur in zip(corridor.mainline_segments, corridor.mainline_segments[1:]):
        assert cur.pm_start == pytest.approx(prev.pm_end, abs=1e-9)
    # First segment starts at zero.
    assert corridor.mainline_segments[0].pm_start == 0.0
    # 0.01° of longitude at lat=34° is ~0.575 mi; total ~2.3 mi for 4 segments.
    assert corridor.total_length == pytest.approx(2.3, abs=0.05)


def test_ramps_identified_with_correct_kind_and_postmile():
    corridor = corridor_from_graph(
        _build_westbound_freeway(), ref="I 210", direction="W"
    )
    by_kind = {r.kind: r for r in corridor.ramp_junctions}
    assert set(by_kind) == {"on", "off"}

    on = by_kind["on"]
    assert on.mainline_node == 2
    assert on.ramp_terminus_node == 10
    # On-ramp attaches at node 2: postmile = length of segment 1->2.
    seg_1_2 = corridor.mainline_segments[0]
    assert on.postmile == pytest.approx(seg_1_2.pm_end, abs=1e-9)
    assert on.name == "Hill On"

    off = by_kind["off"]
    assert off.mainline_node == 3
    assert off.ramp_terminus_node == 20
    seg_2_3 = corridor.mainline_segments[1]
    assert off.postmile == pytest.approx(seg_2_3.pm_end, abs=1e-9)
    assert off.name == "Allen Off"


def test_lane_change_postmile_detected():
    corridor = corridor_from_graph(
        _build_westbound_freeway(), ref="I 210", direction="W"
    )
    lane_changes = corridor.lane_change_postmiles()
    # 4 segments: lanes = 4, 4, 4, 3 -> one change at start of segment index 3.
    assert len(lane_changes) == 1
    expected_pm = corridor.mainline_segments[2].pm_end
    assert lane_changes[0] == pytest.approx(expected_pm, abs=1e-9)


def test_pm_range_crop():
    corridor = corridor_from_graph(
        _build_westbound_freeway(), ref="I 210", direction="W"
    )
    # Drop the first and last segment by cropping to (pm_start of seg 1, pm_end of seg 2).
    lo = corridor.mainline_segments[1].pm_start
    hi = corridor.mainline_segments[2].pm_end
    cropped = corridor_from_graph(
        _build_westbound_freeway(), ref="I 210", direction="W",
        pm_range=(lo, hi),
    )
    assert cropped.n_mainline == 2
    assert cropped.pm_start == pytest.approx(lo, abs=1e-9)
    assert cropped.pm_end == pytest.approx(hi, abs=1e-9)
    # Ramps at node 2 (on, pm = lo) and node 3 (off, pm = hi) both lie on the
    # boundary, so both are retained.
    assert {r.kind for r in cropped.ramp_junctions} == {"on", "off"}


def test_numeric_direction_accepted():
    """A numeric bearing (clockwise from north) should work like the compass label."""
    g = _build_westbound_freeway()
    by_label = corridor_from_graph(g, ref="I 210", direction="W")
    by_number = corridor_from_graph(g, ref="I 210", direction=270.0)
    assert by_label.n_mainline == by_number.n_mainline
    assert by_label.target_bearing == by_number.target_bearing


def test_unknown_ref_raises():
    g = _build_westbound_freeway()
    with pytest.raises(ValueError, match="No mainline edges"):
        corridor_from_graph(g, ref="US 101", direction="W")


def test_invalid_direction_string_raises():
    g = _build_westbound_freeway()
    with pytest.raises(ValueError, match="direction must be one of"):
        corridor_from_graph(g, ref="I 210", direction="NW")


def test_wrong_direction_filters_everything_out():
    """Asking for the eastbound carriageway should pick up only edge 100->101."""
    g = _build_westbound_freeway()
    corridor = corridor_from_graph(g, ref="I 210", direction="E")
    # Eastbound stub is a single edge; corridor has one segment, no ramps.
    assert corridor.n_mainline == 1
    assert corridor.mainline_segments[0].u == 100
    assert corridor.mainline_segments[0].v == 101
    assert corridor.ramp_junctions == []


def test_branching_mainline_follows_target_bearing():
    """At a branch with equal lanes, the walker should follow the successor
    whose bearing is closest to the target direction.

    Node 2 has two equal-lane successors: westward to node 3 (bearing ~270°,
    matches direction='W') and south to node 4 (bearing ~180°). The walker
    should pick 2→3.
    """
    g = _empty_wgs84_graph()
    _add_node(g, 1, -118.40, 34.05)
    _add_node(g, 2, -118.41, 34.05)
    _add_node(g, 3, -118.42, 34.05)
    _add_node(g, 4, -118.42, 34.04)
    _add_edge(g, 1, 2, highway="motorway", ref="I 210", lanes=4)
    _add_edge(g, 2, 3, highway="motorway", ref="I 210", lanes=4)
    _add_edge(g, 2, 4, highway="motorway", ref="I 210", lanes=4)  # branch off-axis
    corridor = corridor_from_graph(g, ref="I 210", direction="W")
    main_nodes = [seg.u for seg in corridor.mainline_segments] + [
        corridor.mainline_segments[-1].v
    ]
    assert main_nodes == [1, 2, 3]


def test_parallel_carriageway_sinks_get_collapsed():
    """A parallel HOV / auxiliary mainline that ends at the bbox edge alongside
    the main carriageway should not derail extraction.

    The Hayward I-880 N case: the bbox truncates two parallel motorway-tagged
    ways at the north edge -- the through carriageway (4-3 lanes) and a
    short auxiliary lane (1 lane) -- whose downstream ends are ~10 m apart.
    The walker should pick the lane-rich carriageway and ignore the aux stub.
    """
    g = _empty_wgs84_graph()
    # Through carriageway: 1 -> 2 -> 3 -> 4_main, 4 lanes throughout.
    _add_node(g, 1, -122.0447, 37.5699)
    _add_node(g, 2, -122.0500, 37.5800)
    _add_node(g, 3, -122.0700, 37.6100)
    _add_node(g, 4, -122.0761, 37.6202)
    _add_edge(g, 1, 2, highway="motorway", ref="I 880", lanes=4)
    _add_edge(g, 2, 3, highway="motorway", ref="I 880", lanes=4)
    _add_edge(g, 3, 4, highway="motorway", ref="I 880", lanes=4)
    # Parallel auxiliary lane branching off near node 3, ending ~10 m from
    # node 4 (the bbox-edge truncation effect).
    _add_node(g, 5, -122.0762, 37.6202)
    _add_edge(g, 3, 5, highway="motorway", ref="I 880", lanes=1)

    corridor = corridor_from_graph(g, ref="I 880", direction="N")
    # Mainline should be 1->2->3->4 (the high-lane carriageway), not the
    # 1->2->3->5 aux branch.
    main_nodes = [seg.u for seg in corridor.mainline_segments] + [
        corridor.mainline_segments[-1].v
    ]
    assert main_nodes == [1, 2, 3, 4]
    assert {s.lanes for s in corridor.mainline_segments} == {4}


def test_ramp_without_ref_still_identified():
    """A motorway_link without a matching ref still counts as a ramp."""
    g = _build_westbound_freeway()
    # Strip the ramp's ref tag (an osmnx graph after simplification may have
    # ramps with `ref` missing because the source/destination road is the only
    # ref-bearing piece).
    for u, v, k, data in list(g.edges(keys=True, data=True)):
        if data.get("highway") == "motorway_link":
            data.pop("ref", None)
    corridor = corridor_from_graph(g, ref="I 210", direction="W")
    kinds = sorted(r.kind for r in corridor.ramp_junctions)
    assert kinds == ["off", "on"]


def test_lanes_tag_can_be_list():
    """``lanes`` may be a list after osmnx simplify_graph -- handle gracefully."""
    g = _empty_wgs84_graph()
    _add_node(g, 1, -118.40, 34.05)
    _add_node(g, 2, -118.41, 34.05)
    _add_node(g, 3, -118.42, 34.05)
    g.add_edge(1, 2, highway="motorway", ref="I 210", lanes=["4", "4"],
               geometry=LineString([(-118.40, 34.05), (-118.41, 34.05)]),
               oneway=True, osmid=(1002,))
    g.add_edge(2, 3, highway="motorway", ref="I 210", lanes=["3", "4"],
               geometry=LineString([(-118.41, 34.05), (-118.42, 34.05)]),
               oneway=True, osmid=(2003,))
    corridor = corridor_from_graph(g, ref="I 210", direction="W")
    # Modes: [4] -> 4 for first segment, [3, 4] (tie) -> min = 3 for second.
    assert corridor.mainline_segments[0].lanes == 4
    assert corridor.mainline_segments[1].lanes == 3
