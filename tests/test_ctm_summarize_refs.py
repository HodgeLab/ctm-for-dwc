"""Tests for ``summarize_refs`` -- the ``ref``-discovery helper used to figure
out the right ``--ref`` for an unfamiliar region.

Unit tests use small in-memory graphs so the logic is exercised independently;
an integration test runs against the cached I-210 OSM fixture.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import osmnx as ox
import pytest
from shapely.geometry import LineString

from ctm_for_dwc.utils.ctm.osm import summarize_refs

REPO = Path(__file__).resolve().parents[1]
GRAPHML = REPO / "tests/fixtures/osm/i210_bbox.graphml"


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
    attrs.setdefault("length", 921.3)   # ~0.01° lon at lat 34, in meters
    attrs.setdefault("oneway", True)
    attrs.setdefault("osmid", u * 1000 + v)
    g.add_edge(u, v, **attrs)


def _two_freeway_graph() -> nx.MultiDiGraph:
    """A graph with three refs: 'I 880' (long), 'I 580' (medium), one ramp."""
    g = _wgs84_graph()
    # I 880 mainline -- 3 edges.
    _add_node(g, 1, -122.30, 37.75)
    _add_node(g, 2, -122.29, 37.75)
    _add_node(g, 3, -122.28, 37.75)
    _add_node(g, 4, -122.27, 37.75)
    for u, v in [(1, 2), (2, 3), (3, 4)]:
        _add_edge(g, u, v, highway="motorway", ref="I 880", name="Nimitz Freeway")

    # I 580 mainline -- 2 edges.
    _add_node(g, 5, -122.25, 37.80)
    _add_node(g, 6, -122.24, 37.80)
    _add_node(g, 7, -122.23, 37.80)
    for u, v in [(5, 6), (6, 7)]:
        _add_edge(g, u, v, highway="motorway", ref="I 580", name="MacArthur Freeway")

    # An on-ramp tagged as motorway_link (should be excluded by default).
    _add_node(g, 8, -122.305, 37.745)
    _add_edge(g, 8, 1, highway="motorway_link", ref="I 880")
    return g


# ---- Unit tests -----------------------------------------------------------


def test_summarize_refs_lists_each_ref_once():
    df = summarize_refs(_two_freeway_graph())
    assert sorted(df["ref"]) == ["I 580", "I 880"]


def test_summarize_refs_sorted_by_length_descending():
    df = summarize_refs(_two_freeway_graph())
    # I 880 has 3 edges, I 580 has 2 -> I 880 first.
    assert df["ref"].iloc[0] == "I 880"
    assert df["ref"].iloc[1] == "I 580"
    assert df["length_mi"].iloc[0] > df["length_mi"].iloc[1]


def test_summarize_refs_carries_human_readable_top_names():
    df = summarize_refs(_two_freeway_graph())
    row880 = df[df["ref"] == "I 880"].iloc[0]
    row580 = df[df["ref"] == "I 580"].iloc[0]
    assert "Nimitz Freeway" in row880["top_names"]
    assert "MacArthur Freeway" in row580["top_names"]


def test_summarize_refs_excludes_ramps_by_default():
    """The motorway_link ramp from node 8 to node 1 should not bump I 880's
    edge count when ``highway_types`` defaults to mainline only."""
    df = summarize_refs(_two_freeway_graph())
    row880 = df[df["ref"] == "I 880"].iloc[0]
    assert row880["n_edges"] == 3   # mainline only, not 3 + 1 ramp


def test_summarize_refs_includes_ramps_when_requested():
    df = summarize_refs(
        _two_freeway_graph(),
        highway_types=("motorway", "motorway_link"),
    )
    row880 = df[df["ref"] == "I 880"].iloc[0]
    assert row880["n_edges"] == 4   # 3 mainline + 1 ramp


def test_summarize_refs_returns_bbox_per_ref():
    df = summarize_refs(_two_freeway_graph())
    row880 = df[df["ref"] == "I 880"].iloc[0]
    assert row880["lon_min"] == pytest.approx(-122.30)
    assert row880["lon_max"] == pytest.approx(-122.27)
    row580 = df[df["ref"] == "I 580"].iloc[0]
    assert row580["lat_min"] == pytest.approx(37.80)


def test_summarize_refs_handles_list_valued_ref_tag():
    """osmnx may collapse multi-ref edges into a list -- each ref should count."""
    g = _wgs84_graph()
    _add_node(g, 1, -118.40, 34.05)
    _add_node(g, 2, -118.39, 34.05)
    _add_edge(g, 1, 2, highway="motorway", ref=["I 210", "CA 134"], name="X")
    df = summarize_refs(g)
    assert set(df["ref"]) == {"I 210", "CA 134"}


def test_summarize_refs_returns_empty_frame_when_no_motorway():
    g = _wgs84_graph()
    _add_node(g, 1, 0, 0)
    _add_node(g, 2, 1, 0)
    _add_edge(g, 1, 2, highway="primary", ref="X")  # not a motorway
    df = summarize_refs(g)
    assert df.empty


# ---- Integration on the I-210 OSM fixture --------------------------------


@pytest.fixture(scope="module")
def i210_graph():
    if not GRAPHML.exists():
        pytest.skip(f"I-210 OSM fixture not found at {GRAPHML}")
    return ox.load_graphml(GRAPHML)


def test_summarize_refs_finds_i_210_with_freeway_name(i210_graph):
    df = summarize_refs(i210_graph)
    refs = set(df["ref"])
    assert "I 210" in refs
    i210 = df[df["ref"] == "I 210"].iloc[0]
    # I-210 should be the longest motorway in this bbox.
    assert i210["ref"] == df["ref"].iloc[0]
    # OSM tags the I-210 mainline as "Foothill Freeway".
    assert "Foothill Freeway" in i210["top_names"]
