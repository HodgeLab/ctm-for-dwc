"""Unit tests for ``utils/ctm/cells.py``: cell layout along a Corridor.

These tests bypass osmnx entirely and construct :class:`Corridor` instances
by hand so the surface under test is just :func:`cells_from_corridor`'s
breakpoint logic.
"""

from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from transportation_models.utils.ctm.cells import (
    cells_from_corridor,
    cells_to_geodataframe,
    ramp_junctions_to_geodataframe,
)
from transportation_models.utils.ctm.osm import (
    Corridor,
    MainlineSegment,
    RampJunction,
)


# ---- Fixture builders -----------------------------------------------------


def _seg(pm_start: float, pm_end: float, lanes: int) -> MainlineSegment:
    """A MainlineSegment with synthetic coords; only postmiles/lanes matter here."""
    return MainlineSegment(
        u=int(pm_start * 1000),
        v=int(pm_end * 1000),
        geometry=LineString([(0.0, 0.0), (1.0, 0.0)]),  # placeholder
        length=pm_end - pm_start,
        lanes=lanes,
        pm_start=pm_start,
        pm_end=pm_end,
    )


def _ramp(kind: str, postmile: float, name: str | None = None) -> RampJunction:
    return RampJunction(
        kind=kind,                              # type: ignore[arg-type]
        postmile=postmile,
        mainline_node=int(postmile * 1000),
        ramp_terminus_node=-1,
        geometry=Point(0.0, 0.0),
        lanes=1,
        name=name,
    )


def _corridor(
    segs: list[MainlineSegment],
    ramps: list[RampJunction] | None = None,
) -> Corridor:
    return Corridor(
        ref="I 210",
        direction="W",
        target_bearing=270.0,
        mainline_segments=segs,
        ramp_junctions=ramps or [],
    )


# ---- Tests ----------------------------------------------------------------


def test_no_ramps_no_lane_changes_yields_a_single_cell():
    corridor = _corridor([_seg(0.0, 1.0, 4)])
    cells = cells_from_corridor(corridor)
    assert len(cells) == 1
    row = cells.iloc[0]
    assert row.pm_start == 0.0
    assert row.pm_end == 1.0
    assert row.length == pytest.approx(1.0)
    assert row.lanes == 4
    assert not row.on_ramp
    assert not row.off_ramp
    assert row.on_ramp_name is None
    assert row.off_ramp_name is None


def test_on_ramp_in_the_middle_creates_two_cells():
    """An on-ramp at PM 0.5 splits a 1-mile segment; cell 1 owns the on-ramp."""
    corridor = _corridor(
        [_seg(0.0, 1.0, 4)],
        ramps=[_ramp("on", 0.5, name="Hill")],
    )
    cells = cells_from_corridor(corridor)
    assert list(cells["pm_start"]) == [0.0, 0.5]
    assert list(cells["pm_end"]) == [0.5, 1.0]
    assert list(cells["on_ramp"]) == [False, True]
    assert list(cells["off_ramp"]) == [False, False]
    assert cells.iloc[1].on_ramp_name == "Hill"


def test_off_ramp_in_the_middle_creates_two_cells():
    """An off-ramp at PM 0.5; cell 0 owns the off-ramp (gore is its pm_end)."""
    corridor = _corridor(
        [_seg(0.0, 1.0, 4)],
        ramps=[_ramp("off", 0.5, name="Allen")],
    )
    cells = cells_from_corridor(corridor)
    assert list(cells["pm_end"]) == [0.5, 1.0]
    assert list(cells["off_ramp"]) == [True, False]
    assert list(cells["on_ramp"]) == [False, False]
    assert cells.iloc[0].off_ramp_name == "Allen"


def test_ramp_at_corridor_upstream_end_belongs_to_first_cell():
    """An on-ramp at the very first postmile lives in the first cell."""
    corridor = _corridor(
        [_seg(0.0, 1.0, 4)],
        ramps=[_ramp("on", 0.0, name="UpstreamOn")],
    )
    cells = cells_from_corridor(corridor)
    assert len(cells) == 1
    assert cells.iloc[0].on_ramp
    assert cells.iloc[0].on_ramp_name == "UpstreamOn"


def test_ramp_at_corridor_downstream_end_belongs_to_last_cell():
    """An off-ramp at the very last postmile lives in the last cell."""
    corridor = _corridor(
        [_seg(0.0, 1.0, 4)],
        ramps=[_ramp("off", 1.0, name="DownstreamOff")],
    )
    cells = cells_from_corridor(corridor)
    assert len(cells) == 1
    assert cells.iloc[0].off_ramp
    assert cells.iloc[0].off_ramp_name == "DownstreamOff"


def test_on_and_off_at_same_postmile_split_across_adjacent_cells():
    """On + off at the same gore: off goes to cell upstream, on goes to cell downstream."""
    corridor = _corridor(
        [_seg(0.0, 1.0, 4)],
        ramps=[
            _ramp("off", 0.5, name="OffAtJunction"),
            _ramp("on", 0.5, name="OnAtJunction"),
        ],
    )
    cells = cells_from_corridor(corridor)
    assert len(cells) == 2
    # Upstream cell: ends at the off-ramp.
    assert cells.iloc[0].off_ramp and not cells.iloc[0].on_ramp
    assert cells.iloc[0].off_ramp_name == "OffAtJunction"
    # Downstream cell: starts at the on-ramp.
    assert cells.iloc[1].on_ramp and not cells.iloc[1].off_ramp
    assert cells.iloc[1].on_ramp_name == "OnAtJunction"


def test_lane_change_creates_a_break_even_without_ramps():
    """A mid-corridor lane drop produces a cell boundary."""
    corridor = _corridor([
        _seg(0.0, 1.0, 4),
        _seg(1.0, 2.0, 3),     # lane drop at PM 1.0
    ])
    cells = cells_from_corridor(corridor)
    assert list(cells["pm_start"]) == [0.0, 1.0]
    assert list(cells["lanes"]) == [4, 3]
    assert not cells["on_ramp"].any()
    assert not cells["off_ramp"].any()


def test_extra_break_postmiles_force_boundaries():
    """Caller-supplied breakpoints (Step 2's VDS postmiles) create cell boundaries."""
    corridor = _corridor([_seg(0.0, 3.0, 4)])
    cells = cells_from_corridor(corridor, extra_break_postmiles=[1.0, 2.0])
    assert list(cells["pm_start"]) == [0.0, 1.0, 2.0]
    assert list(cells["pm_end"]) == [1.0, 2.0, 3.0]
    # No ramps -> no ramp flags.
    assert not cells["on_ramp"].any()
    assert not cells["off_ramp"].any()


def test_extra_break_postmiles_outside_corridor_are_ignored():
    corridor = _corridor([_seg(10.0, 11.0, 4)])
    cells = cells_from_corridor(
        corridor, extra_break_postmiles=[5.0, 10.5, 20.0],
    )
    assert list(cells["pm_start"]) == [10.0, 10.5]
    assert list(cells["pm_end"]) == [10.5, 11.0]


def test_lengths_sum_to_corridor_total():
    corridor = _corridor(
        [_seg(0.0, 1.5, 4), _seg(1.5, 2.7, 3)],
        ramps=[_ramp("on", 0.4), _ramp("off", 2.1)],
    )
    cells = cells_from_corridor(corridor)
    assert cells["length"].sum() == pytest.approx(2.7, abs=1e-9)
    # And every length equals pm_end - pm_start exactly.
    for _, row in cells.iterrows():
        assert row.length == pytest.approx(row.pm_end - row.pm_start, abs=1e-12)


def test_floating_point_noise_in_postmiles_dedupes_to_one_break():
    """Two breakpoints within rounding tolerance collapse to one cell boundary."""
    corridor = _corridor(
        [_seg(0.0, 1.0, 4), _seg(1.0 + 1e-9, 2.0, 4)],
        ramps=[_ramp("on", 1.0 + 5e-10)],
    )
    cells = cells_from_corridor(corridor)
    # Without dedup we'd get three nearly-identical breaks; with dedup -> two cells.
    assert len(cells) == 2
    assert cells.iloc[1].on_ramp


def test_columns_and_dtypes_match_freeway_schema_subset():
    """Output should slot into io.freeway_from_dataframe with FD params joined in."""
    corridor = _corridor(
        [_seg(0.0, 1.0, 4)],
        ramps=[_ramp("on", 0.5, name="X")],
    )
    cells = cells_from_corridor(corridor)
    expected_cols = {
        "pm_start", "pm_end", "length", "lanes",
        "on_ramp", "off_ramp", "on_ramp_name", "off_ramp_name",
    }
    assert set(cells.columns) == expected_cols
    assert cells["length"].dtype == float
    assert cells["lanes"].dtype == int
    assert cells["on_ramp"].dtype == bool
    assert cells["off_ramp"].dtype == bool


# ---- cells_to_geodataframe (GIS overlay) ---------------------------------


def _geo_seg(pm_start, pm_end, lanes, coords) -> MainlineSegment:
    """A MainlineSegment with explicit lat/lon coords (degree units)."""
    return MainlineSegment(
        u=int(pm_start * 1000), v=int(pm_end * 1000),
        geometry=LineString(coords),
        length=pm_end - pm_start, lanes=lanes,
        pm_start=pm_start, pm_end=pm_end,
    )


def test_cells_to_geodataframe_returns_one_linestring_per_cell():
    # Single segment running 0..2 mi along a horizontal line at lat=34.
    corridor = _corridor(
        [_geo_seg(0.0, 2.0, 4, [(-118.40, 34.05), (-118.42, 34.05)])],
    )
    cells = cells_from_corridor(
        corridor, extra_break_postmiles=[0.5, 1.5],
    )
    gdf = cells_to_geodataframe(cells, corridor)
    assert len(gdf) == 3
    assert str(gdf.crs).endswith(":4326")
    # All three cells get LineString geometries.
    for geom in gdf.geometry:
        assert geom.geom_type == "LineString"
    # Cell 0 covers postmiles 0..0.5; its LineString should start at the
    # corridor's upstream end (-118.40, 34.05) and end at the 25% point.
    first = gdf.geometry.iloc[0]
    start_x, start_y = list(first.coords)[0]
    assert start_x == pytest.approx(-118.40, abs=1e-6)
    assert start_y == pytest.approx(34.05, abs=1e-6)
    # First cell length spans 25% of the segment -> end at -118.405 (= 25%
    # between -118.40 and -118.42).
    end_x, end_y = list(first.coords)[-1]
    assert end_x == pytest.approx(-118.405, abs=1e-6)


def test_cells_to_geodataframe_carries_through_input_columns():
    corridor = _corridor(
        [_geo_seg(0.0, 1.0, 4, [(-118.40, 34.05), (-118.41, 34.05)])],
        ramps=[_ramp("on", 0.5, name="HillOn")],
    )
    cells = cells_from_corridor(corridor)
    gdf = cells_to_geodataframe(cells, corridor)
    # Every column from cells_df is preserved in the GeoDataFrame.
    for col in cells.columns:
        assert col in gdf.columns
    # And the rows are in the same order.
    assert list(gdf["pm_start"]) == list(cells["pm_start"])
    # On-ramp flag carries through.
    assert bool(gdf["on_ramp"].iloc[1]) is True
    assert gdf["on_ramp_name"].iloc[1] == "HillOn"


def test_cells_to_geodataframe_spans_multiple_corridor_segments(tmp_path):
    """A cell that straddles two mainline segments produces a continuous geometry
    after linemerge."""
    corridor = _corridor([
        _geo_seg(0.0, 1.0, 4, [(-118.40, 34.05), (-118.41, 34.05)]),
        _geo_seg(1.0, 2.0, 4, [(-118.41, 34.05), (-118.42, 34.05)]),
    ])
    # One cell covering the full corridor -> spans both segments.
    cells = cells_from_corridor(corridor)
    assert len(cells) == 1
    gdf = cells_to_geodataframe(cells, corridor)
    geom = gdf.geometry.iloc[0]
    # linemerge should yield a single LineString from upstream end to downstream.
    assert geom.geom_type == "LineString"
    start_x, _ = list(geom.coords)[0]
    end_x, _ = list(geom.coords)[-1]
    assert start_x == pytest.approx(-118.40, abs=1e-6)
    assert end_x == pytest.approx(-118.42, abs=1e-6)


def test_cells_to_geodataframe_writes_valid_geojson(tmp_path):
    """End-to-end smoke test: the result loads back via geopandas.read_file."""
    import geopandas as gpd
    corridor = _corridor([
        _geo_seg(0.0, 1.0, 4, [(-118.40, 34.05), (-118.41, 34.05)]),
    ])
    cells = cells_from_corridor(corridor)
    gdf = cells_to_geodataframe(cells, corridor)
    out = tmp_path / "cells.geojson"
    gdf.to_file(out, driver="GeoJSON")
    reread = gpd.read_file(out)
    assert len(reread) == len(gdf)
    assert reread.geometry.iloc[0].geom_type == "LineString"


# ---- ramp_junctions_to_geodataframe --------------------------------------


def test_ramp_junctions_to_geodataframe_one_point_per_junction():
    corridor = _corridor(
        [_geo_seg(0.0, 1.0, 4, [(-118.40, 34.05), (-118.41, 34.05)])],
        ramps=[
            _ramp("on", 0.25, name="UpstreamOn"),
            _ramp("off", 0.75, name="DownstreamOff"),
        ],
    )
    gdf = ramp_junctions_to_geodataframe(corridor)
    assert len(gdf) == 2
    assert set(gdf["kind"]) == {"on", "off"}
    assert {"postmile", "mainline_node", "ramp_terminus_node",
            "lanes", "name"}.issubset(gdf.columns)
    assert all(g.geom_type == "Point" for g in gdf.geometry)


def test_ramp_junctions_to_geodataframe_empty_when_no_ramps():
    corridor = _corridor(
        [_geo_seg(0.0, 1.0, 4, [(-118.40, 34.05), (-118.41, 34.05)])],
    )
    gdf = ramp_junctions_to_geodataframe(corridor)
    assert gdf.empty
