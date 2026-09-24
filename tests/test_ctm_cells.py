"""Unit tests for ``ctm/cells.py``: cell layout along a Corridor.

These tests bypass osmnx entirely and construct :class:`Corridor` instances
by hand so the surface under test is just :func:`cells_from_corridor`'s
breakpoint logic.
"""

from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from ctm_for_dwc.ctm.cells import (
    cells_from_corridor,
    cells_to_geodataframe,
    corridor_from_artifacts,
    corridor_to_artifacts,
    flag_cell_length_warnings,
    ramp_junctions_to_geodataframe,
)
from ctm_for_dwc.ctm.osm import (
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


# ---- flag_cell_length_warnings -------------------------------------------


def _cells_with_lengths(lengths: list[float]) -> pd.DataFrame:
    """Minimal cells DataFrame keyed only by ``length``."""
    return pd.DataFrame({
        "pm_start": [0.0] * len(lengths),
        "pm_end": lengths,
        "length": lengths,
        "lanes": [4] * len(lengths),
    })


def test_flag_cell_length_warnings_assigns_three_levels():
    """Cells get tagged "ok", "too_short", or "too_long" per default bounds."""
    df = _cells_with_lengths([0.05, 0.2, 0.55, 1.0, 1.5])
    out = flag_cell_length_warnings(df)
    assert list(out["length_warning"]) == [
        "too_short",  # 0.05 < 0.2
        "ok",         # 0.2 boundary -> inclusive
        "ok",
        "ok",         # 1.0 boundary -> inclusive
        "too_long",   # 1.5 > 1.0
    ]


def test_flag_cell_length_warnings_preserves_input_columns_and_rows():
    """The helper must not drop rows or mutate existing columns."""
    df = _cells_with_lengths([0.1, 0.5, 1.2])
    df["lanes"] = [3, 4, 5]
    df["foo"] = ["a", "b", "c"]
    out = flag_cell_length_warnings(df)
    assert len(out) == len(df)
    for col in df.columns:
        assert col in out.columns
        if col != "length_warning":
            assert (out[col].to_numpy() == df[col].to_numpy()).all()
    # Original DataFrame is not mutated.
    assert "length_warning" not in df.columns


def test_flag_cell_length_warnings_custom_bounds():
    """``min_length`` / ``max_length`` are user-tunable."""
    df = _cells_with_lengths([0.15, 0.4, 0.8])
    out = flag_cell_length_warnings(df, min_length=0.3, max_length=0.5)
    assert list(out["length_warning"]) == ["too_short", "ok", "too_long"]


def test_flag_cell_length_warnings_invalid_bounds_raises():
    df = _cells_with_lengths([0.5])
    with pytest.raises(ValueError, match="min_length"):
        flag_cell_length_warnings(df, min_length=1.0, max_length=0.5)


def test_flag_cell_length_warnings_missing_length_column_raises():
    df = pd.DataFrame({"pm_start": [0.0], "pm_end": [1.0], "lanes": [4]})
    with pytest.raises(ValueError, match="'length' column"):
        flag_cell_length_warnings(df)


def test_flag_cell_length_warnings_overwrites_existing_column():
    """Re-running the flagger replaces the column rather than failing."""
    df = _cells_with_lengths([0.1, 0.5, 1.5])
    first = flag_cell_length_warnings(df)
    # Simulate a stale annotation: corrupt the column and re-flag.
    first["length_warning"] = "stale"
    second = flag_cell_length_warnings(first)
    assert list(second["length_warning"]) == ["too_short", "ok", "too_long"]


# ---- corridor_to_artifacts / corridor_from_artifacts round-trip ----------


def _round_trip_corridor() -> Corridor:
    """A 3-segment, 2-ramp corridor with Caltrans PMs set, for sidecar tests."""
    segs = [
        _geo_seg(0.0, 0.5, 4, [(-118.400, 34.05), (-118.405, 34.05)]),
        _geo_seg(0.5, 1.3, 5, [(-118.405, 34.05), (-118.415, 34.05)]),
        _geo_seg(1.3, 2.0, 5, [(-118.415, 34.05), (-118.425, 34.05)]),
    ]
    corridor = _corridor(segs, ramps=[
        _ramp("on", 0.2, name="HillOn"),
        _ramp("off", 1.6, name="AllenOff"),
    ])
    # Caltrans PMs at each mainline node id (decreasing for westbound).
    corridor.caltrans_postmiles = {
        segs[0].u: 38.0, segs[0].v: 37.5,
        segs[1].v: 36.7, segs[2].v: 36.0,
    }
    return corridor


def test_corridor_to_artifacts_writes_three_files(tmp_path):
    corridor = _round_trip_corridor()
    corridor_to_artifacts(corridor, tmp_path)
    assert (tmp_path / "mainline.geojson").exists()
    assert (tmp_path / "ramps.geojson").exists()
    assert (tmp_path / "corridor.json").exists()


def test_corridor_artifacts_round_trip_preserves_scalars(tmp_path):
    """ref / direction / target_bearing / crs survive a write + read cycle."""
    corridor = _round_trip_corridor()
    corridor_to_artifacts(corridor, tmp_path)
    reloaded = corridor_from_artifacts(tmp_path)
    assert reloaded.ref == corridor.ref
    assert reloaded.direction == corridor.direction
    assert reloaded.target_bearing == pytest.approx(corridor.target_bearing)
    assert str(reloaded.crs).endswith(":4326")


def test_corridor_artifacts_round_trip_preserves_mainline(tmp_path):
    corridor = _round_trip_corridor()
    corridor_to_artifacts(corridor, tmp_path)
    reloaded = corridor_from_artifacts(tmp_path)
    assert len(reloaded.mainline_segments) == len(corridor.mainline_segments)
    for orig, reread in zip(corridor.mainline_segments, reloaded.mainline_segments):
        assert reread.u == orig.u
        assert reread.v == orig.v
        assert reread.lanes == orig.lanes
        assert reread.pm_start == pytest.approx(orig.pm_start)
        assert reread.pm_end == pytest.approx(orig.pm_end)
        assert reread.length == pytest.approx(orig.pm_end - orig.pm_start)
        assert list(reread.geometry.coords) == list(orig.geometry.coords)


def test_corridor_artifacts_round_trip_preserves_ramps(tmp_path):
    corridor = _round_trip_corridor()
    corridor_to_artifacts(corridor, tmp_path)
    reloaded = corridor_from_artifacts(tmp_path)
    assert len(reloaded.ramp_junctions) == len(corridor.ramp_junctions)
    # Ramps come out sorted by (postmile, kind) -- same order as input.
    for orig, reread in zip(corridor.ramp_junctions, reloaded.ramp_junctions):
        assert reread.kind == orig.kind
        assert reread.postmile == pytest.approx(orig.postmile)
        assert reread.name == orig.name
        assert reread.mainline_node == orig.mainline_node
        assert reread.ramp_terminus_node == orig.ramp_terminus_node


def test_corridor_artifacts_round_trip_preserves_caltrans_postmiles(tmp_path):
    """Caltrans PM dict survives JSON encoding (int keys, float values)."""
    corridor = _round_trip_corridor()
    corridor_to_artifacts(corridor, tmp_path)
    reloaded = corridor_from_artifacts(tmp_path)
    assert reloaded.caltrans_postmiles is not None
    assert reloaded.caltrans_postmiles.keys() == corridor.caltrans_postmiles.keys()
    for k, v in corridor.caltrans_postmiles.items():
        assert reloaded.caltrans_postmiles[k] == pytest.approx(v)


def test_corridor_artifacts_round_trip_omits_caltrans_when_unset(tmp_path):
    corridor = _corridor(
        [_geo_seg(0.0, 1.0, 4, [(-118.40, 34.05), (-118.41, 34.05)])],
    )
    corridor_to_artifacts(corridor, tmp_path)
    reloaded = corridor_from_artifacts(tmp_path)
    assert reloaded.caltrans_postmiles is None


def test_corridor_artifacts_round_trip_no_ramps(tmp_path):
    """A corridor with no ramps writes no ramps.geojson and still round-trips."""
    corridor = _corridor(
        [_geo_seg(0.0, 1.0, 4, [(-118.40, 34.05), (-118.41, 34.05)])],
    )
    corridor_to_artifacts(corridor, tmp_path)
    assert not (tmp_path / "ramps.geojson").exists()
    reloaded = corridor_from_artifacts(tmp_path)
    assert reloaded.ramp_junctions == []


def test_corridor_artifacts_round_trip_cells_geojson_identical(tmp_path):
    """The reconstructed corridor produces the same cells.geojson geometries."""
    corridor = _round_trip_corridor()
    cells = cells_from_corridor(corridor)
    original_gdf = cells_to_geodataframe(cells, corridor)

    corridor_to_artifacts(corridor, tmp_path)
    reloaded = corridor_from_artifacts(tmp_path)
    rebuilt_gdf = cells_to_geodataframe(cells, reloaded)

    assert len(rebuilt_gdf) == len(original_gdf)
    for o, r in zip(original_gdf.geometry, rebuilt_gdf.geometry):
        assert o.geom_type == r.geom_type
        assert list(o.coords) == list(r.coords)


def test_flag_cell_length_warnings_works_on_real_cells_from_corridor_output():
    """End-to-end: pipe a real cells_from_corridor frame through the flagger.

    cells_from_corridor only splits at ramps + lane changes + endpoints,
    so we stage a 3-cell corridor by varying lane counts across the three
    underlying mainline segments.
    """
    corridor = _corridor([
        _geo_seg(0.0, 0.1, 3, [(-118.400, 34.05), (-118.401, 34.05)]),
        _geo_seg(0.1, 0.7, 4, [(-118.401, 34.05), (-118.410, 34.05)]),
        _geo_seg(0.7, 2.0, 5, [(-118.410, 34.05), (-118.430, 34.05)]),
    ])
    cells = cells_from_corridor(corridor)
    flagged = flag_cell_length_warnings(cells)
    assert len(flagged) == 3
    assert list(flagged["length_warning"]) == ["too_short", "ok", "too_long"]
