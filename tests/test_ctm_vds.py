"""Tests for the Step-4 VDS pipeline in ``ctm/vds.py``.

Unit tests use synthetic CSVs / corridors so the surface under test is just
the logic of each layer. The I-210W integration test (P4e) lives separately
and uses a real PeMS metadata snapshot cached under ``tests/fixtures/pems/``.
"""

from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from ctm_for_dwc.ctm.osm import (
    Corridor,
    MainlineSegment,
    RampJunction,
)
from ctm_for_dwc.ctm.vds import (
    assign_ramp_vds_to_cells,
    assign_vds_to_cells,
    load_pems_station_metadata,
    project_vds_to_corridor,
)


# ---- Synthetic-data builders ----------------------------------------------


def _metadata_csv(rows: list[dict]) -> pd.DataFrame:
    """Build a station_meta-style DataFrame in PeMS's native column casing."""
    return pd.DataFrame(rows)


def _eastbound_corridor() -> Corridor:
    """A straight 3-segment corridor going east along latitude 34.05."""
    def seg(lo, hi, lo_lon, hi_lon, lanes=4):
        return MainlineSegment(
            u=int(lo * 1000), v=int(hi * 1000),
            geometry=LineString([(lo_lon, 34.05), (hi_lon, 34.05)]),
            length=hi - lo, lanes=lanes,
            pm_start=lo, pm_end=hi,
        )
    return Corridor(
        ref="I 210", direction="E", target_bearing=90.0,
        # Three segments, each ~0.575 mi (0.01° lon at lat 34.05 in WGS-84).
        mainline_segments=[
            seg(0.0, 0.575, -118.40, -118.39, lanes=4),
            seg(0.575, 1.150, -118.39, -118.38, lanes=4),
            seg(1.150, 1.725, -118.38, -118.37, lanes=5),  # lane add
        ],
        ramp_junctions=[],
    )


# ---- load_pems_station_metadata -------------------------------------------


_PEMS_ROWS = [
    # Mainline westbound I-210 stations (the ones we want)
    {"ID": 1001, "Fwy": 210, "Dir": "W", "District": 7, "County": "LA",
     "Abs PM": 38.5, "Latitude": 34.13, "Longitude": -117.86,
     "Lanes": 4, "Type": "ML", "Name": "Vernon"},
    {"ID": 1002, "Fwy": 210, "Dir": "W", "District": 7, "County": "LA",
     "Abs PM": 37.8, "Latitude": 34.13, "Longitude": -117.88,
     "Lanes": 4, "Type": "ML", "Name": "Glendora"},
    # Wrong direction
    {"ID": 1003, "Fwy": 210, "Dir": "E", "District": 7, "County": "LA",
     "Abs PM": 38.5, "Latitude": 34.13, "Longitude": -117.86,
     "Lanes": 4, "Type": "ML", "Name": "Vernon EB"},
    # Wrong freeway
    {"ID": 1004, "Fwy": 134, "Dir": "W", "District": 7, "County": "LA",
     "Abs PM": 12.5, "Latitude": 34.15, "Longitude": -118.15,
     "Lanes": 3, "Type": "ML", "Name": "Ventura"},
    # On-ramp detector (right freeway/direction but non-mainline)
    {"ID": 1005, "Fwy": 210, "Dir": "W", "District": 7, "County": "LA",
     "Abs PM": 38.5, "Latitude": 34.13, "Longitude": -117.86,
     "Lanes": 1, "Type": "OR", "Name": "Vernon On"},
]


def test_load_pems_metadata_filters_freeway_direction_and_type():
    df = load_pems_station_metadata(
        _metadata_csv(_PEMS_ROWS), freeway=210, direction="W",
    )
    assert sorted(df["vds_id"]) == [1001, 1002]
    assert set(df["direction"]) == {"W"}
    assert set(df["type"]) == {"ML"}
    # Sorted by abs_pm ascending.
    assert list(df["abs_pm"]) == [37.8, 38.5]


def test_load_pems_metadata_returns_wgs84_points():
    df = load_pems_station_metadata(
        _metadata_csv(_PEMS_ROWS), freeway=210, direction="W",
    )
    assert str(df.crs).endswith(":4326")
    pt = df.iloc[0].geometry
    assert isinstance(pt, Point)
    assert pt.x == pytest.approx(-117.88)
    assert pt.y == pytest.approx(34.13)


def test_load_pems_metadata_keeps_ramp_detectors_when_asked():
    df = load_pems_station_metadata(
        _metadata_csv(_PEMS_ROWS),
        freeway=210, direction="W", mainline_only=False,
    )
    assert set(df["type"]) == {"ML", "OR"}
    assert sorted(df["vds_id"]) == [1001, 1002, 1005]


def test_load_pems_metadata_accepts_freeway_strings():
    df1 = load_pems_station_metadata(
        _metadata_csv(_PEMS_ROWS), freeway="I 210", direction="W",
    )
    df2 = load_pems_station_metadata(
        _metadata_csv(_PEMS_ROWS), freeway="210", direction="W",
    )
    assert sorted(df1["vds_id"]) == sorted(df2["vds_id"]) == [1001, 1002]


def test_load_pems_metadata_handles_underscore_column_names():
    rows = [
        {"ID": 9001, "Fwy": 880, "Dir": "N", "District": 4, "County": "ALA",
         "Abs_PM": 12.3, "Lat": 37.65, "Long": -122.10,
         "Lanes": 4, "Type": "ML", "Name": "Tennyson"},
    ]
    df = load_pems_station_metadata(
        _metadata_csv(rows), freeway=880, direction="N",
    )
    assert df.iloc[0]["abs_pm"] == 12.3
    assert df.iloc[0].geometry.y == pytest.approx(37.65)


def test_load_pems_metadata_reads_a_real_csv_path(tmp_path):
    csv = tmp_path / "station_meta.csv"
    _metadata_csv(_PEMS_ROWS).to_csv(csv, index=False)
    df = load_pems_station_metadata(csv, freeway=210, direction="W")
    assert len(df) == 2


def test_load_pems_metadata_raises_when_nothing_matches():
    with pytest.raises(ValueError, match="No stations found for freeway"):
        load_pems_station_metadata(
            _metadata_csv(_PEMS_ROWS), freeway=5, direction="N",
        )


def test_load_pems_metadata_raises_when_required_column_missing():
    rows = [{"ID": 1, "Fwy": 210, "Dir": "W"}]  # missing lat/lon/abs_pm/lanes/type/...
    with pytest.raises(ValueError, match="missing required columns"):
        load_pems_station_metadata(
            _metadata_csv(rows), freeway=210, direction="W",
        )


# ---- project_vds_to_corridor ----------------------------------------------


def _vds_gdf_at(pm_lon_lat: list[tuple[int, float, float]]) -> gpd.GeoDataFrame:
    """Build a minimal VDS GDF with explicit lat/lon per station."""
    rows = []
    for vds_id, lon, lat in pm_lon_lat:
        rows.append({
            "vds_id": vds_id, "fwy": 210, "direction": "E", "district": 7,
            "county": "LA", "abs_pm": 0.0, "latitude": lat, "longitude": lon,
            "lanes": 4, "type": "ML", "name": f"VDS-{vds_id}",
        })
    df = pd.DataFrame(rows)
    return gpd.GeoDataFrame(
        df, geometry=[Point(r.longitude, r.latitude) for _, r in df.iterrows()],
        crs="EPSG:4326",
    )


def test_project_vds_lands_at_correct_postmile():
    corridor = _eastbound_corridor()
    # One VDS exactly on the corridor at lon=-118.395 (midpoint of seg 0 at
    # lat 34.05). Should project to pm_cum ~ 0.2875 (half of 0.575).
    vds = _vds_gdf_at([(1, -118.395, 34.05)])
    proj = project_vds_to_corridor(vds, corridor)
    assert len(proj) == 1
    assert proj.iloc[0]["pm_cum"] == pytest.approx(0.2875, abs=1e-3)
    assert proj.iloc[0]["seg_index"] == 0
    # Right on the centerline -> lateral_distance_m near zero.
    assert proj.iloc[0]["lateral_distance_m"] < 0.5


def test_project_vds_records_lateral_offset_when_off_centerline():
    corridor = _eastbound_corridor()
    # VDS offset ~50 m south of the corridor (lat 34.05 -> 34.05045 ≈ 50 m).
    vds = _vds_gdf_at([(1, -118.395, 34.0505)])
    proj = project_vds_to_corridor(vds, corridor)
    # ~50 m offset; allow some slack for the great-circle calculation.
    assert 30 < proj.iloc[0]["lateral_distance_m"] < 80


def test_project_vds_picks_closest_segment_when_corridor_curves():
    """A VDS near a segment boundary should snap to whichever segment it's
    perpendicularly closest to."""
    corridor = _eastbound_corridor()
    # Sit on the boundary between segments 0 and 1 (lon = -118.39).
    vds = _vds_gdf_at([(1, -118.39, 34.05)])
    proj = project_vds_to_corridor(vds, corridor)
    # Boundary lands exactly at the start of segment 1, where its pm_cum =
    # pm_end of segment 0 = 0.575. Tie-broken by iteration order; here
    # segment 0 catches it via its endpoint.
    assert proj.iloc[0]["pm_cum"] == pytest.approx(0.575, abs=1e-3)


def test_project_vds_returns_empty_frame_for_empty_input():
    proj = project_vds_to_corridor(
        gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        _eastbound_corridor(),
    )
    assert proj.empty


# ---- assign_vds_to_cells --------------------------------------------------


def _cells_df(pms: list[tuple[float, float]], lanes: int = 4) -> pd.DataFrame:
    """Build a minimal cells DataFrame from (pm_start, pm_end) pairs."""
    rows = []
    for pm_start, pm_end in pms:
        rows.append({
            "pm_start": pm_start, "pm_end": pm_end,
            "length": pm_end - pm_start, "lanes": lanes,
            "on_ramp": False, "off_ramp": False,
            "on_ramp_name": None, "off_ramp_name": None,
        })
    return pd.DataFrame(rows)


def _projected(pm_lanes: list[tuple[int, float]], lane_counts: list[int] | None = None) -> pd.DataFrame:
    """Build a minimal projected-VDS DataFrame from (vds_id, pm_cum) pairs."""
    lanes = lane_counts or [4] * len(pm_lanes)
    return pd.DataFrame([
        {"vds_id": vds_id, "pm_cum": pm_cum, "pm_caltrans": None,
         "seg_index": 0, "lateral_distance_m": 0.0,
         "abs_pm": 0.0, "lanes": ln, "name": f"VDS-{vds_id}"}
        for (vds_id, pm_cum), ln in zip(pm_lanes, lanes)
    ])


def test_assign_direct_vds_one_per_cell():
    cells = _cells_df([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
    vds = _projected([(101, 0.5), (102, 1.5), (103, 2.5)])
    out = assign_vds_to_cells(cells, vds)
    assert list(out["vds_id"]) == [101, 102, 103]
    assert list(out["vds_source"]) == ["direct"] * 3
    assert list(out["vds_n_candidates"]) == [1, 1, 1]
    # vds_distance_mi is exactly zero on direct hits.
    assert list(out["vds_distance_mi"]) == [0.0, 0.0, 0.0]


def test_assign_upstream_fallback_for_empty_middle_cells():
    """Middle cell has no VDS -> inherit from the nearest upstream cell."""
    cells = _cells_df([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
    vds = _projected([(101, 0.5), (103, 2.5)])  # nothing in cell 1
    out = assign_vds_to_cells(cells, vds)
    assert list(out["vds_id"]) == [101, 101, 103]
    assert list(out["vds_source"]) == ["direct", "nearest_upstream", "direct"]
    # Cell 1 inherited VDS 101 (pm_cum=0.5) and has midpoint at 1.5.
    assert out["vds_distance_mi"].iloc[1] == pytest.approx(0.5 - 1.5)


def test_assign_marks_cells_upstream_of_every_vds_as_missing():
    """Cells upstream of the first VDS stay 'missing' (no inherit-from-future)."""
    cells = _cells_df([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
    vds = _projected([(101, 2.5)])  # only a downstream VDS
    out = assign_vds_to_cells(cells, vds)
    assert list(out["vds_source"]) == ["missing", "missing", "direct"]
    assert pd.isna(out["vds_id"].iloc[0])
    assert out["vds_id"].iloc[2] == 101


def test_assign_tiebreaker_midpoint_default():
    """Multiple VDSs in one cell -> closest to midpoint wins by default."""
    cells = _cells_df([(0.0, 1.0)])  # midpoint 0.5
    vds = _projected([(101, 0.3), (102, 0.55), (103, 0.9)])
    out = assign_vds_to_cells(cells, vds)
    assert out["vds_id"].iloc[0] == 102
    assert out["vds_source"].iloc[0] == "direct_tiebreak"
    # All three landed inside the cell -> n_candidates reflects that, so the
    # caller can flag tiebreak situations to inspect later.
    assert out["vds_n_candidates"].iloc[0] == 3


def test_assign_tiebreaker_lowest_id_is_deterministic():
    cells = _cells_df([(0.0, 1.0)])
    vds = _projected([(103, 0.3), (101, 0.55), (102, 0.9)])
    out = assign_vds_to_cells(cells, vds, tiebreaker="lowest_id")
    assert out["vds_id"].iloc[0] == 101
    assert out["vds_source"].iloc[0] == "direct_tiebreak"


def test_assign_lane_match_prefers_same_lane_count():
    """min_lane_match should prefer a VDS whose lanes column matches the cell."""
    cells = _cells_df([(0.0, 1.0)], lanes=5)
    # Two VDSs in the cell; the off-midpoint 5-lane one should beat the
    # midpoint-aligned 4-lane one when min_lane_match=True.
    vds = _projected([(101, 0.5), (102, 0.7)], lane_counts=[4, 5])
    direct = assign_vds_to_cells(cells, vds)  # midpoint -> 101
    assert direct["vds_id"].iloc[0] == 101
    lane_aware = assign_vds_to_cells(cells, vds, min_lane_match=True)
    assert lane_aware["vds_id"].iloc[0] == 102


def test_assign_fallback_none_keeps_holes():
    cells = _cells_df([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
    vds = _projected([(101, 0.5), (103, 2.5)])
    out = assign_vds_to_cells(cells, vds, fallback="none")
    assert list(out["vds_source"]) == ["direct", "missing", "direct"]
    assert pd.isna(out["vds_id"].iloc[1])


def test_assign_boundary_vds_belongs_to_upstream_cell():
    """A VDS at pm == cell.pm_end belongs to the upstream cell (half-open)."""
    cells = _cells_df([(0.0, 1.0), (1.0, 2.0)])
    vds = _projected([(101, 1.0)])  # exactly on the boundary
    out = assign_vds_to_cells(cells, vds)
    assert out["vds_id"].iloc[0] == 101  # upstream cell owns the boundary
    assert out["vds_source"].iloc[0] == "direct"
    # Downstream cell falls back upstream.
    assert out["vds_source"].iloc[1] == "nearest_upstream"


# ---- vds_lanes surfacing --------------------------------------------------


def test_assign_vds_to_cells_propagates_vds_lanes_column():
    """Direct + upstream-inherit cells should both carry the assigned VDS's lane count."""
    cells = _cells_df([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], lanes=4)
    vds = _projected([(101, 0.5), (103, 2.5)], lane_counts=[4, 5])
    out = assign_vds_to_cells(cells, vds)
    assert "vds_lanes" in out.columns
    # Direct hits get the candidate's lane count.
    assert out["vds_lanes"].iloc[0] == 4
    assert out["vds_lanes"].iloc[2] == 5
    # Upstream-inherit cells take the upstream VDS's lane count.
    assert out["vds_lanes"].iloc[1] == 4
    assert out["vds_source"].iloc[1] == "nearest_upstream"


def test_assign_vds_to_cells_vds_lanes_is_NA_for_missing_cells():
    """Cells upstream of every VDS keep vds_lanes as <NA>, matching vds_id."""
    cells = _cells_df([(0.0, 1.0), (1.0, 2.0)])
    vds = _projected([(101, 1.5)])
    out = assign_vds_to_cells(cells, vds)
    assert out["vds_source"].iloc[0] == "missing"
    assert pd.isna(out["vds_id"].iloc[0])
    assert pd.isna(out["vds_lanes"].iloc[0])


# ---- assign_ramp_vds_to_cells --------------------------------------------


def _corridor_with_ramps(ramps):
    """Build a minimal Corridor with the given RampJunction list."""
    return Corridor(
        ref="I 880", direction="N", target_bearing=0.0,
        mainline_segments=[
            MainlineSegment(
                u=1, v=2,
                geometry=LineString([(-122.05, 37.50), (-122.05, 37.52)]),
                length=1.0, lanes=4, pm_start=0.0, pm_end=1.0,
            ),
        ],
        ramp_junctions=ramps,
    )


def _ramp_at(kind, pm, lon, lat) -> RampJunction:
    return RampJunction(
        kind=kind, postmile=pm,
        mainline_node=int(pm * 1000), ramp_terminus_node=-1,
        geometry=Point(lon, lat), lanes=1, name=f"R{int(pm*1000)}",
    )


def _ramp_vds_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    """Build a PeMS-style ramp-VDS GeoDataFrame."""
    df = pd.DataFrame(rows)
    return gpd.GeoDataFrame(
        df, geometry=[Point(r.longitude, r.latitude) for _, r in df.iterrows()],
        crs="EPSG:4326",
    )


def test_assign_ramp_vds_matches_each_gore_to_nearest_typed_detector():
    """On-ramp gores get an OR VDS; off-ramp gores get an FR VDS."""
    corridor = _corridor_with_ramps([
        _ramp_at("on", 0.25, -122.0500, 37.5050),
        _ramp_at("off", 0.75, -122.0500, 37.5150),
    ])
    # OR VDS within 100 m of the on-ramp gore; FR VDS within 100 m of off-ramp.
    vds = _ramp_vds_gdf([
        {"vds_id": 9001, "type": "OR",
         "latitude": 37.5051, "longitude": -122.0500, "lanes": 1, "name": "On"},
        {"vds_id": 9002, "type": "FR",
         "latitude": 37.5149, "longitude": -122.0500, "lanes": 1, "name": "Off"},
        # Decoy: an FR VDS close to the on-ramp gore -- must NOT be picked
        # because of the type mismatch.
        {"vds_id": 9003, "type": "FR",
         "latitude": 37.5050, "longitude": -122.0500, "lanes": 1, "name": "Decoy"},
    ])
    cells = pd.DataFrame([
        {"pm_start": 0.0, "pm_end": 0.5, "length": 0.5, "lanes": 4,
         "on_ramp": False, "off_ramp": False},
        {"pm_start": 0.25, "pm_end": 0.75, "length": 0.5, "lanes": 4,
         "on_ramp": True, "off_ramp": True},
    ])
    out = assign_ramp_vds_to_cells(cells, corridor, vds)
    # Row 0: no ramps -> both ramp-VDS columns are NA.
    assert pd.isna(out["on_ramp_vds_id"].iloc[0])
    assert pd.isna(out["off_ramp_vds_id"].iloc[0])
    # Row 1: on-ramp at pm_start=0.25 -> matches OR 9001;
    #        off-ramp at pm_end=0.75 -> matches FR 9002.
    assert out["on_ramp_vds_id"].iloc[1] == 9001
    assert out["off_ramp_vds_id"].iloc[1] == 9002


def test_assign_ramp_vds_respects_distance_threshold():
    """A ramp VDS farther than max_distance_m from the gore should not match."""
    corridor = _corridor_with_ramps([
        _ramp_at("on", 0.25, -122.0500, 37.5050),
    ])
    # Lat diff of 0.01 degrees -> ~1100 m, well outside the default 200 m.
    far_away_vds = _ramp_vds_gdf([
        {"vds_id": 9001, "type": "OR",
         "latitude": 37.5150, "longitude": -122.0500,
         "lanes": 1, "name": "FarOn"},
    ])
    cells = pd.DataFrame([{
        "pm_start": 0.25, "pm_end": 0.5, "length": 0.25, "lanes": 4,
        "on_ramp": True, "off_ramp": False,
    }])
    out = assign_ramp_vds_to_cells(cells, corridor, far_away_vds)
    # Distance gate triggers -> no match.
    assert pd.isna(out["on_ramp_vds_id"].iloc[0])
    # And bumping the threshold above the actual distance recovers the match.
    out = assign_ramp_vds_to_cells(
        cells, corridor, far_away_vds, max_distance_m=2000.0,
    )
    assert out["on_ramp_vds_id"].iloc[0] == 9001


def test_assign_ramp_vds_empty_input_returns_NA_columns():
    """No ramp VDSs at all -> on/off columns added but all-NA."""
    corridor = _corridor_with_ramps([
        _ramp_at("on", 0.25, -122.0500, 37.5050),
    ])
    cells = pd.DataFrame([{
        "pm_start": 0.25, "pm_end": 0.5, "length": 0.25, "lanes": 4,
        "on_ramp": True, "off_ramp": False,
    }])
    out = assign_ramp_vds_to_cells(
        cells, corridor, gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
    )
    assert "on_ramp_vds_id" in out.columns
    assert pd.isna(out["on_ramp_vds_id"].iloc[0])
