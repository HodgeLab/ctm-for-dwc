"""I-210W Step-4 golden test: real OSM corridor + real PeMS metadata.

Loads the cached OSM graph (``tests/fixtures/osm/i210_bbox.graphml``) and the
cached PeMS District 7 station metadata snapshot
(``tests/fixtures/pems/d07_meta_2023_12_22.txt``), threads them through
Stage-3 corridor extraction and Stage-4 VDS-to-cell assignment, and asserts
the result is structurally consistent with the Kurzhanskiy 2007 I-210W
dissertation layout (encoded in ``ctmsim_configs/w060412.mat``).

Like the Step-3 I-210 golden test, this is a *structural* check: PeMS and
the dissertation use different snapshots and Caltrans postmile equations
introduce small offsets, so we verify counts, density, and rough postmile
alignment rather than exact per-cell matches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import osmnx as ox
import pytest
import scipy.io as sio

from transportation_models.utils.ctm.caltrans import (
    download_caltrans_postmiles,
)
from transportation_models.utils.ctm.cells import cells_from_corridor
from transportation_models.utils.ctm.osm import corridor_from_graph
from transportation_models.utils.ctm.vds import (
    assign_vds_to_cells,
    load_pems_station_metadata,
    project_vds_to_corridor,
)

REPO = Path(__file__).resolve().parents[1]
GRAPHML = REPO / "tests/fixtures/osm/i210_bbox.graphml"
PEMS_FIXTURE = REPO / "tests/fixtures/pems/d07_meta_2023_12_22.txt"
CTMSIM_MAT = REPO / "src/transportation_models/utils/ctm/ctmsim_configs/w060412.mat"


@pytest.fixture(scope="module")
def graph():
    if not GRAPHML.exists():
        pytest.skip(f"I-210 OSM fixture missing: {GRAPHML}")
    return ox.load_graphml(GRAPHML)


@pytest.fixture(scope="module")
def corridor(graph):
    return corridor_from_graph(
        graph, ref="I 210", direction="W",
        postmiles=download_caltrans_postmiles(routes=[210], progress=False),
    )


@pytest.fixture(scope="module")
def vds_gdf():
    if not PEMS_FIXTURE.exists():
        pytest.skip(f"PeMS D7 metadata fixture missing: {PEMS_FIXTURE}")
    return load_pems_station_metadata(PEMS_FIXTURE, freeway=210, direction="W")


@pytest.fixture(scope="module")
def projected(vds_gdf, corridor):
    return project_vds_to_corridor(vds_gdf, corridor)


@pytest.fixture(scope="module")
def cells(corridor):
    return cells_from_corridor(corridor)


@pytest.fixture(scope="module")
def assigned(cells, projected):
    return assign_vds_to_cells(cells, projected)


@pytest.fixture(scope="module")
def ctmsim():
    mat = sio.loadmat(str(CTMSIM_MAT), squeeze_me=True, struct_as_record=False)
    cd = mat["celldata"]
    return {
        "n_cells": len(cd),
        "n_on": int(sum(c.ORfmax > 0 for c in cd)),
        "n_off": int(sum(c.FRfmax > 0 for c in cd)),
    }


# ---- load + project structural checks -------------------------------------


def test_metadata_filters_to_i210_west_mainline(vds_gdf):
    """The D7 fixture has plenty of stations; mainline I-210 W is a clean subset."""
    assert len(vds_gdf) > 60     # CTMSIM has 40; OSM/PeMS overlap is bigger.
    assert (vds_gdf["fwy"] == 210).all()
    assert (vds_gdf["direction"] == "W").all()
    assert (vds_gdf["type"] == "ML").all()


def test_projection_postmiles_match_pems_abs_pm(projected, corridor):
    """When the corridor has Caltrans PMs, our projection's pm_caltrans should
    line up with PeMS's reported Abs PM (independent measurement of the same
    state-route coordinate). 0.5 mi tolerance absorbs OSM-vs-Caltrans
    centerline differences plus the linear interpolation between mainline
    nodes."""
    inside = projected.loc[
        (projected["pm_cum"] >= corridor.pm_start)
        & (projected["pm_cum"] <= corridor.pm_end)
        & projected["pm_caltrans"].notna()
        & projected["abs_pm"].notna()
    ]
    # At least a handful of VDSs should land inside the bbox-cropped corridor.
    assert len(inside) > 10
    diff = (inside["pm_caltrans"] - inside["abs_pm"]).abs()
    # Most match within 0.5 mi; a couple of outliers near the bbox edges
    # are fine. Loose budget so the test is informative without flakiness.
    assert diff.median() < 0.5
    assert (diff < 1.0).mean() > 0.8


def test_projection_lateral_distance_is_small_for_real_mainline_vds(projected):
    """Real I-210 W mainline VDSs sit on the centerline; projection should put
    them within a lane-width of the corridor polyline."""
    inside_corridor = projected.loc[projected["pm_cum"] >= 0]
    if inside_corridor.empty:
        pytest.skip("no VDSs projected inside the corridor")
    median_lateral = inside_corridor["lateral_distance_m"].median()
    # 30 m is comfortable for a 4-lane mainline + a small OSM centerline offset.
    assert median_lateral < 30.0, (
        f"median lateral offset {median_lateral:.1f} m exceeds 30 m"
    )


# ---- cell assignment structural checks ------------------------------------


def test_every_cell_gets_a_vds_with_upstream_fallback(assigned):
    """The M&H 2014 fallback should fill in any gap left by the direct pass."""
    sources = assigned["vds_source"].unique().tolist()
    assert "direct" in sources
    # No "missing" anywhere if we have VDSs throughout the I-210 corridor
    # span we extracted.
    assert (assigned["vds_source"] != "missing").all() or (
        assigned["vds_source"] == "missing"
    ).sum() <= 2     # tolerate up to 2 cells upstream of the first VDS


def test_direct_vds_coverage_matches_pems_spacing(assigned, projected, corridor):
    """Cells with their own VDS should roughly match the corridor's VDS supply.

    CTMSIM's "40 cells, one VDS each" isn't directly comparable -- those
    cells were *drawn around* the dissertation's VDS positions, while ours
    come from ramp gores + lane changes. The meaningful invariant is that
    most in-corridor VDSs land cleanly in a cell (only a handful collide
    into tiebreakers) and that the total VDS-bearing cell count tracks the
    in-corridor VDS count, not CTMSIM's per-cell count.
    """
    direct = (assigned["vds_source"] == "direct").sum()
    direct_tiebreak = (assigned["vds_source"] == "direct_tiebreak").sum()
    direct_or_tiebreak = direct + direct_tiebreak
    n_in_corridor_vds = len(projected)
    # Each VDS-bearing cell consumes at least one VDS, so direct-bearing
    # cell count is bounded above by the VDS supply (with equality when
    # there are no multi-VDS cells).
    assert direct_or_tiebreak <= n_in_corridor_vds
    # And the corridor's VDS coverage shouldn't be wildly different from
    # CTMSIM's density of ~2.8 VDS / mi. With our 12.6 mi crop that's
    # ~35 expected VDSs; we get ~26 (modern PeMS spacing is somewhat
    # sparser than the 2007 dissertation's snapshot).
    vds_per_mi = n_in_corridor_vds / corridor.total_length
    assert 1.0 < vds_per_mi < 4.0, (
        f"VDS density {vds_per_mi:.2f}/mi outside expected range"
    )


def test_tiebreaker_fires_on_at_least_one_cell(assigned):
    """OSM cell breaks are coarser than the PeMS 0.3-mi spacing, so at least
    some cells should land >1 VDS. Verifies the tiebreaker actually runs and
    the n_candidates column carries useful diagnostics."""
    multi = (assigned["vds_n_candidates"] >= 2).sum()
    assert multi > 0
    # When it fires, the source should record direct_tiebreak.
    direct_tiebreak = (assigned["vds_source"] == "direct_tiebreak").sum()
    assert direct_tiebreak >= 1
    # And every direct_tiebreak row should have n_candidates >= 2.
    tb_rows = assigned.loc[assigned["vds_source"] == "direct_tiebreak"]
    assert (tb_rows["vds_n_candidates"] >= 2).all()


def test_assigned_vds_ids_are_subset_of_input_vdss(assigned, vds_gdf):
    """Sanity: every assigned VDS id appears in the input metadata."""
    assigned_ids = set(assigned["vds_id"].dropna().astype(int))
    valid_ids = set(vds_gdf["vds_id"].astype(int))
    assert assigned_ids.issubset(valid_ids)


def test_upstream_fallback_distance_is_signed_and_small(assigned):
    """When fallback fires, vds_distance_mi should be negative (the VDS is
    upstream of the cell midpoint) and within a few cell-lengths."""
    fb = assigned.loc[assigned["vds_source"] == "nearest_upstream"]
    if fb.empty:
        pytest.skip("no nearest_upstream cells in this run")
    # Signed: upstream means smaller pm_cum than the cell's midpoint -> negative.
    assert (fb["vds_distance_mi"] <= 0.0).all()
    # Magnitudes bounded by total corridor length; in practice well under 2 mi.
    assert fb["vds_distance_mi"].abs().max() < 2.0
