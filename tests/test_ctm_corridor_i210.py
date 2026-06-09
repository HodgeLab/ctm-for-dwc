"""Golden test: extracting I-210 West from OSM produces a CTM-ready cell layout
that's structurally faithful to the Kurzhanskiy 2007 manual layout
(as encoded in ``ctmsim_configs/w060412.mat``).

We don't expect pixel-perfect agreement: Kurzhanskiy 2007 uses Caltrans
postmiles (centerline-based, 14.03 mi end to end) while we project geodesic
length along an OSM-derived polyline, and the bbox we cache is wider than
Kurzhanskiy 2007's PM 24.95-38.97 span. So the golden test checks *structural*
properties -- corridor extracts cleanly, cell DataFrame is well-formed, ramp
counts/density agree with CTMSIM to a coarse tolerance -- not specific
postmile values.

Uses a one-time OSM extract cached at ``tests/fixtures/osm/i210_bbox.graphml``
so CI doesn't hit Overpass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import osmnx as ox
import pytest
import scipy.io as sio

from transportation_models.utils.ctm.cells import cells_from_corridor
from transportation_models.utils.ctm.osm import corridor_from_graph

REPO = Path(__file__).resolve().parents[1]
GRAPHML = REPO / "tests/fixtures/osm/i210_bbox.graphml"
CTMSIM_MAT = REPO / "src/transportation_models/utils/ctm/ctmsim_configs/w060412.mat"


@pytest.fixture(scope="module")
def graph():
    if not GRAPHML.exists():
        pytest.skip(f"I-210 OSM fixture not found at {GRAPHML}")
    return ox.load_graphml(GRAPHML)


@pytest.fixture(scope="module")
def corridor(graph):
    return corridor_from_graph(graph, ref="I 210", direction="W")


@pytest.fixture(scope="module")
def cells(corridor):
    return cells_from_corridor(corridor)


@pytest.fixture(scope="module")
def ctmsim():
    """Lift the celldata layout from w060412.mat into plain Python."""
    mat = sio.loadmat(str(CTMSIM_MAT), squeeze_me=True, struct_as_record=False)
    cd = mat["celldata"]
    return {
        "n_cells": len(cd),
        "n_on": int(sum(c.ORfmax > 0 for c in cd)),
        "n_off": int(sum(c.FRfmax > 0 for c in cd)),
        "total_length": float(sum(abs(c.PMstart - c.PMend) for c in cd)),
        "lane_counts": [int(c.lanes) for c in cd],
    }


# ---- Corridor extraction smoke tests --------------------------------------


def test_corridor_extracts_cleanly(corridor):
    """The pipeline runs end-to-end on real OSM data without raising."""
    assert corridor.n_mainline >= 30
    assert corridor.target_bearing == 270.0
    # West-running mainline -> longitude strictly decreases upstream->downstream.
    last_lon = corridor.mainline_segments[0].geometry.coords[0][0]
    for seg in corridor.mainline_segments:
        for x, _y in seg.geometry.coords:
            assert x <= last_lon + 1e-6, "mainline went east somewhere"
            last_lon = x


def test_corridor_total_length_in_expected_range(corridor):
    """Extracted corridor should span ~10-15 mi of I-210W (Vernon Ave area
    westward to the SR-134 split). Exact length differs from CTMSIM's reported
    14.03 mi because that's the Caltrans postmile delta, not the OSM-geodesic
    length -- Caltrans postmile equations on I-210 introduce a few-mile offset
    that needs the SHN shapefile (P3d) to translate cleanly. Width here covers
    the bbox margin we picked.
    """
    assert 10.0 < corridor.total_length < 16.0


def test_corridor_ramp_density_matches_kurzhanskiy_2007(corridor, ctmsim):
    """On- and off-ramp density (per mile) should be in the same ballpark."""
    our_on_density = len(corridor.on_ramp_postmiles) / corridor.total_length
    our_off_density = len(corridor.off_ramp_postmiles) / corridor.total_length
    ctm_on_density = ctmsim["n_on"] / ctmsim["total_length"]
    ctm_off_density = ctmsim["n_off"] / ctmsim["total_length"]
    # I-210W averages ~1.5 on- and ~1.3 off-ramps per mile; allow 0.5 slack.
    assert abs(our_on_density - ctm_on_density) < 0.5, (
        f"on-ramp density {our_on_density:.2f}/mi vs CTMSIM {ctm_on_density:.2f}/mi"
    )
    assert abs(our_off_density - ctm_off_density) < 0.5, (
        f"off-ramp density {our_off_density:.2f}/mi vs CTMSIM {ctm_off_density:.2f}/mi"
    )


# ---- Cell layout structural checks ----------------------------------------


def test_cells_schema_matches_io_consumer(cells):
    """Output must include the columns ``freeway_from_dataframe`` reads from."""
    required_subset = {"length", "on_ramp", "off_ramp"}
    assert required_subset.issubset(cells.columns)
    assert cells["on_ramp"].dtype == bool
    assert cells["off_ramp"].dtype == bool
    assert (cells["length"] > 0).all()


def test_cell_breakpoints_are_monotonic_and_contiguous(cells):
    pm_starts = cells["pm_start"].to_numpy()
    pm_ends = cells["pm_end"].to_numpy()
    assert (pm_starts < pm_ends).all()
    # pm_end of cell k == pm_start of cell k+1.
    np.testing.assert_allclose(pm_ends[:-1], pm_starts[1:], atol=1e-9)


def test_cell_lengths_sum_to_corridor_total(cells, corridor):
    # cells.py rounds postmiles to 6 decimal places (~16 cm) to dedupe near-
    # duplicate breakpoints; allow for that roundoff over many cells.
    assert cells["length"].sum() == pytest.approx(corridor.total_length, abs=1e-4)


def test_off_ramp_count_close_to_kurzhanskiy_2007(cells, ctmsim):
    """Off-ramps stay in Kurzhanskiy 2007's order-of-magnitude.

    Exact match isn't expected: the tight bbox crop trims a few ramps near
    the edges, and OSM occasionally tags multi-exit interchanges differently
    from Kurzhanskiy 2007. The density check below is the load-bearing one.
    """
    n_off = int(cells["off_ramp"].sum())
    assert ctmsim["n_off"] - 6 <= n_off <= ctmsim["n_off"] + 5, (
        f"off-ramp count {n_off} far from CTMSIM {ctmsim['n_off']}"
    )


def test_on_ramp_count_close_to_kurzhanskiy_2007(cells, ctmsim):
    """On-ramps stay in Kurzhanskiy 2007's order-of-magnitude (see off-ramp comment)."""
    n_on = int(cells["on_ramp"].sum())
    assert ctmsim["n_on"] - 8 <= n_on <= ctmsim["n_on"] + 5, (
        f"on-ramp count {n_on} far from CTMSIM {ctmsim['n_on']}"
    )


def test_lane_count_distribution_overlaps_kurzhanskiy_2007(cells, ctmsim):
    """Mainline cells should mostly be 3-, 4-, or 5-lane (matches Kurzhanskiy 2007)."""
    ctm_set = set(ctmsim["lane_counts"])
    our_set = set(int(n) for n in cells["lanes"].unique())
    overlap = ctm_set & our_set
    assert overlap, (
        f"no overlap between our lane counts {sorted(our_set)} and "
        f"CTMSIM's {sorted(ctm_set)}"
    )
    # Bulk of cells should sit in {3,4,5,6} lanes -- CTMSIM uses 4-5 lanes
    # for most cells; OSM also picks up 6-lane sections where the express
    # carriageway runs in parallel with the general-purpose lanes.
    common_lane_share = cells["lanes"].isin([3, 4, 5, 6]).mean()
    assert common_lane_share > 0.85, (
        f"only {common_lane_share:.1%} of our cells have 3-6 lanes"
    )


def test_ramps_interleave_along_corridor(cells):
    """A real freeway has interleaved on/off ramps -- not all on then all off."""
    on_pms = cells.loc[cells["on_ramp"], "pm_start"].to_list()
    off_pms = cells.loc[cells["off_ramp"], "pm_end"].to_list()
    # First half and second half should each have both kinds.
    midpoint = (cells["pm_start"].iloc[0] + cells["pm_end"].iloc[-1]) / 2
    for label, pms in [("on", on_pms), ("off", off_pms)]:
        in_first = sum(1 for p in pms if p < midpoint)
        in_second = len(pms) - in_first
        assert in_first > 0 and in_second > 0, (
            f"{label}-ramps all clustered in one half of the corridor"
        )
