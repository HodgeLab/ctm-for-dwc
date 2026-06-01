"""Tests for ``scripts/regenerate_cell_artifacts.py``.

These exercise the Step-5 manual-edit workflow end-to-end:

1. Build a synthetic Corridor (no osmnx required).
2. Write ``cells.csv`` + sidecars to a temp dir, just like the Step-1 script.
3. Hand-edit ``cells.csv`` (or leave it alone).
4. Run :func:`regenerate_cell_artifacts.regenerate_cell_artifacts`.
5. Verify the rebuilt ``cells.geojson`` matches the edited postmiles and
   that the PNGs got written.
"""

from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from transportation_models.utils.ctm.cells import (
    cells_from_corridor,
    cells_to_geodataframe,
    corridor_to_artifacts,
)
from transportation_models.utils.ctm.osm import (
    Corridor,
    MainlineSegment,
    RampJunction,
)

# Make scripts/ importable so we can call the regenerator directly.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from regenerate_cell_artifacts import regenerate_cell_artifacts  # noqa: E402


# ---- Fixtures -------------------------------------------------------------


def _seg(pm_start, pm_end, lanes, coords) -> MainlineSegment:
    return MainlineSegment(
        u=int(pm_start * 1000), v=int(pm_end * 1000),
        geometry=LineString(coords),
        length=pm_end - pm_start, lanes=lanes,
        pm_start=pm_start, pm_end=pm_end,
    )


def _ramp(kind, postmile, name=None) -> RampJunction:
    return RampJunction(
        kind=kind, postmile=postmile,
        mainline_node=int(postmile * 1000), ramp_terminus_node=-1,
        geometry=Point(0.0, 0.0), lanes=1, name=name,
    )


@pytest.fixture
def step1_dir(tmp_path) -> Path:
    """A Step-1-like output dir: cells.csv + corridor sidecars."""
    corridor = Corridor(
        ref="I 210", direction="W", target_bearing=270.0,
        mainline_segments=[
            _seg(0.0, 1.0, 4, [(-118.40, 34.05), (-118.41, 34.05)]),
            _seg(1.0, 2.0, 5, [(-118.41, 34.05), (-118.42, 34.05)]),
        ],
        ramp_junctions=[_ramp("on", 0.5, name="HillOn")],
    )
    cells_df = cells_from_corridor(corridor)
    cells_df.to_csv(tmp_path / "cells.csv", index=False)
    corridor_to_artifacts(corridor, tmp_path)
    return tmp_path


# ---- Tests ----------------------------------------------------------------


def test_untouched_round_trip_matches_original_cells_geojson(step1_dir):
    """Running the regenerator without edits is geometrically idempotent."""
    # Stash the as-built cells.geojson for comparison.
    from transportation_models.utils.ctm.cells import corridor_from_artifacts
    original = cells_to_geodataframe(
        pd.read_csv(step1_dir / "cells.csv"),
        corridor_from_artifacts(step1_dir),
    )

    written = regenerate_cell_artifacts(step1_dir)
    rebuilt = gpd.read_file(written["cells_geojson"])

    assert len(rebuilt) == len(original)
    for o, r in zip(original.geometry, rebuilt.geometry):
        assert o.geom_type == r.geom_type
        assert list(o.coords) == pytest.approx(list(r.coords), abs=1e-9)
    # PNGs got written.
    assert written["corridor_map_png"].exists()
    assert written["cell_layout_png"].exists()


def test_merge_two_cells_yields_one_merged_geometry(step1_dir):
    """Merging cells 0+1 into a single 0..1 mi cell produces one LineString."""
    cells = pd.read_csv(step1_dir / "cells.csv")
    # Drop cell 1 (0.5..1.0) and stretch cell 0 to cover 0..1.
    cells.loc[0, "pm_end"] = 1.0
    cells.loc[0, "length"] = 1.0
    cells.loc[0, "on_ramp"] = False  # the on-ramp was at the old break
    cells.loc[0, "on_ramp_name"] = ""
    merged = cells.drop(index=1).reset_index(drop=True)
    merged.to_csv(step1_dir / "cells.csv", index=False)

    written = regenerate_cell_artifacts(step1_dir)
    rebuilt = gpd.read_file(written["cells_geojson"])
    # Original had 3 cells (0.0-0.5, 0.5-1.0, 1.0-2.0); now 2.
    assert len(rebuilt) == 2
    first = rebuilt.iloc[0]
    assert first.pm_start == pytest.approx(0.0)
    assert first.pm_end == pytest.approx(1.0)
    assert first.geometry.geom_type == "LineString"
    # First cell's geometry should run end-to-end of the upstream mainline seg.
    coords = list(first.geometry.coords)
    assert coords[0] == pytest.approx((-118.40, 34.05), abs=1e-6)
    assert coords[-1] == pytest.approx((-118.41, 34.05), abs=1e-6)


def test_split_a_cell_yields_two_geometries(step1_dir):
    """Splitting cell 2 (1.0..2.0) at 1.5 produces two half-length cells."""
    cells = pd.read_csv(step1_dir / "cells.csv")
    # Original row 2 = pm_start 1.0, pm_end 2.0, length 1.0, lanes 5.
    cells.loc[2, "pm_end"] = 1.5
    cells.loc[2, "length"] = 0.5
    # Insert a new row for 1.5..2.0 with the same lane count.
    new_row = cells.iloc[2].copy()
    new_row["pm_start"] = 1.5
    new_row["pm_end"] = 2.0
    new_row["length"] = 0.5
    new_row["on_ramp"] = False
    new_row["off_ramp"] = False
    new_row["on_ramp_name"] = ""
    new_row["off_ramp_name"] = ""
    split = pd.concat([cells, new_row.to_frame().T], ignore_index=True)
    split = split.sort_values("pm_start").reset_index(drop=True)
    split.to_csv(step1_dir / "cells.csv", index=False)

    written = regenerate_cell_artifacts(step1_dir)
    rebuilt = gpd.read_file(written["cells_geojson"])
    # Original 3 cells + 1 split -> 4 cells.
    assert len(rebuilt) == 4
    last_two = rebuilt.iloc[-2:]
    assert list(last_two["pm_start"]) == pytest.approx([1.0, 1.5])
    assert list(last_two["pm_end"]) == pytest.approx([1.5, 2.0])
    # Both still LineStrings.
    for geom in last_two.geometry:
        assert geom.geom_type == "LineString"
    # Split midpoint of the 1.0-2.0 mainline seg sits at -118.415 (halfway
    # between -118.41 and -118.42).
    mid_end = list(last_two.iloc[0].geometry.coords)[-1]
    assert mid_end[0] == pytest.approx(-118.415, abs=1e-6)


def test_regenerator_recomputes_length_warning_column(step1_dir):
    """Edits to ``length`` should refresh the ``length_warning`` annotation."""
    cells = pd.read_csv(step1_dir / "cells.csv")
    # Force a stale, wrong warning on a row whose length is clearly "ok".
    if "length_warning" not in cells.columns:
        cells["length_warning"] = "ok"
    cells.loc[0, "length_warning"] = "too_long"   # stale
    # Also shrink cell 0 to 0.05 mi to land it in "too_short".
    cells.loc[0, "pm_end"] = 0.05
    cells.loc[0, "length"] = 0.05
    cells.to_csv(step1_dir / "cells.csv", index=False)

    regenerate_cell_artifacts(step1_dir)
    refreshed = pd.read_csv(step1_dir / "cells.csv")
    assert refreshed.loc[0, "length_warning"] == "too_short"
