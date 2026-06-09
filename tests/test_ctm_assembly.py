"""Unit tests for ``utils/ctm/assembly.py``.

These cover the Step-6 freeway-table assembly: joining Step-1+2 cells with
Step-4 calibration, per-lane → per-cell scaling for q_max / rho_jam /
rho_crit, ramp capacity lookup (with infinite-fallback semantics), and the
CFL Δt advisory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ctm import (
    FREEWAY_COLUMNS,
    assemble_freeway_table,
    compute_cfl_advisory,
    format_cfl_advisory,
    parse_ramp_vds_ids,
)
from transportation_models.utils.ctm.io import freeway_from_dataframe


# ---- Fixtures -------------------------------------------------------------


def _cells_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal cells_df with the columns assemble_freeway_table needs."""
    defaults = dict(
        on_ramp=False, off_ramp=False,
        on_ramp_vds_id=pd.NA, off_ramp_vds_id=pd.NA,
    )
    full = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full)


def _calibrated_df(rows: list[dict]) -> pd.DataFrame:
    """Mainline FD calibration table keyed on Station ID."""
    return pd.DataFrame(rows)


def _ramp_calibrated_df(rows: list[dict]) -> pd.DataFrame:
    """Ramp calibration table keyed on Station ID."""
    return pd.DataFrame(rows)


# ---- assemble_freeway_table ----------------------------------------------


def test_assembles_freeway_table_with_per_cell_scaling():
    """q_max, rho_jam, rho_crit scale by lanes; v_f, w pass through."""
    cells = _cells_df([
        dict(length=0.5, lanes=3, vds_id=100),
        dict(length=0.7, lanes=4, vds_id=200),
    ])
    cal = _calibrated_df([
        dict(**{"Station ID": 100,
                "capacity": 1800.0, "free_flow_speed": 60.0,
                "congestion_wave_speed": 15.0,
                "jam_density": 150.0, "critical_density": 30.0}),
        dict(**{"Station ID": 200,
                "capacity": 2000.0, "free_flow_speed": 65.0,
                "congestion_wave_speed": 16.0,
                "jam_density": 160.0, "critical_density": 31.0}),
    ])
    out = assemble_freeway_table(cells, cal)

    # Schema
    assert list(out.columns) == list(FREEWAY_COLUMNS)
    assert len(out) == 2

    # Cell 0: 3 lanes -> per-cell q_max = 1800 * 3.
    row0 = out.iloc[0]
    assert row0["length"] == pytest.approx(0.5)
    assert row0["q_max"] == pytest.approx(1800.0 * 3)
    assert row0["rho_jam"] == pytest.approx(150.0 * 3)
    assert row0["rho_crit"] == pytest.approx(30.0 * 3)
    # Speeds are intensive -- no scaling.
    assert row0["v_f"] == pytest.approx(60.0)
    assert row0["w"] == pytest.approx(15.0)
    assert row0["lanes"] == 3

    # Cell 1: 4 lanes.
    row1 = out.iloc[1]
    assert row1["q_max"] == pytest.approx(2000.0 * 4)
    assert row1["rho_jam"] == pytest.approx(160.0 * 4)


def test_assembly_carries_ramp_flags_through():
    cells = _cells_df([
        dict(length=0.5, lanes=3, vds_id=100, on_ramp=True, off_ramp=False),
        dict(length=0.5, lanes=3, vds_id=100, on_ramp=False, off_ramp=True),
    ])
    cal = _calibrated_df([dict(**{
        "Station ID": 100,
        "capacity": 1800.0, "free_flow_speed": 60.0,
        "congestion_wave_speed": 15.0,
        "jam_density": 150.0, "critical_density": 30.0,
    })])
    out = assemble_freeway_table(cells, cal)
    assert out.iloc[0]["on_ramp"] is np.True_ or out.iloc[0]["on_ramp"] == True
    assert out.iloc[0]["off_ramp"] == False
    assert out.iloc[1]["on_ramp"] == False
    assert out.iloc[1]["off_ramp"] == True


def test_ramp_capacity_lookup_uses_ramp_calibrated_table():
    """An on-ramp cell with a matched ramp VDS gets the calibrated capacity."""
    cells = _cells_df([
        dict(length=0.5, lanes=3, vds_id=100, on_ramp=True,
             on_ramp_vds_id=900),
    ])
    cal = _calibrated_df([dict(**{
        "Station ID": 100,
        "capacity": 1800.0, "free_flow_speed": 60.0,
        "congestion_wave_speed": 15.0,
        "jam_density": 150.0, "critical_density": 30.0,
    })])
    ramps = _ramp_calibrated_df([
        dict(**{"Station ID": 900, "ramp_capacity_[veh/hr]": 1100.0}),
    ])
    out = assemble_freeway_table(cells, cal, ramp_calibrated_df=ramps)
    assert out.iloc[0]["on_ramp_capacity"] == pytest.approx(1100.0)
    # No off-ramp here, so off_ramp_capacity stays NaN.
    assert np.isnan(out.iloc[0]["off_ramp_capacity"])


def test_missing_ramp_calibration_yields_nan_capacity():
    """An on-ramp cell with no matching entry in ramp_calibrated -> NaN.

    NaN is the signal :func:`freeway_from_dataframe` reads as "fall back
    to the Cell default of infinity".
    """
    cells = _cells_df([
        dict(length=0.5, lanes=3, vds_id=100, on_ramp=True,
             on_ramp_vds_id=900),
    ])
    cal = _calibrated_df([dict(**{
        "Station ID": 100,
        "capacity": 1800.0, "free_flow_speed": 60.0,
        "congestion_wave_speed": 15.0,
        "jam_density": 150.0, "critical_density": 30.0,
    })])
    # Empty ramp calibration table.
    ramps = _ramp_calibrated_df([
        dict(**{"Station ID": 901, "ramp_capacity_[veh/hr]": 1100.0}),
    ])
    out = assemble_freeway_table(cells, cal, ramp_calibrated_df=ramps)
    assert np.isnan(out.iloc[0]["on_ramp_capacity"])


def test_no_ramp_capacity_passed_when_no_ramp_calibration_table():
    """Passing ramp_calibrated_df=None leaves both ramp columns NaN."""
    cells = _cells_df([
        dict(length=0.5, lanes=3, vds_id=100, on_ramp=True,
             on_ramp_vds_id=900),
    ])
    cal = _calibrated_df([dict(**{
        "Station ID": 100,
        "capacity": 1800.0, "free_flow_speed": 60.0,
        "congestion_wave_speed": 15.0,
        "jam_density": 150.0, "critical_density": 30.0,
    })])
    out = assemble_freeway_table(cells, cal)
    assert np.isnan(out.iloc[0]["on_ramp_capacity"])
    assert np.isnan(out.iloc[0]["off_ramp_capacity"])


def test_cell_without_vds_id_raises():
    """A cell with vds_id NaN can't be assembled -- the upstream pipeline
    must resolve it (Step 2's Dervisoglu fallback) first."""
    cells = _cells_df([
        dict(length=0.5, lanes=3, vds_id=100),
        dict(length=0.5, lanes=3, vds_id=pd.NA),
    ])
    cal = _calibrated_df([dict(**{
        "Station ID": 100,
        "capacity": 1800.0, "free_flow_speed": 60.0,
        "congestion_wave_speed": 15.0,
        "jam_density": 150.0, "critical_density": 30.0,
    })])
    with pytest.raises(ValueError, match="no vds_id"):
        assemble_freeway_table(cells, cal)


def test_missing_required_cells_column_raises():
    cells = pd.DataFrame({"length": [0.5], "vds_id": [100]})  # no lanes
    cal = _calibrated_df([dict(**{
        "Station ID": 100,
        "capacity": 1800.0, "free_flow_speed": 60.0,
        "congestion_wave_speed": 15.0,
        "jam_density": 150.0, "critical_density": 30.0,
    })])
    with pytest.raises(KeyError, match="lanes"):
        assemble_freeway_table(cells, cal)


def test_assembled_table_feeds_freeway_from_dataframe():
    """End-to-end: the assembled CSV must build a valid Freeway."""
    cells = _cells_df([
        dict(length=0.25, lanes=3, vds_id=100),
        dict(length=0.30, lanes=3, vds_id=100, on_ramp=True,
             on_ramp_vds_id=900),
    ])
    cal = _calibrated_df([dict(**{
        "Station ID": 100,
        "capacity": 1800.0, "free_flow_speed": 60.0,
        "congestion_wave_speed": 15.0,
        "jam_density": 150.0, "critical_density": 30.0,
    })])
    ramps = _ramp_calibrated_df([dict(**{
        "Station ID": 900, "ramp_capacity_[veh/hr]": 1100.0,
    })])
    out = assemble_freeway_table(cells, cal, ramp_calibrated_df=ramps)

    # Pick a tiny dt well below the CFL bound: 0.25 mi / 60 mi/h = 15 s.
    dt_s = 5.0
    freeway = freeway_from_dataframe(out, dt=dt_s / 3600.0)
    assert freeway.n_cells == 2
    assert freeway.cells[0].q_max == pytest.approx(1800.0 * 3)
    assert freeway.cells[1].on_ramp is True
    assert freeway.cells[1].on_ramp_capacity == pytest.approx(1100.0)
    # Cell 0 has no ramp -> capacity defaults to inf in the Cell.
    assert np.isinf(freeway.cells[0].on_ramp_capacity)


# ---- compute_cfl_advisory + format_cfl_advisory --------------------------


def test_cfl_advisory_finds_the_binding_cell():
    """Δt_max = min_i l_i / v_f_i; the binding cell is the argmin."""
    freeway_df = pd.DataFrame({
        "length": [0.5, 0.2, 0.7],
        "v_f": [60.0, 60.0, 60.0],
    })
    adv = compute_cfl_advisory(freeway_df)
    # The 0.2-mi cell binds: dt_max = 0.2/60 h = 12 s.
    assert adv.binding_cell_idx == 1
    assert adv.binding_dt_max_s == pytest.approx(12.0)
    assert adv.binding_cell_length_mi == pytest.approx(0.2)
    assert adv.binding_cell_v_f_mph == pytest.approx(60.0)


def test_cfl_advisory_per_cell_dt_max_matches_lengths_and_speeds():
    freeway_df = pd.DataFrame({
        "length": [0.5, 0.2],
        "v_f": [60.0, 30.0],
    })
    adv = compute_cfl_advisory(freeway_df)
    # Cell 0: 0.5/60 = 0.00833 h; cell 1: 0.2/30 = 0.00667 h.
    assert adv.per_cell_dt_max_h[0] == pytest.approx(0.5 / 60.0)
    assert adv.per_cell_dt_max_h[1] == pytest.approx(0.2 / 30.0)
    assert adv.binding_cell_idx == 1


def test_cfl_advisory_suggested_dt_rounds_down_to_5s():
    """Suggested Δt is the largest 5-s multiple strictly below the CFL bound."""
    # 0.2 mi at 60 mi/h -> 12 s; expect suggested = 10 s.
    freeway_df = pd.DataFrame({"length": [0.2], "v_f": [60.0]})
    adv = compute_cfl_advisory(freeway_df)
    assert adv.binding_dt_max_s == pytest.approx(12.0)
    assert adv.suggested_dt_s == pytest.approx(10.0)

    # 0.5 mi at 60 mi/h -> 30 s; the largest 5-s multiple strictly below
    # 30.0 is 25 s.
    freeway_df_2 = pd.DataFrame({"length": [0.5], "v_f": [60.0]})
    adv_2 = compute_cfl_advisory(freeway_df_2)
    assert adv_2.suggested_dt_s == pytest.approx(25.0)


def test_cfl_advisory_handles_sub_5s_bound():
    """When the CFL bound is below 5 s, return the bound itself."""
    # 0.01 mi at 60 mi/h -> 0.6 s.
    freeway_df = pd.DataFrame({"length": [0.01], "v_f": [60.0]})
    adv = compute_cfl_advisory(freeway_df)
    assert adv.binding_dt_max_s == pytest.approx(0.6)
    assert adv.suggested_dt_s == pytest.approx(0.6)


def test_format_cfl_advisory_renders_all_key_fields():
    freeway_df = pd.DataFrame({
        "length": [0.5, 0.2, 0.7],
        "v_f": [60.0, 60.0, 60.0],
    })
    adv = compute_cfl_advisory(freeway_df)
    rendered = format_cfl_advisory(adv)
    assert "CFL" in rendered
    assert "binding cell idx : 1" in rendered
    assert "12.0 s" in rendered or "12.0" in rendered
    assert "suggested" in rendered.lower()


# ---- parse_ramp_vds_ids --------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (None, []),
    (np.nan, []),
    (pd.NA, []),
    (12345, [12345]),
    (np.int64(12345), [12345]),
    (12345.0, [12345]),
    ("12345", [12345]),
    ("12345;67890", [12345, 67890]),
    ("12345,67890", [12345, 67890]),
    ("12345, 67890", [12345, 67890]),
    ("[12345, 67890]", [12345, 67890]),
    ("12345 67890", [12345, 67890]),
    ([12345, 67890], [12345, 67890]),
    ((12345, 67890), [12345, 67890]),
    (np.array([12345, 67890]), [12345, 67890]),
    ("", []),
    ("   ", []),
])
def test_parse_ramp_vds_ids_handles_every_input_shape(value, expected):
    assert parse_ramp_vds_ids(value) == expected


# ---- ramp-capacity assembly with list-valued cells -----------------------


def test_ramp_capacity_sums_across_multiple_vdss_on_one_cell():
    """A Step-5 manual edit can pile multiple on-ramp VDSs into one cell;
    the cell's R_i = sum of per-VDS calibrated capacities."""
    cells = _cells_df([
        dict(length=0.5, lanes=3, vds_id=100,
             on_ramp=True, on_ramp_vds_id="900;901"),
    ])
    cal = _calibrated_df([dict(**{
        "Station ID": 100, "capacity": 1800.0, "free_flow_speed": 60.0,
        "congestion_wave_speed": 15.0,
        "jam_density": 150.0, "critical_density": 30.0,
    })])
    ramps = _ramp_calibrated_df([
        dict(**{"Station ID": 900, "ramp_capacity_[veh/hr]": 1100.0}),
        dict(**{"Station ID": 901, "ramp_capacity_[veh/hr]":  400.0}),
    ])
    out = assemble_freeway_table(cells, cal, ramp_calibrated_df=ramps)
    assert out.iloc[0]["on_ramp_capacity"] == pytest.approx(1500.0)


def test_ramp_capacity_partial_coverage_warns_and_sums_present():
    """Some ids in the list are calibrated, others aren't; the present ones
    are summed and a warning flags the partial coverage."""
    cells = _cells_df([
        dict(length=0.5, lanes=3, vds_id=100,
             on_ramp=True, on_ramp_vds_id="900;902"),
    ])
    cal = _calibrated_df([dict(**{
        "Station ID": 100, "capacity": 1800.0, "free_flow_speed": 60.0,
        "congestion_wave_speed": 15.0,
        "jam_density": 150.0, "critical_density": 30.0,
    })])
    ramps = _ramp_calibrated_df([
        dict(**{"Station ID": 900, "ramp_capacity_[veh/hr]": 1100.0}),
        # 902 deliberately absent.
    ])
    with pytest.warns(UserWarning, match="partial"):
        out = assemble_freeway_table(cells, cal, ramp_calibrated_df=ramps)
    assert out.iloc[0]["on_ramp_capacity"] == pytest.approx(1100.0)


# ---- fd_vds_id (Step-5 FD-override column) -------------------------------


def _two_cal_rows() -> pd.DataFrame:
    """Two stations with deliberately different FD parameters so a redirect
    is observable in the assembled table."""
    return _calibrated_df([
        dict(**{"Station ID": 100, "capacity": 1800.0,
                "free_flow_speed": 60.0, "congestion_wave_speed": 15.0,
                "jam_density": 150.0, "critical_density": 30.0}),
        dict(**{"Station ID": 200, "capacity": 2200.0,
                "free_flow_speed": 65.0, "congestion_wave_speed": 16.0,
                "jam_density": 170.0, "critical_density": 32.0}),
    ])


def test_fd_vds_id_redirects_fd_lookup_to_override():
    """When fd_vds_id is set, the row pulls FD params from the override."""
    cells = _cells_df([
        dict(length=0.5, lanes=3, vds_id=100, fd_vds_id=200),
    ])
    out = assemble_freeway_table(cells, _two_cal_rows())
    # Override -> station 200's FD, scaled by lanes=3.
    assert out.iloc[0]["q_max"] == pytest.approx(2200.0 * 3)
    assert out.iloc[0]["v_f"] == pytest.approx(65.0)
    assert out.iloc[0]["rho_jam"] == pytest.approx(170.0 * 3)


def test_fd_vds_id_nan_falls_back_to_vds_id():
    """Rows where fd_vds_id is NaN keep using vds_id (per-row fallback)."""
    cells = _cells_df([
        dict(length=0.5, lanes=3, vds_id=100, fd_vds_id=pd.NA),
        dict(length=0.5, lanes=3, vds_id=100, fd_vds_id=200),
    ])
    out = assemble_freeway_table(cells, _two_cal_rows())
    # Row 0: fd_vds_id NaN -> falls back to vds_id=100 (capacity 1800).
    assert out.iloc[0]["q_max"] == pytest.approx(1800.0 * 3)
    # Row 1: fd_vds_id=200 -> override (capacity 2200).
    assert out.iloc[1]["q_max"] == pytest.approx(2200.0 * 3)


def test_no_fd_vds_id_column_keeps_existing_behavior():
    """A cells_df without the override column behaves exactly as before."""
    cells = _cells_df([dict(length=0.5, lanes=3, vds_id=100)])
    assert "fd_vds_id" not in cells.columns
    out = assemble_freeway_table(cells, _two_cal_rows())
    assert out.iloc[0]["q_max"] == pytest.approx(1800.0 * 3)


def test_fd_vds_id_set_when_vds_id_is_nan_resolves_for_fd_lookup():
    """If vds_id is NaN but fd_vds_id is set, assembly succeeds (the FD
    lookup has a target). Downstream consumers that need vds_id directly
    will still raise their own errors when they hit that cell."""
    cells = _cells_df([
        dict(length=0.5, lanes=3, vds_id=pd.NA, fd_vds_id=200),
    ])
    out = assemble_freeway_table(cells, _two_cal_rows())
    assert out.iloc[0]["q_max"] == pytest.approx(2200.0 * 3)


def test_both_vds_id_and_fd_vds_id_nan_raises():
    """If neither column has a value for a row, the error message
    mentions both columns so the user knows the override is an option."""
    cells = _cells_df([
        dict(length=0.5, lanes=3, vds_id=100),
        dict(length=0.5, lanes=3, vds_id=pd.NA, fd_vds_id=pd.NA),
    ])
    with pytest.raises(ValueError, match=r"fd_vds_id"):
        assemble_freeway_table(cells, _two_cal_rows())


def test_ramp_capacity_all_missing_in_list_falls_back_to_nan():
    """If every id in the list is missing from the calibration table, the
    cell's capacity stays NaN -> freeway_from_dataframe defaults to ∞."""
    cells = _cells_df([
        dict(length=0.5, lanes=3, vds_id=100,
             on_ramp=True, on_ramp_vds_id="900;901"),
    ])
    cal = _calibrated_df([dict(**{
        "Station ID": 100, "capacity": 1800.0, "free_flow_speed": 60.0,
        "congestion_wave_speed": 15.0,
        "jam_density": 150.0, "critical_density": 30.0,
    })])
    ramps = _ramp_calibrated_df([
        # No ids 900 or 901 present.
        dict(**{"Station ID": 902, "ramp_capacity_[veh/hr]": 1100.0}),
    ])
    out = assemble_freeway_table(cells, cal, ramp_calibrated_df=ramps)
    assert np.isnan(out.iloc[0]["on_ramp_capacity"])
