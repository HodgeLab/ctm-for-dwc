"""Tests for the CTM io adapter (P5): DataFrame / CSV <-> Freeway / Scenario."""

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ctm import examples, simulate
from transportation_models.utils.ctm.io import (
    REQUIRED_COLUMNS,
    freeway_from_csv,
    freeway_from_dataframe,
    freeway_to_dataframe,
    scenario_from_dataframes,
    scenario_to_dataframes,
)
from transportation_models.utils.ctm.model import Scenario


def _minimal_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "length": [1.0, 1.0],
            "q_max": [6000.0, 6000.0],
            "v_f": [60.0, 60.0],
            "w": [20.0, 20.0],
            "rho_jam": [400.0, 400.0],
        }
    )


def test_freeway_from_dataframe_required_columns_only():
    fwy = freeway_from_dataframe(_minimal_df(), dt=1.0 / 120.0)
    assert fwy.n_cells == 2
    c = fwy.cells[0]
    assert c.length == 1.0 and c.q_max == 6000.0 and c.v_f == 60.0
    assert c.w == 20.0 and c.rho_jam == 400.0
    # Optional fields default in :class:`Cell`. gamma defaults to 1.0
    # (CTMSIM canonical; matches cells_from_corridor's upstream-edge ramp
    # convention). xi defaults to 1.0 (on-ramp can use the cell's full space).
    assert c.rho_crit == pytest.approx(100.0)  # q_max / v_f
    assert c.gamma == 1.0
    assert c.xi == 1.0
    assert c.on_ramp_capacity == np.inf
    assert c.off_ramp_capacity == np.inf


def test_freeway_from_dataframe_optional_columns_are_used():
    df = _minimal_df()
    df["rho_crit"] = [100.0, 100.0]
    df["on_ramp"] = [False, True]
    df["off_ramp"] = [True, False]
    df["on_ramp_capacity"] = [np.inf, 1500.0]
    df["off_ramp_capacity"] = [800.0, np.inf]
    df["gamma"] = [0.0, 0.4]
    df["xi"] = [1.0, 0.7]
    fwy = freeway_from_dataframe(df, dt=1.0 / 120.0)
    assert fwy.cells[1].on_ramp is True
    assert fwy.cells[1].on_ramp_capacity == 1500.0
    assert fwy.cells[1].gamma == 0.4
    assert fwy.cells[1].xi == 0.7
    assert fwy.cells[0].off_ramp is True
    assert fwy.cells[0].off_ramp_capacity == 800.0


def test_optional_column_with_nan_falls_back_to_default():
    df = _minimal_df()
    df["gamma"] = [np.nan, 0.5]  # cell 0: NaN -> default 1.0; cell 1: 0.5
    fwy = freeway_from_dataframe(df, dt=1.0 / 120.0)
    assert fwy.cells[0].gamma == 1.0
    assert fwy.cells[1].gamma == 0.5


def test_missing_required_column_raises():
    df = _minimal_df().drop(columns=["w"])
    with pytest.raises(ValueError, match="Missing required column"):
        freeway_from_dataframe(df, dt=1.0 / 120.0)


def test_freeway_from_dataframe_runs_freeway_validation():
    # CFL violation: v_f*dt = 60 * (1/30) = 2 mi > length = 1 mi.
    with pytest.raises(ValueError, match="CFL"):
        freeway_from_dataframe(_minimal_df(), dt=1.0 / 30.0)


def test_freeway_from_dataframe_runs_cell_validation():
    df = _minimal_df()
    df.loc[0, "gamma"] = 1.5  # out of range
    with pytest.raises(ValueError, match="gamma"):
        freeway_from_dataframe(df, dt=1.0 / 120.0)


def test_freeway_to_dataframe_has_all_schema_columns():
    fwy = examples.example_1_freeway()
    df = freeway_to_dataframe(fwy)
    assert len(df) == fwy.n_cells
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    # Optional columns are also present (with defaults).
    for col in ("rho_crit", "gamma", "xi", "on_ramp_capacity", "off_ramp_capacity"):
        assert col in df.columns


def test_round_trip_via_dataframe():
    original = examples.example_1_freeway()
    rebuilt = freeway_from_dataframe(freeway_to_dataframe(original), dt=original.dt)
    assert rebuilt.n_cells == original.n_cells
    for orig, new in zip(original.cells, rebuilt.cells):
        for col in REQUIRED_COLUMNS + ("rho_crit", "gamma", "xi"):
            assert getattr(orig, col) == getattr(new, col), f"mismatch on {col}"
        assert orig.on_ramp_capacity == new.on_ramp_capacity
        assert orig.off_ramp_capacity == new.off_ramp_capacity


def test_round_trip_via_csv(tmp_path):
    original = examples.example_1_freeway()
    path = tmp_path / "corridor.csv"
    freeway_to_dataframe(original).to_csv(path, index=False)
    rebuilt = freeway_from_csv(path, dt=original.dt)
    assert rebuilt.n_cells == original.n_cells
    for orig, new in zip(original.cells, rebuilt.cells):
        for col in REQUIRED_COLUMNS:
            assert getattr(orig, col) == getattr(new, col), f"mismatch on {col}"
    # Boolean ramp flags survive the CSV round-trip (string "True"/"False" -> bool).
    assert rebuilt.cells[1].on_ramp is True
    assert rebuilt.cells[0].on_ramp is False


# ---- scenario io --------------------------------------------------------


def test_scenario_from_dataframes_minimal():
    fwy = examples.example_1_freeway()
    inflow = pd.Series([4800.0] * 5, name="inflow")
    scn = scenario_from_dataframes(fwy, inflow=inflow)
    assert scn.n_steps == 5
    assert scn.demand.shape == (2, 5)
    assert (scn.demand == 0).all()
    assert (scn.beta == 0).all()
    np.testing.assert_array_equal(scn.inflow, [4800.0] * 5)


def test_scenario_from_dataframes_sparse_demand_columns():
    """Cells absent from the demand frame's columns default to zero."""
    fwy = examples.example_1_freeway()
    inflow = pd.Series([4800.0] * 5)
    # Only specify demand for cell 1 (the cell that has the on-ramp).
    demand = pd.DataFrame({1: [1200.0] * 5})
    scn = scenario_from_dataframes(fwy, inflow=inflow, demand=demand)
    np.testing.assert_array_equal(scn.demand[0, :], np.zeros(5))
    np.testing.assert_array_equal(scn.demand[1, :], np.full(5, 1200.0))


def test_scenario_from_dataframes_rejects_row_count_mismatch():
    fwy = examples.example_1_freeway()
    inflow = pd.Series([4800.0] * 5)
    demand = pd.DataFrame({1: [1200.0] * 4})  # 4 rows, inflow has 5
    with pytest.raises(ValueError, match="5 steps"):
        scenario_from_dataframes(fwy, inflow=inflow, demand=demand)


def test_scenario_from_dataframes_rejects_out_of_range_column():
    fwy = examples.example_1_freeway()
    inflow = pd.Series([4800.0] * 5)
    demand = pd.DataFrame({5: [1200.0] * 5})  # no cell index 5
    with pytest.raises(ValueError, match=r"out of range \[0, 2\)"):
        scenario_from_dataframes(fwy, inflow=inflow, demand=demand)


def test_scenario_from_dataframes_respects_ramp_consistency():
    """Demand on a cell with on_ramp=False should fail at Scenario.validate."""
    fwy = examples.example_1_freeway()  # cell 0 has no on-ramp
    inflow = pd.Series([4800.0] * 5)
    demand = pd.DataFrame({0: [100.0] * 5})  # demand on cell 0 -- forbidden
    with pytest.raises(ValueError, match="cells without an on-ramp"):
        scenario_from_dataframes(fwy, inflow=inflow, demand=demand)


def test_scenario_round_trip_via_dataframes():
    fwy = examples.example_1_freeway()
    original = examples.example_1_scenario(steps=20, start="empty")
    frames = scenario_to_dataframes(original)
    rebuilt = scenario_from_dataframes(
        fwy,
        inflow=frames["inflow"],
        demand=frames["demand"],
        beta=frames["beta"],
        rho0=frames["rho0"].to_numpy(),
        q0=frames["q0"].to_numpy(),
    )
    np.testing.assert_array_equal(rebuilt.rho0, original.rho0)
    np.testing.assert_array_equal(rebuilt.q0, original.q0)
    np.testing.assert_array_equal(rebuilt.inflow, original.inflow)
    np.testing.assert_array_equal(rebuilt.demand, original.demand)
    np.testing.assert_array_equal(rebuilt.beta, original.beta)


def test_scenario_round_trip_yields_identical_simulation():
    """A round-tripped scenario simulates to the same trajectory as the original."""
    fwy = examples.example_1_freeway()
    original = examples.example_1_scenario(steps=120, start="empty")
    frames = scenario_to_dataframes(original)
    rebuilt = scenario_from_dataframes(
        fwy,
        inflow=frames["inflow"],
        demand=frames["demand"],
        beta=frames["beta"],
        rho0=frames["rho0"].to_numpy(),
        q0=frames["q0"].to_numpy(),
    )
    res_a = simulate(fwy, original)
    res_b = simulate(fwy, rebuilt)
    np.testing.assert_array_equal(res_a.density, res_b.density)
    np.testing.assert_array_equal(res_a.mainline_flow, res_b.mainline_flow)


def test_scenario_to_dataframes_shapes():
    scn = examples.example_1_scenario(steps=10, start="empty")
    frames = scenario_to_dataframes(scn)
    assert frames["demand"].shape == (10, 2)
    assert frames["beta"].shape == (10, 2)
    assert frames["inflow"].shape == (10,)
    assert frames["rho0"].shape == (2,)
    assert frames["q0"].shape == (2,)
    assert frames["demand"].index.name == "step"
    assert frames["demand"].columns.name == "cell"
