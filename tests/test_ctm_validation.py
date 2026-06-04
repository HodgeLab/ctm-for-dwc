"""Tests for ``utils/ctm/validation.py`` -- sim vs historical PeMS comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ctm import compare_against_historical


# ---- Fixture helpers -----------------------------------------------------


@dataclass
class _Cell:
    length: float = 0.5
    v_f: float = 60.0
    rho_crit: float = 30.0
    rho_jam: float = 150.0
    q_max: float = 1800.0


@dataclass
class _Freeway:
    """Just enough of a Freeway to pass through SimulationResult."""
    dt: float
    cells: list[_Cell]

    @property
    def n_cells(self) -> int:
        return len(self.cells)


@dataclass
class _Result:
    """Minimal SimulationResult stand-in carrying the arrays validation reads."""
    freeway: _Freeway
    density: np.ndarray
    mainline_flow: np.ndarray

    @property
    def n_steps(self) -> int:
        return self.mainline_flow.shape[1]


def _write_vds_csv(
    path: Path, *,
    start: pd.Timestamp,
    flows_5min: np.ndarray,
    speeds_mph: np.ndarray,
    station_id: int = 100,
) -> None:
    n = len(flows_5min)
    ts = pd.date_range(start=start, periods=n, freq="5min")
    pd.DataFrame({
        "timestamp": ts,
        "station": station_id,
        "pct_observed": 100.0,
        "total_flow_[veh/5-min]": flows_5min,
        "avg_speed_[mph]": speeds_mph,
    }).to_csv(path, index=False)


def _cells_df(vds_ids: list[int], lanes: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "vds_id": vds_ids,
        "lanes": [lanes] * len(vds_ids),
        "vds_lanes": [lanes] * len(vds_ids),
    })


def _result_constant(
    *, n_cells: int, n_5min: int, steps_per_5min: int,
    density: float, flow_veh_h: float,
) -> _Result:
    """Constant-state result: density and flow flat at the given values."""
    dt_h = (5.0 / 60.0) / steps_per_5min
    n_steps = n_5min * steps_per_5min
    return _Result(
        freeway=_Freeway(dt=dt_h, cells=[_Cell() for _ in range(n_cells)]),
        density=np.full((n_cells, n_steps + 1), density),
        mainline_flow=np.full((n_cells, n_steps), flow_veh_h),
    )


# ---- Exact-match cases ---------------------------------------------------


def test_zero_rmse_and_mape_when_sim_matches_observed(tmp_path):
    """Sim flow = 1200 veh/h, density = 20 veh/mi; PeMS flow = 100/5min
    (= 1200 veh/h), speed = 60 mph -> density = 20 veh/mi. Perfect match.
    """
    start = pd.Timestamp("2022-04-12 06:00")
    n_5min = 6
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=np.full(n_5min, 100.0),
        speeds_mph=np.full(n_5min, 60.0),
    )
    result = _result_constant(
        n_cells=1, n_5min=n_5min, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    out = compare_against_historical(
        result, _cells_df([100]), tmp_path, start=start,
    )
    assert out.shape == (1, 8)
    row = out.iloc[0]
    assert row["cell"] == 0
    assert row["vds_id"] == 100
    assert row["n_density_samples"] == n_5min
    assert row["n_flow_samples"] == n_5min
    assert row["density_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert row["density_mape"] == pytest.approx(0.0, abs=1e-9)
    assert row["flow_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert row["flow_mape"] == pytest.approx(0.0, abs=1e-9)


def test_recovers_known_rmse_and_mape(tmp_path):
    """Constant offset between sim and observed -> known RMSE/MAPE."""
    start = pd.Timestamp("2022-04-12 06:00")
    n_5min = 4
    # Observed: 1200 veh/h. Sim: 1320 veh/h (+10%). RMSE = 120, MAPE = 10%.
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=np.full(n_5min, 100.0),  # -> 1200 veh/h
        speeds_mph=np.full(n_5min, 60.0),   # -> density 20 veh/mi
    )
    result = _result_constant(
        n_cells=1, n_5min=n_5min, steps_per_5min=30,
        density=22.0, flow_veh_h=1320.0,  # +10% on both
    )
    out = compare_against_historical(
        result, _cells_df([100]), tmp_path, start=start,
    )
    row = out.iloc[0]
    assert row["flow_rmse"] == pytest.approx(120.0)
    assert row["flow_mape"] == pytest.approx(10.0)
    assert row["density_rmse"] == pytest.approx(2.0)
    assert row["density_mape"] == pytest.approx(10.0)


# ---- NaN handling --------------------------------------------------------


def test_nan_observed_samples_are_skipped(tmp_path):
    """Observed NaN samples drop out of the RMSE/MAPE; sample count reflects
    the remaining non-NaN points."""
    start = pd.Timestamp("2022-04-12 06:00")
    n_5min = 4
    flows = np.array([100.0, np.nan, 100.0, np.nan])  # 2 valid samples
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=flows,
        speeds_mph=np.full(n_5min, 60.0),
    )
    result = _result_constant(
        n_cells=1, n_5min=n_5min, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    out = compare_against_historical(
        result, _cells_df([100]), tmp_path, start=start,
    )
    # Density's observed value is flow/speed -> NaN where flow is NaN.
    # Flow's observed value is flow * 12 -> NaN where flow is NaN.
    # Both reduce to 2 valid samples.
    assert out.iloc[0]["n_flow_samples"] == 2
    assert out.iloc[0]["n_density_samples"] == 2


def test_zero_observed_samples_excluded_from_mape_only(tmp_path):
    """A zero-observed sample contributes to RMSE but is undefined for MAPE;
    MAPE should still produce a finite value from the non-zero samples."""
    start = pd.Timestamp("2022-04-12 06:00")
    flows = np.array([0.0, 100.0])  # one zero, one nonzero
    speeds = np.array([60.0, 60.0])
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=flows, speeds_mph=speeds,
    )
    # Sim is 0 at step 0 and 1200 veh/h at step 1 -> zero error throughout.
    n_5min = 2
    dt_h = (5.0 / 60.0) / 30
    n_steps = n_5min * 30
    density = np.zeros((1, n_steps + 1))
    mainline = np.zeros((1, n_steps))
    mainline[0, :30] = 0.0
    mainline[0, 30:] = 1200.0
    result = _Result(
        freeway=_Freeway(dt=dt_h, cells=[_Cell()]),
        density=density, mainline_flow=mainline,
    )
    out = compare_against_historical(
        result, _cells_df([100]), tmp_path, start=start,
    )
    # n_flow_samples counts non-NaN observed samples (both are non-NaN).
    assert out.iloc[0]["n_flow_samples"] == 2
    assert out.iloc[0]["flow_rmse"] == pytest.approx(0.0, abs=1e-9)
    # Flow MAPE: only the non-zero observed sample contributes; sim matches,
    # so MAPE = 0.
    assert out.iloc[0]["flow_mape"] == pytest.approx(0.0, abs=1e-9)


def test_all_nan_observed_yields_nan_stats(tmp_path):
    start = pd.Timestamp("2022-04-12 06:00")
    n_5min = 3
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=np.full(n_5min, np.nan),
        speeds_mph=np.full(n_5min, np.nan),
    )
    result = _result_constant(
        n_cells=1, n_5min=n_5min, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    out = compare_against_historical(
        result, _cells_df([100]), tmp_path, start=start,
    )
    row = out.iloc[0]
    assert row["n_flow_samples"] == 0
    assert row["n_density_samples"] == 0
    assert np.isnan(row["flow_rmse"])
    assert np.isnan(row["flow_mape"])


# ---- Cell-vds mapping ----------------------------------------------------


def test_cell_with_nan_vds_id_yields_nan_row(tmp_path):
    """A cell with no assigned VDS gets a NaN-stats row, not an error."""
    start = pd.Timestamp("2022-04-12 06:00")
    n_5min = 3
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=np.full(n_5min, 100.0),
        speeds_mph=np.full(n_5min, 60.0),
    )
    result = _result_constant(
        n_cells=2, n_5min=n_5min, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    cells = pd.DataFrame({
        "vds_id": [100, pd.NA],
        "lanes": [3, 3],
        "vds_lanes": [3, 3],
    })
    out = compare_against_historical(
        result, cells, tmp_path, start=start,
    )
    assert len(out) == 2
    assert out.iloc[1]["n_flow_samples"] == 0
    assert np.isnan(out.iloc[1]["flow_rmse"])


def test_caches_shared_vds_reads_once(tmp_path, monkeypatch):
    """A VDS shared by two cells is only loaded from disk once."""
    start = pd.Timestamp("2022-04-12 06:00")
    n_5min = 3
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=np.full(n_5min, 100.0),
        speeds_mph=np.full(n_5min, 60.0),
    )
    result = _result_constant(
        n_cells=2, n_5min=n_5min, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )

    n_calls = {"count": 0}
    orig = pd.read_csv

    def counting(*args, **kwargs):
        n_calls["count"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(
        "transportation_models.utils.ctm.validation.pd.read_csv",
        counting,
    )
    compare_against_historical(
        result, _cells_df([100, 100]), tmp_path, start=start,
    )
    assert n_calls["count"] == 1


# ---- Error paths ---------------------------------------------------------


def test_lane_mismatch_raises_pointing_at_step_5(tmp_path):
    start = pd.Timestamp("2022-04-12 06:00")
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=np.full(3, 100.0), speeds_mph=np.full(3, 60.0),
    )
    result = _result_constant(
        n_cells=1, n_5min=3, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    cells = pd.DataFrame({"vds_id": [100], "lanes": [4], "vds_lanes": [3]})
    with pytest.raises(ValueError, match="lanes != vds_lanes"):
        compare_against_historical(result, cells, tmp_path, start=start)


def test_misaligned_start_raises(tmp_path):
    start = pd.Timestamp("2022-04-12 06:02")  # not on 5-min grid
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=np.full(3, 100.0), speeds_mph=np.full(3, 60.0),
    )
    result = _result_constant(
        n_cells=1, n_5min=3, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    with pytest.raises(ValueError, match="5-min grid"):
        compare_against_historical(
            result, _cells_df([100]), tmp_path, start=start,
        )


def test_dt_not_dividing_5min_raises(tmp_path):
    start = pd.Timestamp("2022-04-12 06:00")
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=np.full(3, 100.0), speeds_mph=np.full(3, 60.0),
    )
    # dt = 7s -> 300/7 not integer.
    dt_h = 7.0 / 3600.0
    result = _Result(
        freeway=_Freeway(dt=dt_h, cells=[_Cell()]),
        density=np.zeros((1, 100)),
        mainline_flow=np.zeros((1, 99)),
    )
    with pytest.raises(ValueError, match="evenly divide 5 min"):
        compare_against_historical(
            result, _cells_df([100]), tmp_path, start=start,
        )


def test_cells_df_row_count_mismatch_raises(tmp_path):
    start = pd.Timestamp("2022-04-12 06:00")
    result = _result_constant(
        n_cells=2, n_5min=3, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    with pytest.raises(ValueError, match="one row per cell"):
        compare_against_historical(
            result, _cells_df([100]), tmp_path, start=start,
        )


def test_missing_vds_lanes_column_raises(tmp_path):
    start = pd.Timestamp("2022-04-12 06:00")
    result = _result_constant(
        n_cells=1, n_5min=3, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    cells = pd.DataFrame({"vds_id": [100], "lanes": [3]})  # no vds_lanes
    with pytest.raises(KeyError, match="vds_lanes"):
        compare_against_historical(
            result, cells, tmp_path, start=start,
        )
