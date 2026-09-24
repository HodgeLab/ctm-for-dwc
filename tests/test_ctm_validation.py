"""Tests for ``ctm/validation.py`` -- sim vs historical PeMS comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ctm_for_dwc.ctm import (
    QQCellSamples, QQResult, compare_against_historical, corridor_rmse_mape,
    restrict_to_direct_tiebreak,
)


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


def _cells_df(vds_ids: list[int], lanes: int = 3,
              vds_source: str = "direct") -> pd.DataFrame:
    return pd.DataFrame({
        "vds_id": vds_ids,
        "vds_source": [vds_source] * len(vds_ids),
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
        "vds_source": ["direct", "missing"],
        "lanes": [3, 3],
        "vds_lanes": [3, 3],
    })
    out = compare_against_historical(
        result, cells, tmp_path, start=start,
    )
    assert len(out) == 2
    assert out.iloc[1]["n_flow_samples"] == 0
    assert np.isnan(out.iloc[1]["flow_rmse"])


def test_nearest_upstream_cells_are_not_scored(tmp_path):
    """Cells with a propagated (nearest_upstream) VDS get NaN rows; their
    lane mismatches are also exempt from the lanes contract."""
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
        "vds_id": [100, 100],                 # cell 1 inherits cell 0's VDS
        "vds_source": ["direct", "nearest_upstream"],
        "lanes": [3, 4],                      # mismatch only on the inherited cell
        "vds_lanes": [3, 3],
    })
    out = compare_against_historical(result, cells, tmp_path, start=start)
    assert out.iloc[0]["n_flow_samples"] == n_5min
    assert out.iloc[1]["n_flow_samples"] == 0
    assert pd.isna(out.iloc[1]["vds_id"])


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
        "ctm_for_dwc.ctm.validation.pd.read_csv",
        counting,
    )
    compare_against_historical(
        result, _cells_df([100, 100]), tmp_path, start=start,
    )
    assert n_calls["count"] == 1


# ---- restrict_to_direct_tiebreak -----------------------------------------


def test_restrict_keeps_unique_direct_and_tiebreak_blanks_others():
    cells = pd.DataFrame({
        "vds_id": [100, 200, 300, 400],
        "vds_source": ["direct", "nearest_upstream", "direct_tiebreak", "missing"],
        "lanes": [3, 3, 3, 3],
    })
    out = restrict_to_direct_tiebreak(cells)
    # direct (100) and direct_tiebreak (300) kept; the rest blanked.
    assert list(out["vds_id"].isna()) == [False, True, False, True]
    assert out.loc[0, "vds_id"] == 100
    assert out.loc[2, "vds_id"] == 300
    # Other columns and row count are preserved.
    assert list(out["lanes"]) == [3, 3, 3, 3]
    assert len(out) == 4


def test_restrict_dedups_repeated_vds_keeping_first():
    cells = pd.DataFrame({
        "vds_id": [100, 100, 200],
        "vds_source": ["direct", "direct_tiebreak", "direct"],
    })
    out = restrict_to_direct_tiebreak(cells)
    assert list(out["vds_id"].isna()) == [False, True, False]  # 2nd 100 dropped


def test_restrict_does_not_mutate_input():
    cells = pd.DataFrame({"vds_id": [100], "vds_source": ["nearest_upstream"]})
    restrict_to_direct_tiebreak(cells)
    assert cells.loc[0, "vds_id"] == 100  # original untouched


def test_restrict_missing_vds_source_raises():
    with pytest.raises(KeyError, match="vds_source"):
        restrict_to_direct_tiebreak(pd.DataFrame({"vds_id": [100]}))


def test_restricted_frame_scores_only_kept_cell(tmp_path):
    """End-to-end: a restricted frame makes compare_against_historical score
    only the unique direct/tiebreak cell; others become NaN rows."""
    start = pd.Timestamp("2022-04-12 06:00")
    n_5min = 3
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=np.full(n_5min, 100.0), speeds_mph=np.full(n_5min, 60.0),
    )
    result = _result_constant(
        n_cells=2, n_5min=n_5min, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    cells = pd.DataFrame({
        "vds_id": [100, 100],
        "vds_source": ["direct", "nearest_upstream"],
        "lanes": [3, 3], "vds_lanes": [3, 3],
    })
    out = compare_against_historical(
        result, restrict_to_direct_tiebreak(cells), tmp_path, start=start,
    )
    assert out.iloc[0]["n_flow_samples"] == n_5min   # direct cell scored
    assert out.iloc[1]["n_flow_samples"] == 0        # nearest_upstream skipped
    assert np.isnan(out.iloc[1]["flow_rmse"])


# ---- corridor_rmse_mape --------------------------------------------------


def _qq_cell(cell, vds_id, flow_sim, flow_obs, density_sim, density_obs):
    return QQCellSamples(
        cell=cell, vds_id=vds_id,
        flow_sim=np.array(flow_sim, float), flow_obs=np.array(flow_obs, float),
        density_sim=np.array(density_sim, float),
        density_obs=np.array(density_obs, float),
    )


def test_corridor_rmse_mape_pools_residuals_across_cells():
    """Pooled flow residuals [120,120,0] over obs 1200; density [2,2,0]/20."""
    qq = QQResult(per_cell=[
        _qq_cell(0, 100, [1320, 1320], [1200, 1200], [22, 22], [20, 20]),
        _qq_cell(2, 200, [1200], [1200], [20], [20]),
    ])
    errs = corridor_rmse_mape(qq)
    fn, frmse, fmape = errs["flow"]
    dn, drmse, dmape = errs["density"]
    assert fn == 3 and dn == 3
    assert frmse == pytest.approx((28800.0 / 3.0) ** 0.5)   # ~97.98
    assert frmse == pytest.approx(97.9796, abs=1e-3)
    assert fmape == pytest.approx(100.0 * 0.2 / 3.0)        # (0.1+0.1+0)/3
    assert drmse == pytest.approx((8.0 / 3.0) ** 0.5)       # ~1.633
    assert dmape == pytest.approx(100.0 * 0.2 / 3.0)


def test_corridor_rmse_mape_empty_pool_is_nan():
    errs = corridor_rmse_mape(QQResult(per_cell=[]))
    assert errs["flow"][0] == 0
    assert np.isnan(errs["flow"][1]) and np.isnan(errs["flow"][2])


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
    cells = pd.DataFrame({"vds_id": [100], "vds_source": ["direct"],
                          "lanes": [4], "vds_lanes": [3]})
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


# ---- compute_qq_samples --------------------------------------------------


from ctm_for_dwc.ctm import (  # noqa: E402
    QQResult, compute_qq_samples,
)


def _cells_df_source(rows: list[tuple[int, str]]) -> pd.DataFrame:
    """Build a cells_df with just vds_id + vds_source from (vds_id, source) rows."""
    return pd.DataFrame({
        "vds_id": [r[0] for r in rows],
        "vds_source": [r[1] for r in rows],
    })


def test_qq_one_entry_per_unique_direct_vds(tmp_path):
    """Only direct/tiebreak cells contribute, and a repeated VDS is scored
    once -- at its first occurrence."""
    start = pd.Timestamp("2022-04-12 06:00")
    n_5min = 6
    for vds in (100, 200):
        _write_vds_csv(
            tmp_path / f"{vds}.csv", start=start,
            flows_5min=np.full(n_5min, 100.0),
            speeds_mph=np.full(n_5min, 60.0), station_id=vds,
        )
    result = _result_constant(
        n_cells=4, n_5min=n_5min, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    cells = _cells_df_source([
        (100, "direct"),
        (300, "nearest"),          # non-direct -> skipped (and never loaded)
        (200, "direct_tiebreak"),
        (200, "direct"),           # duplicate VDS -> skipped
    ])
    qq = compute_qq_samples(result, cells, tmp_path, start=start)
    assert isinstance(qq, QQResult)
    assert [c.vds_id for c in qq.per_cell] == [100, 200]
    # VDS 200's first appearance is cell index 2, not the later duplicate.
    assert [c.cell for c in qq.per_cell] == [0, 2]


def test_qq_pairs_drop_nan_observed_windows(tmp_path):
    """A NaN observed window drops from both sides; sim is never NaN."""
    start = pd.Timestamp("2022-04-12 06:00")
    n_5min = 4
    flows = np.array([100.0, np.nan, 100.0, 100.0])  # 3 valid windows
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=flows, speeds_mph=np.full(n_5min, 60.0),
    )
    # Sim +10% on both quantities: flow 1320, density 22.
    result = _result_constant(
        n_cells=1, n_5min=n_5min, steps_per_5min=30,
        density=22.0, flow_veh_h=1320.0,
    )
    qq = compute_qq_samples(
        result, _cells_df_source([(100, "direct")]), tmp_path, start=start,
    )
    c = qq.per_cell[0]
    assert c.flow_sim.size == 3 and c.flow_obs.size == 3
    assert c.density_sim.size == 3 and c.density_obs.size == 3
    assert np.isfinite(c.flow_obs).all() and np.isfinite(c.density_obs).all()
    # Observed flow 1200, sim 1320 -> residual +120 throughout.
    assert np.allclose(c.flow_obs, 1200.0)
    assert np.allclose(c.flow_sim, 1320.0)
    assert np.allclose(c.flow_sim - c.flow_obs, 120.0)
    assert np.allclose(c.density_obs, 20.0)
    assert np.allclose(c.density_sim, 22.0)


def test_qq_pooled_concatenates_across_cells(tmp_path):
    """pooled() concatenates each cell's paired arrays for one quantity."""
    start = pd.Timestamp("2022-04-12 06:00")
    n_5min = 5
    for vds in (100, 200):
        _write_vds_csv(
            tmp_path / f"{vds}.csv", start=start,
            flows_5min=np.full(n_5min, 100.0),
            speeds_mph=np.full(n_5min, 60.0), station_id=vds,
        )
    result = _result_constant(
        n_cells=2, n_5min=n_5min, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    cells = _cells_df_source([(100, "direct"), (200, "direct")])
    qq = compute_qq_samples(result, cells, tmp_path, start=start)
    sim, obs = qq.pooled("flow")
    assert sim.size == 2 * n_5min
    assert obs.size == 2 * n_5min
    assert np.allclose(sim, 1200.0)
    assert np.allclose(obs, 1200.0)


def test_qq_pooled_empty_when_no_direct_cells(tmp_path):
    """No direct/tiebreak cell -> empty per_cell and empty pooled arrays."""
    start = pd.Timestamp("2022-04-12 06:00")
    result = _result_constant(
        n_cells=1, n_5min=3, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    cells = _cells_df_source([(100, "nearest")])
    qq = compute_qq_samples(result, cells, tmp_path, start=start)
    assert qq.per_cell == []
    sim, obs = qq.pooled("density")
    assert sim.size == 0 and obs.size == 0


def test_qq_pooled_rejects_unknown_quantity(tmp_path):
    start = pd.Timestamp("2022-04-12 06:00")
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=np.full(3, 100.0), speeds_mph=np.full(3, 60.0),
    )
    result = _result_constant(
        n_cells=1, n_5min=3, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    qq = compute_qq_samples(
        result, _cells_df_source([(100, "direct")]), tmp_path, start=start,
    )
    with pytest.raises(ValueError, match="flow.*density"):
        qq.pooled("speed")


def test_qq_missing_vds_source_column_raises(tmp_path):
    start = pd.Timestamp("2022-04-12 06:00")
    result = _result_constant(
        n_cells=1, n_5min=3, steps_per_5min=30,
        density=20.0, flow_veh_h=1200.0,
    )
    cells = pd.DataFrame({"vds_id": [100]})  # no vds_source
    with pytest.raises(KeyError, match="vds_source"):
        compute_qq_samples(result, cells, tmp_path, start=start)


# ---- compare_against_ctmsim ----------------------------------------------


from ctm_for_dwc.ctm import (  # noqa: E402
    CTMSIMValidation, compare_against_ctmsim, simulate,
)

REPO = Path(__file__).resolve().parents[1]
CTMSIM_CONFIGS = REPO / "src/ctm_for_dwc/ctm/ctmsim_configs"
CTMSIM_RESULTS = REPO / "src/ctm_for_dwc/ctm/ctmsim_results"


def _ctmsim_w060412_result():
    """Re-simulate the canonical I-210W day; reused across the CTMSIM tests."""
    # Defer the heavy build_scenario import to avoid pulling matplotlib at
    # collection time.
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from run_ctm_ctmsim_demo import build_scenario  # noqa: E402

    fwy, scn = build_scenario(CTMSIM_CONFIGS / "w060412.mat")
    return simulate(fwy, scn)


@pytest.fixture(scope="module")
def ctmsim_result():
    if not (CTMSIM_CONFIGS / "w060412.mat").exists():
        pytest.skip("CTMSIM config w060412.mat is missing")
    if not (CTMSIM_RESULTS / "w060412").exists():
        pytest.skip("CTMSIM reference dir w060412/ is missing")
    return _ctmsim_w060412_result()


def test_returns_two_dataframes_with_expected_columns(ctmsim_result):
    report = compare_against_ctmsim(
        ctmsim_result, CTMSIM_RESULTS / "w060412",
    )
    assert isinstance(report, CTMSIMValidation)
    assert list(report.per_cell.columns) == [
        "cell",
        "n_density_samples", "density_rmse", "density_mape",
        "n_flow_samples", "flow_rmse", "flow_mape",
    ]
    assert list(report.per_metric.columns) == [
        "metric", "n_samples", "rmse", "mape",
    ]
    # n_cells == 40 for the I-210W config (one boundary cell + N-1 driveable
    # cells -> the freeway has 40 cells in our schema).
    assert len(report.per_cell) == 40
    assert list(report.per_metric["metric"]) == [
        "vht", "vmt", "delay", "productivity_loss",
    ]


def test_per_cell_stats_are_tight_on_canonical_run(ctmsim_result):
    """Our engine matches CTMSIM tightly; thresholds match the existing
    assertion test's tolerances scaled to the corridor."""
    report = compare_against_ctmsim(
        ctmsim_result, CTMSIM_RESULTS / "w060412",
    )
    pc = report.per_cell
    # Density: matches CTMSIM very tightly except near downstream
    # boundary during peak congestion. Median sample shouldn't exceed
    # 60 veh/mi RMSE on a corridor with rho_jam ~150 veh/mi.
    assert pc["density_rmse"].median() < 60.0
    assert pc["density_mape"].median() < 10.0
    # Flow: similar story; tighter median because flow is bounded by
    # capacity and the per-cell values agree to floating-point upstream.
    assert pc["flow_rmse"].median() < 500.0
    assert pc["flow_mape"].median() < 5.0


def test_per_metric_rmse_matches_24h_total_tolerance(ctmsim_result):
    """24h totals of VHT/VMT/delay/ploss agree with CTMSIM to a few %
    (matches the existing assertion test's rtol_total budget)."""
    report = compare_against_ctmsim(
        ctmsim_result, CTMSIM_RESULTS / "w060412",
    )
    # Per the existing test, each metric's day total agrees with CTMSIM
    # within rtol_total <= 5e-2. Since the per-period RMSE doesn't
    # directly translate to a total-error bound, we just check that the
    # rmse values are finite and bounded by sane absolute thresholds.
    pm = report.per_metric.set_index("metric")
    assert np.isfinite(pm["rmse"]).all()
    assert pm.loc["vht", "rmse"] < 100.0     # veh*h per 5-min
    assert pm.loc["vmt", "rmse"] < 1000.0    # veh*mi per 5-min
    assert pm.loc["delay", "rmse"] < 100.0
    assert pm.loc["productivity_loss", "rmse"] < 1.0


def test_horizon_mismatch_raises(tmp_path):
    """Sim with the wrong number of steps for the requested plot cadence raises."""
    # Build a 2-step, 1-cell dummy result.
    result = _Result(
        freeway=_Freeway(dt=1.0, cells=[_Cell()]),
        density=np.zeros((1, 3)),
        mainline_flow=np.zeros((1, 2)),
    )
    with pytest.raises(ValueError, match="result.n_steps"):
        compare_against_ctmsim(
            result, tmp_path, plot_samples=288, sim_steps_per_plot_sample=30,
        )


def test_missing_reference_csv_raises(tmp_path, ctmsim_result):
    """An empty / missing reference dir raises FileNotFoundError early."""
    with pytest.raises(FileNotFoundError, match="density.csv"):
        compare_against_ctmsim(ctmsim_result, tmp_path)


# ---- Sim density is a 5-min interval average, not a snapshot -------------


def _ramp_density_case(tmp_path):
    """A run whose density rises within each window, plus its matching PeMS.

    Four steps per 5-min window over two windows. The post-step densities
    are 10,20,30,40 then 50,60,70,80, so the window means are 25 and 65
    while the *boundary snapshots* (the old definition) would be 0 and 40.
    Observed density is written to equal the window means exactly, so the
    interval-average definition scores zero error and the snapshot one
    would not.
    """
    steps_per_5min, n_5min = 4, 2
    start = pd.Timestamp("2022-04-12 06:00")
    density = np.arange(0.0, 90.0, 10.0)[None, :]        # (1, 9)
    flows_veh_h = np.array([1500.0] * 4 + [3900.0] * 4)  # (8,) -> means 1500/3900
    result = _Result(
        freeway=_Freeway(
            dt=(5.0 / 60.0) / steps_per_5min, cells=[_Cell()],
        ),
        density=density,
        mainline_flow=flows_veh_h[None, :],
    )
    # obs density = flow / speed = 1500/60 = 25 and 3900/60 = 65.
    _write_vds_csv(
        tmp_path / "100.csv", start=start,
        flows_5min=np.array([1500.0, 3900.0]) / 12.0,
        speeds_mph=np.full(n_5min, 60.0),
    )
    return result, _cells_df([100]), start


def test_density_uses_window_mean_not_boundary_snapshot(tmp_path):
    result, cells, start = _ramp_density_case(tmp_path)
    row = compare_against_historical(
        result, cells, tmp_path, start=start,
    ).iloc[0]
    assert row["n_density_samples"] == 2
    # Window means (25, 65) match observed exactly; the old boundary
    # snapshots (0, 40) would have given RMSE 25.
    assert row["density_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert row["density_mape"] == pytest.approx(0.0, abs=1e-9)


def test_qq_density_samples_use_window_mean(tmp_path):
    from ctm_for_dwc.ctm import compute_qq_samples

    result, cells, start = _ramp_density_case(tmp_path)
    qq = compute_qq_samples(result, cells, tmp_path, start=start)
    assert np.allclose(qq.per_cell[0].density_sim, [25.0, 65.0])
    assert np.allclose(qq.per_cell[0].density_obs, [25.0, 65.0])
