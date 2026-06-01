"""Tests for ``utils/ctm/scenario.py`` -- the PeMS -> scenario adapters.

Covers :func:`inflow_from_vds` (resampling, timestamp bounds, alignment
errors, NaN rejection) and :func:`initial_state_from_vds` (lane-match
density recovery, lane-mismatch rejection, missing-vds errors).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ctm import (
    inflow_from_vds,
    initial_state_from_vds,
)


# ---- Fixture helpers -----------------------------------------------------


def _write_vds_csv(
    path: Path,
    *,
    start: pd.Timestamp,
    n_rows: int,
    flows_5min: np.ndarray | None = None,
    speeds_mph: np.ndarray | None = None,
    station_id: int = 100,
) -> None:
    """Write a PeMS-style per-VDS CSV with a contiguous 5-min grid."""
    timestamps = pd.date_range(start=start, periods=n_rows, freq="5min")
    if flows_5min is None:
        flows_5min = np.full(n_rows, 100.0)  # 1200 veh/h flat
    if speeds_mph is None:
        speeds_mph = np.full(n_rows, 60.0)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "station": station_id,
        "pct_observed": 100.0,
        "total_flow_[veh/5-min]": flows_5min,
        "avg_speed_[mph]": speeds_mph,
    })
    df.to_csv(path, index=False)


# ---- inflow_from_vds: sub-5-min branch -----------------------------------


def test_inflow_sub_5min_constant_hold(tmp_path):
    """dt = 10 s -> 30 sim steps per 5-min sample, each carrying flow * 12."""
    path = tmp_path / "100.csv"
    flows = np.array([100.0, 150.0, 200.0])  # veh/5-min -> 1200, 1800, 2400 veh/h
    _write_vds_csv(
        path, start=pd.Timestamp("2022-01-01 12:00"),
        n_rows=3, flows_5min=flows,
    )
    dt_h = 10.0 / 3600.0  # 10 s
    out = inflow_from_vds(
        path, dt=dt_h,
        start=pd.Timestamp("2022-01-01 12:00"),
        end=pd.Timestamp("2022-01-01 12:15"),
    )
    assert out.shape == (90,)  # 15 min / 10 s
    # First 30 steps = 1200 veh/h, next 30 = 1800, last 30 = 2400.
    np.testing.assert_allclose(out[:30], 1200.0)
    np.testing.assert_allclose(out[30:60], 1800.0)
    np.testing.assert_allclose(out[60:], 2400.0)


def test_inflow_dt_equals_5min_one_step_per_sample(tmp_path):
    """dt = 5 min hits the boundary -- exactly one sim step per 5-min sample."""
    path = tmp_path / "100.csv"
    flows = np.array([100.0, 150.0])
    _write_vds_csv(
        path, start=pd.Timestamp("2022-01-01 12:00"),
        n_rows=2, flows_5min=flows,
    )
    out = inflow_from_vds(
        path, dt=5.0 / 60.0,
        start=pd.Timestamp("2022-01-01 12:00"),
        end=pd.Timestamp("2022-01-01 12:10"),
    )
    np.testing.assert_allclose(out, [1200.0, 1800.0])


def test_inflow_end_is_exclusive(tmp_path):
    """end=12:05 with dt=10s -> 30 steps, not 31 (samples at 12:00..12:00:50)."""
    path = tmp_path / "100.csv"
    _write_vds_csv(
        path, start=pd.Timestamp("2022-01-01 12:00"),
        n_rows=2, flows_5min=np.array([100.0, 999.0]),
    )
    out = inflow_from_vds(
        path, dt=10.0 / 3600.0,
        start=pd.Timestamp("2022-01-01 12:00"),
        end=pd.Timestamp("2022-01-01 12:05"),
    )
    assert out.shape == (30,)
    # 12:05's sample (flow 999) must not appear in the output.
    np.testing.assert_allclose(out, 1200.0)


def test_inflow_respects_start_offset_into_csv(tmp_path):
    """Asking for [12:10, 12:20) inside a 4-hour CSV returns just that window."""
    path = tmp_path / "100.csv"
    # 1 hour of data with distinct per-5-min flows so the window is identifiable.
    flows = np.arange(48, dtype=float) * 10.0  # 4 hours = 48 samples
    _write_vds_csv(
        path, start=pd.Timestamp("2022-01-01 12:00"),
        n_rows=48, flows_5min=flows,
    )
    out = inflow_from_vds(
        path, dt=5.0 / 60.0,
        start=pd.Timestamp("2022-01-01 12:10"),
        end=pd.Timestamp("2022-01-01 12:20"),
    )
    # Samples at 12:10 and 12:15 -> indices 2 and 3 of the flows array.
    np.testing.assert_allclose(out, [20.0 * 12, 30.0 * 12])


# ---- inflow_from_vds: super-5-min branch ---------------------------------


def test_inflow_super_5min_hourly_aggregation(tmp_path):
    """dt=10 min: 12 5-min counts/hour summed -> hourly veh/h, replicated 6x."""
    path = tmp_path / "100.csv"
    # Two hours: hour 0 has all 100/5-min counts (sum=1200 veh/h), hour 1 has
    # all 200/5-min counts (sum=2400 veh/h).
    flows = np.concatenate([np.full(12, 100.0), np.full(12, 200.0)])
    _write_vds_csv(
        path, start=pd.Timestamp("2022-01-01 12:00"),
        n_rows=24, flows_5min=flows,
    )
    out = inflow_from_vds(
        path, dt=10.0 / 60.0,
        start=pd.Timestamp("2022-01-01 12:00"),
        end=pd.Timestamp("2022-01-01 14:00"),
    )
    # 2 hours / 10 min = 12 sim steps; first 6 = 1200, next 6 = 2400.
    assert out.shape == (12,)
    np.testing.assert_allclose(out[:6], 1200.0)
    np.testing.assert_allclose(out[6:], 2400.0)


def test_inflow_super_5min_requires_hourly_aligned_bounds(tmp_path):
    """dt > 5 min must align to hourly grid; misalignment raises."""
    path = tmp_path / "100.csv"
    _write_vds_csv(
        path, start=pd.Timestamp("2022-01-01 12:00"),
        n_rows=24,
    )
    with pytest.raises(ValueError, match="1-hour grid"):
        inflow_from_vds(
            path, dt=10.0 / 60.0,
            start=pd.Timestamp("2022-01-01 12:30"),  # not on the hour
            end=pd.Timestamp("2022-01-01 13:30"),
        )


# ---- inflow_from_vds: error paths ----------------------------------------


def test_inflow_raises_on_misaligned_start(tmp_path):
    path = tmp_path / "100.csv"
    _write_vds_csv(path, start=pd.Timestamp("2022-01-01 12:00"), n_rows=12)
    with pytest.raises(ValueError, match="5-minute grid"):
        inflow_from_vds(
            path, dt=10.0 / 3600.0,
            start=pd.Timestamp("2022-01-01 12:02"),  # not on a 5-min tick
            end=pd.Timestamp("2022-01-01 12:07"),
        )


def test_inflow_raises_on_dt_not_dividing_5min(tmp_path):
    """dt=7 s doesn't evenly divide 5 min (300/7 is not an integer)."""
    path = tmp_path / "100.csv"
    _write_vds_csv(path, start=pd.Timestamp("2022-01-01 12:00"), n_rows=12)
    with pytest.raises(ValueError, match="evenly divide 5 min"):
        inflow_from_vds(
            path, dt=7.0 / 3600.0,
            start=pd.Timestamp("2022-01-01 12:00"),
            end=pd.Timestamp("2022-01-01 12:05"),
        )


def test_inflow_raises_on_nan_in_window(tmp_path):
    path = tmp_path / "100.csv"
    flows = np.array([100.0, np.nan, 200.0])
    _write_vds_csv(
        path, start=pd.Timestamp("2022-01-01 12:00"),
        n_rows=3, flows_5min=flows,
    )
    with pytest.raises(ValueError, match="NaN"):
        inflow_from_vds(
            path, dt=10.0 / 3600.0,
            start=pd.Timestamp("2022-01-01 12:00"),
            end=pd.Timestamp("2022-01-01 12:15"),
        )


def test_inflow_nan_outside_window_is_fine(tmp_path):
    """NaN at 12:10 is fine if we only ask for [12:00, 12:10)."""
    path = tmp_path / "100.csv"
    flows = np.array([100.0, 150.0, np.nan])
    _write_vds_csv(
        path, start=pd.Timestamp("2022-01-01 12:00"),
        n_rows=3, flows_5min=flows,
    )
    out = inflow_from_vds(
        path, dt=5.0 / 60.0,
        start=pd.Timestamp("2022-01-01 12:00"),
        end=pd.Timestamp("2022-01-01 12:10"),
    )
    np.testing.assert_allclose(out, [1200.0, 1800.0])


def test_inflow_raises_on_short_csv(tmp_path):
    """Asking for 30 min when the CSV only has 15 min raises."""
    path = tmp_path / "100.csv"
    _write_vds_csv(path, start=pd.Timestamp("2022-01-01 12:00"), n_rows=3)
    with pytest.raises(ValueError, match="expected 6"):
        inflow_from_vds(
            path, dt=5.0 / 60.0,
            start=pd.Timestamp("2022-01-01 12:00"),
            end=pd.Timestamp("2022-01-01 12:30"),
        )


def test_inflow_raises_on_dt_above_one_hour(tmp_path):
    path = tmp_path / "100.csv"
    _write_vds_csv(path, start=pd.Timestamp("2022-01-01 12:00"), n_rows=24)
    with pytest.raises(ValueError, match="in \\(0, 1 hour\\]"):
        inflow_from_vds(
            path, dt=2.0,  # 2 hours
            start=pd.Timestamp("2022-01-01 12:00"),
            end=pd.Timestamp("2022-01-01 14:00"),
        )


def test_inflow_raises_when_end_before_start(tmp_path):
    path = tmp_path / "100.csv"
    _write_vds_csv(path, start=pd.Timestamp("2022-01-01 12:00"), n_rows=12)
    with pytest.raises(ValueError, match="strictly after start"):
        inflow_from_vds(
            path, dt=5.0 / 60.0,
            start=pd.Timestamp("2022-01-01 12:30"),
            end=pd.Timestamp("2022-01-01 12:00"),
        )


# ---- initial_state_from_vds ----------------------------------------------


def _cells_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_initial_state_when_lanes_match(tmp_path):
    """ρ_i = total_flow / speed when lanes == vds_lanes."""
    # Two cells, each with its own VDS, lanes=3 everywhere.
    _write_vds_csv(
        tmp_path / "100.csv", start=pd.Timestamp("2022-01-01 12:00"),
        n_rows=2,
        flows_5min=np.array([100.0, 150.0]),    # 1200, 1800 veh/h
        speeds_mph=np.array([60.0, 30.0]),
        station_id=100,
    )
    _write_vds_csv(
        tmp_path / "200.csv", start=pd.Timestamp("2022-01-01 12:00"),
        n_rows=2,
        flows_5min=np.array([50.0, 75.0]),       # 600, 900 veh/h
        speeds_mph=np.array([60.0, 30.0]),
        station_id=200,
    )
    cells = _cells_df([
        dict(vds_id=100, lanes=3, vds_lanes=3),
        dict(vds_id=200, lanes=3, vds_lanes=3),
    ])
    rho = initial_state_from_vds(
        cells, tmp_path, at_time=pd.Timestamp("2022-01-01 12:00"),
    )
    # Cell 0: 1200 / 60 = 20 veh/mi; Cell 1: 600 / 60 = 10 veh/mi.
    np.testing.assert_allclose(rho, [20.0, 10.0])


def test_initial_state_raises_when_lanes_mismatch(tmp_path):
    """Any cell with lanes != vds_lanes must raise pointing to fix it."""
    _write_vds_csv(tmp_path / "100.csv",
                   start=pd.Timestamp("2022-01-01 12:00"), n_rows=2)
    cells = _cells_df([
        dict(vds_id=100, lanes=4, vds_lanes=3),  # mismatch
    ])
    with pytest.raises(ValueError, match="lanes != vds_lanes"):
        initial_state_from_vds(
            cells, tmp_path, at_time=pd.Timestamp("2022-01-01 12:00"),
        )


def test_initial_state_raises_on_missing_vds_id(tmp_path):
    cells = _cells_df([
        dict(vds_id=100, lanes=3, vds_lanes=3),
        dict(vds_id=pd.NA, lanes=3, vds_lanes=3),
    ])
    with pytest.raises(ValueError, match="no `vds_id`"):
        initial_state_from_vds(
            cells, tmp_path, at_time=pd.Timestamp("2022-01-01 12:00"),
        )


def test_initial_state_raises_when_vds_lanes_column_missing(tmp_path):
    """cells_df without `vds_lanes` -> KeyError pointing at Step 2."""
    cells = _cells_df([
        dict(vds_id=100, lanes=3),
    ])
    with pytest.raises(KeyError, match="vds_lanes"):
        initial_state_from_vds(
            cells, tmp_path, at_time=pd.Timestamp("2022-01-01 12:00"),
        )


def test_initial_state_raises_when_at_time_missing(tmp_path):
    """If the VDS CSV has no row at at_time -> ValueError."""
    _write_vds_csv(tmp_path / "100.csv",
                   start=pd.Timestamp("2022-01-01 12:00"), n_rows=2)
    cells = _cells_df([dict(vds_id=100, lanes=3, vds_lanes=3)])
    with pytest.raises(ValueError, match="no sample at"):
        initial_state_from_vds(
            cells, tmp_path,
            at_time=pd.Timestamp("2022-01-01 13:00"),  # not in the file
        )


def test_initial_state_raises_when_flow_or_speed_is_nan(tmp_path):
    _write_vds_csv(
        tmp_path / "100.csv", start=pd.Timestamp("2022-01-01 12:00"),
        n_rows=2,
        flows_5min=np.array([np.nan, 100.0]),
        speeds_mph=np.array([60.0, 60.0]),
    )
    cells = _cells_df([dict(vds_id=100, lanes=3, vds_lanes=3)])
    with pytest.raises(ValueError, match="NaN"):
        initial_state_from_vds(
            cells, tmp_path, at_time=pd.Timestamp("2022-01-01 12:00"),
        )


def test_initial_state_caches_shared_vds(tmp_path, monkeypatch):
    """A VDS shared by multiple cells (Dervisoglu fallback) is read once."""
    _write_vds_csv(tmp_path / "100.csv",
                   start=pd.Timestamp("2022-01-01 12:00"), n_rows=2)
    cells = _cells_df([
        dict(vds_id=100, lanes=3, vds_lanes=3),
        dict(vds_id=100, lanes=3, vds_lanes=3),  # same VDS
    ])
    n_calls = {"count": 0}
    orig_read_csv = pd.read_csv

    def counting_read_csv(*args, **kwargs):
        n_calls["count"] += 1
        return orig_read_csv(*args, **kwargs)

    monkeypatch.setattr(
        "transportation_models.utils.ctm.scenario.pd.read_csv",
        counting_read_csv,
    )
    rho = initial_state_from_vds(
        cells, tmp_path, at_time=pd.Timestamp("2022-01-01 12:00"),
    )
    assert n_calls["count"] == 1
    assert rho.shape == (2,)
    assert rho[0] == rho[1]
