"""Tests for utils/ramp_flow_estimation/zhang_features.py (preprocessing +
per-stretch sample extraction for the Zhang 2024 estimator)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ramp_flow_estimation.zhang_features import (
    FEATURES_PER_STEP,
    Preprocessed,
    ZhangSamples,
    build_stretch_samples,
    preprocess_mainline,
    stretch_samples,
)

_TS0 = "2023-06-05 00:00"   # a Monday


def _grid(n, start=_TS0):
    return pd.date_range(start, periods=n, freq="5min")


def _pre(flow, speed=None, pct=None, *, threshold=50.0, start=_TS0):
    flow = np.asarray(flow, dtype=float)
    n = len(flow)
    speed = np.full(n, 60.0) if speed is None else np.asarray(speed, float)
    pct = np.full(n, 100.0) if pct is None else np.asarray(pct, float)
    return preprocess_mainline(flow, speed, pct, _grid(n, start),
                               pct_threshold=threshold)


# --------------------------------------------------------------------------- #
# preprocess_mainline
# --------------------------------------------------------------------------- #
def test_short_gaps_filled_by_persistence():
    flow = [100.0, np.nan, np.nan, 200.0, np.nan, 300.0]
    out = _pre(flow)
    # 2-step gap and 1-step gap both copy the last observed value
    np.testing.assert_allclose(out.flow, [100, 100, 100, 200, 200, 300])
    np.testing.assert_array_equal(out.filled, [0, 1, 1, 0, 1, 0])
    assert out.usable.all()


_WEEK = 7 * 288   # 5-min samples per week (slot means are keyed by weekday)


def test_long_gaps_filled_by_slot_historical_mean():
    # two Mondays a week apart; the second has a 3-step gap -> filled by the
    # same (day-of-week, time-of-day) slot's observed values, i.e. the first
    # Monday's.
    flow = np.arange(_WEEK + 288, dtype=float)
    flow[_WEEK:_WEEK + 3] = np.nan
    out = _pre(flow, start=_TS0)
    np.testing.assert_allclose(out.flow[_WEEK:_WEEK + 3], flow[:3])  # slot means
    assert out.filled[_WEEK:_WEEK + 3].all() and out.usable.all()


def test_leading_gap_uses_historical_mean_not_persistence():
    # a 2-step gap at the record start has no previous value; falls through to
    # the slot mean (observed a week later at the same weekday/time slot).
    flow = np.full(_WEEK + 288, 500.0)
    flow[:2] = np.nan
    out = _pre(flow)
    np.testing.assert_allclose(out.flow[:2], [500.0, 500.0])
    assert out.filled[:2].all()


def test_below_threshold_pct_is_treated_missing():
    flow = [100.0, 150.0, 200.0]
    pct = [100.0, 30.0, 100.0]                     # middle sample low quality
    out = _pre(flow, pct=pct, threshold=50.0)
    assert out.filled[1] and not out.filled[0] and not out.filled[2]
    np.testing.assert_allclose(out.flow, [100, 100, 200])   # persistence


def test_never_observed_slot_stays_nan_and_unusable():
    flow = np.full(6, np.nan)                      # nothing to fill from
    out = _pre(flow)
    assert np.isnan(out.flow).all() and not out.usable.any()


def test_density_is_flow_over_speed():
    out = _pre([6000.0, 1200.0], speed=[60.0, 30.0])
    np.testing.assert_allclose(out.density, [100.0, 40.0])


def test_zero_speed_gives_unusable_density():
    out = _pre([6000.0, 1200.0], speed=[60.0, 0.0])
    assert out.usable[0] and not out.usable[1]


# --------------------------------------------------------------------------- #
# stretch_samples
# --------------------------------------------------------------------------- #
def _station(flow, **kw):
    return _pre(flow, **kw)


def _ramp(flow, observed=None):
    flow = np.asarray(flow, dtype=float)
    obs = np.ones(len(flow), bool) if observed is None else np.asarray(observed, bool)
    return [(flow, obs)]


def test_feature_layout_and_targets():
    n, H = 7, 5
    up = _station(np.arange(n, dtype=float) + 1000)
    down = _station(np.arange(n, dtype=float) + 2000)
    on = _ramp(np.arange(n, dtype=float) + 10)
    off = _ramp(np.arange(n, dtype=float) + 20)
    out = stretch_samples(up, down, on, off, _grid(n), window_size=H)
    assert out.X.shape == (n - H + 1, FEATURES_PER_STEP * H)
    # first window, first step: [q_up, q_down, v_up, v_down, rho_up, rho_down,
    # hour, minute] at t-4
    np.testing.assert_allclose(
        out.X[0, :FEATURES_PER_STEP],
        [1000, 2000, 60, 60, 1000 / 60, 2000 / 60, 0, 0])
    # first window, last step (t = index 4, 00:20)
    np.testing.assert_allclose(
        out.X[0, -FEATURES_PER_STEP:],
        [1004, 2004, 60, 60, 1004 / 60, 2004 / 60, 0, 20])
    np.testing.assert_allclose(out.r, np.arange(H - 1, n) + 10)
    np.testing.assert_allclose(out.s, np.arange(H - 1, n) + 20)
    assert (out.date == np.datetime64("2023-06-05")).all()
    np.testing.assert_array_equal(out.t_index, np.arange(H - 1, n))


def test_window_dropped_when_ramp_unmeasured_at_t():
    n, H = 6, 5
    up, down = _station(np.full(n, 1000.0)), _station(np.full(n, 1100.0))
    obs = np.ones(n, bool)
    obs[H - 1] = False                             # t of the first window
    out = stretch_samples(up, down, _ramp(np.full(n, 50.0), obs),
                          _ramp(np.full(n, 30.0)), _grid(n), window_size=H)
    np.testing.assert_array_equal(out.t_index, [H])   # only the second window


def test_absent_ramp_is_zero_flow():
    n, H = 5, 5
    up, down = _station(np.full(n, 1000.0)), _station(np.full(n, 1100.0))
    out = stretch_samples(up, down, [], _ramp(np.full(n, 30.0)),
                          _grid(n), window_size=H)
    np.testing.assert_allclose(out.r, [0.0])
    np.testing.assert_allclose(out.s, [30.0])


def test_entirely_filled_window_dropped_partially_filled_kept():
    n, H = 12, 5
    # both stations observed through week 1; week 2 opens with a long gap
    # (fillable from week 1's matching slots)
    m = _WEEK + n
    flow = np.full(m, 1000.0)
    flow[_WEEK:_WEEK + H] = np.nan                 # one window is all filled
    up, down = _station(flow), _station(flow)
    on, off = _ramp(np.full(m, 50.0)), _ramp(np.full(m, 30.0))
    out = stretch_samples(up, down, on, off, _grid(m), window_size=H)
    t = out.t_index
    assert up.usable.all()                         # the gap was fillable
    assert _WEEK + H - 1 not in t                  # all-filled window dropped
    assert _WEEK + H in t                          # 4 filled + 1 observed kept


def test_weekend_windows_dropped_by_default():
    # record starts Friday: Friday windows kept, Sat/Sun dropped, Monday kept
    n, H = 4 * 288, 5
    friday = "2023-06-09 00:00"
    up, down = (_pre(np.full(n, 1000.0), start=friday),
                _pre(np.full(n, 1100.0), start=friday))
    on, off = _ramp(np.full(n, 50.0)), _ramp(np.full(n, 30.0))
    grid = _grid(n, start=friday)
    out = stretch_samples(up, down, on, off, grid, window_size=H)
    dows = pd.DatetimeIndex(grid[out.t_index]).dayofweek
    assert (dows < 5).all()
    assert {4, 0} <= set(dows)                     # Friday and Monday survive
    kept_all = stretch_samples(up, down, on, off, grid, window_size=H,
                               weekdays_only=False)
    assert len(kept_all.r) == n - H + 1            # opt-out keeps weekends


def test_unusable_mainline_window_dropped():
    n, H = 6, 5
    flow = np.full(n, np.nan)                      # unfillable -> unusable
    flow[H:] = 1000.0
    up, down = _station(flow), _station(np.full(n, 1100.0))
    out = stretch_samples(up, down, _ramp(np.full(n, 50.0)),
                          _ramp(np.full(n, 30.0)), _grid(n), window_size=H)
    assert len(out.r) == 0


# --------------------------------------------------------------------------- #
# build_stretch_samples (IO assembler)
# --------------------------------------------------------------------------- #
def _write_station(ts_dir, sid, flow, n, *, pct=100.0, start=_TS0):
    ts = pd.date_range(start, periods=n, freq="5min")
    pd.DataFrame({
        "timestamp": ts,
        "total_flow_[veh/5-min]": np.asarray(flow, float),
        "avg_speed_[mph]": np.full(n, 60.0),
        "avg_occupancy_[%]": np.full(n, 10.0),
        "pct_observed": np.full(n, float(pct)),
    }).to_csv(ts_dir / f"{sid}.csv", index=False)


def _stretches():
    return pd.DataFrame([{
        "stretch_id": "880_N:0", "up_ml_id": 100, "down_ml_id": 200,
        "on_ids": "10", "off_ids": "20", "config_type": "c",
    }])


def test_build_stretch_samples_scales_flow_to_veh_hr(tmp_path):
    n = 8
    _write_station(tmp_path, 100, np.full(n, 100.0), n)   # veh/5-min
    _write_station(tmp_path, 200, np.full(n, 120.0), n)
    _write_station(tmp_path, 10, np.full(n, 10.0), n)
    _write_station(tmp_path, 20, np.full(n, 5.0), n)
    out = build_stretch_samples(_stretches(), "880_N:0", tmp_path,
                                pct_threshold=50.0)
    assert len(out.r) == n - 5 + 1
    np.testing.assert_allclose(out.X[0, 0], 1200.0)       # q_up veh/hr
    np.testing.assert_allclose(out.r, 120.0)              # ramp veh/hr
    np.testing.assert_allclose(out.s, 60.0)


def test_build_stretch_samples_unknown_stretch_raises(tmp_path):
    with pytest.raises(ValueError, match="matches 0 rows"):
        build_stretch_samples(_stretches(), "880_N:9", tmp_path,
                              pct_threshold=50.0)


def test_build_stretch_samples_missing_mainline_raises(tmp_path):
    n = 8
    _write_station(tmp_path, 100, np.full(n, 100.0), n)   # no 200.csv
    with pytest.raises(ValueError, match="mainline timeseries missing"):
        build_stretch_samples(_stretches(), "880_N:0", tmp_path,
                              pct_threshold=50.0)
