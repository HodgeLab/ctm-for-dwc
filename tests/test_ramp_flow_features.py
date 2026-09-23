"""Tests for utils/ramp_flow_estimation/features.py (method-agnostic extractor).

Exercises the pure per-stretch sample extractor with tiny synthetic aligned
arrays (no timeseries files). One stretch: up ML 100, down ML 200, on-ramp 10,
off-ramp 20. Feature layout is 2 stations x {flow, speed, occ} x 3 lags
(t-2, t-1, t), lags oldest-first, then [day-of-week, hour, 5-min slot] at t:
21 columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ctm_for_dwc.utils.ramp_flow_estimation.features import (
    mainline_flows,
    stretch_samples,
)


def _inputs(n=6, **over):
    """Fully-measured scenario over ``n`` samples: q_up=1000, q_down=1200,
    r=500, s=300. Override any array by name; override an ``observed`` mask via
    ``obs_<id>``."""
    a = lambda v: np.full(n, float(v))
    flow = {100: a(1000), 200: a(1200), 10: a(500), 20: a(300)}
    speed = {100: a(60), 200: a(55)}
    occ = {100: a(10), 200: a(12)}
    observed = {i: np.ones(n, dtype=bool) for i in (100, 200, 10, 20)}
    for k, v in over.items():
        if k.startswith("obs_"):
            observed[int(k[4:])] = np.asarray(v, dtype=bool)
        elif k in ("100", "200", "10", "20"):
            flow[int(k)] = np.asarray(v, dtype=float)
    # 2023-06-01 is a Thursday (dayofweek 3)
    ts = pd.date_range("2023-06-01 00:00", periods=n, freq="5min")
    return dict(flow=flow, speed=speed, occ=occ, observed=observed, timestamps=ts)


def _call(**over):
    return stretch_samples(
        up_id=100, down_id=200, on_ids=[10], off_ids=[20],
        window_size=3, **_inputs(**over),
    )


def test_fully_measured_yields_one_sample_per_window():
    s = _call()                       # 6 samples, window 3 -> 4 windows
    assert len(s.r_true) == 4
    np.testing.assert_allclose(s.r_true, 500.0)
    np.testing.assert_allclose(s.s_true, 300.0)
    np.testing.assert_allclose(s.q_up, 1000.0)
    np.testing.assert_allclose(s.q_down, 1200.0)


def test_feature_matrix_shape_and_current_step_values():
    s = _call()
    assert s.X.shape == (4, 21)
    # newest lag (t) block starts at col 12: [up_flow, up_speed, up_occ, down_flow,...]
    np.testing.assert_allclose(s.X[:, 12], 1000.0)   # up flow at t
    np.testing.assert_allclose(s.X[:, 15], 1200.0)   # down flow at t


def test_time_features_taken_at_prediction_step():
    # windows predict at t = 2..5 -> 00:10 .. 00:25 on Thursday 2023-06-01
    s = _call()
    np.testing.assert_allclose(s.X[:, 18], 3.0)               # day-of-week
    np.testing.assert_allclose(s.X[:, 19], 0.0)               # hour
    np.testing.assert_allclose(s.X[:, 20], [2, 3, 4, 5])      # 5-min slot


def test_time_features_cross_hour_and_day_boundaries():
    ts = pd.date_range("2023-06-01 23:50", periods=6, freq="5min")
    s = stretch_samples(
        up_id=100, down_id=200, on_ids=[10], off_ids=[20],
        window_size=3, **{**_inputs(), "timestamps": ts},
    )
    # t = 2..5 -> 00:00, 00:05, 00:10, 00:15 on Friday 2023-06-02
    np.testing.assert_allclose(s.X[:, 18], 4.0)
    np.testing.assert_allclose(s.X[:, 19], 0.0)
    np.testing.assert_allclose(s.X[:, 20], [0, 1, 2, 3])


def test_mainline_flows_recovered_from_last_lag_block():
    s = _call()
    q_up, q_down = mainline_flows(s.X)
    np.testing.assert_allclose(q_up, s.q_up)
    np.testing.assert_allclose(q_down, s.q_down)


def test_no_estimator_specific_filtering():
    # Flows that violate conservation, or exceed any plausible capacity, are
    # kept -- feasibility is an estimator concern (kan.py), not the corpus's.
    s = _call(**{"10": np.full(6, 100.0)})           # r far below q_down - q_up
    assert len(s.r_true) == 4
    np.testing.assert_allclose(s.r_true, 100.0)


def test_conservation_windows_excluded_by_default():
    # off-ramp detector never reports -> only conservation could recover s, and
    # the default keeps fully-measured windows only.
    s = _call(obs_20=np.zeros(6, dtype=bool))
    assert len(s.r_true) == 0


def test_conservation_windows_included_when_opted_in():
    # require_both_measured=False -> s recovered from q_up + r - q_down.
    s = stretch_samples(
        up_id=100, down_id=200, on_ids=[10], off_ids=[20],
        window_size=3, require_both_measured=False,
        **_inputs(obs_20=np.zeros(6, dtype=bool)),
    )
    assert len(s.r_true) == 4
    np.testing.assert_allclose(s.r_true, 500.0)      # measured
    np.testing.assert_allclose(s.s_true, 300.0)      # 1000 + 500 - 1200


def test_mainline_gap_drops_windows_that_span_it():
    # up ML unobserved at sample index 3 -> windows [1,2,3],[2,3,4],[3,4,5] drop.
    obs = np.ones(6, dtype=bool); obs[3] = False
    s = _call(obs_100=obs)
    assert len(s.r_true) == 1                         # only window [0,1,2]


def test_two_unmeasured_ramps_yield_no_samples():
    s = _call(obs_10=np.zeros(6, dtype=bool), obs_20=np.zeros(6, dtype=bool))
    assert len(s.r_true) == 0
