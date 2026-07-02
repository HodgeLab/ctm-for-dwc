"""Tests for utils/ramp_flow_estimation/features.py.

Exercises the pure per-stretch sample extractor with tiny synthetic aligned
arrays (no timeseries files). One stretch: up ML 100, down ML 200, on-ramp 10,
off-ramp 20. Feature layout is 2 stations x {flow, speed, occ} x 3 lags
(t-2, t-1, t), 18 columns, lags oldest-first.
"""
from __future__ import annotations

import numpy as np

from transportation_models.utils.ramp_flow_estimation import bounds
from transportation_models.utils.ramp_flow_estimation.features import stretch_samples

_C_W = 5000.0  # total mainline capacity for the stretch


def _inputs(n=6, **over):
    """Fully-measured, conservation-consistent scenario over ``n`` samples:
    q_up=1000, q_down=1200, r=500, s=300 (1000 + 500 - 300 = 1200). Override any
    array by name; override an ``observed`` mask via ``obs_<id>``."""
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
    return dict(flow=flow, speed=speed, occ=occ, observed=observed)


def _call(**over):
    return stretch_samples(
        up_id=100, down_id=200, on_ids=[10], off_ids=[20],
        c_w=_C_W, window_size=3, **_inputs(**over),
    )


def test_fully_measured_yields_one_sample_per_window():
    s = _call()                       # 6 samples, window 3 -> 4 windows
    assert len(s.alpha) == 4
    np.testing.assert_allclose(s.r_true, 500.0)
    np.testing.assert_allclose(s.s_true, 300.0)
    np.testing.assert_allclose(s.q_up, 1000.0)
    np.testing.assert_allclose(s.q_down, 1200.0)


def test_alpha_matches_bounds_module():
    s = _call()
    b = bounds.compute_bounds(1000.0, 1200.0, _C_W)
    expected, ok = bounds.alpha_target(b, 1000.0, 1200.0, 500.0, 300.0)
    assert ok
    np.testing.assert_allclose(s.alpha, expected)


def test_feature_matrix_shape_and_current_step_values():
    s = _call()
    assert s.X.shape == (4, 18)
    # newest lag (t) block starts at col 12: [up_flow, up_speed, up_occ, down_flow,...]
    np.testing.assert_allclose(s.X[:, 12], 1000.0)   # up flow at t
    np.testing.assert_allclose(s.X[:, 15], 1200.0)   # down flow at t


def test_conservation_windows_excluded_by_default():
    # off-ramp detector never reports -> only conservation could recover s, and
    # the default keeps fully-measured windows only.
    s = _call(obs_20=np.zeros(6, dtype=bool))
    assert len(s.alpha) == 0


def test_conservation_windows_included_when_opted_in():
    # require_both_measured=False -> s recovered from q_up + r - q_down.
    s = stretch_samples(
        up_id=100, down_id=200, on_ids=[10], off_ids=[20],
        c_w=_C_W, window_size=3, require_both_measured=False,
        **_inputs(obs_20=np.zeros(6, dtype=bool)),
    )
    assert len(s.alpha) == 4
    np.testing.assert_allclose(s.r_true, 500.0)      # measured
    np.testing.assert_allclose(s.s_true, 300.0)      # 1000 + 500 - 1200


def test_mainline_gap_drops_windows_that_span_it():
    # up ML unobserved at sample index 3 -> windows [1,2,3],[2,3,4],[3,4,5] drop.
    obs = np.ones(6, dtype=bool); obs[3] = False
    s = _call(obs_100=obs)
    assert len(s.alpha) == 1                          # only window [0,1,2]


def test_two_unmeasured_ramps_yield_no_samples():
    s = _call(obs_10=np.zeros(6, dtype=bool), obs_20=np.zeros(6, dtype=bool))
    assert len(s.alpha) == 0


def test_degenerate_band_windows_dropped():
    # q_up above total capacity -> C_w - q_up < 0 -> infeasible band, dropped.
    s = stretch_samples(
        up_id=100, down_id=200, on_ids=[10], off_ids=[20],
        c_w=800.0, window_size=3, **_inputs(),
    )
    assert len(s.alpha) == 0


def test_r_demand_tightens_alpha_target():
    tight = stretch_samples(
        up_id=100, down_id=200, on_ids=[10], off_ids=[20],
        c_w=_C_W, window_size=3, r_demand=600.0, **_inputs(),
    )
    loose = _call()
    # smaller band -> larger alpha for the same true flow
    assert (tight.alpha > loose.alpha).all()
