"""Tests for utils/ramp_flow_estimation/bounds.py (Kan Appendix A)."""
from __future__ import annotations

import numpy as np

from transportation_models.utils.ramp_flow_estimation import (
    alpha_target,
    compute_bounds,
    feasible_band,
    predicts_on_ramp,
    reconstruct_pair,
)


def test_default_bounds_are_the_weaving_cap():
    # caps = inf -> R_max = c_w - q_up, S_max = c_w - q_down.
    b = compute_bounds(q_up=1000.0, q_down=1200.0, c_w=2000.0)
    assert b.r_min == 200.0 and b.s_min == 0.0
    assert b.r_max == 1000.0                 # min(200+800, 1000)
    assert b.s_max == 800.0                  # min(-200+1000, 800)


def test_r_demand_tightens_on_ramp_and_couples_to_off_ramp():
    b = compute_bounds(1000.0, 1200.0, 2000.0, r_demand=600.0)
    assert b.r_max == 600.0                  # R_max = min(1000, 600)
    # s_max = min(q_up - q_down + R_max, S_max) = min(-200 + 600, 800) -> coupled
    assert b.s_max == 400.0


def test_on_ramp_round_trip_recovers_flows():
    q_up, q_down, c_w, r = 1000.0, 1200.0, 2000.0, 500.0
    s = r + q_up - q_down                     # conservation -> 300
    b = compute_bounds(q_up, q_down, c_w)
    alpha, ok = alpha_target(b, q_up, q_down, r, s)
    assert ok and np.isclose(alpha, (500 - 200) / (1000 - 200))
    r_hat, s_hat = reconstruct_pair(alpha, q_up, q_down, c_w)
    assert np.isclose(r_hat, r) and np.isclose(s_hat, s)


def test_off_ramp_round_trip_recovers_flows():
    q_up, q_down, c_w, s = 1200.0, 1000.0, 2000.0, 400.0
    r = s + q_down - q_up                     # conservation -> 200
    b = compute_bounds(q_up, q_down, c_w)
    assert not predicts_on_ramp(q_up, q_down)  # q_up > q_down -> predict s
    alpha, ok = alpha_target(b, q_up, q_down, r, s)
    r_hat, s_hat = reconstruct_pair(alpha, q_up, q_down, c_w)
    assert np.isclose(r_hat, r) and np.isclose(s_hat, s)


def test_reconstruction_conserves_flow():
    q_up, q_down, c_w = 900.0, 1100.0, 2500.0
    r_hat, s_hat = reconstruct_pair(0.42, q_up, q_down, c_w)
    assert np.isclose(q_up + r_hat - s_hat, q_down)   # A.1


def test_degenerate_band_flagged_in_congestion():
    # q_up above weaving capacity -> c_w - q_up < 0 -> collapsed on-ramp band.
    b = compute_bounds(q_up=2100.0, q_down=2000.0, c_w=2000.0)
    assert not feasible_band(b, 2100.0, 2000.0)
    _, ok = alpha_target(b, 2100.0, 2000.0, r_true=50.0, s_true=150.0)
    assert not ok


def test_vectorized_mixed_directions_round_trip():
    q_up = np.array([1000.0, 1200.0, 800.0])
    q_down = np.array([1200.0, 1000.0, 800.0])
    c_w = np.array([2000.0, 2000.0, 2000.0])
    r = np.array([500.0, 200.0, 300.0])
    s = r + q_up - q_down                      # conservation for all
    b = compute_bounds(q_up, q_down, c_w)
    alpha, ok = alpha_target(b, q_up, q_down, r, s)
    assert ok.all()
    r_hat, s_hat = reconstruct_pair(alpha, q_up, q_down, c_w)
    np.testing.assert_allclose(r_hat, r)
    np.testing.assert_allclose(s_hat, s)
