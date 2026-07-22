"""Tests for vehicle seeding (T3, CP2).

Inflow (veh/h per microsim step) -> Poisson entry steps + per-vehicle v_max
drawn from U(v_f - halfwidth, v_f + halfwidth). Reproducible under a seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from transportation_models.utils.microsim import MPH_TO_MS, MicrosimSpec
from transportation_models.utils.microsim.seeding import seed_vehicles


def _spec(dt=1.0, seed=0):
    return MicrosimSpec(
        a=1.5, b=2.0, s_o=25.0, vehicle_length=4.69, dt=dt, seed=seed
    )


def test_seed_count_matches_expected_for_constant_inflow():
    # 600 veh/h for 1 hour (3600 steps of 1 s) -> expect ~600 vehicles.
    spec = _spec()
    rate = np.full(3600, 600.0)
    veh = seed_vehicles(rate, spec, v_f_ms=60 * MPH_TO_MS, eta_ev=0.5, n_lanes=3)
    # Poisson(600): 3-sigma is ~73; allow generous tolerance.
    assert abs(veh.n_vehicles - 600) < 100


def test_seed_mean_headway_matches_rate():
    # At 600 veh/h, mean inter-entry spacing ~ 3600/600 = 6 steps.
    spec = _spec()
    rate = np.full(3600, 600.0)
    veh = seed_vehicles(rate, spec, v_f_ms=60 * MPH_TO_MS, eta_ev=0.5, n_lanes=3)
    headways = np.diff(np.sort(veh.entry_step))
    assert np.mean(headways) == pytest.approx(6.0, rel=0.15)


def test_seed_v_max_within_band():
    spec = _spec()
    rate = np.full(2000, 800.0)
    v_f = 60 * MPH_TO_MS
    hw = spec.speed_halfwidth_ms
    veh = seed_vehicles(rate, spec, v_f_ms=v_f, eta_ev=0.3, n_lanes=3)
    assert np.all(veh.v_max >= v_f - hw)
    assert np.all(veh.v_max <= v_f + hw)


def test_seed_ev_fraction_matches_eta():
    spec = _spec()
    rate = np.full(5000, 1000.0)
    veh = seed_vehicles(rate, spec, v_f_ms=60 * MPH_TO_MS, eta_ev=0.25, n_lanes=3)
    assert veh.is_ev.mean() == pytest.approx(0.25, abs=0.05)


def test_seed_is_reproducible_under_seed():
    rate = np.full(1000, 700.0)
    a = seed_vehicles(rate, _spec(seed=42), v_f_ms=60 * MPH_TO_MS, eta_ev=0.4, n_lanes=3)
    b = seed_vehicles(rate, _spec(seed=42), v_f_ms=60 * MPH_TO_MS, eta_ev=0.4, n_lanes=3)
    np.testing.assert_array_equal(a.entry_step, b.entry_step)
    np.testing.assert_array_equal(a.v_max, b.v_max)
    np.testing.assert_array_equal(a.is_ev, b.is_ev)
    np.testing.assert_array_equal(a.entry_lane, b.entry_lane)


def test_seed_entry_lane_in_range():
    spec = _spec()
    rate = np.full(3000, 900.0)
    veh = seed_vehicles(rate, spec, v_f_ms=60 * MPH_TO_MS, eta_ev=0.5, n_lanes=4)
    assert np.all(veh.entry_lane >= 0)
    assert np.all(veh.entry_lane < 4)
    # All lanes used at this volume.
    assert set(np.unique(veh.entry_lane)) == {0, 1, 2, 3}


def test_seed_different_seeds_differ():
    rate = np.full(1000, 700.0)
    a = seed_vehicles(rate, _spec(seed=1), v_f_ms=60 * MPH_TO_MS, eta_ev=0.4, n_lanes=3)
    b = seed_vehicles(rate, _spec(seed=2), v_f_ms=60 * MPH_TO_MS, eta_ev=0.4, n_lanes=3)
    # Overwhelmingly likely to differ in count or attributes.
    assert a.n_vehicles != b.n_vehicles or not np.array_equal(a.v_max, b.v_max)


def test_seed_entry_steps_sorted_and_in_range():
    spec = _spec()
    rate = np.full(500, 600.0)
    veh = seed_vehicles(rate, spec, v_f_ms=60 * MPH_TO_MS, eta_ev=0.5, n_lanes=3)
    assert np.all(np.diff(veh.entry_step) >= 0)
    assert veh.entry_step.min() >= 0
    assert veh.entry_step.max() < 500
