"""Tests for the single-lane microsim main loop (T5, CP3).

Invariants on a synthetic corridor: no overlap (gap >= 0), velocities in the
per-vehicle [v_min, v_max] envelope, and a controlled follower-decelerates case.
"""

from __future__ import annotations

import numpy as np
import pytest

from transportation_models.utils.microsim import MicrosimSpec, Vehicles
from transportation_models.utils.microsim.seeding import seed_vehicles
from transportation_models.utils.microsim.simulate import simulate


def _spec(dt=1.0):
    return MicrosimSpec(
        a=1.5, b=2.0, s_o=25.0, vehicle_length=5.0, dt=dt, seed=0
    )


def _no_overlap(result, corridor_length, vehicle_length):
    """No two co-present vehicles are closer than one vehicle length."""
    pos = result.position
    for k in range(pos.shape[1]):
        col = pos[:, k]
        present = col[(~np.isnan(col)) & (col >= 0) & (col <= corridor_length)]
        s = np.sort(present)
        if s.size >= 2:
            assert np.all(np.diff(s) >= vehicle_length - 1e-6)


def test_simulate_smoke_and_completes():
    spec = _spec()
    rate = np.full(400, 600.0)
    veh = seed_vehicles(rate, spec, v_f_ms=26.0, eta_ev=0.5)
    res = simulate(spec, veh, corridor_length_m=300.0, n_steps=400)
    assert res.n_steps == 400
    assert res.n_vehicles == veh.n_vehicles
    # At least one vehicle reaches the far end.
    assert np.nanmax(res.position) >= 300.0


def test_simulate_no_overlap_invariant():
    spec = _spec()
    rate = np.full(400, 900.0)  # heavier inflow stresses spacing
    veh = seed_vehicles(rate, spec, v_f_ms=26.0, eta_ev=0.5)
    res = simulate(spec, veh, corridor_length_m=300.0, n_steps=400)
    _no_overlap(res, 300.0, spec.vehicle_length)


def test_simulate_velocity_envelope():
    spec = _spec()
    rate = np.full(400, 900.0)
    veh = seed_vehicles(rate, spec, v_f_ms=26.0, eta_ev=0.5)
    res = simulate(spec, veh, corridor_length_m=300.0, n_steps=400)
    vel = res.velocity
    active = ~np.isnan(vel)
    assert np.all(vel[active] >= spec.v_min - 1e-9)
    # per-vehicle cap
    cap = np.broadcast_to(veh.v_max[:, None], vel.shape)
    assert np.all(vel[active] <= cap[active] + 1e-9)


def test_simulate_follower_decelerates_behind_slow_leader():
    # Leader slow (v_max 10), follower fast (v_max 30) entering just behind.
    spec = _spec()
    veh = Vehicles(
        entry_step=np.array([0, 2]),
        v_max=np.array([10.0, 30.0]),
        is_ev=np.array([True, True]),
    )
    res = simulate(spec, veh, corridor_length_m=500.0, n_steps=120)
    _no_overlap(res, 500.0, spec.vehicle_length)
    # Follower's velocity is pulled below its own desired speed at some point.
    follower_v = res.velocity[1]
    active = ~np.isnan(follower_v)
    assert np.nanmin(follower_v[active]) < 30.0


def test_guard_stats_reported_and_activates_in_overshoot():
    # Fast follower behind slow leader forces the displacement-cap guard.
    spec = _spec()
    veh = Vehicles(
        entry_step=np.array([0, 2]),
        v_max=np.array([10.0, 30.0]),
        is_ev=np.array([True, True]),
    )
    res = simulate(spec, veh, corridor_length_m=500.0, n_steps=120)
    gs = res.guard_stats
    assert gs is not None
    assert gs.n_activations > 0
    assert 0.0 < gs.activation_rate <= 1.0
    assert gs.max_speed_reduction_ms > 0.0
    assert gs.max_accel_delta_ms2 == pytest.approx(
        gs.max_speed_reduction_ms / spec.dt
    )


def test_guard_rarely_activates_in_homogeneous_free_flow():
    # Low inflow + the guard => collision-free with few/no activations.
    spec = _spec()
    rate = np.full(400, 300.0)
    veh = seed_vehicles(rate, spec, v_f_ms=26.0, eta_ev=0.5)
    res = simulate(spec, veh, corridor_length_m=300.0, n_steps=400)
    _no_overlap(res, 300.0, spec.vehicle_length)
    assert res.guard_stats.activation_rate < 0.05
