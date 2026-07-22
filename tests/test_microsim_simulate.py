"""Tests for the N-lane microsim main loop (R3, CP3).

Invariants on a synthetic corridor: no within-lane overlap, velocities in the
per-vehicle [v_min, v_max] envelope, lanes in range, lane-changing relieves
congestion, and guard stats are reported.
"""

from __future__ import annotations

import numpy as np
import pytest

from transportation_models.utils.microsim import MicrosimSpec, Vehicles
from transportation_models.utils.microsim.simulate import simulate


def _spec(dt=1.0, **kw):
    base = dict(a=1.5, b=2.0, s_o=25.0, vehicle_length=5.0, dt=dt, seed=0)
    base.update(kw)
    return MicrosimSpec(**base)


def _seed_lanes(n, n_lanes, rng, every=4):
    return Vehicles(
        entry_step=np.arange(n) * every,
        v_max=rng.uniform(22.0, 30.0, size=n),
        is_ev=rng.random(n) < 0.5,
        entry_lane=rng.integers(0, n_lanes, size=n),
    )


def _no_overlap_per_lane(result, vehicle_length):
    """Within each lane, co-present vehicles keep >= one vehicle length."""
    pos, lane = result.position, result.lane
    L = result.corridor_length_m
    for k in range(lane.shape[1]):
        for ln in range(result.n_lanes):
            col = pos[:, k]
            here = (lane[:, k] == ln) & (~np.isnan(col)) & (col >= 0) & (col <= L)
            s = np.sort(col[here])
            if s.size >= 2:
                assert np.all(np.diff(s) >= vehicle_length - 1e-6)


def test_simulate_multilane_smoke():
    spec = _spec()
    veh = _seed_lanes(30, 3, np.random.default_rng(1))
    res = simulate(spec, veh, corridor_length_m=400.0, n_steps=300, n_lanes=3)
    assert res.n_steps == 300
    assert res.n_lanes == 3
    assert np.nanmax(res.position) >= 400.0
    active = res.lane >= 0
    assert np.all(res.lane[active] < 3)


def test_no_overlap_within_each_lane():
    spec = _spec()
    veh = _seed_lanes(40, 3, np.random.default_rng(2), every=2)
    res = simulate(spec, veh, corridor_length_m=400.0, n_steps=300, n_lanes=3)
    _no_overlap_per_lane(res, spec.vehicle_length)


def test_velocity_envelope():
    spec = _spec()
    veh = _seed_lanes(40, 3, np.random.default_rng(3), every=2)
    res = simulate(spec, veh, corridor_length_m=400.0, n_steps=300, n_lanes=3)
    vel = res.velocity
    active = ~np.isnan(vel)
    assert np.all(vel[active] >= spec.v_min - 1e-9)
    cap = np.broadcast_to(veh.v_max[:, None], vel.shape)
    assert np.all(vel[active] <= cap[active] + 1e-9)


def test_lane_change_relieves_congestion():
    # Fast follower behind a slow leader in lane 0; lane 1 free -> it should
    # change lanes at some point.
    spec = _spec(lane_change_prob=1.0)
    veh = Vehicles(
        entry_step=np.array([0, 3]),
        v_max=np.array([10.0, 30.0]),
        is_ev=np.array([True, True]),
        entry_lane=np.array([0, 0]),
    )
    res = simulate(spec, veh, corridor_length_m=600.0, n_steps=150, n_lanes=2)
    follower_lanes = res.lane[1][res.lane[1] >= 0]
    assert np.any(follower_lanes == 1)  # moved out of lane 0


def test_single_lane_has_no_lane_changes():
    spec = _spec(lane_change_prob=1.0)
    veh = _seed_lanes(20, 1, np.random.default_rng(4), every=3)
    res = simulate(spec, veh, corridor_length_m=400.0, n_steps=300, n_lanes=1)
    active = res.lane >= 0
    assert np.all(res.lane[active] == 0)


def test_guard_stats_reported():
    spec = _spec()
    veh = _seed_lanes(30, 2, np.random.default_rng(5), every=2)
    res = simulate(spec, veh, corridor_length_m=400.0, n_steps=300, n_lanes=2)
    assert res.guard_stats is not None
    assert 0.0 <= res.guard_stats.activation_rate <= 1.0
