"""N-lane microsim main loop (R3/V4; Newbolt 2026 Alg 1, sequential).

Faithful to Newbolt's per-vehicle loop: each step, after admitting due vehicles
at the upstream boundary, iterate the active vehicles in index (entry) order and
for each one decide its lane change (``lanechange.decide_lane_change``), then its
modified-Gipps velocity, then advance its position -- so a vehicle sees the
already-updated state of the vehicles processed before it this step (the
"trailing EV reacts to the leading EV" behavior of the paper).

This sequential loop is deliberately *not* vectorized: reproducing Newbolt's
algorithm is the point, and it yields an **honest micro compute cost** for the
Stage-1 efficiency comparison against the CTM (a vectorized loop would
understate it). Cost is ~O(active^2) per step; fine for tests, and the
slowness on full 24 h runs is itself the finding.

DEVIATION FROM THE PUBLISHED MODEL -- displacement-cap collision guard
----------------------------------------------------------------------
Newbolt's simplified Gipps safe-velocity (2026 eq 5) is not collision-free at
practical ``dt``; we keep the velocity formula verbatim, then cap each
vehicle's per-step displacement at the gap to its same-lane leader so positions
never overlap. The guard's footprint is reported in
``MicrosimResult.guard_stats``. See ``docs/dwpt_validation_spec.md``.
"""

from __future__ import annotations

import numpy as np

from .gipps import next_velocity
from .lanechange import decide_lane_change
from .model import GuardStats, MicrosimResult, MicrosimSpec, Vehicles


def _admit_lane(cur_x, cur_lane, act_idx, entry_lane_pref, n_lanes, length, s_o):
    """Pick a lane with room at the boundary, or -1 if all are blocked.

    Prefers the seeded entry lane; otherwise the free lane whose nearest active
    vehicle is farthest downstream (Newbolt bumps boundary conflicts).
    """
    best_lane, best_room = -1, -np.inf
    order = [entry_lane_pref] + [L for L in range(n_lanes) if L != entry_lane_pref]
    for L in order:
        in_lane = act_idx[cur_lane[act_idx] == L]
        room = cur_x[in_lane].min() - length if in_lane.size else np.inf
        if room > s_o:
            if L == entry_lane_pref:
                return L
            if room > best_room:
                best_lane, best_room = L, room
    return best_lane


def simulate(
    spec: MicrosimSpec,
    vehicles: Vehicles,
    *,
    corridor_length_m: float,
    n_steps: int,
    n_lanes: int,
) -> MicrosimResult:
    """Run the N-lane sequential microsim and return per-vehicle trajectories."""
    n = vehicles.n_vehicles
    dt = spec.dt
    length = spec.vehicle_length
    L = corridor_length_m
    rng = np.random.default_rng(spec.seed + 1)  # distinct from seeding stream

    position = np.full((n, n_steps + 1), np.nan)
    velocity = np.full((n, n_steps), np.nan)
    lane_traj = np.full((n, n_steps), -1, dtype=int)

    cur_x = np.full(n, np.nan)
    cur_v = np.full(n, np.nan)
    cur_lane = np.full(n, -1, dtype=int)
    entered = np.zeros(n, dtype=bool)
    exited = np.zeros(n, dtype=bool)
    entry_step = vehicles.entry_step
    entry_lane = vehicles.entry_lane
    v_max = vehicles.v_max

    n_activations = 0
    n_vehicle_steps = 0
    max_speed_reduction = 0.0
    max_accel_delta = 0.0
    max_abs_accel = 0.0

    next_to_enter = 0
    for k in range(n_steps):
        # 1. Admit due vehicles while a lane has room at the boundary.
        while next_to_enter < n and entry_step[next_to_enter] <= k:
            idx = next_to_enter
            act_idx = np.flatnonzero(entered & ~exited)
            L_in = _admit_lane(
                cur_x, cur_lane, act_idx, int(entry_lane[idx]), n_lanes,
                length, spec.s_o,
            )
            if L_in < 0:
                break
            cur_x[idx] = 0.0
            cur_v[idx] = v_max[idx]
            cur_lane[idx] = L_in
            entered[idx] = True
            position[idx, k] = 0.0
            next_to_enter += 1

        # 2-3. Sequential per-vehicle update (Newbolt Alg 1), index order.
        for i in np.flatnonzero(entered & ~exited):
            others = np.flatnonzero(entered & ~exited)
            others = others[others != i]
            o_pos, o_lane = cur_x[others], cur_lane[others]

            if n_lanes > 1:
                cur_lane[i] = decide_lane_change(
                    cur_x[i], cur_lane[i], o_pos, o_lane,
                    length=length, s_o=spec.s_o, n_lanes=n_lanes,
                    lane_change_prob=spec.lane_change_prob, rng=rng,
                )

            # Leader in the (possibly new) lane.
            same = o_pos[o_lane == cur_lane[i]]
            ahead = same[same > cur_x[i]]
            gap = (ahead.min() - cur_x[i] - length) if ahead.size else np.inf

            v_prev = cur_v[i]
            v_pref = next_velocity(
                v_prev, gap, a=spec.a, b=spec.b, dt=dt, s_o=spec.s_o,
                v_max=v_max[i], v_min=spec.v_min,
            )
            v_act = max(min(float(v_pref), gap / dt), 0.0)

            n_vehicle_steps += 1
            if v_act < v_pref - 1e-12:
                n_activations += 1
                max_speed_reduction = max(max_speed_reduction, v_pref - v_act)
                max_accel_delta = max(max_accel_delta, (v_pref - v_act) / dt)
            max_abs_accel = max(max_abs_accel, abs(v_act - v_prev) / dt)

            cur_v[i] = v_act
            cur_x[i] = cur_x[i] + v_act * dt
            velocity[i, k] = v_act
            lane_traj[i, k] = cur_lane[i]
            position[i, k + 1] = cur_x[i]
            if cur_x[i] > L:
                exited[i] = True

    guard_stats = GuardStats(
        n_activations=n_activations,
        n_vehicle_steps=n_vehicle_steps,
        max_speed_reduction_ms=max_speed_reduction,
        max_accel_delta_ms2=max_accel_delta,
        max_abs_accel_ms2=max_abs_accel,
    )
    return MicrosimResult(
        position=position,
        velocity=velocity,
        lane=lane_traj,
        is_ev=vehicles.is_ev,
        dt=dt,
        corridor_length_m=L,
        n_lanes=n_lanes,
        guard_stats=guard_stats,
    )
