"""Single-lane microsim main loop (T5; Newbolt 2026 Alg 1-2, single-lane).

Vehicles enter at the upstream boundary at their seeded entry step (subject to
room), follow the modified-Gipps longitudinal rule each step, and exit past the
corridor end. Per-step work is vectorized across the active vehicles (no
lane-changing; that is deferred per the spec).

DEVIATION FROM THE PUBLISHED MODEL -- displacement-cap collision guard
----------------------------------------------------------------------
Newbolt's simplified Gipps safe-velocity (2026 eq 5) is **not collision-free**
at practical time steps: because it uses the *subject's* speed and a hard
binary accelerate/decelerate switch at ``s_o``, a fast follower closing on a
slow leader can overshoot and pass through it (observed bumper overlaps of a
few meters at ``dt = 0.25-1 s`` under ordinary ``U(v_f +/- 15)`` speed
heterogeneity; clean only near ``dt = 0.1 s``). To keep the simulation
physical at ``dt = 1 s``, this implementation **diverges from the paper**: it
keeps the paper's velocity formula verbatim, then caps each vehicle's per-step
displacement at the gap to its leader so positions can never overlap. The
guard binds only in the rare overshoot; its footprint is reported in
``MicrosimResult.guard_stats`` (activation count/rate, max speed reduction,
max acceleration delta) so the divergence is auditable, never silent. See
``docs/dwpt_validation_spec.md`` "Deviations from the published model".
"""

from __future__ import annotations

import numpy as np

from .gipps import gap as gap_fn
from .gipps import next_velocity
from .model import GuardStats, MicrosimResult, MicrosimSpec, Vehicles


def simulate(
    spec: MicrosimSpec,
    vehicles: Vehicles,
    *,
    corridor_length_m: float,
    n_steps: int,
) -> MicrosimResult:
    """Run the single-lane microsim and return per-vehicle trajectories.

    Parameters
    ----------
    spec : MicrosimSpec
        Gipps params, ``dt``, ``v_min``.
    vehicles : Vehicles
        Seeded vehicles (entry steps must be ascending).
    corridor_length_m : float
        Corridor length, meters; vehicles exit once their position exceeds it.
    n_steps : int
        Number of simulation steps to run.

    Returns
    -------
    MicrosimResult
        ``position`` (N, n_steps+1) and ``velocity`` (N, n_steps), NaN where a
        vehicle has not yet entered or has already exited.
    """
    n = vehicles.n_vehicles
    dt = spec.dt
    length = spec.vehicle_length
    L = corridor_length_m

    position = np.full((n, n_steps + 1), np.nan)
    velocity = np.full((n, n_steps), np.nan)

    cur_x = np.full(n, np.nan)
    cur_v = np.full(n, np.nan)
    entered = np.zeros(n, dtype=bool)
    exited = np.zeros(n, dtype=bool)
    entry_step = vehicles.entry_step
    v_max = vehicles.v_max

    # Displacement-cap guard accumulators (see module docstring).
    n_activations = 0
    n_vehicle_steps = 0
    max_speed_reduction = 0.0
    max_accel_delta = 0.0
    max_abs_accel = 0.0

    next_to_enter = 0
    for k in range(n_steps):
        # 1. Admit due vehicles while there is room at the boundary.
        active = entered & ~exited
        while (
            next_to_enter < n and entry_step[next_to_enter] <= k
        ):
            min_x = cur_x[active].min() if active.any() else np.inf
            if min_x - length > spec.s_o:
                idx = next_to_enter
                cur_x[idx] = 0.0
                cur_v[idx] = v_max[idx]
                entered[idx] = True
                position[idx, k] = 0.0
                next_to_enter += 1
                active = entered & ~exited
            else:
                break  # boundary occupied; wait for a later step

        # 2. Leader gaps among active vehicles (sorted by position).
        act_idx = np.flatnonzero(active)
        if act_idx.size == 0:
            continue
        xs = cur_x[act_idx]
        order = np.argsort(xs)
        sorted_idx = act_idx[order]
        sorted_x = xs[order]
        gaps_sorted = np.full(sorted_x.size, np.inf)
        if sorted_x.size >= 2:
            gaps_sorted[:-1] = gap_fn(
                leader_back=sorted_x[1:],
                subject_back=sorted_x[:-1],
                subject_length=length,
            )

        # 3. Gipps-preferred velocity (Newbolt eq 2-4).
        v_prev = cur_v[sorted_idx]
        v_pref = next_velocity(
            v_prev,
            gaps_sorted,
            a=spec.a,
            b=spec.b,
            dt=dt,
            s_o=spec.s_o,
            v_max=v_max[sorted_idx],
            v_min=spec.v_min,
        )
        # Displacement-cap guard (DEVIATION; see module docstring): a vehicle
        # may advance at most ``gap`` this step, so it can never pass the
        # leader's current back. Frontmost vehicles have gap = inf (no cap).
        disp_cap = gaps_sorted / dt
        v_act = np.maximum(np.minimum(v_pref, disp_cap), 0.0)

        # Guard statistics.
        bound = v_act < v_pref - 1e-12
        n_vehicle_steps += sorted_idx.size
        if bound.any():
            n_activations += int(bound.sum())
            max_speed_reduction = max(
                max_speed_reduction, float((v_pref - v_act)[bound].max())
            )
            max_accel_delta = max(
                max_accel_delta, float(((v_pref - v_act)[bound] / dt).max())
            )
        max_abs_accel = max(
            max_abs_accel, float((np.abs(v_act - v_prev) / dt).max())
        )

        cur_v[sorted_idx] = v_act
        cur_x[sorted_idx] = cur_x[sorted_idx] + v_act * dt
        velocity[sorted_idx, k] = v_act
        position[sorted_idx, k + 1] = cur_x[sorted_idx]

        # 4. Retire vehicles that have left the corridor.
        just_exited = sorted_idx[cur_x[sorted_idx] > L]
        exited[just_exited] = True

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
        is_ev=vehicles.is_ev,
        dt=dt,
        corridor_length_m=L,
        guard_stats=guard_stats,
    )
