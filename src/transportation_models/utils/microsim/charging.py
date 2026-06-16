"""Original (per-vehicle) mCONV charging over micro trajectories (T7).

Newbolt's charging convolution (2026 eq 30-32) reduces, for a single vehicle,
to indexing the spatial power profile ``P_y`` at the vehicle's position each
step and integrating over time. Summed over EVs this gives the corridor's
DWPT demand profile -- the *original* mCONV side of the Stage-2 comparison
(no SOC / discharge, per the spec).

Each in-corridor position maps to the **interior** ``P_y`` index via the same
center-reference convention ``M_CTM`` uses (``j = off + round(x/dx_grid)`` with
``off = (delta_grid-1)//2``, clipped to the corridor). Mapping only to interior
positions means the micro charges exactly where ``M_CTM`` is non-zero, so the
adapted and original methods agree in the uniform-density limit (CP5).
"""

from __future__ import annotations

import numpy as np

_SECONDS_PER_HOUR = 3600.0


def micro_demand(result, corridor) -> np.ndarray:
    """Per-position DWPT demand [Wh] from the original mCONV over trajectories.

    Parameters
    ----------
    result : MicrosimResult
        Micro trajectories (only ``is_ev`` vehicles draw charge).
    corridor : CorridorSpec
        Provides ``build_P_y`` and the position grid.

    Returns
    -------
    ndarray (m,)
        Energy delivered at each traversal position, watt-hours, summed over
        time and over all EVs. Non-interior (approach/exit) positions stay 0.
    """
    P_y = corridor.build_P_y()
    dx = corridor.dx_grid
    off = (corridor.delta_grid - 1) // 2
    n_corr = corridor.n_corridor
    L = corridor.corridor_length_m

    a = result.position[:, :-1]  # start-of-interval positions (N, T)
    v = result.velocity
    dt_h = result.dt / _SECONDS_PER_HOUR

    in_corridor = (~np.isnan(v)) & (a >= 0.0) & (a < L)
    ev_mask = result.is_ev[:, None] & in_corridor

    E = np.zeros_like(P_y)
    positions = a[ev_mask]
    if positions.size:
        grid = np.clip(np.round(positions / dx).astype(int), 0, n_corr - 1)
        idx = grid + off
        np.add.at(E, idx, P_y[idx] * dt_h)
    return E


def micro_power_timeseries(result, corridor) -> np.ndarray:
    """Instantaneous corridor charging power [W] at each micro step.

    For each step, the sum of ``P_y`` over the EVs in the corridor. The total
    energy ``power.sum() * dt_h`` matches :func:`micro_demand`'s total; the
    peak is the grid-load peak (resolution-dependent: finer ``dt`` -> higher).
    """
    P_y = corridor.build_P_y()
    dx = corridor.dx_grid
    off = (corridor.delta_grid - 1) // 2
    n_corr = corridor.n_corridor
    L = corridor.corridor_length_m

    a = result.position[:, :-1]
    v = result.velocity
    in_corridor = (~np.isnan(v)) & (a >= 0.0) & (a < L)
    ev_mask = result.is_ev[:, None] & in_corridor

    power = np.zeros_like(v)
    if ev_mask.any():
        grid = np.clip(np.round(a[ev_mask] / dx).astype(int), 0, n_corr - 1)
        power[ev_mask] = P_y[grid + off]
    return power.sum(axis=0)
