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

from ..dwpt.model import DemandResult

_SECONDS_PER_HOUR = 3600.0


def micro_demand(result, corridor) -> DemandResult:
    """Spatiotemporal DWPT demand [Wh] from the original mCONV over trajectories.

    Builds the per-(timestep, position) energy array the same way the adapted
    :func:`~transportation_models.utils.dwpt.demand.compute` does, returning the
    same :class:`DemandResult` container so the two are a like-for-like
    comparison. Both reductions follow from ``E``: corridor energy per position
    is ``E.sum(axis=0)``, the corridor power series [W] is ``E.sum(axis=1)/dt_h``,
    and total energy is ``E.sum()``.

    Only ``is_ev`` vehicles in the corridor deposit charge; each in-corridor
    position maps to the interior ``P_y`` index via the same center-reference
    convention ``M_CTM`` uses (``j = off + round(x/dx_grid)``, clipped to the
    corridor), so the micro charges exactly where ``M_CTM`` is non-zero and the
    adapted/original methods agree in the uniform-density limit (CP5).

    Parameters
    ----------
    result : MicrosimResult
        Micro trajectories (only ``is_ev`` vehicles draw charge).
    corridor : CorridorSpec
        Provides ``build_P_y`` and the position grid.

    Returns
    -------
    DemandResult
        ``E`` has shape ``(T, m)`` in watt-hours: rows are micro timesteps,
        columns Rx-pad positions. Non-interior (approach/exit) positions and
        non-EV vehicles contribute zero.
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

    n_steps = v.shape[1]
    E = np.zeros((n_steps, P_y.size))  # (T, m), Wh
    if ev_mask.any():
        veh_i, step_i = np.nonzero(ev_mask)
        grid = np.clip(np.round(a[veh_i, step_i] / dx).astype(int), 0, n_corr - 1)
        idx = grid + off
        np.add.at(E, (step_i, idx), P_y[idx] * dt_h)

    return DemandResult(
        E=E, position_m=corridor.position_m, timesteps=np.arange(n_steps)
    )
