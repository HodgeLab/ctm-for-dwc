"""Performance metrics for a CTM simulation: VHT, VMT, delay, productivity loss.

Implements the per-step metrics from Kurzhanskiy diss. §4.2.1 (eqs. 4.9-4.17,
p. 86-87). Each metric at step index ``k`` (``k = 0..T-1``) corresponds to
the dissertation's "period k+1" value: it is computed from the *post-step*
state (``rho_i(k+1)``, ``q_i(k+1)``, ``V_i(k+1)``) and the step's mainline
flow ``f_i(k+1)``.

==========================  ==========  ==============================
metric                      unit        source
==========================  ==========  ==============================
travel_time                 h           eq. 4.9
vht_per_cell, vht           veh*h       eqs. 4.10, 4.11
vmt_per_cell, vmt           veh*mi      eqs. 4.12, 4.13
delay_per_cell, delay       veh*h       eqs. 4.14, 4.15
productivity_loss_per_cell, mi*h        eqs. 4.16, 4.17
productivity_loss
==========================  ==========  ==============================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .results import SimulationResult


@dataclass
class Metrics:
    """Per-cell and total CTM performance metrics over a simulation horizon.

    Per-cell arrays are shaped ``(n_cells, T)``; the total properties sum over
    cells to shape ``(T,)``.
    """

    travel_time: np.ndarray                  # (T,)         [h]      eq. 4.9
    vht_per_cell: np.ndarray                 # (n_cells, T) [veh*h]  eq. 4.10
    vmt_per_cell: np.ndarray                 # (n_cells, T) [veh*mi] eq. 4.12
    delay_per_cell: np.ndarray               # (n_cells, T) [veh*h]  eq. 4.14
    productivity_loss_per_cell: np.ndarray   # (n_cells, T) [mi*h]   eq. 4.16

    @property
    def vht(self) -> np.ndarray:
        """Total VHT per step (eq. 4.11), shape ``(T,)``."""
        return self.vht_per_cell.sum(axis=0)

    @property
    def vmt(self) -> np.ndarray:
        """Total VMT per step (eq. 4.13), shape ``(T,)``."""
        return self.vmt_per_cell.sum(axis=0)

    @property
    def delay(self) -> np.ndarray:
        """Total delay per step (eq. 4.15), shape ``(T,)``."""
        return self.delay_per_cell.sum(axis=0)

    @property
    def productivity_loss(self) -> np.ndarray:
        """Total productivity loss per step (eq. 4.17), shape ``(T,)``."""
        return self.productivity_loss_per_cell.sum(axis=0)


def compute_metrics(res: SimulationResult) -> Metrics:
    """Compute the per-step performance metrics (eqs. 4.9-4.17) for ``res``."""
    L = np.array([c.length for c in res.freeway.cells])
    v_f = np.array([c.v_f for c in res.freeway.cells])
    rho_crit = np.array([c.rho_crit for c in res.freeway.cells])
    q_max = np.array([c.q_max for c in res.freeway.cells])
    dt = res.freeway.dt

    # Align all per-step quantities to step index k = 0..T-1:
    #   density and queue use the *post-step* state at index k+1;
    #   speed and flow are already indexed by step k in SimulationResult.
    rho = res.density[:, 1:]        # (n_cells, T)  rho_i(k+1)
    queue = res.queue[:, 1:]        # (n_cells, T)  q_i(k+1)
    speed = res.speed               # (n_cells, T)  V_i(k+1)
    flow = res.mainline_flow        # (n_cells, T)  f_i(k+1)

    # eq. 4.9 -- instantaneous travel time across the freeway.
    # V_i is set to v_f when rho_i == 0 (see engine.step, eq. 4.8), so speed > 0
    # for any non-empty cell; guard the rare pathological case (stuck cell).
    with np.errstate(divide="ignore", invalid="ignore"):
        per_cell_tt = np.where(speed > 0.0, L[:, None] / speed, np.inf)
    travel_time = per_cell_tt.sum(axis=0)

    # eq. 4.10 -- VHT includes vehicles waiting in on-ramp queues.
    vht_per_cell = (rho * L[:, None] + queue) * dt

    # eq. 4.12 -- VMT counts vehicle-miles traveled in each cell.
    vmt_per_cell = rho * speed * dt * L[:, None]

    # eq. 4.14 / 4.16 -- delay and productivity loss only accrue in congestion
    # (rho > rho_crit, the right branch of the triangular FD).
    congested = rho > rho_crit[:, None]

    delay_per_cell = np.where(
        congested, vht_per_cell - vmt_per_cell / v_f[:, None], 0.0
    )

    productivity_loss_per_cell = np.where(
        congested, (1.0 - flow / q_max[:, None]) * dt * L[:, None], 0.0
    )

    return Metrics(
        travel_time=travel_time,
        vht_per_cell=vht_per_cell,
        vmt_per_cell=vmt_per_cell,
        delay_per_cell=delay_per_cell,
        productivity_loss_per_cell=productivity_loss_per_cell,
    )
