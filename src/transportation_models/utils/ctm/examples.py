"""Hand-built CTM systems for demos and golden tests."""

from __future__ import annotations

import numpy as np

from .model import Cell, Freeway, Scenario


def example_1_freeway() -> Freeway:
    """Two-cell freeway from Kurzhanskiy diss. Ch. 3, Example 1 (p. 46).

    v_f = 60 mph, w = 20 mph, rho_crit = 100, rho_jam = 400 veh/mi, capacity
    6000 veh/h. dt = 1/120 h (30 s) satisfies the CFL rule (v_f*dt = 0.5 mi <= 1).
    """

    def cell() -> Cell:
        return Cell(
            length=1.0,
            q_max=6000.0,
            v_f=60.0,
            w=20.0,
            rho_jam=400.0,
            rho_crit=100.0,
        )

    return Freeway(dt=1.0 / 120.0, cells=[cell(), cell()]).validate()


def example_1_scenario(steps: int, start: str = "empty") -> Scenario:
    """Constant-demand scenario for Example 1.

    Upstream inflow r_0 = 4800 veh/h, on-ramp r_2 = 1200 veh/h at the downstream
    cell, no off-ramps. ``start="empty"`` (rho0 = 0) is expected to converge to the
    uncongested equilibrium (80, 100); ``start="jam"`` (rho0 = rho_jam) to the most
    congested equilibrium (160, 160) (diss. p. 46-47). The convergence check itself
    lands in P3.
    """
    if start not in ("empty", "jam"):
        raise ValueError("start must be 'empty' or 'jam'")
    fwy = example_1_freeway()
    rho0 = 0.0 if start == "empty" else 400.0
    demand = np.zeros((fwy.n_cells, steps))
    demand[1, :] = 1200.0  # on-ramp at the downstream cell (index 1)
    return Scenario.build(fwy, steps, rho0=rho0, inflow=4800.0, demand=demand, beta=0.0)
