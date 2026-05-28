"""Hand-built CTM systems for demos and golden tests."""

from __future__ import annotations

import numpy as np

from .model import Cell, Freeway, Scenario


def example_1_freeway() -> Freeway:
    """Two-cell freeway from Kurzhanskiy diss. Ch. 3, Example 1 (p. 46).

    v_f = 60 mph, w = 20 mph, rho_crit = 100, rho_jam = 400 veh/mi, capacity
    6000 veh/h. dt = 1/120 h (30 s) satisfies the CFL rule (v_f*dt = 0.5 mi <= 1).
    """

    def cell(**overrides) -> Cell:
        return Cell(
            length=1.0,
            q_max=6000.0,
            v_f=60.0,
            w=20.0,
            rho_jam=400.0,
            rho_crit=100.0,
            **overrides,
        )

    # Cell 0 has no ramps (it receives the upstream boundary inflow); cell 1
    # carries the on-ramp r_2 = 1200 veh/h from Example 1.
    return Freeway(
        dt=1.0 / 120.0,
        cells=[cell(), cell(on_ramp=True)],
    ).validate()


def four_cell_freeway() -> Freeway:
    """Four-cell freeway from Kurzhanskiy diss. §3.4 (p. 54, Fig. 3.10).

    Identical cells share the dissertation's standard FD (v_f = 60, w = 20,
    rho_crit = 100, rho_jam = 400, q_max = 6000). On-ramps live on cells 0,
    1, and 3 (the example's r_1, r_2, r_4); cell 2 has none (r_3 = 0). All
    upstream cells carry an off-ramp with split beta = 0.2 in the scenario;
    cell 3 has no off-ramp (the dissertation's beta_N = 0 convention, which
    is what makes f_4 = (f_3 + r_4) instead of 0.8*(f_3 + r_4)).
    """

    def cell(**overrides) -> Cell:
        return Cell(
            length=1.0,
            q_max=6000.0,
            v_f=60.0,
            w=20.0,
            rho_jam=400.0,
            rho_crit=100.0,
            **overrides,
        )

    return Freeway(
        dt=1.0 / 120.0,
        cells=[
            cell(on_ramp=True, off_ramp=True),    # cell 0: r_1 = 2000, beta = 0.2
            cell(on_ramp=True, off_ramp=True),    # cell 1: r_2 = 2700, beta = 0.2
            cell(on_ramp=False, off_ramp=True),   # cell 2: no on-ramp, beta = 0.2
            cell(on_ramp=True, off_ramp=False),   # cell 3: r_4 (1200 / 1300), beta_N = 0
        ],
    ).validate()


def four_cell_scenario(steps: int, *, r_4: float = 1200.0) -> Scenario:
    """Scenario for the §3.4 example: feasible by default, infeasible at r_4 = 1300.

    Demand pattern follows Fig. 3.10:
        boundary inflow r_0 = 4000;
        on-ramps   r_1 = 2000  (cell 0)
                   r_2 = 2700  (cell 1)
                   r_3 = 0     (cell 2 has no on-ramp)
                   r_4 = `r_4` (cell 3, default 1200 = feasible; 1300 = infeasible)
        off-ramps  beta = 0.2 on cells 0, 1, 2;  beta_N = 0 on cell 3.

    Starts from an empty freeway.
    """
    fwy = four_cell_freeway()
    n = fwy.n_cells
    demand = np.zeros((n, steps))
    demand[0, :] = 2000.0
    demand[1, :] = 2700.0
    demand[3, :] = r_4
    beta = np.zeros((n, steps))
    beta[0:3, :] = 0.2
    return Scenario.build(
        fwy,
        steps=steps,
        rho0=0.0,
        inflow=4000.0,
        demand=demand,
        beta=beta,
    )


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
