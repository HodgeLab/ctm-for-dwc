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
        # Dissertation Example 1 is stated in the Gomes & Horowitz / four-mode
        # form, which sets gamma = 0. Pin it explicitly so the reported
        # equilibria rho^u = (80, 100) and rho^con = (160, 160) hold
        # regardless of the (now CTMSIM-canonical, gamma = 1) Cell default.
        return Cell(
            length=1.0,
            q_max=6000.0,
            v_f=60.0,
            w=20.0,
            rho_jam=400.0,
            rho_crit=100.0,
            gamma=0.0,
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

    Identical cells share Kurzhanskiy 2007's standard FD (v_f = 60, w = 20,
    rho_crit = 100, rho_jam = 400, q_max = 6000). On-ramps live on cells 0,
    1, and 3 (the example's r_1, r_2, r_4); cell 2 has none (r_3 = 0). All
    upstream cells carry an off-ramp with split beta = 0.2 in the scenario;
    cell 3 has no off-ramp (Kurzhanskiy 2007's beta_N = 0 convention, which
    is what makes f_4 = (f_3 + r_4) instead of 0.8*(f_3 + r_4)).
    """

    def cell(**overrides) -> Cell:
        # §3.4 example also follows Gomes & Horowitz (gamma = 0); pin it so
        # the golden test's feasible / infeasible flows match Kurzhanskiy 2007.
        return Cell(
            length=1.0,
            q_max=6000.0,
            v_f=60.0,
            w=20.0,
            rho_jam=400.0,
            rho_crit=100.0,
            gamma=0.0,
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


def metered_on_ramp_freeway() -> Freeway:
    """Two-cell freeway whose downstream on-ramp is capacity-limited.

    Standard Kurzhanskiy 2007 FD (v_f = 60, w = 20, rho_crit = 100,
    rho_jam = 400, q_max = 6000). Cell 0 has no ramps (it takes the upstream
    boundary inflow); cell 1 carries an on-ramp with a finite capacity
    ``R = 600 veh/h``. That finite ``R`` is what lets the on-ramp queue
    (eq. 4.3) grow: when arrival demand exceeds ``R`` the admitted flow ``r``
    is clipped to ``R`` and the surplus accumulates in the queue. The other
    examples leave ``on_ramp_capacity`` at its default (inf), so their queues
    stay pinned at zero.
    """
    return Freeway(
        dt=1.0 / 120.0,
        cells=[
            Cell(length=1.0, q_max=6000.0, v_f=60.0, w=20.0, rho_jam=400.0,
                 rho_crit=100.0),
            Cell(length=1.0, q_max=6000.0, v_f=60.0, w=20.0, rho_jam=400.0,
                 rho_crit=100.0, on_ramp=True, on_ramp_capacity=600.0),
        ],
    ).validate()


def metered_on_ramp_scenario(steps: int) -> Scenario:
    """Scenario that drives the metered on-ramp queue up, then drains it.

    On-ramp demand at cell 1 is held at 1000 veh/h (above the 600 veh/h ramp
    capacity) for the first half of the horizon, then dropped to 0 for the
    second half. During the first half the queue grows at (1000 - 600)*dt
    veh/step; during the second half it drains at 600*dt veh/step (the ramp
    keeps admitting at capacity until the queue empties). Since the drain rate
    exceeds the build rate, an equal-length second half fully empties the
    queue. Boundary inflow is a modest 3000 veh/h so the mainline stays
    uncongested and the cell's receiving room never binds -- isolating the
    capacity mechanism (eq. 4.2) as the sole cause of the queue.

    Starts from an empty freeway with an empty queue.
    """
    fwy = metered_on_ramp_freeway()
    n = fwy.n_cells
    demand = np.zeros((n, steps))
    demand[1, : steps // 2] = 1000.0  # above R for the first half, then 0
    return Scenario.build(
        fwy,
        steps=steps,
        rho0=0.0,
        inflow=3000.0,
        demand=demand,
        beta=0.0,
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
