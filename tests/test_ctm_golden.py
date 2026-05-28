"""Golden tests: the engine vs. published behavior from the dissertation (P3).

These tests cover behaviors that single-step hand-calcs (P1) and structural
invariants (P2) can't verify on their own: the time-dependent propagation of
density waves at the LWR characteristic speeds, and the global structure of
the equilibrium set under strictly feasible demand (Theorem 3.3.1).

Targets are stated in the dissertation, not derived from our implementation,
so any match is genuinely independent verification.

Source: A. Kurzhanskiy, "Modeling and Software Tools for Freeway Operational
Planning," UCB/EECS-2007-148, 2007.
"""

import numpy as np

from transportation_models.utils.ctm import examples, simulate
from transportation_models.utils.ctm.model import Cell, Freeway, Scenario


def test_freeflow_pulse_propagates_at_v_f():
    """A free-flow density pulse moves downstream at v_f, one cell per step at Courant=1.

    Setup: 5 identical cells, v_f*dt = l = 1 mi (Courant=1 on the free-flow
    side). One-step boundary inflow pulse of 3000 veh/h injected at step 0
    (below q_max so the pulse stays free-flow at rho = 3000/v_f = 50 veh/mi).

    Expected: at state index k+1 (just after step k), the pulse density sits
    in cell k and every other cell is empty; at state index n+1 the pulse
    has exited the downstream sink.

    Source: diss. §2.2.1 (LWR characteristic speed in free flow, p. 11-12),
    §2.2.2 (Godunov discretization, p. 17-21), Fig. 2.7 (triangular FD,
    p. 20); CFL rule, eq. 4.1, p. 84.
    """
    n = 5
    cells = [
        Cell(length=1.0, q_max=6000.0, v_f=60.0, w=20.0, rho_jam=400.0, rho_crit=100.0)
        for _ in range(n)
    ]
    fwy = Freeway(dt=1.0 / 60.0, cells=cells).validate()  # v_f*dt = 1 mi = l

    steps = n + 1
    inflow = np.zeros(steps)
    inflow[0] = 3000.0  # one-step pulse
    scn = Scenario.build(fwy, steps=steps, rho0=0.0, inflow=inflow, demand=0.0, beta=0.0)
    res = simulate(fwy, scn)

    pulse_rho = 3000.0 / 60.0  # = 50 veh/mi
    # After step k (state at index k+1), the pulse lives in cell k only.
    for k in range(n):
        for i in range(n):
            expected = pulse_rho if i == k else 0.0
            np.testing.assert_allclose(
                res.density[i, k + 1], expected, atol=1e-9,
                err_msg=(
                    f"state {k + 1}: expected pulse in cell {k}, "
                    f"got density[{i}]={res.density[i, k + 1]}"
                ),
            )
    # State n+1 = steps: pulse has exited via the downstream sink.
    np.testing.assert_allclose(res.density[:, steps], 0.0, atol=1e-9)


def test_congestion_wave_propagates_at_w():
    """A jam wave moves upstream at w, one cell per step at Courant=1 on the congestion side.

    Setup: 4 identical cells with a symmetric FD (v_f = w = 20 mph,
    rho_crit = 100, rho_jam = 200, q_max = 2000); w*dt = l = 1 mi. Initial
    state has cells 0..n-2 at critical density (capacity throughput) and the
    last cell at jam density, with boundary inflow = q_max sustained.

    Expected: at state index k (k = 0..n-1), the jammed cell is at index
    n-1-k; all other cells remain at critical density. At state index n the
    jam has propagated past the upstream boundary and every cell is back at
    critical.

    Source: diss. §2.2.1 (LWR characteristic speed in congestion, p. 11-12);
    direct statement on p. 47 ("the congestion wave speed is 20 mph it
    takes about 3 minutes for the congestion wave to travel the one
    mile-long cell").
    """
    n = 4
    cells = [
        Cell(length=1.0, q_max=2000.0, v_f=20.0, w=20.0, rho_jam=200.0, rho_crit=100.0)
        for _ in range(n)
    ]
    fwy = Freeway(dt=1.0 / 20.0, cells=cells).validate()  # w*dt = 1 mi = l

    rho0 = np.array([100.0] * (n - 1) + [200.0])  # critical, ..., critical, jam
    steps = n + 1
    scn = Scenario.build(fwy, steps=steps, rho0=rho0, inflow=2000.0, demand=0.0, beta=0.0)
    res = simulate(fwy, scn)

    # At state index k, the jam sits in cell n-1-k; all others at critical.
    for k in range(n):
        jam_cell = n - 1 - k
        for i in range(n):
            expected = 200.0 if i == jam_cell else 100.0
            np.testing.assert_allclose(
                res.density[i, k], expected, atol=1e-9,
                err_msg=(
                    f"state {k}: expected jam in cell {jam_cell}, "
                    f"got density[{i}]={res.density[i, k]}"
                ),
            )
    # State n: jam has propagated past the upstream boundary; all cells back at critical.
    np.testing.assert_allclose(res.density[:, n], 100.0, atol=1e-9)


def test_strictly_feasible_demand_yields_unique_equilibrium():
    """Theorem 3.3.1: strictly feasible demand yields a single equilibrium n^u.

    Example 2 (diss. p. 47-48, Fig. 3.8): the Example 1 freeway with upstream
    inflow reduced from 4800 to 4750 vph. Total demand 5950 < capacity 6000,
    so the equilibrium set collapses to n^u alone and every trajectory
    converges to it -- including the from-jam trajectory which in Example 1
    reaches the distinct equilibrium (160, 160).

    With r_0 = 4750 and r_2 = 1200 (the on-ramp), the algebraic equilibrium
    is n^u = (4750/v_f, 5950/v_f) = (79.166..., 99.166...) veh/mi.

    Source: Theorem 3.3.1, p. 45; Example 2, p. 47-48; Fig. 3.8.
    """
    fwy = examples.example_1_freeway()
    steps = 2400  # 20 h: plenty for the from-jam trajectory to drain to free flow
    demand = np.zeros((2, steps))
    demand[1, :] = 1200.0  # on-ramp at the downstream cell, unchanged from Example 1

    scn_empty = Scenario.build(
        fwy, steps=steps, rho0=0.0, inflow=4750.0, demand=demand, beta=0.0
    )
    scn_jam = Scenario.build(
        fwy, steps=steps, rho0=400.0, inflow=4750.0, demand=demand, beta=0.0
    )
    res_empty = simulate(fwy, scn_empty)
    res_jam = simulate(fwy, scn_jam)

    n_u = np.array([4750.0 / 60.0, 5950.0 / 60.0])
    np.testing.assert_allclose(res_empty.density[:, -1], n_u, atol=1e-6)
    np.testing.assert_allclose(res_jam.density[:, -1], n_u, atol=1e-6)
    # Both trajectories collapse onto the same point -- the heart of Theorem 3.3.1.
    np.testing.assert_allclose(res_empty.density[:, -1], res_jam.density[:, -1], atol=1e-9)
