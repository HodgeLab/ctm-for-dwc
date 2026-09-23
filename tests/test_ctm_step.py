"""Per-equation hand-calc tests for the canonical CTMSIM update (P1, layer 1).

Each test pins one or more of the six update equations of
``docs/ctm_module.md`` (Kurzhanskiy 2007, eqs. 4.2-4.8) by computing the step
result by hand on a small system and asserting the engine matches.
"""

import numpy as np
import pytest

from ctm_for_dwc.utils.ctm.engine import step
from ctm_for_dwc.utils.ctm.model import Cell, Freeway


def _two_identical_cells(dt: float = 1.0 / 120.0) -> Freeway:
    """Two cells with the Kurzhanskiy 2007 Example-1 FD; default dt = 30 s."""
    cell = lambda: Cell(  # noqa: E731
        length=1.0, q_max=6000.0, v_f=60.0, w=20.0, rho_jam=400.0, rho_crit=100.0
    )
    return Freeway(dt=dt, cells=[cell(), cell()]).validate()


# ---------------------------------------------------------------------------
# Mainline flow (eq. 4.4) + density (eq. 4.7) + speed (eq. 4.8): free flow
# ---------------------------------------------------------------------------

def test_freeflow_step_matches_hand_calc():
    fwy = _two_identical_cells()
    res = step(
        fwy.arrays(), fwy.dt,
        rho=[50.0, 30.0], queue=[0.0, 0.0],
        demand=[0.0, 0.0], beta=[0.0, 0.0], inflow=3000.0,
    )
    # cell 0 sends 60*50 = 3000 (< recv_1 = 20*(400-30) = 7400 and q_max);
    # cell 1 sends 60*30 = 1800 (free downstream).
    np.testing.assert_allclose(res.mainline_flow, [3000.0, 1800.0])
    np.testing.assert_allclose(res.on_ramp, [0.0, 0.0])
    np.testing.assert_allclose(res.off_ramp, [0.0, 0.0])
    # Boundary inflow 3000 == cell-0 outflow -> rho_0 unchanged.
    # Cell 1 net + (3000 - 1800) = 1200 veh/h * (1/120 h) / 1 mi = +10 veh/mi.
    np.testing.assert_allclose(res.rho, [50.0, 40.0])
    # Mean speed = min(v_f, (f+s)/rho_new). cell 0: 3000/50=60; cell 1: 1800/40=45.
    np.testing.assert_allclose(res.speed, [60.0, 45.0])


# ---------------------------------------------------------------------------
# Mainline flow (eq. 4.4): downstream-supply-limited (congested out)
# ---------------------------------------------------------------------------

def test_congested_outflow_step_matches_hand_calc():
    fwy = _two_identical_cells()
    res = step(
        fwy.arrays(), fwy.dt,
        rho=[50.0, 380.0], queue=[0.0, 0.0],
        demand=[0.0, 0.0], beta=[0.0, 0.0], inflow=3000.0,
    )
    # recv_1 = 20*(400-380) = 400 -> cell-0 outflow = min(3000, 400, q_max) = 400.
    # cell 1 outflow capped at q_max (= 6000) since send = 60*380 = 22800 and
    # downstream is free.
    np.testing.assert_allclose(res.mainline_flow, [400.0, 6000.0])
    # rho_0_new = 50 + (1/120)*(3000 + 0 - 400 - 0)
    np.testing.assert_allclose(res.rho[0], 50.0 + 2600.0 / 120.0)
    # rho_1_new = 380 + (1/120)*(400 + 0 - 6000 - 0)
    np.testing.assert_allclose(res.rho[1], 380.0 - 5600.0 / 120.0)


# ---------------------------------------------------------------------------
# On-ramp flow (eq. 4.2) + queue (eq. 4.3): ramp-capacity-limited
# ---------------------------------------------------------------------------

def test_on_ramp_flow_capped_by_R_grows_queue():
    dt = 1.0 / 120.0
    fwy = Freeway(
        dt=dt,
        cells=[
            Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0),
            Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0, on_ramp=True, on_ramp_capacity=600.0),
        ],
    ).validate()
    res = step(
        fwy.arrays(), fwy.dt,
        rho=[0.0, 0.0], queue=[0.0, 0.0],
        demand=[0.0, 1000.0], beta=[0.0, 0.0], inflow=0.0,
    )
    # r_1 = min(1000 + 0, xi*(rho_jam-rho)*L/dt = large, R=600) = 600.
    np.testing.assert_allclose(res.on_ramp[1], 600.0)
    # queue_1_new = max(0 + (1000 - 600)*dt, 0) = 400*dt.
    np.testing.assert_allclose(res.queue[1], 400.0 * dt)


def test_on_ramp_queue_drains_when_admitted_exceeds_arrivals():
    dt = 1.0 / 120.0
    fwy = Freeway(dt=dt, cells=[Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0)]).validate()
    # Existing queue of 5 veh; arrivals 0; capacity-limited intake = min(0+5/dt, large, inf)=5/dt veh/h
    res = step(
        fwy.arrays(), fwy.dt,
        rho=[0.0], queue=[5.0],
        demand=[0.0], beta=[0.0], inflow=0.0,
    )
    np.testing.assert_allclose(res.on_ramp[0], 5.0 / dt)
    # queue_new = max(5 + (0 - 5/dt)*dt, 0) = 0
    np.testing.assert_allclose(res.queue[0], 0.0)


# ---------------------------------------------------------------------------
# Off-ramp flow (eq. 4.6): proportional split (beta < 1)
# ---------------------------------------------------------------------------

def test_off_ramp_split_beta_less_than_one():
    fwy = _two_identical_cells()
    res = step(
        fwy.arrays(), fwy.dt,
        rho=[100.0, 0.0], queue=[0.0, 0.0],
        demand=[0.0, 0.0], beta=[0.25, 0.0], inflow=0.0,
    )
    # send_0 = 0.75 * 60 * 100 = 4500; recv_1 = 8000; q_max = 6000 -> f_0 = 4500.
    # s_0 = (0.25/0.75) * 4500 = 1500; total leaving cell 0 = 6000 = v_f*rho.
    np.testing.assert_allclose(res.mainline_flow[0], 4500.0)
    np.testing.assert_allclose(res.off_ramp[0], 1500.0)


# ---------------------------------------------------------------------------
# Off-ramp flow (eq. 4.6): full diversion (beta == 1)
# ---------------------------------------------------------------------------

def test_full_off_ramp_beta_one():
    fwy = _two_identical_cells()
    res = step(
        fwy.arrays(), fwy.dt,
        rho=[50.0, 0.0], queue=[0.0, 0.0],
        demand=[0.0, 0.0], beta=[1.0, 0.0], inflow=0.0,
    )
    # beta = 1 -> mainline f_0 = 0 (beta_bar = 0). Off-ramp s_0 = min(v_f*rho, S)
    # = min(60*50, inf) = 3000.
    np.testing.assert_allclose(res.mainline_flow[0], 0.0)
    np.testing.assert_allclose(res.off_ramp[0], 3000.0)


# ---------------------------------------------------------------------------
# Boundary handling: f_0 capped by cell-0 receiving (single cell, near jam)
# ---------------------------------------------------------------------------

def test_single_cell_boundary_inflow_capped_by_receiving():
    fwy = Freeway(dt=1.0 / 120.0, cells=[Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0)]).validate()
    res = step(
        fwy.arrays(), fwy.dt,
        rho=[390.0], queue=[0.0],
        demand=[0.0], beta=[0.0], inflow=5000.0,
    )
    # recv_0 = 20*(400-390) = 200 -> admitted inflow = min(5000, 200) = 200.
    # Outflow capped by q_max = 6000 (free downstream).
    np.testing.assert_allclose(res.mainline_flow[0], 6000.0)
    np.testing.assert_allclose(res.rho[0], 390.0 - 5800.0 / 120.0)


# ---------------------------------------------------------------------------
# On-ramp blending (gamma): r_i adds to sending in the same step
# ---------------------------------------------------------------------------

def test_gamma_blends_on_ramp_into_same_step_sending():
    dt = 1.0 / 120.0
    # Single cell, no off-ramp, free downstream. gamma=1 fully blends r_i.
    fwy = Freeway(
        dt=dt,
        cells=[Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0, gamma=1.0)],
    ).validate()
    res = step(
        fwy.arrays(), fwy.dt,
        rho=[30.0], queue=[0.0],
        demand=[1200.0], beta=[0.0], inflow=0.0,
    )
    # r_0 = min(1200, large, inf) = 1200.
    # rho_eff = 30 + 1*1200*(1/120)/1 = 30 + 10 = 40. send = 60*40 = 2400.
    # f_0 = min(send=2400, inf, inf, q_max=6000) = 2400.
    np.testing.assert_allclose(res.on_ramp[0], 1200.0)
    np.testing.assert_allclose(res.mainline_flow[0], 2400.0)
