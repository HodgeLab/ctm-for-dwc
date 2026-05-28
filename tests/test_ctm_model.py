"""Config validation tests for the CTM model (P0)."""

import numpy as np
import pytest

from transportation_models.utils.ctm import examples
from transportation_models.utils.ctm.model import Cell, Freeway, Scenario


def test_rho_crit_defaults_to_qmax_over_vf():
    c = Cell(length=1.0, q_max=6000.0, v_f=60.0, w=20.0, rho_jam=400.0)
    assert c.rho_crit == pytest.approx(100.0)


def test_example1_freeway_validates():
    # Triggers Cell + Freeway validation (FD consistency, CFL); no raise.
    examples.example_1_freeway()


def test_example1_cells_are_fd_consistent():
    fwy = examples.example_1_freeway()
    for c in fwy.cells:
        free, cong = c.fd_residuals()
        assert free == pytest.approx(0.0, abs=1e-12)
        assert cong == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(length=-1.0, q_max=6000.0, v_f=60.0, w=20.0, rho_jam=400.0),
        dict(length=1.0, q_max=0.0, v_f=60.0, w=20.0, rho_jam=400.0),
        dict(length=1.0, q_max=6000.0, v_f=60.0, w=20.0, rho_jam=400.0, gamma=1.5),
        dict(length=1.0, q_max=6000.0, v_f=60.0, w=20.0, rho_jam=400.0, gamma=-0.1),
        dict(length=1.0, q_max=6000.0, v_f=60.0, w=20.0, rho_jam=400.0, xi=0.0),
        # rho_crit > rho_jam:
        dict(length=1.0, q_max=6000.0, v_f=60.0, w=20.0, rho_jam=50.0, rho_crit=100.0),
    ],
)
def test_invalid_cell_raises(kwargs):
    with pytest.raises(ValueError):
        Cell(**kwargs)


def test_cell_rejects_finite_on_ramp_capacity_without_on_ramp_flag():
    # A finite capacity is meaningless if the cell has no on-ramp; catch it loudly.
    with pytest.raises(ValueError, match="on_ramp_capacity"):
        Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0, on_ramp_capacity=600.0)
    # Pairing the flag with the capacity is fine.
    Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0, on_ramp=True, on_ramp_capacity=600.0)


def test_cell_rejects_finite_off_ramp_capacity_without_off_ramp_flag():
    with pytest.raises(ValueError, match="off_ramp_capacity"):
        Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0, off_ramp_capacity=800.0)
    Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0, off_ramp=True, off_ramp_capacity=800.0)


def test_scenario_rejects_demand_on_cell_without_on_ramp():
    fwy = examples.example_1_freeway()  # cell 0: no on-ramp; cell 1: has on-ramp
    demand = np.zeros((2, 5))
    demand[0, :] = 100.0  # demand on the no-ramp cell
    with pytest.raises(ValueError, match="cells without an on-ramp"):
        Scenario.build(fwy, steps=5, inflow=4800.0, demand=demand, beta=0.0)


def test_scenario_rejects_beta_on_cell_without_off_ramp():
    fwy = examples.example_1_freeway()  # neither cell has an off-ramp
    beta = np.zeros((2, 5))
    beta[1, :] = 0.1  # split on a no-off-ramp cell
    with pytest.raises(ValueError, match="cells without an off-ramp"):
        Scenario.build(fwy, steps=5, inflow=4800.0, demand=0.0, beta=beta)


def test_fd_inconsistency_detected():
    # v_f*rho_crit = 6000 (consistent) but w*(rho_jam-rho_crit) = 20*200 = 4000 != 6000
    c = Cell(length=1.0, q_max=6000.0, v_f=60.0, w=20.0, rho_jam=300.0, rho_crit=100.0)
    free, cong = c.fd_residuals()
    assert free == pytest.approx(0.0)
    assert cong > 0.1
    with pytest.raises(ValueError, match="FD inconsistent"):
        c.check_fd_consistency()


def test_freeway_validate_raises_on_cfl_violation():
    # v_f*dt = 60 * (1/30) = 2 mi > length = 1 mi
    fwy = Freeway(dt=1.0 / 30.0, cells=[Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0)])
    with pytest.raises(ValueError, match="CFL"):
        fwy.validate()


def test_freeway_arrays_match_cell_order():
    fwy = examples.example_1_freeway()
    arr = fwy.arrays()
    assert arr.length.shape == (2,)
    np.testing.assert_allclose(arr.v_f, [60.0, 60.0])
    np.testing.assert_allclose(arr.rho_jam, [400.0, 400.0])
    np.testing.assert_allclose(arr.q_max, [6000.0, 6000.0])


def test_scenario_build_broadcasts_scalars():
    fwy = examples.example_1_freeway()
    scn = Scenario.build(fwy, steps=10, rho0=0.0, inflow=4800.0, demand=0.0, beta=0.0)
    assert scn.rho0.shape == (2,)
    assert scn.q0.shape == (2,)
    assert scn.inflow.shape == (10,)
    assert scn.demand.shape == (2, 10)
    assert scn.beta.shape == (2, 10)
    assert scn.n_steps == 10


def test_scenario_rejects_beta_out_of_range():
    fwy = examples.example_1_freeway()
    with pytest.raises(ValueError, match="beta"):
        Scenario.build(fwy, steps=5, beta=1.5)


def test_scenario_rejects_negative_demand():
    fwy = examples.example_1_freeway()
    with pytest.raises(ValueError, match="demand"):
        Scenario.build(fwy, steps=5, demand=-1.0)


def test_scenario_rejects_rho0_above_jam():
    fwy = examples.example_1_freeway()
    with pytest.raises(ValueError, match="rho_jam"):
        Scenario.build(fwy, steps=5, rho0=500.0)
