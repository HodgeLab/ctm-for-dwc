"""Parity: the torch CTM step (diffsim) matches the numpy engine to ~1e-10.

M1 of the differentiable forward-sim ramp optimizer. The torch reimplementation
(``utils.ctm.diffsim.step``) is only trustworthy if it reproduces the exact
dynamics of ``utils.ctm.engine.step``; this pins them together on the per-
equation hand-calc systems from ``test_ctm_step.py`` plus randomized inputs
that exercise every branch (finite/inf ramp caps, the beta=0/1 edges,
congestion, boundary receiving). A single-cell + a multi-cell freeway cover the
``recv_down``/``f_up`` array-shift paths.
"""

import numpy as np
import pytest
import torch

from transportation_models.utils.ctm import diffsim
from transportation_models.utils.ctm.engine import step as np_step
from transportation_models.utils.ctm.model import Cell, Freeway

RTOL, ATOL = 0.0, 1e-9


def _ramp_freeway() -> Freeway:
    """3 cells: plain / on+off-ramp with finite caps / on-ramp with inf cap."""
    return Freeway(
        dt=1.0 / 120.0,
        cells=[
            Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0),
            Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0, on_ramp=True,
                 on_ramp_capacity=800.0, off_ramp=True, off_ramp_capacity=1200.0),
            Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0, on_ramp=True),
        ],
    ).validate()


def _single_cell() -> Freeway:
    return Freeway(dt=1.0 / 120.0, cells=[Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0)]).validate()


def _assert_step_parity(fwy, rho, queue, demand, beta, inflow):
    arr = fwy.arrays()
    np_res = np_step(arr, fwy.dt, rho, queue, demand, beta, inflow)

    tp = diffsim.freeway_arrays_to_torch(arr)
    to_t = lambda x: torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)  # noqa: E731
    t_res = diffsim.step(
        tp, fwy.dt, to_t(rho), to_t(queue), to_t(demand), to_t(beta), float(inflow),
    )

    for field in ("rho", "queue", "mainline_flow", "off_ramp", "on_ramp", "speed"):
        np.testing.assert_allclose(
            getattr(t_res, field).detach().numpy(), getattr(np_res, field),
            rtol=RTOL, atol=ATOL, err_msg=f"mismatch in {field}",
        )
    np.testing.assert_allclose(
        float(t_res.boundary_inflow), np_res.boundary_inflow, rtol=RTOL, atol=ATOL,
    )


# --- the per-equation hand-calc systems, replayed against torch --------------

@pytest.mark.parametrize(
    "rho, queue, demand, beta, inflow",
    [
        ([50.0, 30.0, 20.0], [0.0, 0.0, 0.0], [0.0, 500.0, 300.0], [0.0, 0.25, 0.0], 3000.0),  # free flow + ramps
        ([50.0, 380.0, 20.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 3000.0),       # congested outflow
        ([0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 2000.0, 1000.0], [0.0, 0.0, 0.0], 0.0),         # ramp-cap + queue drain
        ([100.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0),             # full diversion beta=1
        ([390.0, 390.0, 390.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 5000.0),      # boundary + near-jam recv
    ],
)
def test_hand_calc_systems_match(rho, queue, demand, beta, inflow):
    _assert_step_parity(_ramp_freeway(), rho, queue, demand, beta, inflow)


def test_single_cell_boundary_and_beta_one():
    _assert_step_parity(_single_cell(), [390.0], [0.0], [0.0], [0.0], 5000.0)
    _assert_step_parity(_single_cell(), [50.0], [0.0], [0.0], [1.0], 0.0)


# --- randomized branch coverage ---------------------------------------------

@pytest.mark.parametrize("seed", range(25))
def test_randomized_parity(seed):
    rng = np.random.default_rng(seed)
    fwy = _ramp_freeway()
    n = fwy.n_cells
    rho = rng.uniform(0.0, 400.0, n)          # spans free-flow through jam
    queue = rng.uniform(0.0, 20.0, n)
    demand = rng.uniform(0.0, 2500.0, n)
    beta = rng.uniform(0.0, 0.99, n)          # interior split (1.0 covered above)
    inflow = float(rng.uniform(0.0, 8000.0))
    _assert_step_parity(fwy, rho, queue, demand, beta, inflow)


def test_step_is_differentiable():
    """Gradients flow from density back to the genuine ramp decision variables.

    The optimizer's free variables are on-ramp-cell demand and off-ramp-cell
    beta (the latter reparameterized by a sigmoid, so strictly in (0,1)). We
    assert finite, nonzero grads on exactly those. The beta = 0 / beta = 1
    *exact* edges can yield NaN grads via torch.where masking the inf-valued
    (unselected) offcap branch -- but those cells are never decision variables
    (a cell with no off-ramp has beta fixed at 0), so the path is unreachable.
    """
    fwy = _ramp_freeway()  # on-ramp cells: 1, 2; off-ramp cell: 1
    tp = diffsim.freeway_arrays_to_torch(fwy.arrays())
    rho = torch.full((fwy.n_cells,), 50.0, dtype=torch.float64)
    queue = torch.zeros(fwy.n_cells, dtype=torch.float64)
    demand = torch.tensor([0.0, 600.0, 400.0], dtype=torch.float64, requires_grad=True)
    beta_off = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    beta = torch.stack([torch.zeros(()), beta_off, torch.zeros(())]).to(torch.float64)
    res = diffsim.step(tp, fwy.dt, rho, queue, demand, beta, 3000.0)
    res.rho.sum().backward()
    assert torch.isfinite(demand.grad[[1, 2]]).all()
    assert torch.isfinite(beta_off.grad)
    assert demand.grad[1] != 0.0  # on-ramp demand moves downstream density
    assert beta_off.grad != 0.0   # off-ramp split moves density
