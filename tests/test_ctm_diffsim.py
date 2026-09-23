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

from ctm_for_dwc.utils.ctm import diffsim
from ctm_for_dwc.utils.ctm.engine import simulate as np_simulate
from ctm_for_dwc.utils.ctm.engine import step as np_step
from ctm_for_dwc.utils.ctm.model import Cell, Freeway, Scenario

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
    """Gradients flow from density back to demand/beta -- finite everywhere.

    Because every inf in the engine (unbounded R/S, the boundary receiving, the
    downstream-free receiving) is a *constant*, no masked min/where branch ever
    backprops 0*inf = NaN. So even the full demand/beta vectors -- including the
    beta=0 rows of non-off-ramp cells and the S=inf uncapacitated off-ramp --
    are safe grad leaves. cell 1 is on+off-ramp with finite caps; cell 2 is an
    on-ramp with S/R = inf, exercising the constant-inf path.
    """
    fwy = _ramp_freeway()  # on-ramp cells: 1, 2 (cell 2 caps inf); off-ramp cell: 1
    tp = diffsim.freeway_arrays_to_torch(fwy.arrays())
    rho = torch.full((fwy.n_cells,), 50.0, dtype=torch.float64)
    queue = torch.zeros(fwy.n_cells, dtype=torch.float64)
    demand = torch.tensor([0.0, 600.0, 400.0], dtype=torch.float64, requires_grad=True)
    beta = torch.tensor([0.0, 0.3, 0.0], dtype=torch.float64, requires_grad=True)
    res = diffsim.step(tp, fwy.dt, rho, queue, demand, beta, 3000.0)
    res.rho.sum().backward()
    assert torch.isfinite(demand.grad).all()
    assert torch.isfinite(beta.grad).all()
    assert demand.grad[1] != 0.0  # on-ramp demand moves downstream density
    assert beta.grad[1] != 0.0    # off-ramp split moves density


# ===========================================================================
# M2: differentiable trajectory unroll (diffsim.simulate)
# ===========================================================================

def _to_t(x):
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)


def test_trajectory_matches_numpy_simulate():
    """Full-horizon parity: diffsim.simulate == engine.simulate over T steps."""
    fwy = _ramp_freeway()
    n, T = fwy.n_cells, 40
    rng = np.random.default_rng(7)
    demand = np.zeros((n, T))
    demand[[1, 2], :] = rng.uniform(0.0, 1500.0, (2, T))   # on-ramp cells only
    beta = np.zeros((n, T))
    beta[1, :] = rng.uniform(0.0, 0.6, T)                  # off-ramp cell only
    inflow = rng.uniform(1000.0, 6000.0, T)
    rho0 = rng.uniform(10.0, 120.0, n)

    scn = Scenario.build(fwy, T, rho0=rho0, inflow=inflow, demand=demand, beta=beta)
    np_res = np_simulate(fwy, scn)

    traj = diffsim.simulate(
        diffsim.freeway_arrays_to_torch(fwy.arrays()), fwy.dt,
        _to_t(rho0), _to_t(np.zeros(n)), _to_t(demand), _to_t(beta), _to_t(inflow),
    )
    for field in ("density", "queue", "mainline_flow", "off_ramp", "on_ramp", "speed",
                  "boundary_inflow"):
        np.testing.assert_allclose(
            getattr(traj, field).detach().numpy(), getattr(np_res, field),
            rtol=0.0, atol=1e-8, err_msg=f"trajectory mismatch in {field}",
        )


def _single_ramp_freeway() -> Freeway:
    """cell 0 on-ramp only, cell 1 off-ramp only -> each ramp flow is uniquely
    identifiable from observed mainline density+flow (no r-minus-s ambiguity)."""
    return Freeway(
        dt=1.0 / 120.0,
        cells=[
            Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0, on_ramp=True),
            Cell(1.0, 6000.0, 60.0, 20.0, 400.0, 100.0, off_ramp=True),
        ],
    ).validate()


def test_density_flow_gradient_matches_finite_difference():
    """autograd d(loss)/d(demand,beta) matches central finite differences."""
    fwy = _single_ramp_freeway()
    params = diffsim.freeway_arrays_to_torch(fwy.arrays())
    n, T = fwy.n_cells, 12
    rho0 = _to_t(np.full(n, 40.0))
    q0 = _to_t(np.zeros(n))
    inflow = _to_t(np.full(T, 3000.0))
    # arbitrary "observed" targets so the loss has a nontrivial gradient
    rng = np.random.default_rng(3)
    rho_obs = _to_t(rng.uniform(30.0, 60.0, (n, T)))
    f_obs = _to_t(rng.uniform(2000.0, 4000.0, (n, T)))

    demand0 = np.zeros((n, T)); demand0[0, :] = 500.0
    beta0 = np.zeros((n, T)); beta0[1, :] = 0.2

    def loss_from(demand_t, beta_t):
        traj = diffsim.simulate(params, fwy.dt, rho0, q0, demand_t, beta_t, inflow)
        drho = traj.density[:, 1:] - rho_obs
        df = traj.mainline_flow - f_obs
        return (drho**2).mean() + (df**2).mean()

    demand = _to_t(demand0).requires_grad_(True)
    beta = _to_t(beta0).requires_grad_(True)
    loss_from(demand, beta).backward()

    # Central FD on a representative free entry of each: on-ramp demand[0,t]
    # and off-ramp beta[1,t]. Perturb one entry, hold the other matrix fixed.
    for (name, base, grad, idx, h) in [
        ("demand", demand0, demand.grad, (0, 5), 1e-2),
        ("beta", beta0, beta.grad, (1, 5), 1e-5),
    ]:
        plus = base.copy(); plus[idx] += h
        minus = base.copy(); minus[idx] -= h
        other = _to_t(beta0) if name == "demand" else _to_t(demand0)
        lp = loss_from(_to_t(plus), other) if name == "demand" else loss_from(other, _to_t(plus))
        lm = loss_from(_to_t(minus), other) if name == "demand" else loss_from(other, _to_t(minus))
        fd = (lp.item() - lm.item()) / (2 * h)
        ana = grad[idx].item()
        assert abs(ana - fd) <= 1e-6 + 1e-4 * abs(fd), f"{name}: autograd {ana} vs FD {fd}"


def test_identifiability_recovers_ramp_flows():
    """Exact-dynamics analog of the QP's synthetic identifiability check.

    Known freeway -> forward-sim density+flow ('observed') -> recover the ramp
    inputs from a perturbed init by descending the density+flow error. With
    single-ramp cells the inverse is unique, so the recovered on/off-ramp flows
    must return to the truth (not merely some density-equivalent solution).
    """
    torch.manual_seed(0)
    fwy = _single_ramp_freeway()
    params = diffsim.freeway_arrays_to_torch(fwy.arrays())
    n, T = fwy.n_cells, 30
    rho0 = _to_t(np.full(n, 40.0))
    q0 = _to_t(np.zeros(n))
    inflow = _to_t(np.full(T, 3000.0))

    t = np.linspace(0, 1, T)
    demand_true = np.zeros((n, T)); demand_true[0, :] = 800.0 + 400.0 * np.sin(2 * np.pi * t)
    beta_true = np.zeros((n, T)); beta_true[1, :] = 0.30 + 0.15 * np.sin(2 * np.pi * t + 1.0)
    truth = diffsim.simulate(params, fwy.dt, rho0, q0, _to_t(demand_true), _to_t(beta_true), inflow)
    rho_obs = truth.density[:, 1:].detach()
    f_obs = truth.mainline_flow.detach()

    # Free variables: on-ramp demand[0,:] (>=0) and off-ramp beta[1,:] (sigmoid),
    # scattered into constant-zero rows for the non-ramp entries -- exactly how
    # the M3 optimizer will parameterize. Loss is scale-normalized so the
    # flow term's ~1e6 magnitude doesn't dominate the density term.
    RHO_SCALE, F_SCALE = 50.0, 2000.0
    d_raw = torch.tensor(demand_true[0] + 300.0, dtype=torch.float64, requires_grad=True)
    b_logit = torch.logit(_to_t(np.clip(beta_true[1] + 0.15, 1e-3, 1 - 1e-3))).requires_grad_(True)
    opt = torch.optim.Adam([{"params": [d_raw], "lr": 20.0}, {"params": [b_logit], "lr": 0.1}])

    zero_row = torch.zeros(T, dtype=torch.float64)
    build = lambda: (  # noqa: E731
        torch.stack([torch.clamp(d_raw, min=0.0), zero_row]),
        torch.stack([zero_row, torch.sigmoid(b_logit)]),
    )
    for _ in range(1000):
        opt.zero_grad()
        demand, beta = build()
        traj = diffsim.simulate(params, fwy.dt, rho0, q0, demand, beta, inflow)
        loss = ((traj.density[:, 1:] - rho_obs) ** 2).mean() / RHO_SCALE**2 + \
               ((traj.mainline_flow - f_obs) ** 2).mean() / F_SCALE**2
        loss.backward()
        opt.step()

    with torch.no_grad():
        demand, beta = build()
        traj = diffsim.simulate(params, fwy.dt, rho0, q0, demand, beta, inflow)
    r_rmse = torch.sqrt(((traj.on_ramp[0] - truth.on_ramp[0]) ** 2).mean()).item()
    s_rmse = torch.sqrt(((traj.off_ramp[1] - truth.off_ramp[1]) ** 2).mean()).item()
    rho_rmse = torch.sqrt(((traj.density[:, 1:] - rho_obs) ** 2).mean()).item()
    assert rho_rmse < 1e-4, f"density not matched: {rho_rmse}"
    assert r_rmse < 1.0, f"on-ramp flow not recovered: {r_rmse}"   # truth ~800 veh/h
    assert s_rmse < 1.0, f"off-ramp flow not recovered: {s_rmse}"
