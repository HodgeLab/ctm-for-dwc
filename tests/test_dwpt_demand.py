"""Hand-calc and invariant tests for compute_demand (P4).

``compute_demand`` assembles E = eta_EV * (VHT^T M_CTM) * P_y. Verified by
the plan's central energy-conservation identity (layer 2) plus hand calc,
linearity, and non-negativity (layers 1-2).
"""

import numpy as np

from transportation_models.utils.dwpt.demand import compute_demand
from transportation_models.utils.dwpt.mapping import build_M_CTM
from transportation_models.utils.dwpt.spatial import (
    build_P_y,
    build_S_R,
    build_S_T,
)


def _profile(ppc, alpha_grid, lambda_grid, delta_grid, gamma):
    """A P_y of length m = sum(ppc) + delta_grid - 1, plus its M_CTM."""
    n = sum(ppc)
    S_T = build_S_T(n=n, alpha_grid=alpha_grid, lambda_grid=lambda_grid, beta_prime=1.0)
    S_R = build_S_R(delta_grid=delta_grid, beta=1.0)
    P_y = build_P_y(S_T, S_R, gamma)
    M = build_M_CTM(positions_per_cell=ppc, delta_grid=delta_grid)
    return M, P_y


# ---------------------------------------------------------------------------
# Hand calc (layer 1)
# ---------------------------------------------------------------------------


def test_compute_demand_single_cell_constant_inputs():
    """1 cell, constant VHT and P_y: E[j] = eta * VHT * P_y / n_i everywhere."""
    M = build_M_CTM(positions_per_cell=(4,), delta_grid=1)  # (1, 4), all 0.25
    P_y = np.full(4, 2.0)
    VHT = np.array([[10.0]])  # (N=1, T=1)
    E = compute_demand(VHT, M, P_y, eta_EV=0.5)
    expected = 0.5 * (10.0 / 4) * 2.0  # = 2.5 at every position
    np.testing.assert_allclose(E, np.full((1, 4), expected))


def test_compute_demand_shape_is_T_by_m():
    M, P_y = _profile((3, 6, 3), 4, 2, 4, gamma=1.0)
    VHT = np.ones((3, 5))  # N=3, T=5
    E = compute_demand(VHT, M, P_y, eta_EV=0.3)
    assert E.shape == (5, M.shape[1])


# ---------------------------------------------------------------------------
# Energy conservation (layer 2) — the plan's central identity
# ---------------------------------------------------------------------------


def test_compute_demand_conserves_energy():
    """Sum over position == eta * sum_i VHT[i,k] * mean(P_y over cell i)."""
    ppc = (3, 6, 3)
    M, P_y = _profile(ppc, alpha_grid=4, lambda_grid=2, delta_grid=4, gamma=150_000.0)
    VHT = np.array([[10.0, 5.0], [20.0, 0.0], [5.0, 8.0]])  # (N=3, T=2)
    eta = 0.3

    E = compute_demand(VHT, M, P_y, eta_EV=eta)

    # Independent expectation: per-cell mean power computed directly from P_y.
    cell_mean_power = np.array([P_y[M[i] > 0].mean() for i in range(len(ppc))])
    expected_row_energy = eta * VHT.T @ cell_mean_power  # (T,)
    np.testing.assert_allclose(E.sum(axis=1), expected_row_energy)


# ---------------------------------------------------------------------------
# Invariants (layer 2)
# ---------------------------------------------------------------------------


def test_compute_demand_linear_in_eta():
    M, P_y = _profile((3, 6, 3), 4, 2, 4, gamma=1.0)
    VHT = np.array([[10.0], [20.0], [5.0]])
    base = compute_demand(VHT, M, P_y, eta_EV=0.2)
    doubled = compute_demand(VHT, M, P_y, eta_EV=0.4)
    np.testing.assert_allclose(doubled, 2.0 * base)


def test_compute_demand_linear_in_vht():
    M, P_y = _profile((3, 6, 3), 4, 2, 4, gamma=1.0)
    VHT = np.array([[10.0], [20.0], [5.0]])
    base = compute_demand(VHT, M, P_y, eta_EV=0.3)
    doubled = compute_demand(2.0 * VHT, M, P_y, eta_EV=0.3)
    np.testing.assert_allclose(doubled, 2.0 * base)


def test_compute_demand_is_nonnegative():
    M, P_y = _profile((3, 6, 3), 4, 2, 4, gamma=150_000.0)
    VHT = np.array([[10.0, 5.0], [20.0, 0.0], [5.0, 8.0]])
    E = compute_demand(VHT, M, P_y, eta_EV=0.3)
    assert np.all(E >= 0.0)


def test_compute_demand_is_zero_where_vht_is_zero():
    """A timestep with no vehicle-hours anywhere draws no demand."""
    M, P_y = _profile((3, 6, 3), 4, 2, 4, gamma=150_000.0)
    VHT = np.array([[10.0, 0.0], [20.0, 0.0], [5.0, 0.0]])  # timestep 1 is empty
    E = compute_demand(VHT, M, P_y, eta_EV=0.3)
    np.testing.assert_allclose(E[1], 0.0)
