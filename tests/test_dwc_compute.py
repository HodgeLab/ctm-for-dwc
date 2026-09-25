"""Tests for the P5 orchestration: CorridorSpec.build_P_y() and compute().

``compute(VHT, corridor, eta_EV)`` composes the spatial build, the M_CTM
map, and compute_demand into a labeled ``DemandResult``. Verified end-to-end
by the energy-conservation identity plus shape/label/validation checks.
"""

import numpy as np
import pytest

from ctm_for_dwc.dwc.demand import compute
from ctm_for_dwc.dwc.mapping import build_M_CTM
from ctm_for_dwc.dwc.model import CorridorSpec, PadSpec


def _pad(delta: float = 3.0) -> PadSpec:
    return PadSpec(
        alpha=2.0, delta=delta, lambda_gap=1.0,
        beta=150_000.0, beta_prime=150_000.0,
    )


# ---------------------------------------------------------------------------
# CorridorSpec.build_P_y
# ---------------------------------------------------------------------------


def test_corridor_spec_build_P_y_shape_and_peak():
    corridor = CorridorSpec(cell_lengths_m=(9.0,), pad=_pad(delta=2.0), dx_grid=1.0)
    P_y = corridor.build_P_y()
    assert P_y.shape == (corridor.m_traversal,)
    assert P_y.max() == pytest.approx(corridor.pad.gamma)


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------


def test_compute_returns_demand_result_with_correct_shapes():
    corridor = CorridorSpec(cell_lengths_m=(4.0, 6.0), pad=_pad(), dx_grid=1.0)
    VHT = np.ones((2, 5))  # N=2 cells, T=5 steps
    result = compute(VHT, corridor, eta_EV=0.3)
    assert result.E.shape == (5, corridor.m_traversal)
    assert result.position_m.shape == (corridor.m_traversal,)
    assert result.timesteps.shape == (5,)


def test_compute_position_m_is_center_reference():
    """position_m[j] = (j - off) * dx_grid with off = (delta_grid-1)//2."""
    corridor = CorridorSpec(cell_lengths_m=(10.0,), pad=_pad(delta=3.0), dx_grid=1.0)
    result = compute(np.ones((1, 1)), corridor, eta_EV=0.5)
    off = (corridor.delta_grid - 1) // 2
    expected = (np.arange(corridor.m_traversal) - off) * corridor.dx_grid
    np.testing.assert_allclose(result.position_m, expected)


def test_compute_conserves_energy_end_to_end():
    corridor = CorridorSpec(cell_lengths_m=(4.0, 6.0), pad=_pad(), dx_grid=1.0)
    VHT = np.array([[10.0, 5.0], [20.0, 8.0]])  # (N=2, T=2)
    result = compute(VHT, corridor, eta_EV=0.3)

    M = build_M_CTM(corridor.positions_per_cell, corridor.delta_grid)
    P_y = corridor.build_P_y()
    cell_mean_power = np.array([P_y[M[i] > 0].mean() for i in range(2)])
    expected = 0.3 * VHT.T @ cell_mean_power
    np.testing.assert_allclose(result.E.sum(axis=1), expected)


def test_compute_rejects_eta_out_of_range():
    corridor = CorridorSpec(cell_lengths_m=(4.0,), pad=_pad(), dx_grid=1.0)
    with pytest.raises(ValueError, match="eta_EV"):
        compute(np.ones((1, 1)), corridor, eta_EV=1.5)


def test_compute_rejects_vht_cell_count_mismatch():
    corridor = CorridorSpec(cell_lengths_m=(4.0, 6.0), pad=_pad(), dx_grid=1.0)
    with pytest.raises(ValueError, match="cells"):
        compute(np.ones((3, 1)), corridor, eta_EV=0.3)
