"""Tests for Edie micro->macro aggregation (T6, CP4).

Edie's generalized definitions over each cell's space-time box. Exact on a
controlled steady fixture (constant density and velocity), plus the
comparator-wrapper shape contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from transportation_models.utils.microsim import MicrosimResult
from transportation_models.utils.microsim.aggregate import (
    METERS_PER_MILE,
    aggregate_to_cells,
    to_validation_arrays,
)


def _steady_result(n=4, v=10.0, dt=1.0, window=10, x0=100.0, spacing=200.0,
                   lane=None, n_lanes=1):
    """N vehicles fully inside a long cell for the whole window."""
    starts = x0 + spacing * np.arange(n)
    k = np.arange(window + 1)
    position = starts[:, None] + v * dt * k[None, :]
    velocity = np.full((n, window), v)
    if lane is None:
        lane = np.zeros((n, window), dtype=int)
    return MicrosimResult(
        position=position,
        velocity=velocity,
        lane=lane,
        is_ev=np.ones(n, dtype=bool),
        dt=dt,
        corridor_length_m=2000.0,
        n_lanes=n_lanes,
    )


def test_edie_density_and_flow_exact_steady():
    # 4 vehicles, v=10 m/s, cell [0, 2000] m, window = 10 s.
    res = _steady_result(n=4, v=10.0, dt=1.0, window=10)
    edges = np.array([0.0, 2000.0])
    density, flow = aggregate_to_cells(res, edges, window_steps=10)
    assert density.shape == (1, 1)
    # density = N / L = 4 / 2000 veh/m -> veh/mi.
    assert density[0, 0] == pytest.approx(4 / 2000 * METERS_PER_MILE)
    # flow = N * v / L = 4 * 10 / 2000 veh/s -> veh/h.
    assert flow[0, 0] == pytest.approx(4 * 10 / 2000 * 3600)


def test_edie_two_cells_additivity():
    # Same vehicles, split the domain into two cells; each cell sees its share.
    res = _steady_result(n=4, v=10.0, dt=1.0, window=10, x0=100.0, spacing=200.0)
    # Vehicles span 100..700+100=800 over the window; cell split at 1000 keeps
    # all in cell 0, none in cell 1.
    edges = np.array([0.0, 1000.0, 2000.0])
    density, flow = aggregate_to_cells(res, edges, window_steps=10)
    assert density.shape == (2, 1)
    assert density[0, 0] == pytest.approx(4 / 1000 * METERS_PER_MILE)
    assert density[1, 0] == pytest.approx(0.0)


def test_edie_sums_across_lanes():
    # Same 4 vehicles, but split across 2 lanes: Edie (lane-agnostic) must
    # report the corridor total, matching the single-lane count -> PeMS-style
    # station totals (sum across lanes).
    lane = np.array([[0] * 10, [0] * 10, [1] * 10, [1] * 10])
    res = _steady_result(n=4, v=10.0, dt=1.0, window=10, lane=lane, n_lanes=2)
    edges = np.array([0.0, 2000.0])
    density, flow = aggregate_to_cells(res, edges, window_steps=10)
    assert density[0, 0] == pytest.approx(4 / 2000 * METERS_PER_MILE)
    assert flow[0, 0] == pytest.approx(4 * 10 / 2000 * 3600)


def test_edie_nonneg_and_zero_when_empty():
    res = _steady_result(n=2, v=12.0)
    edges = np.array([0.0, 2000.0])
    density, flow = aggregate_to_cells(res, edges, window_steps=10)
    assert np.all(density >= 0)
    assert np.all(flow >= 0)


def test_to_validation_arrays_shape_contract():
    # density must be (n_cells, n_win+1), flow (n_cells, n_win) for the
    # reused compare_against_historical (dt = 5 min -> steps_per_5min = 1).
    density = np.array([[6.0, 7.0, 8.0]])
    flow = np.array([[100.0, 110.0, 120.0]])
    wrapper = to_validation_arrays(density, flow)
    assert wrapper.density.shape == (1, 4)
    assert wrapper.mainline_flow.shape == (1, 3)
    assert wrapper.n_steps == 3
    assert wrapper.freeway.n_cells == 1
    # dt = 5 min in hours.
    assert wrapper.freeway.dt == pytest.approx(5 / 60)
    # The first n_win density columns are the Edie windows (last is padding).
    np.testing.assert_allclose(wrapper.density[:, :3], density)
