"""Tests for microsim examples / fixtures (T4).

The uniform-density scenario is the controlled fixture behind the CP5
equivalence test: all-EV vehicles traversing a single cell at constant
velocity, grid-aligned (one grid cell per step) so the charging integral is
exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from transportation_models.utils.microsim.examples import (
    default_pad,
    uniform_density_scenario,
)


def test_default_pad_resolves_on_grid():
    pad = default_pad()
    dx = 0.5
    for v in (pad.alpha, pad.delta, pad.lambda_gap):
        assert (v / dx) == pytest.approx(round(v / dx))


def test_uniform_scenario_constant_velocity():
    sc = uniform_density_scenario(n_vehicles=8)
    vel = sc.micro_result.velocity
    active = ~np.isnan(vel)
    # Every active interval moves at exactly v_ms.
    np.testing.assert_allclose(vel[active], sc.v_ms)


def test_uniform_scenario_full_traversals():
    sc = uniform_density_scenario(n_vehicles=8)
    pos = sc.micro_result.position
    L = sc.corridor.corridor_length_m
    # Each vehicle starts at 0 and reaches the corridor end exactly.
    for n in range(sc.micro_result.n_vehicles):
        track = pos[n][~np.isnan(pos[n])]
        assert track[0] == pytest.approx(0.0)
        assert track[-1] == pytest.approx(L)


def test_uniform_scenario_grid_aligned_steps():
    sc = uniform_density_scenario(n_vehicles=4)
    # One grid cell per step: v_ms * dt == dx_grid.
    assert sc.v_ms * sc.dt == pytest.approx(sc.corridor.dx_grid)


def test_uniform_scenario_all_ev():
    sc = uniform_density_scenario(n_vehicles=6)
    assert np.all(sc.micro_result.is_ev)


def test_uniform_scenario_vht_total_matches_traversals():
    sc = uniform_density_scenario(n_vehicles=10)
    n = sc.corridor.positions_per_cell[0]
    expected_total = sc.micro_result.n_vehicles * n * (sc.dt / 3600.0)
    assert sc.vht.sum() == pytest.approx(expected_total)
