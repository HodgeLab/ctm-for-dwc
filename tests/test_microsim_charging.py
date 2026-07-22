"""Tests for original mCONV charging over trajectories (T7).

Single-vehicle hand calc + EV-only and non-negativity properties. The
macro-vs-micro uniform-limit equivalence (CP5) lives in
``test_dwpt_validation.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from transportation_models.utils.microsim.charging import micro_demand
from transportation_models.utils.microsim.examples import uniform_density_scenario


def test_single_vehicle_deposits_interior_p_y_times_dt():
    sc = uniform_density_scenario(n_vehicles=1)
    corridor = sc.corridor
    E = micro_demand(sc.micro_result, corridor).E.sum(axis=0)

    P_y = corridor.build_P_y()
    off = (corridor.delta_grid - 1) // 2
    n = corridor.n_corridor
    dt_h = sc.micro_result.dt / 3600.0

    # Interior positions get exactly P_y * dt_h (one visit each); the
    # approach/exit boundary positions stay zero.
    expected = np.zeros_like(P_y)
    expected[off : off + n] = P_y[off : off + n] * dt_h
    np.testing.assert_allclose(E, expected)


def test_non_ev_vehicles_contribute_no_demand():
    sc = uniform_density_scenario(n_vehicles=4)
    # Flip all vehicles to non-EV.
    res = sc.micro_result
    res_non_ev = type(res)(
        position=res.position,
        velocity=res.velocity,
        lane=res.lane,
        is_ev=np.zeros(res.n_vehicles, dtype=bool),
        dt=res.dt,
        corridor_length_m=res.corridor_length_m,
        n_lanes=res.n_lanes,
    )
    E = micro_demand(res_non_ev, sc.corridor).E
    assert np.all(E == 0.0)


def test_demand_non_negative():
    sc = uniform_density_scenario(n_vehicles=5)
    E = micro_demand(sc.micro_result, sc.corridor).E
    assert np.all(E >= 0.0)
