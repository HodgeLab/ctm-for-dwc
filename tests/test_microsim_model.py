"""Tests for the microsim data model (T1): MicrosimSpec, Vehicles, MicrosimResult."""

from __future__ import annotations

import numpy as np
import pytest

from transportation_models.utils.microsim import (
    MPH_TO_MS,
    MicrosimResult,
    MicrosimSpec,
    Vehicles,
)


def _spec(**overrides):
    kwargs = dict(a=4.79, b=2.50, s_o=25.0, vehicle_length=4.69, dt=1.0, seed=0)
    kwargs.update(overrides)
    return MicrosimSpec(**kwargs)


# --- MicrosimSpec ---------------------------------------------------------


def test_microsim_spec_paper_defaults():
    # Defaults are Newbolt L1 (Tesla Model 3) values, CLI-overridable.
    spec = MicrosimSpec()
    assert spec.a == pytest.approx(4.79)
    assert spec.b == pytest.approx(2.50)
    assert spec.s_o == pytest.approx(25.0)
    assert spec.vehicle_length == pytest.approx(4.69)
    assert spec.dt == 1.0
    assert spec.seed == 0
    assert spec.v_min == 0.0
    assert spec.speed_halfwidth_ms == pytest.approx(15 * MPH_TO_MS)
    assert 0.0 <= spec.lane_change_prob <= 1.0


@pytest.mark.parametrize("field", ["a", "b", "s_o", "vehicle_length", "dt"])
def test_microsim_spec_rejects_non_positive(field):
    with pytest.raises(ValueError, match=field):
        _spec(**{field: 0.0})


def test_microsim_spec_rejects_negative_v_min():
    with pytest.raises(ValueError, match="v_min"):
        _spec(v_min=-1.0)


def test_microsim_spec_rejects_non_positive_halfwidth():
    with pytest.raises(ValueError, match="speed_halfwidth_ms"):
        _spec(speed_halfwidth_ms=0.0)


def test_mph_to_ms_constant():
    # 60 mph == 26.8224 m/s exactly (1 mph = 0.44704 m/s).
    assert 60 * MPH_TO_MS == pytest.approx(26.8224)


# --- Vehicles -------------------------------------------------------------


def _vehicles(n=3):
    return Vehicles(
        entry_step=np.array([0, 2, 5]),
        v_max=np.array([26.0, 27.0, 28.0]),
        is_ev=np.array([True, False, True]),
        entry_lane=np.array([0, 1, 3]),
    )


def test_vehicles_fields_and_count():
    veh = _vehicles()
    assert veh.n_vehicles == 3
    assert veh.is_ev.sum() == 2
    np.testing.assert_array_equal(veh.entry_lane, [0, 1, 3])


def test_vehicles_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        Vehicles(
            entry_step=np.array([0, 1]),
            v_max=np.array([26.0]),
            is_ev=np.array([True, False]),
            entry_lane=np.array([0, 1]),
        )


def test_vehicles_rejects_non_positive_v_max():
    with pytest.raises(ValueError, match="v_max"):
        Vehicles(
            entry_step=np.array([0]),
            v_max=np.array([0.0]),
            is_ev=np.array([True]),
            entry_lane=np.array([0]),
        )


# --- MicrosimResult -------------------------------------------------------


def _result(n=2, t=4):
    return MicrosimResult(
        position=np.zeros((n, t + 1)),
        velocity=np.full((n, t), 26.0),
        lane=np.zeros((n, t), dtype=int),
        is_ev=np.array([True, False]),
        dt=1.0,
        corridor_length_m=1609.34,
        n_lanes=2,
    )


def test_microsim_result_shapes_and_props():
    res = _result(n=2, t=4)
    assert res.n_vehicles == 2
    assert res.n_steps == 4
    assert res.n_lanes == 2
    assert res.lane.shape == (2, 4)


def test_microsim_result_rejects_inconsistent_shapes():
    with pytest.raises(ValueError, match="velocity"):
        MicrosimResult(
            position=np.zeros((2, 5)),
            velocity=np.zeros((3, 4)),  # wrong vehicle count
            lane=np.zeros((2, 4), dtype=int),
            is_ev=np.array([True, False]),
            dt=1.0,
            corridor_length_m=1000.0,
            n_lanes=2,
        )


def test_microsim_result_rejects_state_flow_length_mismatch():
    with pytest.raises(ValueError, match="T \\+ 1"):
        MicrosimResult(
            position=np.zeros((2, 4)),  # should be T+1 = 5
            velocity=np.zeros((2, 4)),
            lane=np.zeros((2, 4), dtype=int),
            is_ev=np.array([True, False]),
            dt=1.0,
            corridor_length_m=1000.0,
            n_lanes=2,
        )


def test_microsim_result_rejects_lane_shape_mismatch():
    with pytest.raises(ValueError, match="lane"):
        MicrosimResult(
            position=np.zeros((2, 5)),
            velocity=np.zeros((2, 4)),
            lane=np.zeros((2, 3), dtype=int),  # should match velocity (2,4)
            is_ev=np.array([True, False]),
            dt=1.0,
            corridor_length_m=1000.0,
            n_lanes=2,
        )
