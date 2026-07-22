"""Tests for the modified-Gipps longitudinal update (T2, CP1).

Newbolt 2026 eq 1-5. Hand calcs use round numbers; properties check the
[v_min, v_max] envelope and the collision-free radicand guard.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from transportation_models.utils.microsim.gipps import (
    gap,
    next_velocity,
    safe_velocity,
)


# --- gap (eq 1) -----------------------------------------------------------


def test_gap_hand_calc():
    # leader back at 100, subject back at 50, subject length 4 ->
    # bumper-to-bumper gap = 100 - (50 + 4) = 46.
    assert gap(leader_back=100.0, subject_back=50.0, subject_length=4.0) == 46.0


def test_gap_vectorized():
    g = gap(
        leader_back=np.array([100.0, 30.0]),
        subject_back=np.array([50.0, 20.0]),
        subject_length=4.0,
    )
    np.testing.assert_allclose(g, [46.0, 6.0])


# --- safe_velocity (eq 5) -------------------------------------------------


def test_safe_velocity_hand_calc():
    # v=20, gap=10, s_o=25, b=2, dt=1:
    # v_SAFE = -2 + sqrt(4 + 400 + 2*2*(10-25)) = -2 + sqrt(344).
    expected = -2.0 + math.sqrt(4.0 + 400.0 + 4.0 * (10.0 - 25.0))
    got = safe_velocity(v=20.0, gap=10.0, b=2.0, dt=1.0, s_o=25.0)
    assert got == pytest.approx(expected)


def test_safe_velocity_clamps_negative_radicand():
    # Tiny gap and low speed drive the radicand negative; guard -> sqrt(0),
    # so v_SAFE = -b*dt (which the decel branch then floors at v_min).
    got = safe_velocity(v=0.0, gap=0.0, b=2.0, dt=1.0, s_o=25.0)
    assert got == pytest.approx(-2.0)


# --- next_velocity (eq 2-4) ----------------------------------------------


def test_next_velocity_accelerates_when_leader_far():
    # gap > s_o -> v_a = min(v_max, v + a*dt) = min(30, 20 + 1.5) = 21.5.
    v = next_velocity(
        v=20.0, gap=1000.0, a=1.5, b=2.0, dt=1.0, s_o=25.0, v_max=30.0, v_min=0.0
    )
    assert v == pytest.approx(21.5)


def test_next_velocity_clamps_acceleration_to_v_max():
    # v + a*dt = 21.5 but v_max = 21 -> clamped to 21.
    v = next_velocity(
        v=20.0, gap=1000.0, a=1.5, b=2.0, dt=1.0, s_o=25.0, v_max=21.0, v_min=0.0
    )
    assert v == pytest.approx(21.0)


def test_next_velocity_decelerates_when_leader_near():
    # gap <= s_o -> v_b = max(v_min, v_SAFE).
    expected = -2.0 + math.sqrt(4.0 + 400.0 + 4.0 * (10.0 - 25.0))
    v = next_velocity(
        v=20.0, gap=10.0, a=1.5, b=2.0, dt=1.0, s_o=25.0, v_max=30.0, v_min=0.0
    )
    assert v == pytest.approx(expected)


def test_next_velocity_floors_at_v_min():
    # Tiny gap, low speed -> v_SAFE < v_min -> floored at v_min.
    v = next_velocity(
        v=0.0, gap=0.0, a=1.5, b=2.0, dt=1.0, s_o=25.0, v_max=30.0, v_min=0.0
    )
    assert v == pytest.approx(0.0)


def test_next_velocity_stays_in_envelope_vectorized():
    rng = np.random.default_rng(0)
    v = rng.uniform(0, 30, size=200)
    gaps = rng.uniform(0, 200, size=200)
    out = next_velocity(
        v=v, gap=gaps, a=1.5, b=2.0, dt=1.0, s_o=25.0, v_max=30.0, v_min=0.0
    )
    assert np.all(out >= 0.0 - 1e-9)
    assert np.all(out <= 30.0 + 1e-9)
