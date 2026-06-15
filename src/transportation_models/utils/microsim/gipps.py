"""Modified-Gipps longitudinal car-following update (Newbolt 2026, eq 1-5).

Pure, NumPy-vectorizable functions: a vehicle accelerates toward its
``v_max`` when the gap to the leader exceeds the desired safe spacing, and
otherwise decelerates to the collision-free safe velocity (floored at
``v_min``). All quantities are SI (m, s, m/s).
"""

from __future__ import annotations

import numpy as np


def gap(leader_back, subject_back, subject_length):
    """Bumper-to-bumper gap to the leader (Newbolt eq 1).

    ``s_i = x_{i+1} - (x_i + l_i)``: the leader's back position minus the
    subject's front (back + length).

    Parameters
    ----------
    leader_back : float or ndarray
        Leading vehicle's back position, meters.
    subject_back : float or ndarray
        Subject vehicle's back position, meters.
    subject_length : float or ndarray
        Subject vehicle length ``l_i``, meters.

    Returns
    -------
    float or ndarray
        Gap distance, meters.
    """
    return leader_back - (subject_back + subject_length)


def safe_velocity(v, gap, *, b, dt, s_o):
    """Gipps collision-free safe velocity (Newbolt eq 5).

    ``v_SAFE = -b*dt + sqrt(b^2 dt^2 + v^2 + 2 b (gap - s_o))``. The radicand
    is clamped at zero (a vehicle with too little space relative to ``s_o``
    yields ``v_SAFE = -b*dt``, which the deceleration branch floors at
    ``v_min``).

    Parameters
    ----------
    v : float or ndarray
        Subject velocity this step, m/s.
    gap : float or ndarray
        Bumper-to-bumper gap to the leader, meters (Newbolt eq 1).
    b, dt, s_o : float
        Deceleration magnitude [m/s^2], step [s], desired safe spacing [m].

    Returns
    -------
    float or ndarray
        Collision-free velocity, m/s.
    """
    radicand = b**2 * dt**2 + v**2 + 2.0 * b * (gap - s_o)
    return -b * dt + np.sqrt(np.maximum(radicand, 0.0))


def next_velocity(v, gap, *, a, b, dt, s_o, v_max, v_min):
    """Next-step velocity (Newbolt eq 2-4).

    If ``gap > s_o``: accelerate, ``v_a = min(v_max, v + a*dt)`` (eq 3).
    Otherwise: decelerate, ``v_b = max(v_min, v_SAFE)`` (eq 4-5).

    Parameters
    ----------
    v : float or ndarray
        Subject velocity this step, m/s.
    gap : float or ndarray
        Gap to the leader, meters.
    a, b, dt, s_o : float
        Acceleration [m/s^2], deceleration [m/s^2], step [s], safe spacing [m].
    v_max : float or ndarray
        Per-vehicle desired/max velocity, m/s.
    v_min : float
        Minimum velocity, m/s.

    Returns
    -------
    float or ndarray
        Updated velocity, m/s.
    """
    v_a = np.minimum(v_max, v + a * dt)
    v_b = np.maximum(v_min, safe_velocity(v, gap, b=b, dt=dt, s_o=s_o))
    return np.where(gap > s_o, v_a, v_b)
