"""Kan 2021 ramp-flow bounds and the alpha-between-bounds map (Appendix A).

For a unit stretch with upstream/downstream mainline flows ``q_up``, ``q_down``
and a weaving-area capacity ``c_w``, the missing on-ramp flow ``r`` and off-ramp
flow ``s`` are bounded by flow conservation and capacity:

    r_min = max(q_down - q_up, 0)             (A.3-A.4)
    s_min = max(q_up - q_down, 0)             (A.6-A.7)
    R_max = min(c_w - q_up,   r_cap, r_demand)  (A.10)
    S_max = min(c_w - q_down, s_cap, s_qmax)    (A.11)
    r_max = min(q_down - q_up + S_max, R_max)   (A.14)
    s_max = min(q_up - q_down + R_max, S_max)   (A.15)

The four ramp-specific caps default to ``+inf``, so ``R_max``/``S_max`` default
to the weaving cap ``c_w - q`` -- a band that is computable for *target*
(unmeasured) ramps, since ``y_min`` comes from conservation and ``y_max`` from
``c_w`` and mainline flow alone. ``r_demand``/``s_qmax`` may be set from a ramp's
historical peak observed flow; ``r_cap``/``s_cap`` are left at ``inf`` pending an
HCM geometry-based capacity (follow-up). See docs/ramp_flow_estimation_module.md.

Note: A.11 as printed uses ``c_w - q_up`` for ``S_max``; per A.9 and conservation
that is an apparent typo, and we use ``c_w - q_down``.

Kan predicts the *determined-first* ramp -- the one with a positive conservation
floor -- via a coefficient ``alpha in [0, 1]`` blended over ``[y_min, y_max]``,
then recovers its pair by conservation (A.18-A.19):

    q_down >= q_up : r = alpha*r_max + (1-alpha)*r_min ; s = max(r + q_up - q_down, 0)
    q_up  > q_down : s = alpha*s_max + (1-alpha)*s_min ; r = max(s + q_down - q_up, 0)

All functions accept scalars or numpy arrays (one entry per 5-min step).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Bounds:
    """Per-step ramp-flow bounds (arrays broadcast to the inputs' shape)."""

    r_min: np.ndarray
    r_max: np.ndarray
    s_min: np.ndarray
    s_max: np.ndarray


def compute_bounds(
    q_up, q_down, c_w, *,
    r_cap=np.inf, r_demand=np.inf, s_cap=np.inf, s_qmax=np.inf,
) -> Bounds:
    """Kan A.3-A.15. Ramp-specific caps default to ``+inf`` (band = ``c_w - q``)."""
    q_up = np.asarray(q_up, dtype=float)
    q_down = np.asarray(q_down, dtype=float)
    c_w = np.asarray(c_w, dtype=float)

    r_min = np.maximum(q_down - q_up, 0.0)
    s_min = np.maximum(q_up - q_down, 0.0)
    cap_r = np.minimum(np.minimum(c_w - q_up, r_cap), r_demand)     # R_max (A.10)
    cap_s = np.minimum(np.minimum(c_w - q_down, s_cap), s_qmax)     # S_max (A.11, q_down)
    r_max = np.minimum(q_down - q_up + cap_s, cap_r)               # A.14
    s_max = np.minimum(q_up - q_down + cap_r, cap_s)               # A.15
    return Bounds(r_min, r_max, s_min, s_max)


def predicts_on_ramp(q_up, q_down) -> np.ndarray:
    """True where the on-ramp is the determined-first ramp (``q_down >= q_up``)."""
    return np.asarray(q_down, dtype=float) >= np.asarray(q_up, dtype=float)


def feasible_band(bounds: Bounds, q_up, q_down) -> np.ndarray:
    """True where the determined-first ramp has a non-degenerate band
    (``y_max > y_min``). In congestion ``c_w - q <= 0`` collapses it."""
    on = predicts_on_ramp(q_up, q_down)
    width = np.where(on, bounds.r_max - bounds.r_min, bounds.s_max - bounds.s_min)
    return width > 0.0


def alpha_target(bounds: Bounds, q_up, q_down, r_true, s_true):
    """Invert the blend to get the training target ``alpha`` for the
    determined-first ramp. Returns ``(alpha, feasible)``; ``alpha`` is clipped to
    ``[0, 1]`` and is meaningless where ``feasible`` is False (degenerate band)."""
    on = predicts_on_ramp(q_up, q_down)
    r_true = np.asarray(r_true, dtype=float)
    s_true = np.asarray(s_true, dtype=float)
    width_r = bounds.r_max - bounds.r_min
    width_s = bounds.s_max - bounds.s_min
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_r = (r_true - bounds.r_min) / width_r
        alpha_s = (s_true - bounds.s_min) / width_s
    alpha = np.where(on, alpha_r, alpha_s)
    feasible = np.where(on, width_r, width_s) > 0.0
    alpha = np.clip(np.where(feasible, alpha, 0.0), 0.0, 1.0)
    return alpha, feasible


def reconstruct_pair(
    alpha, q_up, q_down, c_w, *,
    r_cap=np.inf, r_demand=np.inf, s_cap=np.inf, s_qmax=np.inf,
):
    """Reconstruct ``(r_hat, s_hat)`` from the predicted ``alpha`` (Kan A.18-A.19):
    blend the determined-first ramp, then recover its pair by conservation
    (clipped at 0)."""
    q_up = np.asarray(q_up, dtype=float)
    q_down = np.asarray(q_down, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    b = compute_bounds(q_up, q_down, c_w, r_cap=r_cap, r_demand=r_demand,
                       s_cap=s_cap, s_qmax=s_qmax)
    on = predicts_on_ramp(q_up, q_down)
    r_blend = alpha * b.r_max + (1.0 - alpha) * b.r_min
    s_blend = alpha * b.s_max + (1.0 - alpha) * b.s_min
    r_hat = np.where(on, r_blend, np.maximum(s_blend + q_down - q_up, 0.0))
    s_hat = np.where(on, np.maximum(r_blend + q_up - q_down, 0.0), s_blend)
    return r_hat, s_hat
