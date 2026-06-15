"""Seed vehicles from a boundary-inflow series (T3).

Converts an inflow rate series (veh/h, one value per microsim step) into
discrete vehicle entry steps via a per-step Poisson draw (the Monte-Carlo
hourly->fine-grid conversion in the spirit of Newbolt ref [31]), and draws
each vehicle's desired/max velocity from ``U(v_f - halfwidth, v_f +
halfwidth)`` (see ``docs/dwpt_validation_spec.md``).
"""

from __future__ import annotations

import numpy as np

from .model import MicrosimSpec, Vehicles

_SECONDS_PER_HOUR = 3600.0


def seed_vehicles(
    inflow_rate_per_step: np.ndarray,
    spec: MicrosimSpec,
    *,
    v_f_ms: float,
    eta_ev: float,
    rng: np.random.Generator | None = None,
) -> Vehicles:
    """Seed vehicles from a per-step inflow rate series.

    Parameters
    ----------
    inflow_rate_per_step : ndarray (n_steps,)
        Upstream mainline inflow rate, veh/h, at each microsim step.
    spec : MicrosimSpec
        Provides ``dt`` (s), ``seed``, and ``speed_halfwidth_ms``.
    v_f_ms : float
        Corridor free-flow speed, m/s; the desired-speed band center.
    eta_ev : float
        EV penetration fraction in [0, 1]; each vehicle is independently an
        EV with this probability.
    rng : numpy.random.Generator, optional
        Override RNG; defaults to ``np.random.default_rng(spec.seed)``.

    Returns
    -------
    Vehicles
        Entry steps (sorted), per-vehicle ``v_max`` [m/s], and EV flags.
    """
    if not 0.0 <= eta_ev <= 1.0:
        raise ValueError(f"eta_ev must be in [0, 1], got {eta_ev}")
    if rng is None:
        rng = np.random.default_rng(spec.seed)

    rate = np.asarray(inflow_rate_per_step, dtype=float)
    # Expected arrivals per step = rate [veh/h] * dt [s] / 3600 [s/h].
    lam = rate * spec.dt / _SECONDS_PER_HOUR
    counts = rng.poisson(lam)
    entry_step = np.repeat(np.arange(rate.size), counts)

    n = entry_step.size
    hw = spec.speed_halfwidth_ms
    v_max = rng.uniform(v_f_ms - hw, v_f_ms + hw, size=n)
    is_ev = rng.random(n) < eta_ev

    return Vehicles(entry_step=entry_step, v_max=v_max, is_ev=is_ev)
