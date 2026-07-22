"""Synthetic corridors and the uniform-density fixture (T4).

The uniform-density scenario underpins the CP5 macro-vs-micro equivalence
test (``docs/dwpt_validation_spec.md``, Risk #2): all-EV vehicles traverse a
single cell at constant velocity, *grid-aligned* (``v_ms * dt == dx_grid``) so
each vehicle advances exactly one grid cell per step and the charging integral
is exact. The analytic VHT is constructed so a CTM-equivalent
``dwpt.compute`` and the original mCONV over these trajectories agree to
float round-off.

This is a *kinematic* fixture (positions placed analytically, no Gipps
dynamics): its purpose is the mCONV equivalence, not traffic-flow realism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dwpt.model import CorridorSpec, PadSpec
from .model import MPH_TO_MS, MicrosimResult, MicrosimSpec


def default_pad() -> PadSpec:
    """A representative pad that resolves cleanly on a 0.5 m grid (350 kW)."""
    return PadSpec(
        alpha=3.0, delta=1.5, lambda_gap=0.5, beta=350e3, beta_prime=350e3
    )


def demo_spec(dt: float = 1.0, seed: int = 0) -> MicrosimSpec:
    """A representative single-class Gipps spec (Tesla Model 3 length)."""
    return MicrosimSpec(
        a=1.5, b=2.0, s_o=25.0, vehicle_length=4.69, dt=dt, seed=seed
    )


@dataclass(frozen=True)
class UniformScenario:
    """Controlled uniform-density fixture for the CP5 equivalence test.

    Attributes
    ----------
    micro_result : MicrosimResult
        All-EV vehicles each fully traversing the single cell at constant v.
    corridor : CorridorSpec
        Single-cell corridor (meters) with ``default_pad``.
    vht : ndarray (1, T)
        Analytic vehicle-hours matching the micro traversals exactly.
    v_ms : float
        Constant vehicle velocity, m/s.
    dt : float
        Time step, s (grid-aligned: ``v_ms * dt == dx_grid``).
    eta_ev : float
        EV fraction (1.0 for this fixture).
    """

    micro_result: MicrosimResult
    corridor: CorridorSpec
    vht: np.ndarray
    v_ms: float
    dt: float
    eta_ev: float


def uniform_density_scenario(
    *,
    n_vehicles: int = 10,
    length_m: float = 100.0,
    dx_grid: float = 0.5,
    v_mph: float = 60.0,
    gap_steps: int = 20,
) -> UniformScenario:
    """Build the all-EV, constant-velocity, grid-aligned single-cell fixture.

    Parameters
    ----------
    n_vehicles : int
        Number of (all-EV) vehicles, each completing a full traversal.
    length_m : float
        Single-cell corridor length, meters (multiple of ``dx_grid``).
    dx_grid : float
        Position-grid spacing, meters.
    v_mph : float
        Constant velocity, mph.
    gap_steps : int
        Entry spacing between consecutive vehicles, in steps.
    """
    pad = default_pad()
    corridor = CorridorSpec(
        cell_lengths_m=(length_m,), pad=pad, dx_grid=dx_grid
    )
    n = corridor.positions_per_cell[0]  # interior grid positions = steps/traversal

    v_ms = v_mph * MPH_TO_MS
    dt = dx_grid / v_ms  # grid-aligned: one cell per step

    # Horizon covers the last vehicle's full traversal.
    last_entry = (n_vehicles - 1) * gap_steps
    n_steps = last_entry + n + 1
    position = np.full((n_vehicles, n_steps + 1), np.nan)
    velocity = np.full((n_vehicles, n_steps), np.nan)

    for idx in range(n_vehicles):
        entry = idx * gap_steps
        # Boundaries entry..entry+n hold positions 0, dx, ..., n*dx = L.
        steps = np.arange(n + 1)
        position[idx, entry : entry + n + 1] = steps * dx_grid
        # Intervals entry..entry+n-1 move at v_ms.
        velocity[idx, entry : entry + n] = v_ms

    micro_result = MicrosimResult(
        position=position,
        velocity=velocity,
        lane=np.zeros((n_vehicles, n_steps), dtype=int),
        is_ev=np.ones(n_vehicles, dtype=bool),
        dt=dt,
        corridor_length_m=length_m,
        n_lanes=1,
    )

    # Analytic VHT: total vehicle-hours = N * n * dt_h (= N * L_mi / v_mph),
    # spread uniformly across the horizon. Only the sum matters for the
    # time-invariant P_y, so a flat profile is exact.
    dt_h = dt / 3600.0
    vht_total = n_vehicles * n * dt_h
    vht = np.full((1, n_steps), vht_total / n_steps)

    return UniformScenario(
        micro_result=micro_result,
        corridor=corridor,
        vht=vht,
        v_ms=v_ms,
        dt=dt,
        eta_ev=1.0,
    )
