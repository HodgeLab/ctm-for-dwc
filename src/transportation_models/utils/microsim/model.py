"""Data model for the microsim validation harness: ``MicrosimSpec``,
``Vehicles``, ``MicrosimResult``.

``MicrosimSpec`` carries the corridor-independent vehicle/driver model and
numerical parameters for the single-lane modified-Gipps microsimulation
(Newbolt 2026, eq 1-5). ``Vehicles`` is the structure-of-arrays produced by
seeding (one entry per simulated vehicle). ``MicrosimResult`` holds the
trajectory arrays the loop produces.

All quantities are **SI (meters, seconds, m/s)** inside the microsim; the
single mph->m/s conversion happens at the case-study seam. See
``docs/dwpt_validation_spec.md`` for the harness design and the velocity
sampling decision (``v_max ~ U(v_f - 15, v_f + 15) mph``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: 1 mph in m/s (exact: 1609.344 m / 3600 s).
MPH_TO_MS = 0.44704


@dataclass(frozen=True)
class GuardStats:
    """Activation statistics for the displacement-cap collision guard.

    The guard is a **deliberate divergence from Newbolt's published model**
    (see ``docs/dwpt_validation_spec.md`` "Deviations from the published
    model"): the paper's simplified Gipps safe-velocity (2026 eq 5) admits
    vehicle overlaps at practical ``dt`` under speed heterogeneity, so the
    microsim caps each vehicle's per-step displacement at the gap to its
    leader. These statistics quantify how often and how hard that guard binds,
    so its footprint is auditable rather than silent.

    Attributes
    ----------
    n_activations : int
        Number of (vehicle, step) updates where the cap bound (the Gipps
        preferred velocity exceeded the displacement cap).
    n_vehicle_steps : int
        Total active (vehicle, step) updates, for an activation rate.
    max_speed_reduction_ms : float
        Largest ``v_preferred - v_actual`` [m/s] imposed by the guard.
    max_accel_delta_ms2 : float
        Largest ``(v_preferred - v_actual)/dt`` [m/s^2] -- the gap between the
        Gipps-preferred and guard-actual acceleration.
    max_abs_accel_ms2 : float
        Largest absolute single-step acceleration ``|v_actual - v_prev|/dt``
        [m/s^2] across all updates (guarded or not).
    """

    n_activations: int
    n_vehicle_steps: int
    max_speed_reduction_ms: float
    max_accel_delta_ms2: float
    max_abs_accel_ms2: float

    @property
    def activation_rate(self) -> float:
        """Fraction of active vehicle-steps where the guard bound."""
        if self.n_vehicle_steps == 0:
            return 0.0
        return self.n_activations / self.n_vehicle_steps


@dataclass(frozen=True)
class MicrosimSpec:
    """Vehicle/driver model and numerical parameters (corridor-independent).

    Parameters
    ----------
    a : float
        Maximum acceleration, m/s^2 (Newbolt eq 3).
    b : float
        Maximum deceleration magnitude, m/s^2 (Newbolt eq 4-5).
    s_o : float
        Desired safe spacing between leader and subject, meters.
    vehicle_length : float
        Vehicle length ``l``, meters.
    dt : float
        Simulation time step, seconds.
    seed : int
        RNG seed for reproducible seeding/sampling.
    v_min : float, default 0.0
        Minimum velocity, m/s (floor of the Gipps deceleration branch).
    speed_halfwidth_ms : float, default 15 mph
        Half-width of the per-vehicle desired-speed band; ``v_max`` is drawn
        from ``U(v_f - halfwidth, v_f + halfwidth)`` (see spec).
    lane_change_prob : float, default 0.5
        Weighted probability (Newbolt ``w_o``) that an incentivized vehicle
        actually changes lanes on a given step.

    Notes
    -----
    Defaults are Newbolt's L1 (Tesla Model 3) parameters (Tables II/IV;
    ``a=4.79``, ``b=2.50``, ``s_o=25``, ``length=4.69``), CLI-overridable.
    """

    a: float = 4.79
    b: float = 2.50
    s_o: float = 25.0
    vehicle_length: float = 4.69
    dt: float = 1.0
    seed: int = 0
    v_min: float = 0.0
    speed_halfwidth_ms: float = 15 * MPH_TO_MS
    lane_change_prob: float = 0.5

    def __post_init__(self) -> None:
        for name in ("a", "b", "s_o", "vehicle_length", "dt"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(
                    f"MicrosimSpec.{name} must be positive, got {value}"
                )
        if self.v_min < 0:
            raise ValueError(
                f"MicrosimSpec.v_min must be non-negative, got {self.v_min}"
            )
        if self.speed_halfwidth_ms <= 0:
            raise ValueError(
                "MicrosimSpec.speed_halfwidth_ms must be positive, got "
                f"{self.speed_halfwidth_ms}"
            )
        if not 0.0 <= self.lane_change_prob <= 1.0:
            raise ValueError(
                "MicrosimSpec.lane_change_prob must be in [0, 1], got "
                f"{self.lane_change_prob}"
            )


@dataclass(frozen=True)
class Vehicles:
    """Seeded vehicles as a structure-of-arrays (one entry per vehicle).

    Each vehicle's desired/free-flow speed ``v_max`` doubles as its initial
    velocity (sampled speed = that driver's desired speed); position starts at
    the corridor entry (0 m) at ``entry_step``.

    Parameters
    ----------
    entry_step : ndarray (N,) int
        Simulation step at which each vehicle enters the corridor.
    v_max : ndarray (N,) float
        Per-vehicle desired/max velocity, m/s (Gipps eq 3 cap).
    is_ev : ndarray (N,) bool
        Whether each vehicle is an EV (only EVs draw DWPT charging load).
    entry_lane : ndarray (N,) int
        Lane each vehicle enters on (0-based; randomized at seeding).
    """

    entry_step: np.ndarray
    v_max: np.ndarray
    is_ev: np.ndarray
    entry_lane: np.ndarray

    def __post_init__(self) -> None:
        n = self.entry_step.shape[0]
        if (
            self.v_max.shape[0] != n
            or self.is_ev.shape[0] != n
            or self.entry_lane.shape[0] != n
        ):
            raise ValueError(
                "Vehicles.entry_step, v_max, is_ev, entry_lane must have the "
                f"same length; got {self.entry_step.shape[0]}, "
                f"{self.v_max.shape[0]}, {self.is_ev.shape[0]}, "
                f"{self.entry_lane.shape[0]}"
            )
        if np.any(self.v_max <= 0):
            raise ValueError("Vehicles.v_max entries must be positive")

    @property
    def n_vehicles(self) -> int:
        return int(self.entry_step.shape[0])


@dataclass(frozen=True)
class MicrosimResult:
    """Microsim trajectories: per-vehicle position and velocity over time.

    Inactive entries (before a vehicle enters or after it exits) are NaN, so
    the array width is the full horizon for every vehicle.

    Parameters
    ----------
    position : ndarray (N, T+1) float
        Per-vehicle longitudinal position, meters, at step boundaries.
    velocity : ndarray (N, T) float
        Per-vehicle velocity, m/s, over each step interval.
    lane : ndarray (N, T) int
        Per-vehicle lane (0-based) over each step interval; ``-1`` when the
        vehicle is inactive (not yet entered or already exited).
    is_ev : ndarray (N,) bool
        EV flag per vehicle.
    dt : float
        Time step, seconds.
    corridor_length_m : float
        Corridor length, meters.
    n_lanes : int
        Number of lanes modeled.
    """

    position: np.ndarray
    velocity: np.ndarray
    lane: np.ndarray
    is_ev: np.ndarray
    dt: float
    corridor_length_m: float
    n_lanes: int
    guard_stats: Optional[GuardStats] = None

    def __post_init__(self) -> None:
        n, t_plus_1 = self.position.shape
        if self.velocity.shape[0] != n:
            raise ValueError(
                f"velocity has {self.velocity.shape[0]} vehicles but position "
                f"has {n}; they must match"
            )
        if self.velocity.shape[1] != t_plus_1 - 1:
            raise ValueError(
                f"velocity must have T columns where position has T + 1; got "
                f"velocity T={self.velocity.shape[1]}, position T+1={t_plus_1}"
            )
        if self.lane.shape != self.velocity.shape:
            raise ValueError(
                f"lane shape {self.lane.shape} must match velocity shape "
                f"{self.velocity.shape}"
            )
        if self.is_ev.shape[0] != n:
            raise ValueError(
                f"is_ev has {self.is_ev.shape[0]} entries but position has {n} "
                "vehicles"
            )

    @property
    def n_vehicles(self) -> int:
        return int(self.position.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.velocity.shape[1])
