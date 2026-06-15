"""Single-lane microscopic traffic simulation for DWPT-demand validation.

A from-scratch implementation of the Newbolt 2026 modified-Gipps microsim
(single-lane equivalent), used to validate this repo's macroscopic CTM +
adapted-mCONV DWPT-demand pipeline. See ``docs/dwpt_validation_spec.md``.

Public API:

* :class:`MicrosimSpec`, :class:`Vehicles`, :class:`MicrosimResult`,
  :class:`GuardStats` -- the data model.
* :func:`seed_vehicles` -- inflow series -> seeded vehicles.
* :func:`simulate` -- single-lane Gipps loop -> trajectories.
* :func:`aggregate_to_cells` -- trajectories -> per-cell density/flow (Edie).
* :func:`micro_demand` -- original mCONV charging over trajectories.
"""

from __future__ import annotations

from .aggregate import aggregate_to_cells
from .charging import micro_demand
from .model import (
    MPH_TO_MS,
    GuardStats,
    MicrosimResult,
    MicrosimSpec,
    Vehicles,
)
from .seeding import seed_vehicles
from .simulate import simulate

__all__ = [
    "MPH_TO_MS",
    "GuardStats",
    "MicrosimResult",
    "MicrosimSpec",
    "Vehicles",
    "aggregate_to_cells",
    "micro_demand",
    "seed_vehicles",
    "simulate",
]
