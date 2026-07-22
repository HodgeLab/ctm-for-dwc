"""CTM -> DWPT adapter: build a ``DemandResult`` from a CTM run.

This is the module's single coupling to the CTM engine. It reads the
post-step density and cell geometry off a ``SimulationResult``, forms the
mainline (queue-free) VHT, snaps cell lengths to the position grid, and
hands off to ``compute``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .demand import compute
from .model import CorridorSpec, DemandResult, PadSpec

if TYPE_CHECKING:
    from ..ctm.results import SimulationResult

_METERS_PER_MILE = 1609.344


def mainline_vht(result: SimulationResult) -> np.ndarray:
    """Queue-free mainline VHT per cell, shape ``(n_cells, T)`` in veh*h.

    The CTM's VHT (metrics eq. 4.10) is ``(rho * L + queue) * dt``; DWPT demand
    is generated only on the mainline, so we drop the on-ramp ``queue`` term
    and use ``rho * L * dt``.
    """
    rho = result.density[:, 1:]  # (n_cells, T), post-step density
    L = np.array([c.length for c in result.freeway.cells])  # miles
    return rho * L[:, None] * result.freeway.dt


def corridor_from_ctm(
    result: SimulationResult, pad_spec: PadSpec, dx_grid: float
) -> CorridorSpec:
    """Build a ``CorridorSpec`` from the CTM freeway geometry.

    Cell lengths (miles -> meters) are snapped to the nearest ``dx_grid``
    multiple so the corridor satisfies the spatial submodule's integer-grid
    requirement (error <= ``dx_grid``/2 per cell).
    """
    L_m = np.array([c.length for c in result.freeway.cells]) * _METERS_PER_MILE
    cell_lengths_m = tuple((np.round(L_m / dx_grid) * dx_grid).tolist())
    return CorridorSpec(
        cell_lengths_m=cell_lengths_m, pad=pad_spec, dx_grid=dx_grid
    )


def from_ctm(
    result: SimulationResult,
    pad_spec: PadSpec,
    dx_grid: float,
    eta_EV: float,
) -> DemandResult:
    """Assemble DWPT demand from a CTM ``SimulationResult``.

    Mainline (queue-free) VHT drives the demand over the snapped corridor.
    """
    corridor = corridor_from_ctm(result, pad_spec, dx_grid)
    return compute(mainline_vht(result), corridor, eta_EV)
