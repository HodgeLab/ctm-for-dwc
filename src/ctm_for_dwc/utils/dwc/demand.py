"""Demand assembly for the DWC module: ``compute_demand``.

Combines the temporal (``VHT``), mapping (``M_CTM``), and spatial (``P_y``)
pieces into the spatiotemporal demand
``E = eta_EV * (VHT^T M_CTM) * P_y``. See ``docs/dwc_demand_module.md``
§"Putting It All Together".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .mapping import build_M_CTM
from .model import DemandResult

if TYPE_CHECKING:
    from .model import CorridorSpec


def compute_demand(
    VHT: np.ndarray, M_CTM: np.ndarray, P_y: np.ndarray, eta_EV: float
) -> np.ndarray:
    """Spatiotemporal DWC demand, shape ``(T, m)`` in Wh.

    Parameters
    ----------
    VHT : ndarray (N, T)
        Vehicle-hours per cell per timestep (mainline only; veh-hours).
    M_CTM : ndarray (N, m)
        Cell->position partition of unity.
    P_y : ndarray (m,)
        Power-vs-position profile, watts.
    eta_EV : float
        EV fraction in ``[0, 1]``.

    Notes
    -----
    Units: veh-hours * watts = Wh, so ``VHT`` must be in vehicle-hours for
    ``E`` to come out in watt-hours.
    """
    return eta_EV * (VHT.T @ M_CTM) * P_y


def compute(
    VHT: np.ndarray, corridor: CorridorSpec, eta_EV: float
) -> DemandResult:
    """Assemble a labeled ``DemandResult`` from a corridor and its VHT.

    Composes the spatial build (``corridor.build_P_y``), the cell->position
    map (``build_M_CTM``), and ``compute_demand``. Column positions use the
    center-reference convention, ``position_m[j] = (j - off) * dx_grid`` with
    ``off = (delta_grid - 1)//2``; rows are timestep indices.
    """
    if not 0.0 <= eta_EV <= 1.0:
        raise ValueError(f"eta_EV must be in [0, 1], got {eta_EV}")
    n_cells = len(corridor.positions_per_cell)
    if VHT.shape[0] != n_cells:
        raise ValueError(
            f"VHT has {VHT.shape[0]} cells but corridor has {n_cells}"
        )

    M_CTM = build_M_CTM(corridor.positions_per_cell, corridor.delta_grid)
    P_y = corridor.build_P_y()
    E = compute_demand(VHT, M_CTM, P_y, eta_EV)

    timesteps = np.arange(VHT.shape[1])
    return DemandResult(E=E, position_m=corridor.position_m, timesteps=timesteps)
