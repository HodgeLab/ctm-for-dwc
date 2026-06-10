"""Demand assembly for the DWPT module: ``compute_demand``.

Combines the temporal (``VHT``), mapping (``M_CTM``), and spatial (``P_y``)
pieces into the spatiotemporal demand
``E = eta_EV * (VHT^T M_CTM) * P_y``. See ``docs/dwpt_demand_module.md``
§"Putting It All Together".
"""

from __future__ import annotations

import numpy as np


def compute_demand(
    VHT: np.ndarray, M_CTM: np.ndarray, P_y: np.ndarray, eta_EV: float
) -> np.ndarray:
    """Spatiotemporal DWPT demand, shape ``(T, m)`` in Wh.

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
