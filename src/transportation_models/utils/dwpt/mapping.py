"""Cell->position mapping for the DWPT demand module: ``build_M_CTM``.

``M_CTM`` is the ``(N, m)`` row-partition-of-unity that relates CTM cell
index to Rx-pad traversal position and distributes each cell's vehicle-hours
across its positions. See ``docs/dwpt_demand_module.md`` for the definition.

Convention (center-reference): traversal position ``j`` carries the Rx pad
spanning corridor positions ``[j - delta_grid + 1, j]``, whose center sits at
corridor position ``j - off`` with ``off = (delta_grid - 1) // 2``. Position
``j`` is attributed to the cell containing that center; positions whose center
falls off-corridor (the approach/exit boundary) get zero columns. The mapping
is 1:1 over the ``n`` corridor positions, so ``n_i = positions_per_cell[i]``
exactly and every row sums to 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def build_M_CTM(
    positions_per_cell: Sequence[int], delta_grid: int
) -> np.ndarray:
    """Cell->traversal-position map, shape ``(N, n + delta_grid - 1)``.

    ``M[i, j] = 1 / positions_per_cell[i]`` when traversal position ``j``'s
    Rx-pad center lies in cell ``i``, else 0.
    """
    positions_per_cell = tuple(positions_per_cell)
    n = sum(positions_per_cell)
    m = n + delta_grid - 1
    off = (delta_grid - 1) // 2
    cuts = np.cumsum(positions_per_cell)

    M = np.zeros((len(positions_per_cell), m))
    for c in range(n):  # corridor position c <-> traversal position c + off
        i = int(np.searchsorted(cuts, c, side="right"))
        M[i, c + off] = 1.0 / positions_per_cell[i]
    return M
