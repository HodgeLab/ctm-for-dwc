"""Micro->macro aggregation via Edie's generalized definitions (T6).

Turns per-vehicle trajectories into per-cell density (veh/mi) and flow (veh/h)
on a fixed time-window grid, so the microsim can be scored against PeMS with
the *same* ``compare_against_historical`` path the CTM uses. Edie's definitions
over a space-time box A = (cell x window):

    density = (total time vehicles spend in A) / |A|
    flow    = (total distance vehicles travel in A) / |A|

with ``|A| = L_cell * window`` (see ``docs/dwpt_validation_spec.md``: per-cell
Edie box chosen over point-detector emulation to match the CTM's per-cell
quantity and the ``validation.csv`` schema).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

#: Meters per mile (exact), matching ``MPH_TO_MS`` so units cancel cleanly.
METERS_PER_MILE = 1609.344

_FIVE_MIN_H = 5.0 / 60.0
_EPS = 1e-9


def aggregate_to_cells(
    result,
    cell_edges_m: np.ndarray,
    *,
    window_steps: int,
):
    """Per-cell density [veh/mi] and flow [veh/h] via Edie's box.

    Parameters
    ----------
    result : MicrosimResult
        Trajectories (``position`` (N, T+1), ``velocity`` (N, T), ``dt``).
    cell_edges_m : ndarray (n_cells+1,)
        Cumulative cell boundaries along the corridor, meters (ascending).
    window_steps : int
        Microsim steps per aggregation window (e.g. ``round(300 / dt)`` for
        5-minute windows).

    Returns
    -------
    density : ndarray (n_cells, n_windows)
        Edie density, veh/mi.
    flow : ndarray (n_cells, n_windows)
        Edie flow, veh/h.
    """
    pos = result.position
    vel = result.velocity
    dt = result.dt
    a = pos[:, :-1]
    b = pos[:, 1:]
    valid = ~np.isnan(vel)

    n_steps = vel.shape[1]
    n_windows = n_steps // window_steps
    if n_windows < 1:
        raise ValueError(
            f"window_steps={window_steps} exceeds n_steps={n_steps}"
        )
    keep = n_windows * window_steps

    edges = np.asarray(cell_edges_m, dtype=float)
    n_cells = edges.size - 1
    window_s = window_steps * dt

    density = np.empty((n_cells, n_windows))
    flow = np.empty((n_cells, n_windows))

    for ci in range(n_cells):
        c0, c1 = edges[ci], edges[ci + 1]
        # Distance each vehicle travels inside the cell during each interval.
        seg = np.clip(np.minimum(b, c1) - np.maximum(a, c0), 0.0, None)
        seg = np.where(valid, seg, 0.0)
        # Time in cell: dist / v while moving; for (rare) v~0, dt if inside.
        with np.errstate(invalid="ignore", divide="ignore"):
            moving = valid & (vel > _EPS)
            time_in = np.where(moving, seg / np.where(moving, vel, 1.0), 0.0)
            stationary = valid & (vel <= _EPS) & (a >= c0) & (a < c1)
            time_in = np.where(stationary, dt, time_in)

        dist_w = (
            seg[:, :keep].reshape(-1, n_windows, window_steps).sum(axis=(0, 2))
        )
        time_w = (
            time_in[:, :keep]
            .reshape(-1, n_windows, window_steps)
            .sum(axis=(0, 2))
        )
        area = (c1 - c0) * window_s
        density[ci] = time_w / area * METERS_PER_MILE  # veh/m -> veh/mi
        flow[ci] = dist_w / area * 3600.0  # veh/s -> veh/h

    return density, flow


def to_validation_arrays(
    density: np.ndarray, flow: np.ndarray
) -> SimpleNamespace:
    """Wrap Edie aggregates for ``ctm.compare_against_historical``.

    The comparator reads ``result.freeway.n_cells``, ``result.freeway.dt``,
    ``result.density`` (n_cells, T+1), ``result.mainline_flow`` (n_cells, T),
    and ``result.n_steps``. We emit one window per 5-minute interval
    (``dt = 5 min`` so ``steps_per_5min = 1``); the density array gets a
    duplicated trailing column to honor the (T+1) state-frame convention (the
    comparator drops it).
    """
    n_cells, n_windows = density.shape
    density_tp1 = np.concatenate([density, density[:, -1:]], axis=1)
    return SimpleNamespace(
        freeway=SimpleNamespace(n_cells=n_cells, dt=_FIVE_MIN_H),
        density=density_tp1,
        mainline_flow=flow,
        n_steps=n_windows,
    )
