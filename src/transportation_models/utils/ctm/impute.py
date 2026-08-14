"""PeMS-based imputation of unobserved cells on the CTM ``(n_cells, n_5min)`` grid.

Two kinds of unobserved index are filled before the diffsim ramp optimizer
scores the grid (docs/ramp_flow_estimation_module.md, "Model-based
optimization"):

* **Type 2** -- a cell has a direct VDS assignment but the measurement is
  missing at some 5-min steps. Filled from the detector's own history via the
  ``(day-of-week, time-of-day)`` slot mean (:func:`historical_slot_fill`) function.
* **Type 1** -- a cell has no direct VDS at all (a full column of missing
  data). Filled by a spatial-temporal KNN over the surrounding *observed* and
  Type-2-filled cells (:func:`knn_impute_grid`); other Type-1 cells are never
  used as neighbors, so there is no imputation-on-imputation.

Both fills are pure array ops; the augment script (``scripts/augment_observed_grid.py``)
wires them to the bundle's CSVs and the raw PeMS timeseries.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _slot_means(values, observed, slot_key) -> np.ndarray:
    """Historical mean per (day-of-week, time-of-day) slot, broadcast back to
    the full record; NaN for slots with no observed sample."""
    obs = pd.Series(values[observed])
    means = obs.groupby(slot_key[observed]).mean()
    return means.reindex(slot_key).to_numpy(dtype=float)


def historical_slot_fill(values, observed, timestamps):
    """Fill ``~observed`` entries with their ``(dow, time-of-day)`` slot mean.

    The slot mean for each 5-min-of-day-x-weekday bin is computed over the
    ``observed`` entries of ``values`` (so a multi-day history gives a robust
    mean); missing entries are then set to the mean of their own slot. Entries
    whose slot was never observed stay NaN.

    Parameters
    ----------
    values : array-like of float, shape (n,)
        The series (e.g. one VDS's density or flow over its full record).
    observed : array-like of bool, shape (n,)
        True where ``values`` is a genuine measurement.
    timestamps : array-like of datetime64, shape (n,)
        Wallclock time of each entry, used to form the slot key.

    Returns
    -------
    filled : ndarray of float, shape (n,)
        ``values`` with fillable missing entries replaced by their slot mean.
    filled_mask : ndarray of bool, shape (n,)
        True exactly where an entry was missing *and* got a slot mean.
    """
    values = np.asarray(values, dtype=float).copy()
    observed = np.asarray(observed, dtype=bool)
    ts = pd.DatetimeIndex(timestamps)
    slot_key = (ts.dayofweek * 10_000 + ts.hour * 100 + ts.minute).to_numpy()
    means = _slot_means(values, observed, slot_key)  # per-row slot mean, NaN if unseen
    filled_mask = ~observed & ~np.isnan(means)
    values[filled_mask] = means[filled_mask]
    return values, filled_mask


def _shift(arr, dc, dt, fill):
    """``out[c, t] = arr[c + dc, t + dt]``, out-of-grid positions set to ``fill``."""
    out = np.full_like(arr, fill, dtype=float)
    n, m = arr.shape
    cs0, cs1 = max(0, dc), min(n, n + dc)
    cd0, cd1 = max(0, -dc), min(n, n - dc)
    ts0, ts1 = max(0, dt), min(m, m + dt)
    td0, td1 = max(0, -dt), min(m, m - dt)
    if cs0 < cs1 and ts0 < ts1:
        out[cd0:cd1, td0:td1] = arr[cs0:cs1, ts0:ts1]
    return out


def knn_impute_grid(values, usable, target, k, decay):
    """Impute ``target`` grid entries from a Manhattan-radius-``k`` KNN.

    For each target ``(c, t)`` the estimate is a weighted mean of the
    ``usable`` neighbors at cell/time offsets ``(dc, dt)`` with
    ``1 <= |dc| + |dt| <= k`` and ``dc != 0`` (the target's own column is
    excluded -- for a Type-1 cell that column is unobserved at every step).
    A neighbor at Manhattan distance ``d`` gets raw weight
    ``exp(-decay * (d - 1))``; the raw weights of the in-grid, usable neighbors
    are normalized to sum to 1. With ``k == 1`` every neighbor is at ``d == 1``,
    so the result is the plain upstream/downstream average regardless of
    ``decay``. Targets with no usable neighbor in range are left NaN.

    Parameters
    ----------
    values : ndarray of float, shape (n_cells, n_time)
        Grid whose ``usable`` entries hold measurements (NaN elsewhere).
    usable : ndarray of bool, shape (n_cells, n_time)
        Entries allowed to serve as neighbors (observed + Type-2-filled).
    target : ndarray of bool, shape (n_cells, n_time)
        Entries to impute (Type-1). Disjoint from ``usable``.
    k : int
        Manhattan radius of the neighborhood (>= 1).
    decay : float
        Exponential decay rate ``lambda`` (>= 0).

    Returns
    -------
    filled : ndarray of float, shape (n_cells, n_time)
        Copy of ``values`` with imputable target entries filled.
    imputed_mask : ndarray of bool, shape (n_cells, n_time)
        True exactly where a target entry received an estimate.
    """
    values = np.asarray(values, dtype=float)
    usable_f = np.asarray(usable, dtype=float)
    vsum = np.zeros_like(values)
    wsum = np.zeros_like(values)
    for dc in range(-k, k + 1):
        if dc == 0:  # exclude the target's own column
            continue
        for dt in range(-(k - abs(dc)), (k - abs(dc)) + 1):
            d = abs(dc) + abs(dt)  # >= 1 since dc != 0
            w = math.exp(-decay * (d - 1))
            valid = _shift(usable_f, dc, dt, fill=0.0)  # 1 where neighbor in-grid & usable
            sval = _shift(values, dc, dt, fill=0.0)
            mass = w * valid
            wsum += mass
            vsum += np.where(valid > 0.0, mass * sval, 0.0)
    imputed_mask = np.asarray(target, dtype=bool) & (wsum > 0.0)
    filled = values.copy()
    filled[imputed_mask] = vsum[imputed_mask] / wsum[imputed_mask]
    return filled, imputed_mask
