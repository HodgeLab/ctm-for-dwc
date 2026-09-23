"""Shared observation-window helpers for ramp-flow estimation.

``sliding_all`` reduces a per-sample boolean mask to window granularity; it is
used by :mod:`features` to decide which windows yield training samples.
"""
from __future__ import annotations

import numpy as np


def sliding_all(observed: np.ndarray, window: int) -> np.ndarray:
    """Window-granular AND: element ``t`` is True iff ``observed[t:t+window]``
    are all True. Returns length ``len(observed) - window + 1`` (empty if the
    record is shorter than the window)."""
    observed = np.asarray(observed, dtype=bool)
    n = observed.shape[0]
    if window <= 0:
        raise ValueError("window must be positive")
    if n < window:
        return np.zeros(0, dtype=bool)
    out = observed[: n - window + 1].copy()
    for k in range(1, window):
        out &= observed[k : n - window + 1 + k]
    return out
