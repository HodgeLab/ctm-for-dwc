"""Unit tests for the PeMS grid imputation helpers (utils.ctm.impute).

Covers the Type-2 ``(dow, time-of-day)`` slot fill and the Type-1 KNN fill:
the k=1 up/down average, exponential-decay ordering, weight normalization,
own-column exclusion, corridor/time-edge renormalization, and the
no-usable-neighbor fallthrough.
"""

import math

import numpy as np
import pandas as pd
import pytest

from ctm_for_dwc.utils.ctm.impute import (
    historical_slot_fill,
    knn_impute_grid,
)


# --- Type-2 historical slot fill ---------------------------------------------

def test_historical_slot_fill_uses_matching_slot_mean():
    # Two Mondays at 08:00 (same slot) observed as 10 and 20; the third Monday
    # 08:00 is missing -> filled with their mean (15). A never-seen slot stays NaN.
    ts = pd.to_datetime([
        "2022-04-04 08:00", "2022-04-11 08:00", "2022-04-18 08:00",  # Mondays 08:00
        "2022-04-05 09:00",                                          # Tue 09:00, unseen
    ])
    values = np.array([10.0, 20.0, np.nan, np.nan])
    observed = np.array([True, True, False, False])
    filled, mask = historical_slot_fill(values, observed, ts)
    assert filled[2] == pytest.approx(15.0)
    assert mask[2]
    assert np.isnan(filled[3]) and not mask[3]  # slot never observed
    # observed entries are never overwritten or marked
    assert filled[0] == 10.0 and not mask[0]


def test_historical_slot_fill_separates_time_of_day_slots():
    ts = pd.to_datetime([
        "2022-04-04 08:00", "2022-04-11 08:00",  # Monday 08:00 -> mean 100
        "2022-04-04 09:00", "2022-04-11 09:00",  # Monday 09:00 -> mean 200
        "2022-04-18 08:00", "2022-04-18 09:00",  # missing, one of each slot
    ])
    values = np.array([100.0, 100.0, 200.0, 200.0, np.nan, np.nan])
    observed = np.array([True, True, True, True, False, False])
    filled, _ = historical_slot_fill(values, observed, ts)
    assert filled[4] == pytest.approx(100.0)
    assert filled[5] == pytest.approx(200.0)


# --- Type-1 KNN fill ---------------------------------------------------------

def _grid(rows):
    return np.asarray(rows, dtype=float)


def test_knn_k1_is_updown_average_and_ignores_decay():
    # 3 cells x 1 time; middle cell is Type-1, neighbors observed 10 and 30.
    values = _grid([[10.0], [np.nan], [30.0]])
    usable = np.array([[True], [False], [True]])
    target = np.array([[False], [True], [False]])
    for decay in (0.0, 1.0, 5.0):
        filled, mask = knn_impute_grid(values, usable, target, k=1, decay=decay)
        assert mask[1, 0]
        assert filled[1, 0] == pytest.approx(20.0)  # (10 + 30) / 2, decay irrelevant


def test_knn_own_column_excluded():
    # k=1: the same-column time neighbors (dc=0) must not be used even when
    # usable; only the cross-cell neighbor feeds the estimate.
    # cells x time = 3 x 3, impute (1,1). Own column (1,0),(1,2) usable=7,9 but
    # dc=0 so excluded; cross neighbors (0,1)=100,(2,1)=200 -> mean 150.
    values = _grid([
        [0.0, 100.0, 0.0],
        [7.0, np.nan, 9.0],
        [0.0, 200.0, 0.0],
    ])
    usable = np.array([
        [False, True, False],
        [True, False, True],   # (1,0),(1,2) usable but same column as target
        [False, True, False],
    ])
    target = np.zeros((3, 3), dtype=bool)
    target[1, 1] = True
    filled, mask = knn_impute_grid(values, usable, target, k=1, decay=1.0)
    assert mask[1, 1]
    assert filled[1, 1] == pytest.approx(150.0)


def test_knn_k2_decay_weights_normalize_to_one():
    # Interior target with all 8 k=2 neighbors usable and equal to 1.0 -> the
    # weighted mean is 1.0 for any decay (weights sum to 1 by construction).
    n_cells, n_time = 5, 5
    values = np.ones((n_cells, n_time))
    usable = np.ones((n_cells, n_time), dtype=bool)
    c, t = 2, 2
    values[c, t] = np.nan
    usable[c, t] = False
    target = np.zeros((n_cells, n_time), dtype=bool)
    target[c, t] = True
    for decay in (0.0, 1.0, 3.0):
        filled, mask = knn_impute_grid(values, usable, target, k=2, decay=decay)
        assert mask[c, t]
        assert filled[c, t] == pytest.approx(1.0)


def test_knn_k2_closer_neighbors_dominate_under_decay():
    # d=1 neighbors carry value 0, d=2 neighbors carry value 10. Higher decay
    # pulls the estimate toward the (nearer) zeros.
    n = 5
    values = np.full((n, n), np.nan)
    c, t = 2, 2
    # d=1 cross-cell neighbors (own-column excluded): (1,2),(3,2)
    for cc, tt in [(1, 2), (3, 2)]:
        values[cc, tt] = 0.0
    # d=2 neighbors: (0,2),(4,2),(1,1),(1,3),(3,1),(3,3)
    for cc, tt in [(0, 2), (4, 2), (1, 1), (1, 3), (3, 1), (3, 3)]:
        values[cc, tt] = 10.0
    usable = ~np.isnan(values)
    target = np.zeros((n, n), dtype=bool)
    target[c, t] = True

    lo, _ = knn_impute_grid(values, usable, target, k=2, decay=0.0)
    hi, _ = knn_impute_grid(values, usable, target, k=2, decay=2.0)
    # uniform: 2 zeros + 6 tens over 8 -> 7.5
    assert lo[c, t] == pytest.approx(60.0 / 8.0)
    assert hi[c, t] < lo[c, t]  # decay upweights the nearer zeros

    # exact check of the normalized weighted mean at decay=1
    w1, w2 = 1.0, math.exp(-1.0)
    expected = (2 * w1 * 0.0 + 6 * w2 * 10.0) / (2 * w1 + 6 * w2)
    mid, _ = knn_impute_grid(values, usable, target, k=2, decay=1.0)
    assert mid[c, t] == pytest.approx(expected)


def test_knn_edge_renormalizes_over_available_neighbors():
    # Target at the corridor edge (cell 0): only the downstream neighbor exists,
    # so the estimate equals it (weights renormalize over what's available).
    values = _grid([[np.nan], [42.0], [99.0]])
    usable = np.array([[False], [True], [True]])
    target = np.array([[True], [False], [False]])
    filled, mask = knn_impute_grid(values, usable, target, k=1, decay=1.0)
    assert mask[0, 0]
    assert filled[0, 0] == pytest.approx(42.0)


def test_knn_no_usable_neighbor_left_unfilled():
    # A Type-1 target whose only in-range neighbors are also Type-1 (unusable)
    # is not imputed.
    values = _grid([[np.nan], [np.nan], [np.nan]])
    usable = np.zeros((3, 1), dtype=bool)
    target = np.array([[False], [True], [False]])
    filled, mask = knn_impute_grid(values, usable, target, k=1, decay=1.0)
    assert not mask[1, 0]
    assert np.isnan(filled[1, 0])
