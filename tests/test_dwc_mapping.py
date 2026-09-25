"""Hand-calc and invariant tests for build_M_CTM (P3).

``build_M_CTM`` uses the center-reference convention: traversal position
``j`` maps to corridor position ``j - off`` with ``off = (delta_grid-1)//2``
(the Rx-pad center, floored). Positions whose center falls off-corridor get
zero columns. Covers layer 1 (hand calc) and layer 2 (row-sum / off-corridor
invariants) of the plan's verification strategy.
"""

import numpy as np

from ctm_for_dwc.dwc.mapping import build_M_CTM


# ---------------------------------------------------------------------------
# Hand calculations (layer 1)
# ---------------------------------------------------------------------------


def test_build_M_CTM_hand_calc_symmetric_delta3():
    """ppc=[2,2], delta_grid=3 -> off=1: one boundary column zeroed each end."""
    M = build_M_CTM(positions_per_cell=(2, 2), delta_grid=3)
    expected = np.array([
        [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
    ])
    np.testing.assert_allclose(M, expected)


def test_build_M_CTM_hand_calc_even_delta_floors_offset():
    """ppc=[2,2], delta_grid=4 -> off=1: 1 column zeroed in front, 2 at back."""
    M = build_M_CTM(positions_per_cell=(2, 2), delta_grid=4)
    expected = np.array([
        [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
    ])
    np.testing.assert_allclose(M, expected)


def test_build_M_CTM_delta_grid_one_is_plain_cell_partition():
    """delta_grid=1 -> off=0, m=n: M is the cell partition with no zeroed columns."""
    M = build_M_CTM(positions_per_cell=(2, 3), delta_grid=1)
    expected = np.array([
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1 / 3, 1 / 3, 1 / 3],
    ])
    np.testing.assert_allclose(M, expected)


# ---------------------------------------------------------------------------
# Invariants (layer 2)
# ---------------------------------------------------------------------------


def test_build_M_CTM_shape_is_N_by_m():
    M = build_M_CTM(positions_per_cell=(3, 6, 3), delta_grid=4)
    assert M.shape == (3, 12 + 4 - 1)


def test_build_M_CTM_rows_sum_to_one():
    M = build_M_CTM(positions_per_cell=(3, 6, 3), delta_grid=4)
    np.testing.assert_allclose(M.sum(axis=1), [1.0, 1.0, 1.0])


def test_build_M_CTM_row_i_has_n_i_entries_each_one_over_n_i():
    ppc = (3, 6, 3)
    M = build_M_CTM(positions_per_cell=ppc, delta_grid=4)
    for i, n_i in enumerate(ppc):
        nonzero = M[i][M[i] > 0]
        assert nonzero.shape == (n_i,)
        np.testing.assert_allclose(nonzero, 1.0 / n_i)


def test_build_M_CTM_zeros_delta_minus_one_boundary_columns():
    ppc, delta_grid = (3, 6, 3), 4
    M = build_M_CTM(positions_per_cell=ppc, delta_grid=delta_grid)
    zero_cols = list(np.where(~M.any(axis=0))[0])
    off = (delta_grid - 1) // 2
    n, m = sum(ppc), sum(ppc) + delta_grid - 1
    assert len(zero_cols) == delta_grid - 1
    assert zero_cols == list(range(off)) + list(range(n + off, m))


def test_build_M_CTM_each_mapped_column_belongs_to_one_cell():
    ppc = (3, 6, 3)
    M = build_M_CTM(positions_per_cell=ppc, delta_grid=4)
    col_nonzeros = (M > 0).sum(axis=0)
    assert set(np.unique(col_nonzeros).tolist()) <= {0, 1}
    assert int((col_nonzeros == 1).sum()) == sum(ppc)
