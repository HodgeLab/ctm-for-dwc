"""Hand-calculation tests for the DWPT spatial submodule (P1).

Covers the three pure array-construction functions of the conventional
mCONV spatial dependence (Newbolt 2024a, Algorithms 1-2): ``build_S_T``,
``build_S_R``, and ``build_T_R``. Each test pins an entry pattern computed
by hand against round-number grid-unit inputs (layer 1 of the plan's
verification strategy).
"""

import numpy as np

from transportation_models.utils.dwpt.spatial import (
    build_S_R,
    build_S_T,
    build_T_R,
)


# ---------------------------------------------------------------------------
# build_S_T  — Tx pad profile (Algorithm 1)
# ---------------------------------------------------------------------------


def test_build_S_T_two_tile_bit_pattern():
    """Two [pad, gap] tiles: beta_prime over alpha_grid, then 0 over lambda_grid."""
    S_T = build_S_T(n=6, alpha_grid=2, lambda_grid=1, beta_prime=5.0)
    np.testing.assert_array_equal(S_T, [5.0, 5.0, 0.0, 5.0, 5.0, 0.0])


def test_build_S_T_starts_with_pad_not_gap():
    S_T = build_S_T(n=4, alpha_grid=1, lambda_grid=1, beta_prime=9.0)
    assert S_T[0] == 9.0


def test_build_S_T_truncates_trailing_partial_tile():
    """n need not be a whole number of tiles; the profile is sliced to n."""
    S_T = build_S_T(n=5, alpha_grid=2, lambda_grid=1, beta_prime=5.0)
    np.testing.assert_array_equal(S_T, [5.0, 5.0, 0.0, 5.0, 5.0])


def test_build_S_T_has_length_n():
    S_T = build_S_T(n=10, alpha_grid=2, lambda_grid=3, beta_prime=1.0)
    assert S_T.shape == (10,)


def test_build_S_T_values_are_zero_or_beta_prime():
    S_T = build_S_T(n=12, alpha_grid=2, lambda_grid=1, beta_prime=7.0)
    assert set(np.unique(S_T).tolist()) <= {0.0, 7.0}


# ---------------------------------------------------------------------------
# build_S_R  — Rx pad kernel (Algorithm 2, rows 1-4)
# ---------------------------------------------------------------------------


def test_build_S_R_is_constant_magnitude_beta():
    S_R = build_S_R(delta_grid=3, beta=4.0)
    np.testing.assert_array_equal(S_R, [4.0, 4.0, 4.0])


def test_build_S_R_has_length_delta_grid():
    S_R = build_S_R(delta_grid=15, beta=2.0)
    assert S_R.shape == (15,)


# ---------------------------------------------------------------------------
# build_T_R  — modified Toeplitz traversal matrix (Algorithm 2, rows 5-14)
# ---------------------------------------------------------------------------


def test_build_T_R_banded_toeplitz_entries_by_hand():
    """delta_grid=3, n=4 -> (6, 4) band; column j is S_R shifted down by j."""
    S_R = np.array([1.0, 2.0, 3.0])
    T_R = build_T_R(S_R, n=4)
    expected = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 0.0, 0.0],
        [3.0, 2.0, 1.0, 0.0],
        [0.0, 3.0, 2.0, 1.0],
        [0.0, 0.0, 3.0, 2.0],
        [0.0, 0.0, 0.0, 3.0],
    ])
    np.testing.assert_array_equal(T_R, expected)


def test_build_T_R_shape_is_m_by_n():
    """m = n + delta_grid - 1 (we drop the paper's trailing zero row)."""
    S_R = build_S_R(delta_grid=3, beta=1.0)
    T_R = build_T_R(S_R, n=4)
    assert T_R.shape == (4 + 3 - 1, 4)


def test_build_T_R_off_band_entries_are_zero():
    S_R = np.array([1.0, 2.0, 3.0])
    T_R = build_T_R(S_R, n=5)
    delta_grid = len(S_R)
    for i in range(T_R.shape[0]):
        for j in range(T_R.shape[1]):
            if not (0 <= i - j <= delta_grid - 1):
                assert T_R[i, j] == 0.0


def test_build_T_R_single_position_kernel_is_diagonal():
    """delta_grid=1: T_R places S_R[0] on the diagonal, so m == n."""
    S_R = np.array([5.0])
    T_R = build_T_R(S_R, n=3)
    np.testing.assert_array_equal(T_R, 5.0 * np.eye(3))
