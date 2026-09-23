"""Hand-calculation tests for the DWC spatial submodule (P1).

Covers the three pure array-construction functions of the conventional
mCONV spatial dependence (Newbolt 2024a, Algorithms 1-2): ``build_S_T``,
``build_S_R``, and ``build_T_R``. Each test pins an entry pattern computed
by hand against round-number grid-unit inputs (layer 1 of the plan's
verification strategy).
"""

import numpy as np
import pytest

from transportation_models.utils.dwc.spatial import (
    build_P_y,
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


# ---------------------------------------------------------------------------
# build_P_y  — power-vs-position profile (Algorithm 3)
# ---------------------------------------------------------------------------


def _build_profile(alpha_grid, lambda_grid, delta_grid, n, gamma):
    """Compose the P1 functions into a P_y, as CorridorSpec.build() will."""
    S_T = build_S_T(n=n, alpha_grid=alpha_grid, lambda_grid=lambda_grid, beta_prime=1.0)
    S_R = build_S_R(delta_grid=delta_grid, beta=1.0)
    return build_P_y(S_T, S_R, gamma)


def test_build_P_y_hand_calc_two_pad_profile():
    """alpha=lambda=delta=2 (grid), n=6: P_A=[1,2,1,0,1,2,1], gamma=10 -> /max=2."""
    P_y = _build_profile(alpha_grid=2, lambda_grid=2, delta_grid=2, n=6, gamma=10.0)
    np.testing.assert_allclose(P_y, [5.0, 10.0, 5.0, 0.0, 5.0, 10.0, 5.0])


def test_build_P_y_peak_equals_gamma_exactly():
    """The /max normalization pins the peak to gamma (Newbolt's 150 kW)."""
    P_y = _build_profile(alpha_grid=3, lambda_grid=1, delta_grid=2, n=12, gamma=150_000.0)
    assert P_y.max() == 150_000.0


def test_build_P_y_never_exceeds_gamma():
    P_y = _build_profile(alpha_grid=3, lambda_grid=1, delta_grid=2, n=12, gamma=150_000.0)
    assert np.all(P_y <= 150_000.0)


def test_build_P_y_is_nonnegative():
    P_y = _build_profile(alpha_grid=3, lambda_grid=1, delta_grid=2, n=12, gamma=150_000.0)
    assert np.all(P_y >= 0.0)


def test_build_P_y_scales_linearly_with_gamma():
    base = _build_profile(alpha_grid=3, lambda_grid=1, delta_grid=2, n=12, gamma=1.0)
    doubled = _build_profile(alpha_grid=3, lambda_grid=1, delta_grid=2, n=12, gamma=2.0)
    np.testing.assert_allclose(doubled, 2.0 * base)


def test_build_P_y_three_pad_system_matches_fig4_shape():
    """Newbolt Fig. 4 qualitative match on a coarse 3-pad analog.

    The Table-1 dimensions at dx_grid=0.0001 m make T_R ~41 GB dense, so we
    use a geometry-preserving coarse analog (alpha > delta, small gap) and
    assert the qualitative shape: one hump per pad, valleys between, tapered
    ends, and a peak equal to gamma.
    """
    alpha_grid, lambda_grid, delta_grid, n_pads = 3, 1, 2, 3
    n = n_pads * (alpha_grid + lambda_grid)
    P_y = _build_profile(alpha_grid, lambda_grid, delta_grid, n, gamma=150_000.0)

    # One hump per Tx pad: count contiguous runs of at-peak positions.
    on_peak = np.isclose(P_y, P_y.max()).astype(int)
    n_humps = int((np.diff(np.concatenate([[0], on_peak, [0]])) == 1).sum())
    assert n_humps == n_pads

    # Valleys between pads dip below the peak, and the ends taper down.
    assert P_y.min() < P_y.max()
    assert P_y[0] < P_y.max()
    assert P_y[-1] < P_y.max()


def test_build_P_y_methods_agree():
    """convolve, fft, and the dense toeplitz reference give the same P_y."""
    S_T = build_S_T(n=60, alpha_grid=4, lambda_grid=2, beta_prime=3.0)
    S_R = build_S_R(delta_grid=5, beta=2.0)
    ref = build_P_y(S_T, S_R, gamma=10.0, method="toeplitz")
    for method in ("convolve", "fft", "auto"):
        # atol covers fft float noise (~1e-15) at the exact-zero end positions.
        np.testing.assert_allclose(
            build_P_y(S_T, S_R, gamma=10.0, method=method), ref, atol=1e-9
        )


def test_build_P_y_scales_without_dense_matrix():
    """A 200k-position corridor runs via convolution; a dense T_R would be ~320 GB."""
    n, delta_grid = 200_000, 3
    S_T = build_S_T(n=n, alpha_grid=7, lambda_grid=1, beta_prime=150_000.0)
    S_R = build_S_R(delta_grid=delta_grid, beta=150_000.0)
    P_y = build_P_y(S_T, S_R, gamma=150_000.0)  # auto -> convolve (small kernel)
    assert P_y.shape == (n + delta_grid - 1,)
    assert P_y.max() == 150_000.0


def test_build_P_y_rejects_unknown_method():
    S_T = build_S_T(n=10, alpha_grid=2, lambda_grid=1, beta_prime=1.0)
    S_R = build_S_R(delta_grid=2, beta=1.0)
    with pytest.raises(ValueError, match="method"):
        build_P_y(S_T, S_R, 1.0, method="bogus")
