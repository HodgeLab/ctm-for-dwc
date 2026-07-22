"""Spatial submodule for the DWPT demand module.

Pure NumPy array-construction functions for the conventional mCONV spatial
dependence (Newbolt 2024a, Algorithms 1-2): the Tx pad profile ``S_T``, the
Rx pad kernel ``S_R``, and the modified Toeplitz traversal matrix ``T_R``.

All functions operate on grid-unit integers (positions), not meters. The
meter->grid conversion and its validation live in ``CorridorSpec`` (see
``model.py``), whose ``*_grid`` properties feed these functions. See
``docs/dwpt_demand_module.md`` §"Reference Algorithms (Newbolt 2024a)".
"""

from __future__ import annotations

import numpy as np


def build_S_T(
    n: int, alpha_grid: int, lambda_grid: int, beta_prime: float
) -> np.ndarray:
    """Tx pad profile over ``n`` grid positions (Algorithm 1).

    Repeats a single tile of ``alpha_grid`` "on" positions (value
    ``beta_prime``) followed by ``lambda_grid`` "off" positions (value 0),
    then slices to length ``n``. The corridor begins with a pad and may end
    mid-tile.
    """
    tile = np.concatenate(
        [np.full(alpha_grid, beta_prime), np.zeros(lambda_grid)]
    )
    repeats = (n + len(tile) - 1) // len(tile)
    return np.tile(tile, repeats)[:n]


def build_S_R(delta_grid: int, beta: float) -> np.ndarray:
    """Constant-magnitude Rx pad kernel of length ``delta_grid`` (Algorithm 2)."""
    return np.full(delta_grid, beta)


def build_T_R(S_R: np.ndarray, n: int) -> np.ndarray:
    """Modified Toeplitz traversal matrix, shape ``(n + delta_grid - 1, n)``.

    Column ``j`` holds ``S_R`` shifted down by ``j`` rows, mimicking the Rx
    kernel sliding across the corridor (Algorithm 2). Entry ``T_R[i, j]`` is
    ``S_R[i - j]`` when ``0 <= i - j < delta_grid``, else 0. Unlike the paper
    we omit the trailing all-zero row, so ``m = n + delta_grid - 1``.

    Reference construction only -- **not on the hot path.** ``build_P_y``
    forms ``P_A = T_R @ S_T`` directly as a convolution (``O(n)`` memory); this
    dense matrix is ``O(n^2)`` and exists for exact reproduction of the
    original mCONV and the ``method="toeplitz"`` comparison path.
    """
    delta_grid = len(S_R)
    T_R = np.zeros((n + delta_grid - 1, n))
    for j in range(n):
        T_R[j : j + delta_grid, j] = S_R
    return T_R


# Empirical crossover (see scripts/benchmark_dwpt_convolution.py): direct
# np.convolve is O(n * delta_grid) and wins for small kernels; fftconvolve is
# O(n log n) and wins once the Rx kernel exceeds ~this many grid positions.
_AUTO_FFT_THRESHOLD = 256


def build_P_y(
    S_T: np.ndarray,
    S_R: np.ndarray,
    gamma: float,
    *,
    method: str = "auto",
) -> np.ndarray:
    """Power-vs-position profile over the m traversal positions (Algorithm 3).

    Forms the raw cross-sectional-area profile ``P_A = S_R * S_T`` (a discrete
    convolution -- the same quantity as ``T_R @ S_T``), then rescales by
    ``gamma / P_A.max()`` so the peak power equals ``gamma = min(beta,
    beta_prime)``. Without that normalization the peak overshoots by
    ``P_A.max()``.

    Parameters
    ----------
    method : {"auto", "convolve", "fft", "toeplitz"}
        How to form ``P_A``:

        - ``"convolve"`` -- ``np.convolve``; ``O(n * delta_grid)``, fastest for
          small kernels (coarse grids).
        - ``"fft"`` -- ``scipy.signal.fftconvolve``; ``O(n log n)``, fastest
          for large kernels (fine grids).
        - ``"toeplitz"`` -- the original dense ``T_R @ S_T``; ``O(n^2)`` memory,
          for reference / exact reproduction only, not production scale.
        - ``"auto"`` (default) -- ``fft`` when ``len(S_R) > 256`` else
          ``convolve`` (empirical crossover; see the benchmark script).
    """
    if method == "auto":
        method = "fft" if len(S_R) > _AUTO_FFT_THRESHOLD else "convolve"

    if method == "convolve":
        P_A = np.convolve(S_T, S_R)
    elif method == "fft":
        from scipy.signal import fftconvolve

        P_A = fftconvolve(S_T, S_R)
    elif method == "toeplitz":
        P_A = build_T_R(S_R, len(S_T)) @ S_T
    else:
        raise ValueError(
            "method must be 'auto', 'convolve', 'fft', or 'toeplitz'; "
            f"got {method!r}"
        )
    return gamma * P_A / P_A.max()
