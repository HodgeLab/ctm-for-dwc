"""Benchmark P_y computation: dense Toeplitz vs np.convolve vs fftconvolve.

Sweeps corridor length and grid resolution (both physical, in meters) and times
forming the cross-sectional-area profile P_A = S_R * S_T three ways, alongside
the dense T_R memory. Demonstrates why ``build_P_y`` avoids materializing the
dense matrix and auto-selects convolve vs fft by kernel size.

    python scripts/benchmark_dwpt_convolution.py

All lengths are physical (miles / meters); the grid-unit position count ``n``
and kernel size are shown only as secondary context.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.signal import fftconvolve

from transportation_models.utils.dwpt.spatial import build_S_R, build_S_T, build_T_R

_METERS_PER_MILE = 1609.344
# Newbolt 2024a pad geometry (meters).
_RX_PAD_M = 1.5
_TX_PAD_M = 3.5
_GAP_M = 0.5

# (corridor_length_m, dx_grid_m) sweep -- physical units.
_SWEEP = [
    (1 * _METERS_PER_MILE, 0.5),
    (10 * _METERS_PER_MILE, 0.5),
    (50 * _METERS_PER_MILE, 0.5),
    (1 * _METERS_PER_MILE, 0.1),
    (10 * _METERS_PER_MILE, 0.1),
    (50 * _METERS_PER_MILE, 0.1),
    (10 * _METERS_PER_MILE, 0.01),
    (1 * _METERS_PER_MILE, 0.001),
    (3 * (_TX_PAD_M + _GAP_M), 0.0001),  # Newbolt small-scale bench, full fidelity
]

_CONV_OPS_LIMIT = 5e9       # skip np.convolve above ~this many multiply-adds
_DENSE_BYTES_LIMIT = 1.5e9  # build the dense T_R only below this


def _best(fn, repeats: int) -> float:
    t = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return t


def _fmt_bytes(b: float) -> str:
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6)):
        if b >= scale:
            return f"{b / scale:.1f} {unit}"
    return f"{b / 1e3:.0f} KB"


def _fmt_corridor(m: float) -> str:
    return f"{m / _METERS_PER_MILE:.2f} mi ({m:,.0f} m)"


def run(sweep=_SWEEP):
    rows = []
    for length_m, dx in sweep:
        n = round(length_m / dx)
        delta_grid = max(1, round(_RX_PAD_M / dx))
        alpha_grid = max(1, round(_TX_PAD_M / dx))
        lambda_grid = max(1, round(_GAP_M / dx))
        S_T = build_S_T(n, alpha_grid, lambda_grid, 150_000.0)
        S_R = build_S_R(delta_grid, 150_000.0)

        t_fft = f"{_best(lambda st=S_T, sr=S_R: fftconvolve(st, sr), 3) * 1e3:.2f} ms"
        if n * delta_grid <= _CONV_OPS_LIMIT:
            t_conv = f"{_best(lambda st=S_T, sr=S_R: np.convolve(st, sr), 3) * 1e3:.2f} ms"
        else:
            t_conv = "(too slow)"
        dense_bytes = (n + delta_grid - 1) * n * 8
        if dense_bytes < _DENSE_BYTES_LIMIT:
            t_dense = f"{_best(lambda st=S_T, sr=S_R: build_T_R(sr, len(st)) @ st, 2) * 1e3:.1f} ms"
        else:
            t_dense = f"OOM ({_fmt_bytes(dense_bytes)})"
        rows.append((length_m, dx, n, delta_grid, t_conv, t_fft, t_dense))
    return rows


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    rows = run()
    print(f"Rx pad delta = {_RX_PAD_M} m;  kernel size = round(delta / dx) grid units\n")
    header = (
        f"{'corridor':>22} {'resolution':>12} {'positions n':>13} "
        f"{'kernel':>8} {'np.convolve':>12} {'fftconvolve':>12} {'dense T_R':>16}"
    )
    print(header)
    print("-" * len(header))
    for length_m, dx, n, delta_grid, conv, fft, dense in rows:
        print(
            f"{_fmt_corridor(length_m):>22} {dx:>10.4g} m {n:>13,} "
            f"{delta_grid:>8,} {conv:>12} {fft:>12} {dense:>16}"
        )


if __name__ == "__main__":
    main()
