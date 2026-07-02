"""Temporal-aggregation analysis for the DWPT demand profile.

The DWPT demand ``E`` (shape ``(T, m)``, Wh) carries one row per CTM timestep
of duration ``dt``. Reporting the corridor load at a coarser temporal
resolution -- averaging power over 5-, 15-, 60-min windows the way a grid
study might -- conserves total energy but *attenuates the peak power* a feeder
must actually carry, and flattens the load shape.

This module holds the pure, NumPy-only numeric core for that analysis:
collapse ``E`` to a corridor-total (or per-mile-chunk) energy-per-step series,
re-bin it to a coarser window, and score peak / load-duration-curve fidelity
against the native-``dt`` baseline. Orchestration, plotting, and I/O live in
``scripts/analyze_dwpt_temporal_aggregation.py``.

All power is in watts, energy in watt-hours, time in hours (``dt_h``), matching
the DWPT module's units.
"""

from __future__ import annotations

import numpy as np

_METERS_PER_MILE = 1609.344


def energy_per_step(E: np.ndarray) -> np.ndarray:
    """Corridor-total energy per timestep, ``(T,)`` Wh, from ``E`` ``(T, m)``."""
    return E.sum(axis=1)


def rebin_sum(x: np.ndarray, steps_per_bin: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum consecutive blocks of ``steps_per_bin`` samples.

    Returns ``(binned, counts)`` where ``binned[b]`` is the sum over bin ``b``
    and ``counts[b]`` is the number of native samples it covers. A trailing
    partial bin (when ``len(x)`` is not a multiple of ``steps_per_bin``) is
    kept and covers ``< steps_per_bin`` samples, so ``binned.sum() == x.sum()``
    exactly -- energy is conserved by construction.
    """
    if steps_per_bin < 1:
        raise ValueError(f"steps_per_bin must be >= 1, got {steps_per_bin}")
    x = np.asarray(x, dtype=float)
    total = x.shape[0]
    n_full = total // steps_per_bin
    parts_binned: list[np.ndarray] = []
    parts_counts: list[np.ndarray] = []
    if n_full:
        head = x[: n_full * steps_per_bin].reshape(n_full, steps_per_bin)
        parts_binned.append(head.sum(axis=1))
        parts_counts.append(np.full(n_full, steps_per_bin, dtype=float))
    rem = total - n_full * steps_per_bin
    if rem:
        parts_binned.append(np.array([x[n_full * steps_per_bin :].sum()]))
        parts_counts.append(np.array([float(rem)]))
    return np.concatenate(parts_binned), np.concatenate(parts_counts)


def window_power(
    energy_per_step_arr: np.ndarray, steps_per_bin: int, dt_h: float
) -> tuple[np.ndarray, np.ndarray]:
    """Average power ``(W)`` per aggregation window.

    Re-bins the energy-per-step series and divides each bin's energy by its
    duration (``counts * dt_h``), giving the mean power over the window.
    ``steps_per_bin == 1`` recovers the native-``dt`` power series. Returns
    ``(power, counts)``; ``counts`` feeds ``bin_center_times_h`` for timing.
    """
    binned, counts = rebin_sum(energy_per_step_arr, steps_per_bin)
    return binned / (counts * dt_h), counts


def bin_center_times_h(counts: np.ndarray, dt_h: float) -> np.ndarray:
    """Bin-center wall-clock times ``(h)`` for a series of bin ``counts``."""
    edges = np.concatenate([[0.0], np.cumsum(counts)])
    centers_steps = (edges[:-1] + edges[1:]) / 2.0
    return centers_steps * dt_h


def peak(power: np.ndarray) -> tuple[float, int]:
    """Peak power and its bin index: ``(max_value, argmax)``."""
    i = int(np.argmax(power))
    return float(power[i]), i


def pct_attenuation(baseline_peak: float, window_peak: float) -> float:
    """Peak understatement ``100 * (baseline - window) / baseline`` in %."""
    if baseline_peak == 0:
        return 0.0
    return 100.0 * (baseline_peak - window_peak) / baseline_peak


def mile_chunk_columns(
    position_m: np.ndarray, corridor_length_m: float, *, chunk_len_mi: float = 1.0
) -> dict[int, np.ndarray]:
    """Group on-corridor ``E`` column indices into fixed-length mile chunks.

    A column is assigned to chunk ``floor(position_mi / chunk_len_mi)`` when its
    center position lies on the corridor (``0 <= position < corridor_length``).
    Off-corridor boundary columns (Rx approach/exit, where ``E`` is zero) are
    dropped. Returns ``{chunk_index: column_indices}`` in ascending chunk order.
    """
    pos = np.asarray(position_m, dtype=float)
    on_corridor = (pos >= 0.0) & (pos < corridor_length_m)
    chunk_of = np.floor((pos / _METERS_PER_MILE) / chunk_len_mi).astype(int)
    out: dict[int, np.ndarray] = {}
    for c in sorted(set(chunk_of[on_corridor].tolist())):
        out[c] = np.nonzero(on_corridor & (chunk_of == c))[0]
    return out


def load_duration_curve(
    power: np.ndarray, weights: np.ndarray, *, n_points: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Load-duration curve sampled on a common exceedance-fraction grid.

    Sorts ``power`` descending, weights each sample by its duration
    (``weights``), and evaluates the resulting step function at ``n_points``
    equally spaced exceedance fractions in ``(0, 1)``. Returning both curves on
    the *same* x-grid lets curves at different temporal resolutions be compared
    directly (see ``ldc_divergence``).
    """
    power = np.asarray(power, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(power)[::-1]
    p = power[order]
    frac = np.cumsum(weights[order]) / weights.sum()  # right edge of each band
    xs = (np.arange(n_points) + 0.5) / n_points
    idx = np.clip(np.searchsorted(frac, xs, side="left"), 0, len(p) - 1)
    return xs, p[idx]


def ldc_divergence(
    power_ref: np.ndarray,
    weights_ref: np.ndarray,
    power_agg: np.ndarray,
    weights_agg: np.ndarray,
    *,
    n_points: int = 512,
) -> float:
    """Normalized area between two load-duration curves (dimensionless).

    Integrates ``|LDC_ref - LDC_agg|`` over the exceedance-fraction axis and
    normalizes by the reference mean power, so the result is a dimensionless
    fraction: ``0`` means identical shapes, larger means the aggregated profile
    departs more from the baseline load-duration shape. Because both LDCs have
    equal area (energy is conserved), this captures purely the reshaping --
    peak shaved, tail lifted.
    """
    _, ref = load_duration_curve(power_ref, weights_ref, n_points=n_points)
    _, agg = load_duration_curve(power_agg, weights_agg, n_points=n_points)
    # LDCs are sampled at equally spaced midpoints over the [0, 1] exceedance
    # axis, so the integral of |ref - agg| over that axis is their mean.
    area = float(np.mean(np.abs(ref - agg)))
    p_mean = float(np.average(power_ref, weights=weights_ref))
    if p_mean == 0:
        return 0.0
    return area / p_mean
