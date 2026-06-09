"""Step 7: gap-filling strategies for ramp VDS timeseries.

Ramp detectors (PeMS Type ∈ {OR, FR}) suffer more outages than mainline
detectors, so the raw 5-min CSVs Step 3 downloads often have NaN samples
that the scenario adapters (:mod:`utils.ctm.scenario`) can't consume
directly. This module provides three pure-function fillers, all sharing
the signature::

    fill(df, *, flow_col, time_col="timestamp", ...) -> pd.DataFrame

Each returns a copy of ``df`` with NaN values in ``flow_col`` filled and
all other columns passed through unchanged. They operate on the **full
downloaded timeseries** (not the sim window) -- the historical-average
versions use every non-NaN sample available in the input to estimate
per-(weekday, time-of-day) statistics, so a wider input window gives
better stats.

Three levels of increasing sophistication
=========================================

* :func:`persistence_fill` (Level 1) -- forward-fill with the last known
  value. Best for short outages (a few samples) where the underlying
  rate is unlikely to have moved much.
* :func:`historical_average_fill` (Level 2) -- per-(weekday,
  time-of-day) mean from non-NaN samples. Captures the typical diurnal
  + weekday pattern; best for longer outages where persistence drifts.
* :func:`stochastic_historical_fill` (Level 2b) -- same binning, but
  draw each fill from ``Normal(mean, stddev)`` of the bin and clip to
  ``>= 0``. Useful for Monte Carlo sensitivity studies.

Outage-duration-aware composition (e.g. use Level 1 for < 30 min
outages, Level 2 for longer ones) is deferred to a follow-up.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# ---- Level 1: persistence ------------------------------------------------


def persistence_fill(
    df: pd.DataFrame, *,
    flow_col: str,
    time_col: str = "timestamp",
    max_gap_minutes: float | None = None,
) -> pd.DataFrame:
    """Forward-fill NaN ``flow_col`` values with the most recent known value.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-VDS 5-minute timeseries; must have ``flow_col`` and
        ``time_col`` columns. Assumed sorted by ``time_col`` ascending.
    flow_col : str
        Column to fill (e.g. ``"total_flow_[veh/5-min]"``).
    time_col : str
        Timestamp column; used only for the ``max_gap_minutes`` check.
    max_gap_minutes : float, optional
        If supplied, NaN runs longer than this many minutes stay NaN
        (the persistence assumption is questionable across long gaps).
        Default: no limit (every gap gets filled regardless of length).

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with NaN ``flow_col`` values forward-filled.
        Leading NaN samples (no prior known value) remain NaN.
    """
    out = df.copy()
    if max_gap_minutes is None:
        out[flow_col] = out[flow_col].ffill()
        return out

    # Build a run-id per consecutive NaN block; ffill only blocks whose
    # duration is <= max_gap_minutes.
    is_na = out[flow_col].isna()
    run_id = (is_na != is_na.shift()).cumsum()
    filled = out[flow_col].ffill()
    for rid in out.loc[is_na, :].groupby(run_id).groups:
        idx = out.loc[is_na & (run_id == rid)].index
        if len(idx) == 0:
            continue
        # 5-min cadence assumed: gap duration = (n_samples - 1) * 5 min
        # is the time between the first and last NaN, but the actual gap
        # spans the inclusive duration of the NaN run.
        gap_min = (len(idx)) * 5.0
        if gap_min <= max_gap_minutes:
            out.loc[idx, flow_col] = filled.loc[idx]
    return out


# ---- Level 2: historical average -----------------------------------------


def _bin_key(timestamps: pd.Series) -> pd.DataFrame:
    """Return (weekday, time_of_day_sec) columns for the (weekday, 5-min) bin."""
    ts = pd.to_datetime(timestamps)
    weekday = ts.dt.weekday.astype(int)
    time_of_day_sec = (
        ts.dt.hour.astype(int) * 3600
        + ts.dt.minute.astype(int) * 60
        + ts.dt.second.astype(int)
    )
    return pd.DataFrame({"_weekday": weekday, "_tod_s": time_of_day_sec})


def _bin_stats(
    history: pd.DataFrame, *, flow_col: str, time_col: str,
) -> pd.DataFrame:
    """Per-(weekday, time-of-day) mean and stddev from non-NaN samples."""
    h = history.copy()
    h[["_weekday", "_tod_s"]] = _bin_key(h[time_col])
    grouped = h.dropna(subset=[flow_col]).groupby(["_weekday", "_tod_s"])
    stats = grouped[flow_col].agg(["mean", "std", "count"]).reset_index()
    return stats


def historical_average_fill(
    df: pd.DataFrame, *,
    flow_col: str,
    time_col: str = "timestamp",
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fill NaN ``flow_col`` with the per-(weekday, time-of-day) mean.

    Statistics are computed from **the full input timeseries** (every
    non-NaN sample in ``df``), not the simulation window. Use
    ``history_df`` to override -- e.g. pass a 3-month baseline CSV when
    filling a 1-week sim window's gaps.

    Bins with zero non-NaN samples can't yield a mean; rows in those
    bins remain NaN and a warning lists the affected bins.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-VDS 5-minute timeseries; must have ``flow_col`` and
        ``time_col`` columns.
    flow_col, time_col : str
        Column names; same defaults as :func:`persistence_fill`.
    history_df : pandas.DataFrame, optional
        Source of bin statistics. Defaults to ``df`` itself.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with NaN ``flow_col`` filled by the matching
        bin's mean.
    """
    src = df if history_df is None else history_df
    stats = _bin_stats(src, flow_col=flow_col, time_col=time_col)
    return _apply_bin_fill(df, stats, flow_col=flow_col, time_col=time_col)


def _apply_bin_fill(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    flow_col: str,
    time_col: str,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Fill NaN rows in ``df`` from per-bin ``stats`` (mean / std / count).

    When ``rng`` is None, fills use the bin mean. When ``rng`` is set,
    fills draw from ``Normal(mean, stddev)`` and clip to ``>= 0`` (flow
    can't be negative). Bins with ``count < 2`` fall back to the mean.
    Bins with ``count == 0`` (no historical samples) leave the row NaN
    and contribute to the missing-bin warning.
    """
    out = df.copy()
    out[["_weekday", "_tod_s"]] = _bin_key(out[time_col])
    nan_mask = out[flow_col].isna()
    if not nan_mask.any():
        out = out.drop(columns=["_weekday", "_tod_s"])
        return out

    # Index stats by bin key for a single vectorized lookup.
    stats_idx = stats.set_index(["_weekday", "_tod_s"])
    nan_rows = out.loc[nan_mask]
    keys = list(zip(nan_rows["_weekday"], nan_rows["_tod_s"]))

    means = np.full(len(nan_rows), np.nan)
    stds = np.full(len(nan_rows), np.nan)
    counts = np.zeros(len(nan_rows), dtype=int)
    for i, k in enumerate(keys):
        if k in stats_idx.index:
            row = stats_idx.loc[k]
            means[i] = row["mean"]
            stds[i] = row["std"]
            counts[i] = int(row["count"])

    if rng is None:
        fills = means
    else:
        # Stochastic: draw from N(mean, std). Fall back to mean if std is
        # NaN (bin had n=1 sample) or zero. Clip negatives to 0.
        use_stoch = (counts >= 2) & np.isfinite(stds) & (stds > 0)
        draws = rng.normal(loc=means, scale=np.where(use_stoch, stds, 0.0))
        fills = np.where(use_stoch, np.maximum(draws, 0.0), means)

    out.loc[nan_mask, flow_col] = fills

    # Warn about bins with no historical samples (means stayed NaN).
    missing_mask = np.isnan(means)
    if missing_mask.any():
        missing_keys = sorted({k for k, m in zip(keys, missing_mask) if m})
        warnings.warn(
            f"{int(missing_mask.sum())} NaN samples sit in bins with no "
            f"non-NaN history; left unfilled. Affected (weekday, "
            f"time-of-day-seconds) bins: {missing_keys[:10]}"
            + (" ..." if len(missing_keys) > 10 else ""),
            RuntimeWarning, stacklevel=3,
        )

    out = out.drop(columns=["_weekday", "_tod_s"])
    return out


# ---- Level 2b: stochastic historical average -----------------------------


def stochastic_historical_fill(
    df: pd.DataFrame, *,
    flow_col: str,
    time_col: str = "timestamp",
    history_df: pd.DataFrame | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Fill NaN ``flow_col`` by sampling from ``Normal(mean, stddev)`` per bin.

    Same binning as :func:`historical_average_fill` and same
    ``history_df`` override. ``seed`` controls the random draws (default
    ``0`` for reproducibility); pass any int for a different reproducible
    sequence.

    Bins with fewer than 2 non-NaN samples have no usable stddev; they
    fall back to the bin mean (deterministic). Negative draws are clipped
    to 0 (flow can't be negative). Bins with no samples emit the same
    warning as :func:`historical_average_fill`.
    """
    src = df if history_df is None else history_df
    stats = _bin_stats(src, flow_col=flow_col, time_col=time_col)
    rng = np.random.default_rng(seed)
    return _apply_bin_fill(
        df, stats, flow_col=flow_col, time_col=time_col, rng=rng,
    )
