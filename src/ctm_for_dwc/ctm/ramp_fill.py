"""Step 7: gap-filling strategies for ramp VDS timeseries.

Ramp detectors (PeMS Type ∈ {OR, FR}) suffer more outages than mainline
detectors, so the raw 5-min CSVs Step 3 downloads often have NaN samples
that the scenario adapters (:mod:`ctm.scenario`) can't consume
directly. This module provides two pure-function fillers, both sharing
the signature::

    fill(df, *, flow_col, time_col="timestamp", ...) -> pd.DataFrame

Each returns a copy of ``df`` with NaN values in ``flow_col`` filled and
all other columns passed through unchanged. They operate on the **full
downloaded timeseries** (not the sim window) -- the historical-average
binning uses every non-NaN sample available in the input to estimate
per-(weekday, time-of-day) statistics, so a wider input window gives
better stats.

The fillers
===========

* :func:`persistence_fill` (Level 1) -- forward-fill with the last known
  value. Best for short outages (a few samples) where the underlying
  rate is unlikely to have moved much.
* :func:`gap_aware_fill` -- outage-duration-aware composition: short gaps
  (<= ``short_gap_max`` samples with a prior value) are persisted,
  everything else takes the per-(weekday, time-of-day) historical
  average computed by :func:`_bin_stats`. This is the always-on gap fill
  used by ``scripts/build_ctm_ramp_scenario.py`` for both ramp and
  mainline series.
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


# ---- Gap-aware composition -----------------------------------------------


def gap_aware_fill(
    df: pd.DataFrame, *,
    flow_col: str,
    time_col: str = "timestamp",
    short_gap_max: int = 2,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Persist short gaps and historical-average the rest, by outage length.

    Composes :func:`persistence_fill` and :func:`historical_average_fill`:
    a NaN run of ``<= short_gap_max`` samples with a prior observed value
    is forward-filled (the rate is unlikely to have moved much), and every
    other gap -- longer runs, and short runs at the very start of the
    record with no prior value -- is filled with the per-(weekday,
    time-of-day) historical mean. Bin statistics come from the **original**
    ``df`` (or ``history_df``), so persisted values don't pollute them.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-VDS 5-minute timeseries; must have ``flow_col`` and
        ``time_col`` columns. Assumed sorted by ``time_col`` ascending.
    flow_col, time_col : str
        Column names; same defaults as :func:`persistence_fill`.
    short_gap_max : int
        Longest NaN run (in 5-min samples) still filled by persistence.
        Default 2 (i.e. gaps of 1-2 samples / up to 10 min).
    history_df : pandas.DataFrame, optional
        Source of the historical-average bin statistics. Defaults to
        ``df`` itself.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with NaN ``flow_col`` values filled. Rows whose
        historical bin has no non-NaN history remain NaN (with the same
        warning as :func:`historical_average_fill`).
    """
    persisted = persistence_fill(
        df, flow_col=flow_col, time_col=time_col,
        max_gap_minutes=short_gap_max * 5.0,
    )
    src = df if history_df is None else history_df
    stats = _bin_stats(src, flow_col=flow_col, time_col=time_col)
    return _apply_bin_fill(persisted, stats, flow_col=flow_col, time_col=time_col)
