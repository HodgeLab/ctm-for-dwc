"""Tests for the three Step-7 gap-fillers in ``utils/ctm/ramp_flow.py``.

The fillers all share signature ``fill(df, *, flow_col, ...) -> df``.
These tests exercise each one on a controlled synthetic series: a
known background pattern + an injected gap whose fill we can predict.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ctm.ramp_flow import (
    gap_aware_fill,
    historical_average_fill,
    persistence_fill,
    stochastic_historical_fill,
)

FLOW_COL = "total_flow_[veh/5-min]"


def _flat_series(
    n_rows: int = 14 * 288,  # 2 weeks at 5-min
    flow: float = 100.0,
    start: str = "2022-01-01",
) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n_rows, freq="5min")
    return pd.DataFrame({"timestamp": ts, FLOW_COL: np.full(n_rows, flow)})


def _diurnal_series(n_weeks: int = 4) -> pd.DataFrame:
    """5-min series with a fixed (weekday, time-of-day) pattern.

    Sets the flow at each bin to ``100 * (weekday + 1) + tod_minutes``,
    so the historical-average filler has a predictable target value.
    """
    ts = pd.date_range("2022-01-03", periods=n_weeks * 7 * 288, freq="5min")
    weekday = ts.weekday.to_numpy()
    tod_min = ts.hour.to_numpy() * 60 + ts.minute.to_numpy()
    flow = 100.0 * (weekday + 1) + tod_min
    return pd.DataFrame({"timestamp": ts, FLOW_COL: flow})


# ---- persistence_fill ----------------------------------------------------


def test_persistence_fills_a_short_gap():
    df = _flat_series()
    df.loc[100:104, FLOW_COL] = np.nan
    out = persistence_fill(df, flow_col=FLOW_COL)
    assert not out[FLOW_COL].isna().any()
    # Filled values equal the last-known value (also 100 here).
    np.testing.assert_allclose(out.loc[100:104, FLOW_COL].to_numpy(), 100.0)


def test_persistence_leaves_leading_nan_alone():
    df = _flat_series()
    df.loc[0:5, FLOW_COL] = np.nan  # No prior known value.
    out = persistence_fill(df, flow_col=FLOW_COL)
    assert out.loc[0:5, FLOW_COL].isna().all()
    assert not out.loc[6:, FLOW_COL].isna().any()


def test_persistence_max_gap_skips_long_runs():
    """A 60-minute gap (12 samples) is filled when max_gap_minutes=60 but
    a 100-minute gap (20 samples) stays NaN under the same setting."""
    df = _flat_series()
    df.loc[100:111, FLOW_COL] = np.nan   # 12 samples = 60 min
    df.loc[500:519, FLOW_COL] = np.nan   # 20 samples = 100 min
    out = persistence_fill(df, flow_col=FLOW_COL, max_gap_minutes=60.0)
    assert not out.loc[100:111, FLOW_COL].isna().any()
    assert out.loc[500:519, FLOW_COL].isna().all()


# ---- historical_average_fill --------------------------------------------


def test_historical_average_recovers_per_bin_mean():
    """A gap at (Tue 10:00) should be filled with the mean of all other
    samples at (Tue 10:00)."""
    df = _diurnal_series(n_weeks=4)
    # Pick one specific (weekday=Tue, tod=10:00) and NaN it out.
    target_mask = (df["timestamp"].dt.weekday == 1) & (
        df["timestamp"].dt.hour == 10
    ) & (df["timestamp"].dt.minute == 0)
    target_value = df.loc[target_mask, FLOW_COL].iloc[0]
    df.loc[df.index[target_mask][0], FLOW_COL] = np.nan
    out = historical_average_fill(df, flow_col=FLOW_COL)
    filled = out.loc[df.index[target_mask][0], FLOW_COL]
    assert filled == pytest.approx(target_value)


def test_historical_average_uses_full_input_not_window():
    """Statistics come from the whole df, not a subset -- so filling a NaN
    near the start of the series uses samples from later weeks too."""
    df = _diurnal_series(n_weeks=4)
    target_idx = df.index[(df["timestamp"].dt.weekday == 0)
                          & (df["timestamp"].dt.hour == 0)
                          & (df["timestamp"].dt.minute == 0)][0]
    expected = df.loc[
        (df["timestamp"].dt.weekday == 0)
        & (df["timestamp"].dt.hour == 0)
        & (df["timestamp"].dt.minute == 0),
        FLOW_COL,
    ].drop(index=target_idx).mean()
    df.loc[target_idx, FLOW_COL] = np.nan
    out = historical_average_fill(df, flow_col=FLOW_COL)
    assert out.loc[target_idx, FLOW_COL] == pytest.approx(expected)


def test_historical_average_warns_on_missing_bin():
    """Bins with zero non-NaN samples leave the row NaN with a warning."""
    df = _diurnal_series(n_weeks=4)
    # Wipe every (Tue, 10:00) sample so the bin has no history.
    target_mask = (df["timestamp"].dt.weekday == 1) & (
        df["timestamp"].dt.hour == 10
    ) & (df["timestamp"].dt.minute == 0)
    df.loc[target_mask, FLOW_COL] = np.nan
    with pytest.warns(RuntimeWarning, match="no non-NaN history"):
        out = historical_average_fill(df, flow_col=FLOW_COL)
    assert out.loc[target_mask, FLOW_COL].isna().all()


def test_historical_average_uses_separate_history_df():
    """history_df= overrides where the bin stats come from."""
    df = _flat_series(n_rows=288)            # 1 day of NaN
    df[FLOW_COL] = np.nan
    history = _flat_series(n_rows=14 * 288, flow=250.0)
    out = historical_average_fill(df, flow_col=FLOW_COL, history_df=history)
    assert not out[FLOW_COL].isna().any()
    np.testing.assert_allclose(out[FLOW_COL].to_numpy(), 250.0)


# ---- stochastic_historical_fill -----------------------------------------


def test_stochastic_fill_is_reproducible_with_default_seed():
    df1 = _diurnal_series(n_weeks=4)
    df2 = df1.copy()
    target_idx = df1.index[(df1["timestamp"].dt.weekday == 1)
                           & (df1["timestamp"].dt.hour == 10)
                           & (df1["timestamp"].dt.minute == 0)][0]
    df1.loc[target_idx, FLOW_COL] = np.nan
    df2.loc[target_idx, FLOW_COL] = np.nan
    out1 = stochastic_historical_fill(df1, flow_col=FLOW_COL)
    out2 = stochastic_historical_fill(df2, flow_col=FLOW_COL)
    assert out1.loc[target_idx, FLOW_COL] == out2.loc[target_idx, FLOW_COL]


def test_stochastic_fill_different_seed_yields_different_draws():
    """Different seeds must produce different draws when bins have nonzero
    variance. _diurnal_series alone has zero per-bin variance (one value
    per (weekday, tod)), so we add Gaussian noise to give the stddev
    something to work with."""
    df = _diurnal_series(n_weeks=4)
    rng = np.random.default_rng(99)
    df[FLOW_COL] = df[FLOW_COL] + rng.normal(0.0, 25.0, size=len(df))
    # Inject 50 NaNs so we have enough samples for a statistical comparison.
    nan_idx = df.index[::200][:50]
    df.loc[nan_idx, FLOW_COL] = np.nan
    out_a = stochastic_historical_fill(df, flow_col=FLOW_COL, seed=0)
    out_b = stochastic_historical_fill(df, flow_col=FLOW_COL, seed=1)
    assert not np.allclose(
        out_a.loc[nan_idx, FLOW_COL].to_numpy(),
        out_b.loc[nan_idx, FLOW_COL].to_numpy(),
    )


def test_stochastic_fill_clips_negative_draws():
    """A bin with a near-zero mean + large stddev would produce negative
    draws; the filler must clip them to 0."""
    # Two-week series: most samples at 1.0 to keep stddev ~ 0, then one
    # bin where stddev is huge so draws can swing negative.
    ts = pd.date_range("2022-01-03", periods=14 * 288, freq="5min")
    flow = np.full(len(ts), 1.0)
    target_mask = (ts.weekday == 1) & (ts.hour == 10) & (ts.minute == 0)
    # Inject extreme values into that bin's history to drive a wide
    # stddev (mean stays small, stddev large -> negative samples likely).
    flow[target_mask] = np.array([0.0, 100.0])  # n=2 samples per bin
    df = pd.DataFrame({"timestamp": ts, FLOW_COL: flow})
    # NaN the first occurrence so the filler has to draw.
    first = np.where(target_mask)[0][0]
    df.loc[first, FLOW_COL] = np.nan
    out = stochastic_historical_fill(df, flow_col=FLOW_COL, seed=0)
    # The filled value must be >= 0 regardless of the draw.
    assert out.loc[first, FLOW_COL] >= 0.0


def test_stochastic_fill_falls_back_to_mean_with_few_samples():
    """A bin with n=1 non-NaN sample has no stddev; the filler must fall
    back to the mean (== the single observed value)."""
    ts = pd.date_range("2022-01-03", periods=2 * 7 * 288, freq="5min")
    flow = np.full(len(ts), 50.0)
    df = pd.DataFrame({"timestamp": ts, FLOW_COL: flow})
    # Pick a (weekday=Sun, tod=23:55) bin where only week 1 has a sample;
    # week 2's value gets NaN'd before being seen.
    target_mask = (df["timestamp"].dt.weekday == 6) & (
        df["timestamp"].dt.hour == 23
    ) & (df["timestamp"].dt.minute == 55)
    target_idx = df.index[target_mask].tolist()
    df.loc[target_idx[1], FLOW_COL] = np.nan
    # Suppress any "missing-bin" RuntimeWarnings from unrelated bins to
    # focus on the deterministic-fallback behavior.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = stochastic_historical_fill(df, flow_col=FLOW_COL, seed=0)
    # With n=1 we have mean=50, stddev=NaN -> fallback to mean.
    assert out.loc[target_idx[1], FLOW_COL] == pytest.approx(50.0)


# ---- gap_aware_fill ------------------------------------------------------

# _diurnal_series gives each (weekday, tod) bin a distinct deterministic
# value and consecutive samples differ by 5 (tod_min step), so a persisted
# fill (== the neighbour) is always distinguishable from a historical-average
# fill (== the sample's own original value).

_INTERIOR = 3 * 288 + 100  # a sample in week 3, away from record edges


def test_gap_aware_persists_one_sample_gap():
    """A single-sample interior gap is forward-filled, not historical-averaged."""
    df = _diurnal_series(n_weeks=4)
    prev_val = df.loc[_INTERIOR - 1, FLOW_COL]
    own_mean = df.loc[_INTERIOR, FLOW_COL]           # bin mean == original value
    df.loc[_INTERIOR, FLOW_COL] = np.nan
    out = gap_aware_fill(df, flow_col=FLOW_COL)
    assert out.loc[_INTERIOR, FLOW_COL] == pytest.approx(prev_val)
    assert prev_val != pytest.approx(own_mean)       # the two paths differ here


def test_gap_aware_persists_two_sample_gap():
    """A 2-sample interior gap is forward-filled from the last known value."""
    df = _diurnal_series(n_weeks=4)
    prev_val = df.loc[_INTERIOR - 1, FLOW_COL]
    df.loc[_INTERIOR:_INTERIOR + 1, FLOW_COL] = np.nan
    out = gap_aware_fill(df, flow_col=FLOW_COL)
    np.testing.assert_allclose(
        out.loc[_INTERIOR:_INTERIOR + 1, FLOW_COL].to_numpy(), prev_val)


def test_gap_aware_historical_averages_long_gap():
    """A 3-sample gap exceeds short_gap_max, so each row takes its bin mean
    (== the original value here), not the pre-gap persistence value."""
    df = _diurnal_series(n_weeks=4)
    original = df[FLOW_COL].copy()
    df.loc[_INTERIOR:_INTERIOR + 2, FLOW_COL] = np.nan
    out = gap_aware_fill(df, flow_col=FLOW_COL)
    np.testing.assert_allclose(
        out.loc[_INTERIOR:_INTERIOR + 2, FLOW_COL].to_numpy(),
        original.loc[_INTERIOR:_INTERIOR + 2].to_numpy())


def test_gap_aware_leading_short_gap_falls_through_to_historical():
    """A short gap at the record start has no prior value to persist, so it
    falls through to the historical average instead of staying NaN."""
    df = _diurnal_series(n_weeks=4)
    original = df[FLOW_COL].copy()
    df.loc[0:1, FLOW_COL] = np.nan
    out = gap_aware_fill(df, flow_col=FLOW_COL)
    np.testing.assert_allclose(
        out.loc[0:1, FLOW_COL].to_numpy(), original.loc[0:1].to_numpy())
