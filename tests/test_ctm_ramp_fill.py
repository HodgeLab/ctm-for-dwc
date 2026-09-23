"""Tests for the Step-7 gap-fillers in ``utils/ctm/ramp_fill.py``.

The fillers share signature ``fill(df, *, flow_col, ...) -> df``.
These tests exercise each one on a controlled synthetic series: a
known background pattern + an injected gap whose fill we can predict.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ctm_for_dwc.utils.ctm.ramp_fill import (
    gap_aware_fill,
    persistence_fill,
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
