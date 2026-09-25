"""Hand-calc and property tests for the DWC temporal-aggregation core.

Covers the pure numeric helpers in ``dwc/aggregation.py``: energy-
conserving re-binning, window-average power (peak attenuation), mile-segment
column grouping, and load-duration-curve divergence.
"""

import numpy as np
import pytest

from ctm_for_dwc.dwc import aggregation as agg

_METERS_PER_MILE = 1609.344


# ---------------------------------------------------------------------------
# rebin_sum (energy conservation)
# ---------------------------------------------------------------------------


def test_rebin_sum_exact_blocks():
    binned, counts = agg.rebin_sum(np.array([1.0, 2, 3, 4, 5, 6]), 2)
    np.testing.assert_array_equal(binned, [3.0, 7.0, 11.0])
    np.testing.assert_array_equal(counts, [2.0, 2.0, 2.0])


def test_rebin_sum_trailing_partial_bin_conserves_total():
    x = np.array([1.0, 2, 3, 4, 5])  # 5 not divisible by 2
    binned, counts = agg.rebin_sum(x, 2)
    np.testing.assert_array_equal(binned, [3.0, 7.0, 5.0])
    np.testing.assert_array_equal(counts, [2.0, 2.0, 1.0])
    assert binned.sum() == x.sum()  # energy conserved exactly


def test_rebin_sum_rejects_bad_bin_size():
    with pytest.raises(ValueError):
        agg.rebin_sum(np.arange(4.0), 0)


# ---------------------------------------------------------------------------
# window_power (peak attenuation is the whole point)
# ---------------------------------------------------------------------------


def test_window_power_native_is_energy_over_dt():
    e = np.array([0.0, 10.0, 0.0, 0.0])
    power, counts = agg.window_power(e, steps_per_bin=1, dt_h=1.0)
    np.testing.assert_allclose(power, [0.0, 10.0, 0.0, 0.0])
    np.testing.assert_array_equal(counts, [1, 1, 1, 1])


def test_window_power_averages_and_shaves_peak():
    """A 10 W spike averaged over a 2-step window drops to 5 W (50% shave)."""
    e = np.array([0.0, 10.0, 0.0, 0.0])  # Wh per step, dt = 1 h
    power, _ = agg.window_power(e, steps_per_bin=2, dt_h=1.0)
    np.testing.assert_allclose(power, [5.0, 0.0])
    native, _ = agg.window_power(e, steps_per_bin=1, dt_h=1.0)
    assert agg.pct_attenuation(native.max(), power.max()) == pytest.approx(50.0)


def test_window_power_conserves_energy():
    e = np.array([1.0, 4.0, 2.0, 7.0, 3.0, 5.0])
    dt_h = 0.25
    power, counts = agg.window_power(e, steps_per_bin=3, dt_h=dt_h)
    assert (power * counts * dt_h).sum() == pytest.approx(e.sum())


def test_bin_center_times():
    counts = np.array([2.0, 2.0, 1.0])
    t = agg.bin_center_times_h(counts, dt_h=1.0)
    np.testing.assert_allclose(t, [1.0, 3.0, 4.5])


# ---------------------------------------------------------------------------
# peak / pct_attenuation
# ---------------------------------------------------------------------------


def test_peak_returns_value_and_index():
    val, idx = agg.peak(np.array([3.0, 9.0, 1.0]))
    assert (val, idx) == (9.0, 1)


def test_pct_attenuation_zero_baseline_is_zero():
    assert agg.pct_attenuation(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# mile_segment_columns
# ---------------------------------------------------------------------------


def test_mile_segment_columns_groups_by_mile():
    # positions at 0, 0.5, 1.0, 1.5 mi over a 2-mi corridor
    position_m = np.array([0.0, 0.5, 1.0, 1.5]) * _METERS_PER_MILE
    segments = agg.mile_segment_columns(position_m, corridor_length_m=2.0 * _METERS_PER_MILE)
    assert set(segments) == {0, 1}
    np.testing.assert_array_equal(segments[0], [0, 1])
    np.testing.assert_array_equal(segments[1], [2, 3])


def test_mile_segment_columns_drops_off_corridor():
    # a negative (approach) column and one past corridor end are dropped
    position_m = np.array([-0.25, 0.25, 0.75, 2.5]) * _METERS_PER_MILE
    segments = agg.mile_segment_columns(position_m, corridor_length_m=1.0 * _METERS_PER_MILE)
    assert set(segments) == {0}
    np.testing.assert_array_equal(segments[0], [1, 2])


# ---------------------------------------------------------------------------
# load-duration curve + divergence
# ---------------------------------------------------------------------------


def test_load_duration_curve_is_non_increasing():
    xs, ldc = agg.load_duration_curve(
        np.array([1.0, 3.0, 2.0]), np.ones(3), n_points=64
    )
    assert np.all(np.diff(ldc) <= 1e-12)
    assert ldc[0] == pytest.approx(3.0)  # highest-power sample at the left


def test_ldc_divergence_zero_for_identical_curves():
    p = np.array([1.0, 5.0, 2.0, 4.0])
    w = np.ones(4)
    assert agg.ldc_divergence(p, w, p, w) == pytest.approx(0.0, abs=1e-12)


def test_ldc_divergence_positive_when_flattened():
    """A peaky profile vs its energy-preserving flat average diverges > 0."""
    peaky = np.array([0.0, 8.0, 0.0, 0.0])
    flat = np.full(4, 2.0)  # same total energy, flat
    w = np.ones(4)
    assert agg.ldc_divergence(peaky, w, flat, w) > 0.0
