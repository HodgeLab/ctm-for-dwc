"""Tests for utils/ctm/ramp_validation.py (historical-average fill accuracy).

Tiny synthetic per-VDS CSVs in a temp dir; stretches frames carry only the
``on_ids`` / ``off_ids`` columns the roster consumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ctm.ramp_validation import ramp_fill_accuracy

_FLOW = "total_flow_[veh/5-min]"


def _write_series(tmp, vds_id, flows, *, start="2023-06-05 00:00", freq="1h"):
    ts = pd.date_range(start, periods=len(flows), freq=freq)
    pd.DataFrame({"timestamp": ts, _FLOW: flows}).to_csv(
        tmp / f"{vds_id}.csv", index=False
    )


def _stretches(rows) -> pd.DataFrame:
    """rows: list of (on_ids, off_ids) cell strings."""
    return pd.DataFrame(
        [{"stretch_id": i, "on_ids": on, "off_ids": off}
         for i, (on, off) in enumerate(rows)]
    )


def test_constant_series_fills_perfectly(tmp_path):
    # 8 weeks of a constant flow: every (weekday, time) bin mean equals the
    # truth, so the fill is exact and all error metrics are zero. (8 weeks =
    # 8 samples per bin, so a 20% random mask emptying a whole bin is
    # vanishingly unlikely.)
    _write_series(tmp_path, 10, np.full(24 * 56, 50.0))
    out = ramp_fill_accuracy(_stretches([("10", "")]), tmp_path, seed=0)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["ramp_type"] == "on"
    assert row["n_observed"] == 24 * 56
    assert row["n_masked"] == int(0.2 * 24 * 56)
    assert row["n_scored"] == row["n_masked"] and row["n_unfilled"] == 0
    for metric in ("rmse", "nrmse", "bias", "nbias"):
        assert row[metric] == pytest.approx(0.0, abs=1e-12)


def test_ramp_type_classification_and_dedupe(tmp_path):
    for vid in (10, 20):
        _write_series(tmp_path, vid, np.full(24 * 14, 30.0))
    # VDS 10 appears in two stretches' on_ids -> one row.
    out = ramp_fill_accuracy(
        _stretches([("10", "20"), ("10", "")]), tmp_path, seed=0
    )
    assert list(out["vds_id"]) == [10, 20]
    assert list(out["ramp_type"]) == ["on", "off"]


def test_conflicting_classification_warns_keeps_first(tmp_path):
    _write_series(tmp_path, 10, np.full(24 * 14, 30.0))
    with pytest.warns(UserWarning, match="both on- and off-ramp"):
        out = ramp_fill_accuracy(
            _stretches([("10", ""), ("", "10")]), tmp_path, seed=0
        )
    assert list(out["ramp_type"]) == ["on"]


def test_virtual_and_missing_ids_skipped(tmp_path):
    _write_series(tmp_path, 10, np.full(24 * 14, 30.0))
    # 'v99' is virtual (never parsed); 55 is real but has no CSV -> warning.
    with pytest.warns(UserWarning, match="no timeseries"):
        out = ramp_fill_accuracy(
            _stretches([("v99;10", "55")]), tmp_path, seed=0
        )
    assert list(out["vds_id"]) == [10]


def test_only_observed_samples_are_masked(tmp_path):
    flows = np.full(24 * 14, 40.0)
    flows[::3] = np.nan                        # a third of the series is a real gap
    _write_series(tmp_path, 10, flows)
    out = ramp_fill_accuracy(_stretches([("10", "")]), tmp_path, seed=0)
    row = out.iloc[0]
    n_obs = int(np.sum(~np.isnan(flows)))
    assert row["n_observed"] == n_obs
    assert row["n_masked"] == int(0.2 * n_obs)


def test_unfillable_bins_counted_not_scored(tmp_path):
    # One week of data: each (weekday, time) bin holds exactly one sample, so
    # masking a sample leaves its bin empty -> unfilled, not scored.
    _write_series(tmp_path, 10, np.full(24 * 7, 25.0))
    out = ramp_fill_accuracy(_stretches([("10", "")]), tmp_path, seed=0)
    row = out.iloc[0]
    assert row["n_unfilled"] == row["n_masked"]
    assert row["n_scored"] == 0
    assert np.isnan(row["rmse"]) and np.isnan(row["nbias"])


def test_same_seed_reproducible(tmp_path):
    rng = np.random.default_rng(3)
    _write_series(tmp_path, 10, rng.uniform(10, 100, 24 * 28))
    stretches = _stretches([("10", "")])
    a = ramp_fill_accuracy(stretches, tmp_path, seed=7)
    b = ramp_fill_accuracy(stretches, tmp_path, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_known_error_two_week_alternation(tmp_path):
    # Two weeks, week1=10 / week2=30 at identical bin times. Masking one
    # sample of a bin makes the fill equal the other week's value, so every
    # scored error is +/-20 -> RMSE exactly 20.
    _write_series(tmp_path, 10, np.concatenate([np.full(24 * 7, 10.0),
                                                np.full(24 * 7, 30.0)]))
    out = ramp_fill_accuracy(_stretches([("10", "")]), tmp_path,
                             mask_rate=0.1, seed=0)
    row = out.iloc[0]
    assert row["n_scored"] > 0
    assert row["rmse"] == pytest.approx(20.0)


def test_bad_mask_rate_raises(tmp_path):
    with pytest.raises(ValueError, match="mask_rate"):
        ramp_fill_accuracy(_stretches([("10", "")]), tmp_path, mask_rate=0.0)


def test_cli_writes_averaged_and_trial_tables(tmp_path):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    from validate_ramp_flow import main

    ts_dir = tmp_path / "ts"; ts_dir.mkdir()
    _write_series(ts_dir, 10, np.full(24 * 28, 50.0))
    _write_series(ts_dir, 20, np.full(24 * 28, 80.0))
    sd = tmp_path / "stretches"; sd.mkdir()
    _stretches([("10", "20")]).to_csv(sd / "880_N.csv", index=False)
    out = tmp_path / "accuracy.csv"

    rc = main(["--stretches", str(sd), "--timeseries-dir", str(ts_dir),
               "--seeds", "0", "1", "--out", str(out)])
    assert rc == 0
    averaged = pd.read_csv(out)
    assert list(averaged["vds_id"]) == [10, 20]
    assert set(averaged["ramp_type"]) == {"on", "off"}
    assert {"rmse", "nrmse", "bias", "nbias"} <= set(averaged.columns)
    trials = pd.read_csv(tmp_path / "accuracy_trials.csv")
    assert sorted(trials["seed"].unique()) == [0, 1]
    assert len(trials) == 4                    # 2 detectors x 2 seeds
