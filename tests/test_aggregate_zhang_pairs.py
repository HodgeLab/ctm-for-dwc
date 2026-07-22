"""Tests for scripts/aggregate_zhang_pairs.py (offline pair-sweep evaluation):
the flow-amplitude-ratio metric, signed-x binning, and aggregation over
result.json / pair_manifest.json fixtures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from aggregate_zhang_pairs import (  # noqa: E402
    METRICS,
    RAMPS,
    STAGES,
    _binned_stats,
    flow_amplitude_ratio,
)
from aggregate_zhang_pairs import main as aggregate_main  # noqa: E402

_ALL_COLS = [*METRICS, *(f"{m}_{r}" for m in METRICS for r in RAMPS), "rmse"]


# --------------------------------------------------------------------------- #
# flow_amplitude_ratio
# --------------------------------------------------------------------------- #
def test_flow_ratio_recovers_amplitude_factor():
    rng = np.random.default_rng(0)
    tgt = rng.uniform(100, 1000, 800)
    src = 2.0 * tgt                                # source uniformly 2x hotter
    np.testing.assert_allclose(flow_amplitude_ratio(src, tgt), np.log(2.0))
    np.testing.assert_allclose(flow_amplitude_ratio(tgt, src), -np.log(2.0))


def test_flow_ratio_uses_top_k_ranks_only():
    # identical upper tails, wildly different bodies -> ratio 0
    src = np.concatenate([np.full(10, 1000.0), np.full(500, 1.0)])
    tgt = np.concatenate([np.full(10, 1000.0), np.full(500, 900.0)])
    assert flow_amplitude_ratio(src, tgt, k=10) == 0.0


def test_flow_ratio_shorter_series_and_zero_ramp():
    assert np.isfinite(flow_amplitude_ratio([10.0, 5.0], [10.0, 5.0, 1.0]))
    assert np.isnan(flow_amplitude_ratio(np.zeros(600), np.ones(600)))


# --------------------------------------------------------------------------- #
# _binned_stats
# --------------------------------------------------------------------------- #
def test_binned_stats_groups_by_fixed_width_bins():
    x = np.array([0.02, 0.04, 0.11, 0.13, 0.31])   # bins [0,.05), [.1,.15), [.3,.35)
    y = np.array([1.0, 3.0, 10.0, 20.0, 7.0])
    centers, means, stds = _binned_stats(x, y, bin_width=0.05)
    np.testing.assert_allclose(centers, [0.025, 0.125, 0.325])
    np.testing.assert_allclose(means, [2.0, 15.0, 7.0])
    np.testing.assert_allclose(stds, [1.0, 5.0, 0.0])   # single-point bin -> 0


def test_binned_stats_handles_negative_x():
    # log-ratio axes are signed; bins align to multiples of the width
    x = np.array([-0.6, -0.55, 0.1])               # bins [-0.75,-0.5), [0,0.25)
    y = np.array([4.0, 6.0, 3.0])
    centers, means, _ = _binned_stats(x, y, bin_width=0.25)
    assert (centers[0] < 0) and len(centers) == 2
    np.testing.assert_allclose(means, [5.0, 3.0])


# --------------------------------------------------------------------------- #
# aggregation over fixtures
# --------------------------------------------------------------------------- #
def _fake_result(out_dir, source, target, mmd_dda, dda_nrmse):
    def metrics(v):
        return {c: v if c.startswith("nrmse") else
                (1 - v if c.startswith("r2") else
                 (v / 10 if c.startswith("nbias") else 100.0))
                for c in _ALL_COLS}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps({
        "config": {"source_stretch": source, "target_stretch": target},
        "source_target_mmd": {"backbone": mmd_dda + 0.1, "dda": mmd_dda},
        "target_metrics": {
            "backbone": metrics(dda_nrmse + 0.2),
            "dda": metrics(dda_nrmse),
            "dda_mt": {
                "per_survey_day": [],
                "mean": metrics(dda_nrmse - 0.05),
                "std": {c: 0.01 for c in _ALL_COLS},
            },
        },
    }))


def test_aggregator_builds_csv_and_plots(tmp_path):
    results = tmp_path / "sweep"
    _fake_result(results / "a__b", "880_N:3", "880_N:11", mmd_dda=0.3, dda_nrmse=0.5)
    _fake_result(results / "a__c", "880_N:3", "880_N:13", mmd_dda=0.8, dda_nrmse=0.7)
    rc = aggregate_main(["--results-dir", str(results)])
    assert rc == 0

    out = results / "aggregate"
    pairs = pd.read_csv(out / "pairs.csv")
    assert len(pairs) == 2 * len(STAGES)
    dda = pairs[pairs["stage"] == "dda"]
    mt = pairs[pairs["stage"] == "dda_mt"]
    # DDA rows carry the stage value with no std; MT rows carry mean + std
    assert dda["nrmse_std"].isna().all()
    assert (mt["nrmse_std"] == 0.01).all()
    row = mt[mt["target"] == "880_N:11"].iloc[0]
    assert row["nrmse"] == 0.45 and row["mmd_dda"] == 0.3
    # without data paths the ratio columns are NaN and no flow-ratio plots
    assert pairs["flow_ratio_on"].isna().all()
    for metric in METRICS:
        assert (out / f"{metric}_vs_mmd.png").exists()
        assert not (out / f"{metric}_on_vs_flowratio.png").exists()


def test_aggregator_skips_incomplete_results(tmp_path, capsys):
    results = tmp_path / "sweep"
    _fake_result(results / "a__b", "880_N:3", "880_N:11", mmd_dda=0.3, dda_nrmse=0.5)
    bad = results / "a__c"; bad.mkdir(parents=True)
    (bad / "result.json").write_text(json.dumps({"config": {}, "arms": {}}))
    rc = aggregate_main(["--results-dir", str(results)])
    assert rc == 0
    assert "skipping" in capsys.readouterr().out
    pairs = pd.read_csv(results / "aggregate" / "pairs.csv")
    assert len(pairs) == len(STAGES)                  # only the complete pair


def test_aggregator_empty_dir_fails_cleanly(tmp_path):
    assert aggregate_main(["--results-dir", str(tmp_path)]) == 1


def test_aggregator_requires_both_data_paths(tmp_path):
    assert aggregate_main(["--results-dir", str(tmp_path),
                           "--stretches", str(tmp_path)]) == 1


# --------------------------------------------------------------------------- #
# flow ratios from pair_manifest.json + timeseries
# --------------------------------------------------------------------------- #
_N = 3 * 288


def _station(ts_dir, sid, base, seed):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-06-05 00:00", periods=_N, freq="5min")
    pd.DataFrame({
        "timestamp": ts,
        "total_flow_[veh/5-min]": base + 5.0 * rng.random(_N),
        "avg_speed_[mph]": np.full(_N, 60.0),
        "avg_occupancy_[%]": np.full(_N, 10.0),
        "pct_observed": np.full(_N, 100.0),
    }).to_csv(ts_dir / f"{sid}.csv", index=False)


def test_aggregator_computes_flow_ratios_from_manifest(tmp_path):
    sd = tmp_path / "stretches"; sd.mkdir()
    ts = tmp_path / "ts"; ts.mkdir()
    # source on-ramp ~half the target's; source off-ramp ~double the target's
    rows = []
    for stid, (up, down, on, off, on_base, off_base) in enumerate(
            [(100, 200, 10, 20, 20.0, 10.0), (300, 400, 30, 40, 40.0, 5.0)]):
        rows.append(dict(stretch_id=stid, up_ml_id=up, down_ml_id=down,
                         on_ids=str(on), off_ids=str(off), review_ids="",
                         n_on=1, n_off=1, config_type="c"))
        _station(ts, up, 100.0, seed=up); _station(ts, down, 120.0, seed=down)
        _station(ts, on, on_base, seed=on); _station(ts, off, off_base, seed=off)
    pd.DataFrame(rows).to_csv(sd / "880_N.csv", index=False)

    results = tmp_path / "sweep"
    pair = results / "880_N-0__880_N-1"
    _fake_result(pair, "880_N:0", "880_N:1", mmd_dda=0.3, dda_nrmse=0.5)
    (pair / "pair_manifest.json").write_text(json.dumps({
        "source_stretch": "880_N:0", "target_stretch": "880_N:1",
        "pct_observed_threshold": 50.0, "ramp_pct_floor": 0.0,
        "window_size": 5, "weekdays_only": False,
        "record_start": None, "record_end": None,
    }))

    rc = aggregate_main(["--results-dir", str(results),
                         "--stretches", str(sd), "--timeseries-dir", str(ts)])
    assert rc == 0
    out = results / "aggregate"
    pairs = pd.read_csv(out / "pairs.csv")
    # on: source ~20 vs target ~40 -> ratio ~ log(0.5) < 0
    # off: source ~10 vs target ~5 -> ratio ~ log(2) > 0
    assert (pairs["flow_ratio_on"] < -0.4).all()
    assert (pairs["flow_ratio_off"] > 0.4).all()
    for metric in METRICS:
        for ramp in RAMPS:
            assert (out / f"{metric}_{ramp}_vs_flowratio.png").exists()
