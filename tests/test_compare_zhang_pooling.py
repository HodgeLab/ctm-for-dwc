"""Integration test for scripts/compare_zhang_pooling.py (the 2x2 pooled-vs-DDA
experiment) and unit test for scripts/aggregate_zhang_pooling.py.

The driver test builds a tiny three-stretch case and runs all four arms
end-to-end with a tiny network; W&B is forced offline. The aggregator test
runs over hand-built result.json fixtures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from aggregate_zhang_pooling import ARMS, METRICS  # noqa: E402
from aggregate_zhang_pooling import main as aggregate_main  # noqa: E402
from compare_zhang_pooling import main as compare_main  # noqa: E402

_DAYS = 3
_N = _DAYS * 288


def _station(ts_dir, sid, base, seed):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-06-05 00:00", periods=_N, freq="5min")
    pd.DataFrame({
        "timestamp": ts,
        "total_flow_[veh/5-min]": base + 20.0 * rng.random(_N),
        "avg_speed_[mph]": 55.0 + 10.0 * rng.random(_N),
        "avg_occupancy_[%]": np.full(_N, 10.0),
        "pct_observed": np.full(_N, 100.0),
    }).to_csv(ts_dir / f"{sid}.csv", index=False)


def _build_case(tmp_path):
    sd = tmp_path / "stretches"; sd.mkdir()
    ts = tmp_path / "ts"; ts.mkdir()
    rows = []
    for stid, (up, down, on, off) in enumerate([(100, 200, 10, 20),
                                                (300, 400, 30, 40),
                                                (500, 600, 50, 60)]):
        rows.append(dict(stretch_id=stid, up_ml_id=up, down_ml_id=down,
                         on_ids=str(on), off_ids=str(off), review_ids="",
                         n_on=1, n_off=1, config_type="c"))
        _station(ts, up, 100.0, seed=up)
        _station(ts, down, 120.0, seed=down)
        _station(ts, on, 20.0, seed=on)
        _station(ts, off, 10.0, seed=off)
    pd.DataFrame(rows).to_csv(sd / "880_N.csv", index=False)
    return sd, ts


# --------------------------------------------------------------------------- #
# compare_zhang_pooling.py
# --------------------------------------------------------------------------- #
def test_cli_runs_all_four_arms(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    sd, ts = _build_case(tmp_path)
    out_dir = tmp_path / "out"
    rc = compare_main([
        "--stretches", str(sd), "--timeseries-dir", str(ts),
        "--stretch-ids", "880_N:0", "880_N:1", "880_N:2",
        "--target-stretch", "880_N:2", "--source-stretch", "880_N:0",
        "--embed-dim", "8", "--hidden-size", "8", "--adapt-dim", "8",
        "--max-epochs", "2", "--dda-epochs", "2",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0

    result = json.loads((out_dir / "result.json").read_text())
    assert result["config"]["pool_stretches"] == ["880_N:0", "880_N:1"]
    arms = result["arms"]
    assert set(arms) == set(ARMS)
    for arm in ARMS:
        for m in METRICS:
            assert np.isfinite(arms[arm]["target_metrics"][m])
        assert np.isfinite(arms[arm]["pair_mmd"]) and arms[arm]["pair_mmd"] >= 0
    # the pooled arms train on strictly more data than the single arms
    assert arms["pooled"]["n_train_rows"] > arms["single"]["n_train_rows"]
    assert arms["pooled_dda"]["n_train_rows"] == arms["pooled"]["n_train_rows"]


def test_cli_rejects_target_not_in_ids(tmp_path):
    sd, ts = _build_case(tmp_path)
    rc = compare_main([
        "--stretches", str(sd), "--timeseries-dir", str(ts),
        "--stretch-ids", "880_N:0", "880_N:1",
        "--target-stretch", "880_N:2", "--source-stretch", "880_N:0",
    ])
    assert rc == 1


def test_cli_rejects_target_equals_source(tmp_path):
    sd, ts = _build_case(tmp_path)
    rc = compare_main([
        "--stretches", str(sd), "--timeseries-dir", str(ts),
        "--stretch-ids", "880_N:0", "880_N:1",
        "--target-stretch", "880_N:0", "--source-stretch", "880_N:0",
    ])
    assert rc == 1


# --------------------------------------------------------------------------- #
# aggregate_zhang_pooling.py
# --------------------------------------------------------------------------- #
def _fake_result(out_dir, target, source, mmd_single, offset=0.0):
    arms = {}
    for k, arm in enumerate(ARMS):
        arms[arm] = {
            "target_metrics": {"nrmse": 0.5 - 0.05 * k + offset,
                               "r2": 0.2 + 0.1 * k - offset,
                               "nbias": 0.01 * k,
                               "rmse": 100.0},
            "pair_mmd": mmd_single + 0.01 * k,
            "n_train_rows": 100 * (k + 1),
        }
    out_dir.mkdir(parents=True)
    (out_dir / "result.json").write_text(json.dumps({
        "config": {"target_stretch": target, "source_stretch": source},
        "arms": arms,
    }))


def test_aggregator_builds_csvs_and_plots(tmp_path, capsys):
    results = tmp_path / "sweep"
    _fake_result(results / "a__b", "880_N:3", "880_N:11", mmd_single=0.2)
    _fake_result(results / "a__c", "880_N:3", "880_N:13", mmd_single=0.9,
                 offset=0.1)
    rc = aggregate_main(["--results-dir", str(results)])
    assert rc == 0

    out = results / "aggregate"
    combos = pd.read_csv(out / "combos.csv")
    assert len(combos) == 2 * len(ARMS)
    # mmd_x is the single arm's pre-DDA MMD, shared by all arms of a combo
    for (_, _), grp in combos.groupby(["target", "source"]):
        assert grp["mmd_x"].nunique() == 1
        single = grp[grp["arm"] == "single"].iloc[0]
        assert single["mmd_x"] == single["pair_mmd"]

    summary = pd.read_csv(out / "summary.csv", index_col=0)
    assert list(summary.index) == list(ARMS)
    assert summary.loc["single", "nrmse_count"] == 2
    # single: mean of 0.5 and 0.6
    assert summary.loc["single", "nrmse_mean"] == 0.55

    for metric in METRICS:
        assert (out / f"{metric}_vs_mmd.png").exists()
        assert (out / f"{metric}_by_arm.png").exists()


def test_aggregator_empty_dir_fails_cleanly(tmp_path):
    assert aggregate_main(["--results-dir", str(tmp_path)]) == 1
