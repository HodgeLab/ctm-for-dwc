"""Integration test for scripts/train_ramp_flow_zhang.py (CLI wiring).

Builds a tiny two-stretch case (source + target, three days each) and runs
the entrypoint end-to-end with a tiny network; W&B is forced offline into the
temp dir so nothing is published.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from transportation_models.utils.ramp_flow_estimation.zhang import ZhangEstimator

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from train_ramp_flow_zhang import _survey_days, main  # noqa: E402

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


def test_cli_pooled_default_skips_dda(tmp_path, monkeypatch):
    """Default recipe: pool sources (--stretch-ids minus target), no DDA, +MT."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    sd, ts = _build_case(tmp_path)
    out_dir = tmp_path / "out"
    rc = main([
        "--stretches", str(sd), "--timeseries-dir", str(ts),
        "--stretch-ids", "880_N:0", "880_N:1", "880_N:2",
        "--target-stretch", "880_N:2",
        "--embed-dim", "8", "--hidden-size", "8", "--adapt-dim", "8",
        "--max-epochs", "2", "--n-survey-days", "2",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0

    manifest = json.loads((out_dir / "pair_manifest.json").read_text())
    assert manifest["mode"] == "pooled" and manifest["dda"] is False
    assert manifest["source_stretch"] is None
    assert manifest["source_stretches"] == ["880_N:0", "880_N:1"]   # target dropped
    assert set(manifest["source_digests"]) == {"880_N:0", "880_N:1"}

    result = json.loads((out_dir / "result.json").read_text())
    # No DDA stage anywhere; MT folds onto the backbone as the "mt" stage.
    assert "dda" not in result["source_target_mmd"]
    tm = result["target_metrics"]
    assert "dda" not in tm and "dda_mt" not in tm
    assert np.isfinite(tm["backbone"]["nrmse"])
    per_day = tm["mt"]["per_survey_day"]
    assert len(per_day) == 2 and len({d["survey_day"] for d in per_day}) == 2
    assert np.isfinite(tm["mt"]["mean"]["nrmse"])

    est = ZhangEstimator.load(out_dir / "zhang.pt")
    assert est._net.bn is None                            # no DDA -> no BN
    assert est._h_mt is not None
    r_hat, s_hat = est.predict_flows(
        np.random.default_rng(0).uniform(0, 1, (5, 50)))
    assert (r_hat >= 0).all() and (s_hat >= 0).all()


def test_cli_single_dda_is_paper_faithful(tmp_path, monkeypatch):
    """--mode single --dda reproduces the backbone -> +DDA -> +DDA+MT pipeline."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    sd, ts = _build_case(tmp_path)
    out_dir = tmp_path / "out"
    rc = main([
        "--stretches", str(sd), "--timeseries-dir", str(ts),
        "--stretch-ids", "880_N:0", "880_N:1", "--mode", "single", "--dda",
        "--source-stretch", "880_N:0", "--target-stretch", "880_N:1",
        "--embed-dim", "8", "--hidden-size", "8", "--adapt-dim", "8",
        "--max-epochs", "2", "--dda-epochs", "2", "--n-survey-days", "2",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0

    manifest = json.loads((out_dir / "pair_manifest.json").read_text())
    assert manifest["mode"] == "single" and manifest["dda"] is True
    assert manifest["source_stretches"] == ["880_N:0"]

    result = json.loads((out_dir / "result.json").read_text())
    for stage in ("backbone", "dda"):
        assert np.isfinite(result["source_target_mmd"][stage])
    tm = result["target_metrics"]
    assert np.isfinite(tm["dda"]["nrmse"])
    assert np.isfinite(tm["dda_mt"]["mean"]["nrmse"])

    est = ZhangEstimator.load(out_dir / "zhang.pt")
    assert est._net.bn is not None                        # DDA inserts BN
    assert est._h_mt is not None


def test_cli_single_mode_requires_source_stretch(tmp_path):
    sd, ts = _build_case(tmp_path)
    try:
        main(["--stretches", str(sd), "--timeseries-dir", str(ts),
              "--stretch-ids", "880_N:0", "880_N:1", "--mode", "single",
              "--target-stretch", "880_N:1"])
    except SystemExit:
        return
    raise AssertionError("expected --mode single without --source-stretch to fail")


def test_survey_days_are_weekdays_only():
    # 9 days starting Friday 2023-06-09: Sat/Sun (x2) must never be drawn
    days = pd.date_range("2023-06-09", periods=9, freq="D").to_numpy("datetime64[D]")
    dates = np.repeat(days, 100)                          # equal coverage
    for seed in range(5):
        chosen = _survey_days(dates, 5, seed)
        assert (pd.DatetimeIndex(chosen).dayofweek < 5).all()
        assert len(chosen) == 5                           # 5 weekdays exist


def test_cli_unknown_stretch_fails_cleanly(tmp_path):
    sd, ts = _build_case(tmp_path)
    try:
        main(["--stretches", str(sd), "--timeseries-dir", str(ts),
              "--stretch-ids", "880_N:9", "880_N:1", "--mode", "single",
              "--source-stretch", "880_N:9", "--target-stretch", "880_N:1"])
    except (SystemExit, ValueError):
        return
    raise AssertionError("expected a failure for an unknown stretch id")
