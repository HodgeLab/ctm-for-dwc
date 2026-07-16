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
                                                (300, 400, 30, 40)]):
        rows.append(dict(stretch_id=stid, up_ml_id=up, down_ml_id=down,
                         on_ids=str(on), off_ids=str(off), review_ids="",
                         n_on=1, n_off=1, config_type="c"))
        _station(ts, up, 100.0, seed=up)
        _station(ts, down, 120.0, seed=down)
        _station(ts, on, 20.0, seed=on)
        _station(ts, off, 10.0, seed=off)
    pd.DataFrame(rows).to_csv(sd / "880_N.csv", index=False)
    return sd, ts


def test_cli_runs_all_stages_and_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    sd, ts = _build_case(tmp_path)
    out_dir = tmp_path / "out"
    rc = main([
        "--stretches", str(sd), "--timeseries-dir", str(ts),
        "--source-stretch", "880_N:0", "--target-stretch", "880_N:1",
        "--embed-dim", "8", "--hidden-size", "8", "--adapt-dim", "8",
        "--max-epochs", "2", "--dda-epochs", "2", "--n-survey-days", "2",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0

    manifest = json.loads((out_dir / "pair_manifest.json").read_text())
    assert manifest["source_stretch"] == "880_N:0"
    assert manifest["n_source_rows"] > 0 and manifest["n_target_rows"] > 0
    assert manifest["source_digest"] != manifest["target_digest"]

    result = json.loads((out_dir / "result.json").read_text())
    for stage in ("backbone", "dda"):                     # adaptation-layer MMD
        assert np.isfinite(result["source_target_mmd"][stage])
        assert result["source_target_mmd"][stage] >= 0.0
    tm = result["target_metrics"]
    for stage in ("backbone", "dda"):
        assert np.isfinite(tm[stage]["nrmse"])
    per_day = tm["dda_mt"]["per_survey_day"]
    assert len(per_day) == 2
    assert len({d["survey_day"] for d in per_day}) == 2   # distinct days
    for d in per_day:
        assert np.isfinite(d["h_r"]) and np.isfinite(d["h_s"])
    assert np.isfinite(tm["dda_mt"]["mean"]["nrmse"])
    assert np.isfinite(result["source_val_metrics"]["nrmse"])

    est = ZhangEstimator.load(out_dir / "zhang.pt")
    assert est._net.bn is not None                        # post-DDA checkpoint
    assert est._h_mt is not None
    r_hat, s_hat = est.predict_flows(
        np.random.default_rng(0).uniform(0, 1, (5, 50)))
    assert (r_hat >= 0).all() and (s_hat >= 0).all()


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
              "--source-stretch", "880_N:9", "--target-stretch", "880_N:1"])
    except (SystemExit, ValueError):
        return
    raise AssertionError("expected a failure for an unknown stretch id")
