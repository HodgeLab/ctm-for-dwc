"""Integration test for scripts/evaluate_ramp_flow_estimation.py (CLI wiring).

Builds a tiny 3-stretch case (temp stretches CSV, station metadata, timeseries)
and runs the entrypoint end-to-end with a small RF.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from evaluate_ramp_flow_estimation import main  # noqa: E402


def _station(ts_dir, sid, flow, n=8):
    ts = pd.date_range("2023-06-01 00:00", periods=n, freq="5min")
    pd.DataFrame({
        "timestamp": ts,
        "total_flow_[veh/5-min]": np.full(n, float(flow)),
        "avg_speed_[mph]": np.full(n, 60.0),
        "avg_occupancy_[%]": np.full(n, 10.0),
        "pct_observed": np.full(n, 100.0),
    }).to_csv(ts_dir / f"{sid}.csv", index=False)


def _build_case(tmp_path):
    sd = tmp_path / "stretches"; sd.mkdir()
    ts = tmp_path / "ts"; ts.mkdir()
    rows, meta_rows = [], []
    # three type-(c) stretches with disjoint stations; q_up=1000, q_down=1200, r=500, s=300
    for stid, (up, down, on, off) in enumerate([(100, 200, 10, 20),
                                                (300, 400, 30, 40),
                                                (500, 600, 50, 60)]):
        rows.append(dict(stretch_id=stid, up_ml_id=up, down_ml_id=down,
                         on_ids=str(on), off_ids=str(off), review_ids="",
                         n_on=1, n_off=1, config_type="c"))
        _station(ts, up, 1000); _station(ts, down, 1200)
        _station(ts, on, 500); _station(ts, off, 300)
        meta_rows.append({"Station ID": up, "capacity": 2000.0, "Lanes": 3})
    pd.DataFrame(rows).to_csv(sd / "880_N.csv", index=False)
    meta = tmp_path / "meta.csv"; pd.DataFrame(meta_rows).to_csv(meta, index=False)
    return sd, ts, meta


def test_cli_writes_scenario_report(tmp_path):
    sd, ts, meta = _build_case(tmp_path)
    out = tmp_path / "report.csv"
    rc = main([
        "--stretches", str(sd), "--timeseries-dir", str(ts), "--station-meta", str(meta),
        "--ks", "1", "2", "--model", "rf", "--n-estimators", "10", "--out", str(out),
    ])
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df["k"]) == [1, 2]
    assert "nrmse_mean" in df.columns and "n_combos" in df.columns


def test_cli_returns_error_when_no_samples(tmp_path):
    sd, ts, meta = _build_case(tmp_path)
    empty = tmp_path / "empty"; empty.mkdir()          # timeseries dir with no files
    rc = main([
        "--stretches", str(sd), "--timeseries-dir", str(empty), "--station-meta", str(meta),
        "--ks", "1", "--out", str(tmp_path / "r.csv"),
    ])
    assert rc == 1
