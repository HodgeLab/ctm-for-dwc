"""Integration test for scripts/train_ramp_flow_gru.py (CLI wiring).

Builds a tiny 3-stretch case and runs the entrypoint end-to-end with a tiny
GRU; W&B is forced offline into the temp dir so nothing is published.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from transportation_models.utils.ramp_flow_estimation.gru import GruEstimator
from transportation_models.utils.ramp_flow_estimation.manifest import load_manifest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from train_ramp_flow_gru import main  # noqa: E402


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
    for stid, (up, down, on, off) in enumerate([(100, 200, 10, 20),
                                                (300, 400, 30, 40),
                                                (500, 600, 50, 60)]):
        rows.append(dict(stretch_id=stid, up_ml_id=up, down_ml_id=down,
                         on_ids=str(on), off_ids=str(off), review_ids="",
                         n_on=1, n_off=1, config_type="c"))
        _station(ts, up, 100); _station(ts, down, 120)
        _station(ts, on, 50); _station(ts, off, 30)
        meta_rows.append({"Station ID": up, "capacity": 2000.0, "Lanes": 3})
    pd.DataFrame(rows).to_csv(sd / "880_N.csv", index=False)
    meta = tmp_path / "meta.csv"; pd.DataFrame(meta_rows).to_csv(meta, index=False)
    return sd, ts, meta


def test_cli_trains_and_writes_manifest_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    sd, ts, meta = _build_case(tmp_path)
    out_dir = tmp_path / "out"
    rc = main([
        "--stretches", str(sd), "--timeseries-dir", str(ts), "--station-meta", str(meta),
        "--hidden-size", "8", "--max-epochs", "3", "--target-transform", "log1p",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    m = load_manifest(out_dir / "split_manifest.json")
    assert m["corpus"]["require_capacity"] is True
    assert m["split"]["val_stretch"] != m["split"]["test_stretch"]
    est = GruEstimator.load(out_dir / "gru.pt")
    assert est.target_transform == "log1p"
    r_hat, s_hat = est.predict_flows(np.random.default_rng(0).uniform(0, 1, (5, 18)))
    assert (r_hat >= 0).all() and (s_hat >= 0).all()
