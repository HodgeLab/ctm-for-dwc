"""Integration test for scripts/compare_ramp_flow_estimators.py (CLI wiring).

Builds a tiny 3-stretch case, writes a split manifest the way the GRU training
script does, and runs the comparison entrypoint end-to-end with a small RF and
a retrained (tiny) GRU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ramp_flow_estimation.features import build_training_data
from transportation_models.utils.ramp_flow_estimation.manifest import (
    load_stretches,
    write_manifest,
)
from transportation_models.utils.ramp_flow_estimation.validation import train_val_test_split

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from compare_ramp_flow_estimators import main  # noqa: E402


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


def _write_case_manifest(tmp_path):
    sd, ts, meta = _build_case(tmp_path)
    corpus_params = {
        "stretches": str(sd), "station_meta": str(meta),
        "timeseries_dir": str(ts), "pct_floor": 0.0, "window_size": 3,
        "require_both_measured": True, "require_capacity": True,
        "record_start": None, "record_end": None,
    }
    data, ctx = build_training_data(load_stretches(sd), ts, pd.read_csv(meta))
    split = train_val_test_split(data, seed=0)
    path = tmp_path / "split_manifest.json"
    write_manifest(path, corpus_params=corpus_params, split_seed=0,
                   val_stretch=split["val_stretch"],
                   test_stretch=split["test_stretch"], data=data, stretch_ctx=ctx)
    return path


def test_cli_scores_both_estimators_on_manifest_split(tmp_path):
    manifest = _write_case_manifest(tmp_path)
    out = tmp_path / "comparison.csv"
    rc = main([
        "--manifest", str(manifest), "--retrain-gru",
        "--n-estimators", "10", "--hidden-size", "8", "--max-epochs", "3",
        "--out", str(out),
    ])
    assert rc == 0
    df = pd.read_csv(out)
    assert set(df["estimator"]) == {"kan_rf", "gru"}
    assert "all" in set(df["slice"])
    assert {"nrmse", "r2", "bias"} <= set(df.columns)
    # one test stretch, identical rows for both estimators
    n_all = df[df["slice"] == "all"]["n"].unique()
    assert len(n_all) == 1


def test_cli_fails_on_drifted_manifest(tmp_path):
    manifest_path = _write_case_manifest(tmp_path)
    _station(tmp_path / "ts", 10, 55)                  # mutate an on-ramp series
    with pytest.raises(ValueError, match="digest"):
        main(["--manifest", str(manifest_path), "--retrain-gru",
              "--out", str(tmp_path / "c.csv")])
