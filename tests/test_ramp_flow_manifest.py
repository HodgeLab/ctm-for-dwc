"""Tests for utils/ramp_flow_estimation/manifest.py (split-manifest contract).

Reuses the tiny per-station CSV fixture style of test_ramp_flow_training_data:
one type-(c) stretch (up ML 100, down ML 200, on 10, off 20) in a temp dir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ramp_flow_estimation.features import build_training_data
from transportation_models.utils.ramp_flow_estimation.manifest import (
    corpus_digest,
    load_manifest,
    rebuild_corpus,
    write_manifest,
)


def _write_station(tmp, sid, flow, *, speed=60.0, occ=10.0, pct=100.0, n=6):
    ts = pd.date_range("2023-06-01 00:00", periods=n, freq="5min")
    pd.DataFrame({
        "timestamp": ts,
        "total_flow_[veh/5-min]": np.full(n, float(flow)),
        "avg_speed_[mph]": np.full(n, float(speed)),
        "avg_occupancy_[%]": np.full(n, float(occ)),
        "pct_observed": np.full(n, float(pct)),
    }).to_csv(tmp / f"{sid}.csv", index=False)


def _case(tmp_path):
    """Timeseries + stretches.csv + station-meta csv on disk; returns the
    corpus_params dict a manifest would record."""
    ts_dir = tmp_path / "timeseries"
    ts_dir.mkdir()
    for sid, flow in ((100, 100), (200, 120), (10, 50), (20, 30)):
        _write_station(ts_dir, sid, flow)
    stretches = tmp_path / "stretches.csv"
    pd.DataFrame([{
        "stretch_id": 0, "up_ml_id": 100, "down_ml_id": 200,
        "on_ids": "10", "off_ids": "20", "review_ids": "",
        "n_on": 1, "n_off": 1, "config_type": "c",
    }]).to_csv(stretches, index=False)
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"Station ID": [100, 200], "capacity": [2000.0, 2000.0],
                  "Lanes": [3, 3]}).to_csv(meta, index=False)
    return {
        "stretches": str(stretches), "station_meta": str(meta),
        "timeseries_dir": str(ts_dir), "pct_floor": 0.0, "window_size": 3,
        "require_both_measured": True, "keep_infeasible": True,
        "record_start": None, "record_end": None,
    }


def _build(params):
    return build_training_data(
        pd.read_csv(params["stretches"]), params["timeseries_dir"],
        pd.read_csv(params["station_meta"]),
        pct_floor=params["pct_floor"], window_size=params["window_size"],
        require_both_measured=params["require_both_measured"],
        keep_infeasible=params["keep_infeasible"],
        record_start=params["record_start"], record_end=params["record_end"],
    )


def test_manifest_roundtrip_rebuilds_identical_corpus(tmp_path):
    params = _case(tmp_path)
    data = _build(params)
    write_manifest(tmp_path / "manifest.json", corpus_params=params,
                   split_seed=0, val_stretch=0, test_stretch=0, data=data)
    m = load_manifest(tmp_path / "manifest.json")
    rebuilt = rebuild_corpus(m)
    assert corpus_digest(rebuilt) == corpus_digest(data)
    np.testing.assert_array_equal(rebuilt.X, data.X)


def test_rebuild_fails_when_parameter_drifts(tmp_path):
    params = _case(tmp_path)
    data = _build(params)
    m = write_manifest(tmp_path / "manifest.json", corpus_params=params,
                       split_seed=0, val_stretch=0, test_stretch=0, data=data)
    m["corpus"]["window_size"] = 2
    with pytest.raises(ValueError, match="digest"):
        rebuild_corpus(m)


def test_rebuild_fails_when_data_drifts(tmp_path):
    params = _case(tmp_path)
    data = _build(params)
    m = write_manifest(tmp_path / "manifest.json", corpus_params=params,
                       split_seed=0, val_stretch=0, test_stretch=0, data=data)
    _write_station(tmp_path / "timeseries", 10, 55)      # on-ramp flow changed
    with pytest.raises(ValueError, match="digest"):
        rebuild_corpus(m)


def test_rebuild_honors_path_overrides(tmp_path):
    params = _case(tmp_path)
    data = _build(params)
    m = write_manifest(tmp_path / "manifest.json", corpus_params=params,
                       split_seed=0, val_stretch=0, test_stretch=0, data=data)
    m["corpus"]["timeseries_dir"] = "/nonexistent/hpc/path"
    rebuilt = rebuild_corpus(m, timeseries_dir=params["timeseries_dir"])
    assert corpus_digest(rebuilt) == m["digest"]
