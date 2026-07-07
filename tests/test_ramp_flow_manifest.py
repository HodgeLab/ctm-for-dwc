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
    load_stretches,
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
        "require_both_measured": True, "require_capacity": True,
        "record_start": None, "record_end": None,
    }


def _build(params):
    return build_training_data(
        load_stretches(params["stretches"]), params["timeseries_dir"],
        pd.read_csv(params["station_meta"]),
        pct_floor=params["pct_floor"], window_size=params["window_size"],
        require_both_measured=params["require_both_measured"],
        require_capacity=params["require_capacity"],
        record_start=params["record_start"], record_end=params["record_end"],
    )


def _write(tmp_path, params, data, ctx):
    return write_manifest(tmp_path / "manifest.json", corpus_params=params,
                          split_seed=0, val_stretch=0, test_stretch=0,
                          data=data, stretch_ctx=ctx)


def _stretch_row(sid, up, down, on, off):
    return {"stretch_id": sid, "up_ml_id": up, "down_ml_id": down,
            "on_ids": str(on), "off_ids": str(off), "review_ids": "",
            "n_on": 1, "n_off": 1, "config_type": "c"}


def test_load_stretches_namespaces_colliding_ids_across_files(tmp_path):
    # Two stretch CSVs both numbered from 0 (the builder numbers per file).
    # Without namespacing their rows merge into one bogus "stretch" and the
    # later stretch's bounds context overwrites the earlier one's.
    ts_dir = tmp_path / "timeseries"; ts_dir.mkdir()
    for sid, flow in ((100, 100), (200, 120), (10, 50), (20, 30),
                      (300, 100), (400, 120), (30, 50), (40, 30)):
        _write_station(ts_dir, sid, flow)
    sd = tmp_path / "stretches"; sd.mkdir()
    pd.DataFrame([_stretch_row(0, 100, 200, 10, 20)]).to_csv(sd / "a.csv", index=False)
    pd.DataFrame([_stretch_row(0, 300, 400, 30, 40)]).to_csv(sd / "b.csv", index=False)
    meta = pd.DataFrame({"Station ID": [100, 300], "capacity": [2000.0, 1500.0],
                         "Lanes": [3, 2]})

    stretches = load_stretches(sd)
    assert set(stretches["stretch_id"]) == {"a:0", "b:0"}

    data, ctx = build_training_data(stretches, ts_dir, meta)
    assert set(np.unique(data.stretch_id)) == {"a:0", "b:0"}
    assert ctx.at["a:0", "c_w"] == 6000.0            # 2000 * 3
    assert ctx.at["b:0", "c_w"] == 3000.0            # 1500 * 2 -- not clobbered


def test_manifest_roundtrip_rebuilds_identical_corpus(tmp_path):
    params = _case(tmp_path)
    data, ctx = _build(params)
    _write(tmp_path, params, data, ctx)
    m = load_manifest(tmp_path / "manifest.json")
    rebuilt, rebuilt_ctx = rebuild_corpus(m)
    assert corpus_digest(rebuilt, rebuilt_ctx) == corpus_digest(data, ctx)
    np.testing.assert_array_equal(rebuilt.X, data.X)


def test_rebuild_fails_when_parameter_drifts(tmp_path):
    params = _case(tmp_path)
    data, ctx = _build(params)
    m = _write(tmp_path, params, data, ctx)
    m["corpus"]["window_size"] = 2
    with pytest.raises(ValueError, match="digest"):
        rebuild_corpus(m)


def test_rebuild_fails_when_data_drifts(tmp_path):
    params = _case(tmp_path)
    data, ctx = _build(params)
    m = _write(tmp_path, params, data, ctx)
    _write_station(tmp_path / "timeseries", 10, 55)      # on-ramp flow changed
    with pytest.raises(ValueError, match="digest"):
        rebuild_corpus(m)


def test_rebuild_fails_when_station_capacity_drifts(tmp_path):
    # capacity only affects the ctx table, not the corpus rows -- the digest
    # must still catch it.
    params = _case(tmp_path)
    data, ctx = _build(params)
    m = _write(tmp_path, params, data, ctx)
    pd.DataFrame({"Station ID": [100, 200], "capacity": [1800.0, 2000.0],
                  "Lanes": [3, 3]}).to_csv(params["station_meta"], index=False)
    with pytest.raises(ValueError, match="digest"):
        rebuild_corpus(m)


def test_rebuild_honors_path_overrides(tmp_path):
    params = _case(tmp_path)
    data, ctx = _build(params)
    m = _write(tmp_path, params, data, ctx)
    m["corpus"]["timeseries_dir"] = "/nonexistent/hpc/path"
    rebuilt, rebuilt_ctx = rebuild_corpus(m, timeseries_dir=params["timeseries_dir"])
    assert corpus_digest(rebuilt, rebuilt_ctx) == m["digest"]
