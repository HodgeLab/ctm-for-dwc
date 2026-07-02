"""Integration tests for build_training_data (features.py IO assembler).

Uses tiny temp per-station CSVs + an in-memory station-metadata frame; no real
timeseries. One type-(c) stretch: up ML 100, down ML 200, on-ramp 10, off-ramp 20.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from transportation_models.utils.ramp_flow_estimation import bounds
from transportation_models.utils.ramp_flow_estimation.features import build_training_data


def _write_station(tmp, sid, flow, *, speed=60.0, occ=10.0, pct=100.0, n=6):
    ts = pd.date_range("2023-06-01 00:00", periods=n, freq="5min")
    pd.DataFrame({
        "timestamp": ts,
        "total_flow_[veh/5-min]": np.full(n, float(flow)) if np.isscalar(flow) else flow,
        "avg_speed_[mph]": np.full(n, float(speed)),
        "avg_occupancy_[%]": np.full(n, float(occ)),
        "pct_observed": np.full(n, float(pct)),
    }).to_csv(tmp / f"{sid}.csv", index=False)


def _stretches():
    return pd.DataFrame([{
        "stretch_id": 0, "up_ml_id": 100, "down_ml_id": 200,
        "on_ids": "10", "off_ids": "20", "review_ids": "",
        "n_on": 1, "n_off": 1, "config_type": "c",
    }])


def _meta(capacity=2000.0, lanes=3):
    return pd.DataFrame({"Station ID": [100, 200], "capacity": [capacity, capacity],
                         "Lanes": [lanes, lanes]})


def test_assembles_samples_with_scaled_c_w_and_stretch_labels(tmp_path):
    _write_station(tmp_path, 100, 1000)     # q_up
    _write_station(tmp_path, 200, 1200)     # q_down
    _write_station(tmp_path, 10, 500)       # on
    _write_station(tmp_path, 20, 300)       # off (1000 + 500 - 300 = 1200)

    td = build_training_data(_stretches(), tmp_path, _meta(capacity=2000.0, lanes=3))

    assert td.X.shape == (4, 18)
    np.testing.assert_array_equal(td.stretch_id, np.zeros(4, dtype=int))
    np.testing.assert_allclose(td.c_w, 6000.0)          # 2000 veh/h/lane * 3 lanes
    # alpha must be consistent with the bounds the assembler actually used
    # (C_w plus each ramp's historical-peak r_demand/s_qmax).
    b = bounds.compute_bounds(1000.0, 1200.0, 6000.0,
                              r_demand=td.r_demand_used[0], s_qmax=td.s_qmax_used[0])
    expected, _ = bounds.alpha_target(b, 1000.0, 1200.0, 500.0, 300.0)
    np.testing.assert_allclose(td.alpha, expected)


def test_stretch_skipped_when_upstream_capacity_missing(tmp_path):
    for sid, f in [(100, 1000), (200, 1200), (10, 500), (20, 300)]:
        _write_station(tmp_path, sid, f)
    meta = pd.DataFrame({"Station ID": [999], "capacity": [2000.0], "Lanes": [3]})  # no 100
    td = build_training_data(_stretches(), tmp_path, meta)
    assert td.X.shape[0] == 0


def test_r_demand_set_from_historical_peak_on_ramp_flow(tmp_path):
    _write_station(tmp_path, 100, 1000)
    _write_station(tmp_path, 200, 1200)
    on = np.array([500, 500, 500, 500, 500, 900.0])     # peak 900
    _write_station(tmp_path, 10, on)
    _write_station(tmp_path, 20, 300)

    td = build_training_data(_stretches(), tmp_path, _meta())
    # r_demand caps R_max at the historical peak 900 -> tighter than C_w - q_up.
    assert np.isclose(td.r_demand_used, 900.0).all()
