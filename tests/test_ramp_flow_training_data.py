"""Integration tests for build_training_data (features.py IO assembler).

Uses tiny temp per-station CSVs + an in-memory station-metadata frame; no real
timeseries. One type-(c) stretch: up ML 100, down ML 200, on-ramp 10, off-ramp 20.
PeMS ``total_flow`` is veh/5-min; the assembler converts it to veh/hr (x12).
The assembler returns ``(data, stretch_ctx)``: the method-agnostic corpus plus
the per-stretch bounds-context table (c_w, historical ramp peaks) for Kan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

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


def _measured_case(tmp):
    # veh/5-min flows; conservation holds (100 + 50 - 30 = 120), and x12 in veh/hr.
    _write_station(tmp, 100, 100)      # q_up  -> 1200 veh/hr
    _write_station(tmp, 200, 120)      # q_down -> 1440 veh/hr
    _write_station(tmp, 10, 50)        # on    -> 600 veh/hr
    _write_station(tmp, 20, 30)        # off   -> 360 veh/hr


def test_flow_is_converted_to_vehicles_per_hour(tmp_path):
    _measured_case(tmp_path)
    td, _ = build_training_data(_stretches(), tmp_path, _meta())
    np.testing.assert_allclose(td.q_up, 1200.0)      # 100 veh/5-min * 12
    np.testing.assert_allclose(td.q_down, 1440.0)
    np.testing.assert_allclose(td.r_true, 600.0)
    np.testing.assert_allclose(td.s_true, 360.0)


def test_assembles_samples_with_stretch_labels_and_ctx(tmp_path):
    _measured_case(tmp_path)
    td, ctx = build_training_data(_stretches(), tmp_path, _meta(capacity=2000.0, lanes=3))

    assert td.X.shape == (4, 18)
    np.testing.assert_array_equal(td.stretch_id, np.zeros(4, dtype=int))
    assert list(ctx.index) == [0]
    assert ctx.at[0, "c_w"] == 6000.0                # 2000 veh/h/lane * 3 lanes


def test_stretch_skipped_when_upstream_capacity_missing(tmp_path):
    _measured_case(tmp_path)
    meta = pd.DataFrame({"Station ID": [999], "capacity": [2000.0], "Lanes": [3]})  # no 100
    td, ctx = build_training_data(_stretches(), tmp_path, meta)
    assert td.X.shape[0] == 0 and len(ctx) == 0


def test_uncalibrated_stretch_kept_with_nan_c_w_when_opted_in(tmp_path):
    _measured_case(tmp_path)
    meta = pd.DataFrame({"Station ID": [999], "capacity": [2000.0], "Lanes": [3]})  # no 100
    td, ctx = build_training_data(_stretches(), tmp_path, meta, require_capacity=False)
    assert td.X.shape[0] == 4
    assert np.isnan(ctx.at[0, "c_w"])
    np.testing.assert_allclose(ctx.at[0, "r_demand"], 600.0)   # peak still computed


def test_r_demand_set_from_historical_peak_on_ramp_flow(tmp_path):
    _write_station(tmp_path, 100, 100)
    _write_station(tmp_path, 200, 120)
    on = np.array([50, 50, 50, 50, 50, 90.0])          # peak 90 veh/5-min
    _write_station(tmp_path, 10, on)
    _write_station(tmp_path, 20, 30)

    _, ctx = build_training_data(_stretches(), tmp_path, _meta())
    np.testing.assert_allclose(ctx.at[0, "r_demand"], 90.0 * 12)   # 1080 veh/hr
