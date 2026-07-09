"""Tests for ``scripts/build_ctm_ramp_scenario.py`` (Step 7 scenario builder).

Covers the virtual-VDS-id paths: stretch-map keying, downstream-mainline
lookup, and measured-sibling subtraction from stretch-total estimates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from build_ctm_ramp_scenario import (  # noqa: E402
    _furthest_downstream_ml_id,
    _parse_stretch_ramp_ids,
    _subtract_measured_siblings,
    _vds_to_stretch_map,
    main,
)

_START = pd.Timestamp("2022-01-03 12:00")  # a Monday
_END = pd.Timestamp("2022-01-03 12:10")


def _write_ramp_csv(path: Path, flows_5min: np.ndarray) -> None:
    timestamps = pd.date_range(start=_START, periods=len(flows_5min), freq="5min")
    pd.DataFrame({
        "timestamp": timestamps,
        "total_flow_[veh/5-min]": flows_5min,
    }).to_csv(path, index=False)


def _stretches(**overrides) -> pd.DataFrame:
    row = {
        "stretch_id": 0, "fwy": 880, "dir": "N",
        "pm_up": 0.0, "pm_down": 1.0,
        "up_ml_id": 100, "down_ml_id": 200,
        "on_ids": np.nan, "off_ids": np.nan,
        "config_type": "c", "notes": np.nan,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ---- Stretch ramp-id parsing / mapping -------------------------------------


def test_parse_stretch_ramp_ids_mixes_ints_and_virtual_strings():
    assert _parse_stretch_ramp_ids("900;v400000") == [900, "v400000"]
    assert _parse_stretch_ramp_ids(" 900 ; 901 ") == [900, 901]
    assert _parse_stretch_ramp_ids(np.nan) == []
    assert _parse_stretch_ramp_ids(None) == []


def test_vds_to_stretch_map_keys_virtual_ids():
    stretches = _stretches(off_ids="900;v400000")
    mapping = _vds_to_stretch_map(stretches, "off_ids")
    assert set(mapping) == {900, "v400000"}
    assert int(mapping["v400000"]["down_ml_id"]) == 200


def test_furthest_downstream_ml_id_resolves_virtual_only_list():
    stretches = _stretches(off_ids="v400000")
    mapping = _vds_to_stretch_map(stretches, "off_ids")
    assert _furthest_downstream_ml_id(["v400000"], stretches, mapping) == 200


# ---- Measured-sibling subtraction ------------------------------------------


def test_subtract_measured_siblings_removes_filled_sibling_flow(tmp_path):
    """est is a stretch-side total; the measured sibling's flow comes out."""
    stretch = _stretches(on_ids="900;v400000").iloc[0]
    # 10 veh/5min = 120 veh/hr, flat over the 2-step window.
    _write_ramp_csv(tmp_path / "900.csv", np.array([10.0, 10.0]))
    est = np.full(2, 200.0)
    out = _subtract_measured_siblings(
        est, stretch, "on_ids", "v400000", tmp_path, start=_START, end=_END)
    np.testing.assert_allclose(out, [80.0, 80.0])


def test_subtract_measured_siblings_clips_at_zero(tmp_path):
    stretch = _stretches(on_ids="900;v400000").iloc[0]
    _write_ramp_csv(tmp_path / "900.csv", np.array([10.0, 10.0]))
    est = np.full(2, 100.0)  # sibling alone is 120 veh/hr
    out = _subtract_measured_siblings(
        est, stretch, "on_ids", "v400000", tmp_path, start=_START, end=_END)
    np.testing.assert_allclose(out, [0.0, 0.0])


def test_subtract_measured_siblings_treats_unfillable_nan_as_zero(tmp_path):
    """A sample the historical-average filler can't fill (empty bin) counts
    as 0 in the subtraction instead of NaN-ing the estimate."""
    stretch = _stretches(off_ids="900;v400000").iloc[0]
    # Single-day CSV: the NaN sample's (weekday, time-of-day) bin has no
    # other history, so historical_average_fill leaves it NaN.
    _write_ramp_csv(tmp_path / "900.csv", np.array([np.nan, 10.0]))
    est = np.full(2, 200.0)
    out = _subtract_measured_siblings(
        est, stretch, "off_ids", "v400000", tmp_path, start=_START, end=_END)
    np.testing.assert_allclose(out, [200.0, 80.0])


def test_subtract_measured_siblings_no_siblings_is_identity(tmp_path):
    stretch = _stretches(off_ids="v400000").iloc[0]
    est = np.array([50.0, -5.0])  # clip still applies
    out = _subtract_measured_siblings(
        est, stretch, "off_ids", "v400000", tmp_path, start=_START, end=_END)
    np.testing.assert_allclose(out, [50.0, 0.0])


# ---- --strategy modes (end-to-end through main) ------------------------------


def _write_mainline_csv(path: Path, flows_5min, *, start) -> None:
    ts = pd.date_range(start=start, periods=len(flows_5min), freq="5min")
    pd.DataFrame({
        "timestamp": ts,
        "total_flow_[veh/5-min]": flows_5min,
        "avg_speed_[mph]": np.full(len(flows_5min), 60.0),
        "avg_occupancy_[%]": np.full(len(flows_5min), 0.05),
    }).to_csv(path, index=False)


def _strategy_case(tmp_path):
    """2-cell corridor: cell 0 has a measured on-ramp (900), cell 1 an absent
    (virtual) off-ramp on a type-(c) stretch. Mainline 100/200 cover the sim
    window plus GRU warmup lags."""
    ts_dir = tmp_path / "ts"; ts_dir.mkdir()
    warmup_start = _START - pd.Timedelta(minutes=10)
    _write_mainline_csv(ts_dir / "100.csv", np.full(4, 100.0), start=warmup_start)
    _write_mainline_csv(ts_dir / "200.csv", np.full(4, 110.0), start=warmup_start)
    _write_ramp_csv(ts_dir / "900.csv", np.array([10.0, 10.0]))

    cells = tmp_path / "cells.csv"
    pd.DataFrame([
        {"on_ramp": True, "off_ramp": False, "on_ramp_vds_id": "900",
         "off_ramp_vds_id": ""},
        {"on_ramp": False, "off_ramp": True, "on_ramp_vds_id": "",
         "off_ramp_vds_id": "v400000"},
    ]).to_csv(cells, index=False)

    stretches = tmp_path / "stretches.csv"
    _stretches(on_ids="900", off_ids="v400000").to_csv(stretches, index=False)
    out_dir = tmp_path / "out"
    base = ["--cells", str(cells), "--stretches", str(stretches),
            "--timeseries-dir", str(ts_dir),
            "--start", str(_START), "--end", str(_END),
            "--dt-seconds", "300", "--out-dir", str(out_dir)]
    return base, out_dir


def test_strategy_historical_average_zeros_absent_ramp(tmp_path, capsys):
    base, out_dir = _strategy_case(tmp_path)
    main(base + ["--strategy", "historical_average"])
    demand = pd.read_csv(out_dir / "demand.csv")
    beta = pd.read_csv(out_dir / "beta.csv")
    np.testing.assert_allclose(demand["0"], 120.0)        # 10 veh/5min * 12
    np.testing.assert_allclose(beta["1"], 0.0)            # absent ramp -> zero
    assert "has no data" in capsys.readouterr().err


def test_strategy_historical_average_uses_conservation_on_type_b(tmp_path):
    """Conservation needs no estimator, so it applies under every strategy:
    an absent off-ramp on a type-(b) stretch is recovered from the mainline
    pair even with --strategy historical_average."""
    ts_dir = tmp_path / "ts"; ts_dir.mkdir()
    # q_up = 110 veh/5min (1320 veh/hr), q_down = 100 (1200) -> s = 120 veh/hr.
    _write_mainline_csv(ts_dir / "100.csv", np.full(2, 110.0), start=_START)
    _write_mainline_csv(ts_dir / "200.csv", np.full(2, 100.0), start=_START)

    cells = tmp_path / "cells.csv"
    pd.DataFrame([
        {"on_ramp": False, "off_ramp": True, "on_ramp_vds_id": "",
         "off_ramp_vds_id": "v400000"},
    ]).to_csv(cells, index=False)
    stretches = tmp_path / "stretches.csv"
    _stretches(off_ids="v400000", config_type="b").to_csv(stretches, index=False)
    out_dir = tmp_path / "out"

    main(["--cells", str(cells), "--stretches", str(stretches),
          "--timeseries-dir", str(ts_dir),
          "--start", str(_START), "--end", str(_END),
          "--dt-seconds", "300", "--out-dir", str(out_dir),
          "--strategy", "historical_average"])
    beta = pd.read_csv(out_dir / "beta.csv")
    # beta = s / (f + s) = 120 / (1200 + 120)
    np.testing.assert_allclose(beta["0"], 120.0 / 1320.0)


def test_strategy_gru_estimates_absent_ramp(tmp_path):
    from transportation_models.utils.ramp_flow_estimation.gru import GruEstimator

    base, out_dir = _strategy_case(tmp_path)
    rng = np.random.default_rng(0)
    X = rng.uniform(50, 150, size=(200, 18))
    est = GruEstimator(hidden_size=8, max_epochs=20, lr=1e-2, seed=0,
                       device="cpu").fit(
        X, np.full(200, 300.0), np.full(200, 240.0))
    ckpt = tmp_path / "gru.pt"; est.save(ckpt)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"corpus": {"window_size": 3}}')

    main(base + ["--strategy", "gru", "--gru-checkpoint", str(ckpt),
                 "--manifest", str(manifest)])
    beta = pd.read_csv(out_dir / "beta.csv")
    assert (beta["1"] > 0).all() and (beta["1"] < 1).all()


def test_strategy_kan_estimates_absent_ramp(tmp_path):
    from transportation_models.utils.ramp_flow_estimation.kan import KanEstimator
    from transportation_models.utils.ramp_flow_estimation import bounds

    base, out_dir = _strategy_case(tmp_path)
    rng = np.random.default_rng(0)
    q_up = rng.uniform(800.0, 1600.0, 200)
    q_down = q_up + 200.0
    X = np.column_stack([q_up, np.full(200, 60.0), np.full(200, 0.05),
                         q_down, np.full(200, 55.0), np.full(200, 0.06)])
    alpha = np.full(200, 0.3)
    r, s = bounds.reconstruct_pair(alpha, q_up, q_down, 6000.0)
    kan = KanEstimator("rf", n_estimators=10, random_state=0).fit(
        X, r, s, ctx={"c_w": np.full(200, 6000.0)})
    ckpt = tmp_path / "kan.joblib"; kan.save(ckpt)
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"Station ID": [100], "capacity": [2000.0],
                  "Lanes": [3]}).to_csv(meta, index=False)

    main(base + ["--strategy", "kan", "--kan-checkpoint", str(ckpt),
                 "--station-meta", str(meta)])
    beta = pd.read_csv(out_dir / "beta.csv")
    assert (beta["1"] >= 0).all() and (beta["1"] < 1).all()


def test_strategy_gru_requires_checkpoint_and_manifest(tmp_path):
    import pytest
    base, _ = _strategy_case(tmp_path)
    with pytest.raises(SystemExit):
        main(base + ["--strategy", "gru"])
