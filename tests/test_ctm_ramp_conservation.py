"""Tests for conservation_fill_accuracy (flow-conservation fill benchmark).

Tiny synthetic per-VDS CSVs in a temp dir; stretch frames carry the columns
the function consumes (stretch_id, config_type, up/down ML ids, ramp ids).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ctm.ramp_validation import conservation_fill_accuracy

_FLOW = "total_flow_[veh/5-min]"


def _write_series(tmp, vds_id, flows, *, start="2023-06-05 00:00", freq="5min"):
    ts = pd.date_range(start, periods=len(flows), freq=freq)
    pd.DataFrame({"timestamp": ts, _FLOW: flows}).to_csv(
        tmp / f"{vds_id}.csv", index=False
    )


def _stretches(rows) -> pd.DataFrame:
    """rows: list of (config_type, up_ml_id, down_ml_id, on_ids, off_ids)."""
    return pd.DataFrame(
        [{"stretch_id": i, "config_type": ct, "up_ml_id": up,
          "down_ml_id": down, "on_ids": on, "off_ids": off}
         for i, (ct, up, down, on, off) in enumerate(rows)]
    )


def test_type_a_perfect_conservation(tmp_path):
    n = 288
    _write_series(tmp_path, 1, np.full(n, 100.0))   # up ML
    _write_series(tmp_path, 2, np.full(n, 130.0))   # down ML
    _write_series(tmp_path, 3, np.full(n, 30.0))    # on-ramp
    table, points = conservation_fill_accuracy(
        _stretches([("a", 1, 2, "3", "")]), tmp_path
    )
    assert len(table) == 1
    row = table.iloc[0]
    assert row["ramp_type"] == "on" and row["config_type"] == "a"
    assert row["n_scored"] == n
    for metric in ("rmse", "nrmse", "bias", "nbias"):
        assert row[metric] == pytest.approx(0.0, abs=1e-12)
    assert len(points) == n
    assert (points["estimated"] == 30.0).all()


def test_type_b_perfect_conservation(tmp_path):
    n = 288
    _write_series(tmp_path, 1, np.full(n, 100.0))
    _write_series(tmp_path, 2, np.full(n, 80.0))
    _write_series(tmp_path, 4, np.full(n, 20.0))    # off-ramp
    table, _ = conservation_fill_accuracy(
        _stretches([("b", 1, 2, "", "4")]), tmp_path
    )
    row = table.iloc[0]
    assert row["ramp_type"] == "off"
    assert row["rmse"] == pytest.approx(0.0, abs=1e-12)


def test_multi_ramp_truth_is_sum(tmp_path):
    n = 100
    _write_series(tmp_path, 1, np.full(n, 100.0))
    _write_series(tmp_path, 2, np.full(n, 130.0))
    _write_series(tmp_path, 3, np.full(n, 10.0))
    _write_series(tmp_path, 4, np.full(n, 20.0))
    table, _ = conservation_fill_accuracy(
        _stretches([("a", 1, 2, "3;4", "")]), tmp_path
    )
    row = table.iloc[0]
    assert row["ramp_ids"] == "3;4"
    assert row["rmse"] == pytest.approx(0.0, abs=1e-12)


def test_negative_estimate_clipped_to_zero(tmp_path):
    # q_down < q_up, so the raw estimate is -10 -> clipped to 0; the measured
    # on-ramp flow is 5, so every scored error is truth - est = 5.
    n = 50
    _write_series(tmp_path, 1, np.full(n, 100.0))
    _write_series(tmp_path, 2, np.full(n, 90.0))
    _write_series(tmp_path, 3, np.full(n, 5.0))
    table, points = conservation_fill_accuracy(
        _stretches([("a", 1, 2, "3", "")]), tmp_path
    )
    row = table.iloc[0]
    assert (points["estimated"] == 0.0).all()
    assert row["rmse"] == pytest.approx(5.0)
    assert row["bias"] == pytest.approx(5.0)
    assert row["nrmse"] == pytest.approx(1.0)
    assert row["nbias"] == pytest.approx(1.0)


def test_virtual_ramp_id_skips_stretch(tmp_path):
    for vid in (1, 2, 3):
        _write_series(tmp_path, vid, np.full(10, 10.0))
    with pytest.warns(UserWarning, match="virtual id"):
        table, points = conservation_fill_accuracy(
            _stretches([("a", 1, 2, "v99;3", "")]), tmp_path
        )
    assert table.empty and points.empty


def test_non_ab_stretches_ignored(tmp_path):
    # Type (c) and untyped stretches never score and never warn.
    table, points = conservation_fill_accuracy(
        _stretches([("c", 1, 2, "3", "4"), (np.nan, 1, 2, "", "")]), tmp_path
    )
    assert table.empty and points.empty


def test_missing_timeseries_warns_and_skips(tmp_path):
    _write_series(tmp_path, 1, np.full(10, 10.0))
    _write_series(tmp_path, 3, np.full(10, 5.0))
    with pytest.warns(UserWarning, match="no timeseries"):
        table, _ = conservation_fill_accuracy(
            _stretches([("a", 1, 2, "3", "")]), tmp_path   # 2.csv missing
        )
    assert table.empty


def test_scores_only_coobserved_timestamps(tmp_path):
    # up covers t0..t99, down t50..t149; the ramp spans both but has 5 NaNs
    # inside the overlap -> 50 common timestamps minus 5 unobserved = 45.
    _write_series(tmp_path, 1, np.full(100, 100.0))
    _write_series(tmp_path, 2, np.full(100, 130.0),
                  start="2023-06-05 04:10")                # t50 at 5-min steps
    ramp = np.full(150, 30.0)
    ramp[60:65] = np.nan
    _write_series(tmp_path, 3, ramp)
    table, points = conservation_fill_accuracy(
        _stretches([("a", 1, 2, "3", "")]), tmp_path
    )
    row = table.iloc[0]
    assert row["n_scored"] == 45
    assert len(points) == 45
    assert row["rmse"] == pytest.approx(0.0, abs=1e-12)


def test_cli_writes_table_and_plots(tmp_path):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    from validate_ramp_flow_conservation import main

    ts_dir = tmp_path / "ts"; ts_dir.mkdir()
    n = 100
    _write_series(ts_dir, 1, np.full(n, 100.0))
    _write_series(ts_dir, 2, np.full(n, 130.0))
    _write_series(ts_dir, 3, np.full(n, 30.0))
    _write_series(ts_dir, 4, np.full(n, 110.0))
    _write_series(ts_dir, 5, np.full(n, 90.0))
    _write_series(ts_dir, 6, np.full(n, 20.0))
    sd = tmp_path / "stretches"; sd.mkdir()
    _stretches([("a", 1, 2, "3", ""),
                ("b", 4, 5, "", "6")]).to_csv(sd / "880_N.csv", index=False)
    out = tmp_path / "conservation.csv"

    rc = main(["--stretches", str(sd), "--timeseries-dir", str(ts_dir),
               "--out", str(out)])
    assert rc == 0
    table = pd.read_csv(out)
    assert list(table["ramp_type"]) == ["on", "off"]
    assert list(table["stretch_id"]) == ["880_N:0", "880_N:1"]
    assert {"rmse", "nrmse", "bias", "nbias"} <= set(table.columns)
    for suffix in ("nrmse", "nbias", "qq_on", "qq_off"):
        assert (tmp_path / f"conservation_{suffix}.png").exists()
