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
