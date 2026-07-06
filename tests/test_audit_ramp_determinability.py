"""Tests for ``scripts/audit_ramp_determinability.py``.

Covers the pure window-scanning core (``sliding_all``,
``count_determinable_windows``) with injected observation masks, plus a light
integration check of ``build_observation_windows`` against tiny temp CSVs -- so
nothing touches the real multi-year record.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from audit_ramp_determinability import (  # noqa: E402
    build_observation_windows,
    count_determinable_windows,
    sliding_all,
)

_M = 5  # window count in the injected-mask tests


def _win(masks):
    false = np.zeros(_M, dtype=bool)
    return lambda sid: np.asarray(masks.get(sid, false), dtype=bool)


def _type_c_row(on="10", off="20", review=""):
    return {"up_ml_id": 100, "down_ml_id": 200, "on_ids": on, "off_ids": off,
            "review_ids": review, "n_on": len(on.split(";")) if on else 0,
            "n_off": len(off.split(";")) if off else 0, "config_type": "c"}


def _all(v=True):
    return np.full(_M, v, dtype=bool)


# --- sliding_all ---------------------------------------------------------- #
def test_sliding_all_ands_over_window():
    arr = [True, True, True, False, True, True]
    np.testing.assert_array_equal(sliding_all(arr, 3), [True, False, False, False])


def test_sliding_all_empty_when_record_shorter_than_window():
    assert sliding_all([True, True], 3).shape == (0,)


# --- count_determinable_windows ------------------------------------------ #
def test_all_observed_type_c_fills_corpus():
    win = _win({100: _all(), 200: _all(), 10: _all(), 20: _all()})
    rec = count_determinable_windows(_type_c_row(), win, _M, min_windows=1)
    assert rec["n_windows"] == _M
    assert rec["training_corpus"] and rec["clean_pair"]


def test_one_unobserved_ramp_still_determinable_by_conservation():
    win = _win({100: _all(), 200: _all(), 10: _all(), 20: _all(False)})  # off never observed
    rec = count_determinable_windows(_type_c_row(), win, _M, min_windows=1)
    assert rec["n_windows"] == _M          # off recovered by conservation
    assert rec["training_corpus"]


def test_two_unobserved_ramps_are_not_determinable():
    win = _win({100: _all(), 200: _all(), 10: _all(False), 20: _all(False)})
    rec = count_determinable_windows(_type_c_row(), win, _M, min_windows=1)
    assert rec["n_windows"] == 0
    assert not rec["training_corpus"]


def test_fully_determined_excludes_conservation_windows():
    # off-ramp never observed: conservation-resolvable (default) but not fully measured.
    win = _win({100: _all(), 200: _all(), 10: _all(), 20: _all(False)})
    default = count_determinable_windows(_type_c_row(), win, _M, min_windows=1)
    strict = count_determinable_windows(_type_c_row(), win, _M, min_windows=1,
                                        fully_determined=True)
    assert default["n_windows"] == _M and default["training_corpus"]
    assert strict["n_windows"] == 0 and not strict["training_corpus"]


def test_fully_determined_keeps_both_measured_windows():
    win = _win({100: _all(), 200: _all(), 10: _all(), 20: _all()})
    rec = count_determinable_windows(_type_c_row(), win, _M, min_windows=1,
                                     fully_determined=True)
    assert rec["n_windows"] == _M and rec["training_corpus"]


def test_mainline_gaps_reduce_window_count():
    down = np.array([True, True, True, False, False])
    win = _win({100: _all(), 200: down, 10: _all(), 20: _all()})
    rec = count_determinable_windows(_type_c_row(), win, _M, min_windows=1)
    assert rec["n_windows"] == 3


def test_virtual_ramp_is_permanent_unknown_but_conservable():
    # off-ramp has no PeMS VDS (virtual string id) -> never observed.
    win = _win({100: _all(), 200: _all(), 10: _all()})
    rec = count_determinable_windows(_type_c_row(off="V_off"), win, _M, min_windows=1)
    assert rec["n_windows"] == _M
    assert rec["training_corpus"]


def test_ff_review_blocks_corpus():
    win = _win({100: _all(), 200: _all(), 10: _all(), 20: _all(), 30: _all()})
    rec = count_determinable_windows(_type_c_row(review="30"), win, _M, min_windows=1)
    assert rec["has_review"]
    assert not rec["training_corpus"]


def test_min_windows_threshold_gates_corpus():
    down = np.array([True, True, True, False, False])   # only 3 determinable windows
    win = _win({100: _all(), 200: down, 10: _all(), 20: _all()})
    rec = count_determinable_windows(_type_c_row(), win, _M, min_windows=5)
    assert rec["n_windows"] == 3
    assert not rec["training_corpus"]


# --- build_observation_windows (light integration) ------------------------ #
def test_build_observation_windows_loads_and_reduces(tmp_path):
    ts = pd.date_range("2023-06-01 00:00", periods=6, freq="5min")
    pd.DataFrame({
        "timestamp": ts,
        "total_flow_[veh/5-min]": [10, 10, np.nan, 10, 10, 10],  # gap at index 2
        "pct_observed": [100, 100, 100, 100, 100, 100],
    }).to_csv(tmp_path / "100.csv", index=False)

    win, n_windows, grid, missing = build_observation_windows(
        [100, 999], tmp_path, pct_floor=0.0, window=3,
        record_start=None, record_end=None,
    )
    assert n_windows == 4 and len(grid) == 6
    assert missing == [999]                                   # no file
    np.testing.assert_array_equal(win(100), [False, False, False, True])
    np.testing.assert_array_equal(win(999), np.zeros(4, dtype=bool))  # missing -> all False
