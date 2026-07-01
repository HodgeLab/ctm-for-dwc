"""Tests for ``scripts/build_pems_stretches.py`` (PeMS-native stretch builder)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from build_pems_stretches import build_stretches, classify_flow  # noqa: E402


def _meta(rows):
    return pd.DataFrame(rows, columns=["ID", "Fwy", "Dir", "Abs_PM", "Type", "Name"])


def test_classify_flow_from_type_and_name():
    assert classify_flow("OR", "Foo on") == "on"
    assert classify_flow("FR", "Foo off") == "off"
    assert classify_flow("FF", "I-238 to San Jose") == "off"   # 'to X' leaves the fwy
    assert classify_flow("FF", "from I-238") == "on"
    assert classify_flow("FF", "ramp off diag") == "off"       # explicit keyword wins
    assert classify_flow("FF", "NB 280") == "review"           # no on/off/to/from


def _corridor(direction):
    # ML at 0,1,2,3; a type-(c) interchange (on+off) between ML at 1 and 2.
    return _meta([
        (100, 880, direction, 0.0, "ML", "ml0"),
        (200, 880, direction, 1.0, "ML", "ml1"),
        (300, 880, direction, 2.0, "ML", "ml2"),
        (400, 880, direction, 3.0, "ML", "ml3"),
        (10, 880, direction, 1.5, "OR", "Foo on"),
        (20, 880, direction, 1.6, "FR", "Foo off"),
    ])


def test_type_c_stretch_from_on_and_off_between_two_ml():
    df = build_stretches(_corridor("N"), 880, "N")
    c = df[df.config_type == "c"]
    assert len(c) == 1
    row = c.iloc[0]
    assert row.on_ids == "10" and row.off_ids == "20"
    assert row.n_on == 1 and row.n_off == 1


def test_direction_sets_upstream_boundary():
    # N: upstream = lower Abs_PM; S: upstream = higher Abs_PM.
    n = build_stretches(_corridor("N"), 880, "N")
    n_c = n[n.config_type == "c"].iloc[0]
    assert n_c.up_ml_id == 200 and n_c.down_ml_id == 300

    s = build_stretches(_corridor("S"), 880, "S")
    s_c = s[s.config_type == "c"].iloc[0]
    assert s_c.up_ml_id == 300 and s_c.down_ml_id == 200


def test_open_end_when_ramp_upstream_of_first_ml():
    meta = _meta([
        (200, 880, "N", 1.0, "ML", "ml1"),
        (300, 880, "N", 2.0, "ML", "ml2"),
        (10, 880, "N", 0.3, "OR", "Foo on"),   # below the first ML
    ])
    df = build_stretches(meta, 880, "N")
    open_rows = df[df.notes.str.contains("open_end", na=False)]
    assert len(open_rows) == 1
    assert pd.isna(open_rows.iloc[0].up_ml_id)   # no upstream measurement
    assert open_rows.iloc[0].on_ids == "10"


def test_ff_lands_in_review_and_is_flagged():
    meta = _meta([
        (100, 880, "N", 0.0, "ML", "ml0"),
        (200, 880, "N", 2.0, "ML", "ml1"),
        (10, 880, "N", 1.0, "FF", "NB 280"),   # unresolvable connector
    ])
    df = build_stretches(meta, 880, "N")
    row = df.iloc[0]
    assert row.review_ids == "10"
    assert row.n_on == 0 and row.n_off == 0
    assert "ff_review" in row.notes
