"""Tests for ``scripts/validate_ctm_corridor.py`` start-resolution logic.

The script now loads the self-contained ``result.npz`` (which carries the sim's
wallclock ``start``) and makes ``--start`` an optional override, so the window
start has a single source of truth. ``resolve_start`` encodes that policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from validate_ctm_corridor import resolve_start  # noqa: E402


def test_resolve_start_prefers_baked_in():
    baked = pd.Timestamp("2023-06-01 06:00")
    assert resolve_start(None, baked) == baked


def test_resolve_start_falls_back_to_cli_for_legacy_npz():
    # Older result.npz with no persisted start: --start is the only source.
    cli = pd.Timestamp("2023-06-01 06:00")
    assert resolve_start(cli, None) == cli


def test_resolve_start_accepts_matching_override():
    ts = pd.Timestamp("2023-06-01 06:00")
    assert resolve_start(ts, ts) == ts


def test_resolve_start_rejects_disagreeing_override():
    with pytest.raises(ValueError, match="disagrees"):
        resolve_start(
            pd.Timestamp("2023-06-01 06:00"), pd.Timestamp("2023-06-01 00:00")
        )


def test_resolve_start_requires_a_source():
    with pytest.raises(ValueError, match="no persisted start"):
        resolve_start(None, None)
