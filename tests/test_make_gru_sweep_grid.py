"""Tests for scripts/make_gru_sweep_grid.py (sweep grid enumeration)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from make_gru_sweep_grid import grid, main  # noqa: E402


def test_grid_has_252_unique_pruned_trials():
    lines = grid()
    assert len(lines) == 252                      # 36 (1-layer) + 216 (2-3 layer)
    assert len(set(lines)) == 252
    for line in lines:
        if "--num-layers 1 " in line:
            assert "--dropout 0.0" in line        # inter-layer dropout pruned


def test_main_writes_grid_file(tmp_path, capsys):
    out = tmp_path / "sweep_grid.txt"
    assert main(["--out", str(out)]) == 0
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 252
    assert "--array=0-251" in capsys.readouterr().out
