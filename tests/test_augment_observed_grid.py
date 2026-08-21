"""Integration test for ``scripts/augment_observed_grid.py`` + the optimizer's
imputed-grid loader.

Builds a tiny 5-cell / 3-step bundle with direct cells 0/2/4 and Type-1 cells
1/3, plus a one-step Type-2 gap at cell 0. Checks that the gap is filled from
the detector's ``(dow, time-of-day)`` history, the Type-1 columns are filled by
the k=1 up/down average, the ``source`` labels are written, and that
``optimize_ramp_flows_diffsim._imputed_grid`` turns them into the right
value/weight/genuine grids.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from augment_observed_grid import augment_bundle  # noqa: E402
from optimize_ramp_flows_diffsim import _imputed_grid  # noqa: E402

# density = flow_veh_5min * 12 / speed; here 100*12/50 = 24, 1000-flow etc below.
_START = "2022-04-04 08:00"  # Monday
_N, _N5MIN = 5, 3


def _write_bundle(tmp_path: Path):
    b = tmp_path / "bundle"
    b.mkdir()
    (b / "meta.json").write_text(json.dumps(
        {"n_cells": _N, "n_5min": _N5MIN, "start": _START}))
    # direct cells 0/2/4 fully present except a Type-2 gap at cell 0, m=1.
    dens = {0: [10.0, None, 10.0], 2: [30.0, 30.0, 30.0], 4: [50.0, 50.0, 50.0]}
    flow = {0: [1000.0, None, 1000.0], 2: [3000.0, 3000.0, 3000.0], 4: [5000.0, 5000.0, 5000.0]}
    drows = [{"cell": c, "m_5min": m, "rho_obs": v}
             for c, col in dens.items() for m, v in enumerate(col) if v is not None]
    frows = [{"cell": c, "m_5min": m, "flow_obs": v}
             for c, col in flow.items() for m, v in enumerate(col) if v is not None]
    pd.DataFrame(drows).to_csv(b / "observed_density.csv", index=False)
    pd.DataFrame(frows).to_csv(b / "observed_flow.csv", index=False)

    cells = pd.DataFrame({
        "vds_id": [100, 100, 200, 200, 300],
        "vds_source": ["direct", "nearest_upstream", "direct",
                       "nearest_upstream", "direct"],
    })
    cells_path = tmp_path / "cells.csv"
    cells.to_csv(cells_path, index=False)

    # VDS 100 history: the window's 08:05 is genuinely missing (NaN speed), but a
    # prior Monday 08:05 is observed, so the slot mean fills the gap to 24 veh/mi
    # (100*12/50) and 1200 veh/h flow. VDS 200/300 have no gaps -> never read.
    ts_dir = tmp_path / "ts"
    ts_dir.mkdir()
    pd.DataFrame({
        "timestamp": ["2022-03-28 08:05", "2022-04-04 08:00",
                      "2022-04-04 08:05", "2022-04-04 08:10"],
        "total_flow_[veh/5-min]": [100.0, 100.0, float("nan"), 100.0],
        "avg_speed_[mph]": [50.0, 50.0, float("nan"), 50.0],
    }).to_csv(ts_dir / "100.csv", index=False)
    return b, cells_path, ts_dir


def _grid(df):
    return df.set_index(["cell", "m_5min"])


def test_augment_fills_type2_history_and_type1_knn(tmp_path):
    b, cells_path, ts_dir = _write_bundle(tmp_path)
    cfg = augment_bundle(b, cells_path, ts_dir, k=1, knn_decay=1.0)
    out = Path(cfg["out_dir"])
    assert out == b / "imputed" / "k1_d1.0"

    dens = _grid(pd.read_csv(out / "observed_density.csv"))
    flow = _grid(pd.read_csv(out / "observed_flow.csv"))

    # Type-2: cell 0, m=1 filled from history (100*12/50 = 24; flow 1200).
    assert dens.loc[(0, 1), "source"] == "hist"
    assert dens.loc[(0, 1), "value"] == 24.0
    assert flow.loc[(0, 1), "source"] == "hist"
    assert flow.loc[(0, 1), "value"] == 1200.0

    # Type-1: cell 1 = up/down average of cells 0 and 2; cell 3 of cells 2 and 4.
    assert dens.loc[(1, 0), "source"] == "knn"
    assert dens.loc[(1, 0), "value"] == (10.0 + 30.0) / 2
    assert dens.loc[(1, 1), "value"] == (24.0 + 30.0) / 2   # uses the hist-filled cell 0
    assert dens.loc[(3, 0), "value"] == (30.0 + 50.0) / 2
    assert flow.loc[(1, 0), "value"] == (1000.0 + 3000.0) / 2

    # Full grid now covered, and the observed cells are untouched.
    assert len(dens) == _N * _N5MIN
    assert dens.loc[(2, 0), "source"] == "observed" and dens.loc[(2, 0), "value"] == 30.0
    assert cfg["counts"]["density"] == {"observed": 8, "hist": 1, "knn": 6, "unfilled": 0}


def test_imputed_grid_loader_weights(tmp_path):
    b, cells_path, ts_dir = _write_bundle(tmp_path)
    cfg = augment_bundle(b, cells_path, ts_dir, k=1, knn_decay=1.0)
    obs, weight, genuine = _imputed_grid(
        Path(cfg["out_dir"]) / "observed_density.csv", _N, _N5MIN, imputed_weight=0.3)

    # genuine observation: weight 1, genuine 1.
    assert weight[2, 0] == 1.0 and genuine[2, 0] == 1.0 and obs[2, 0] == 30.0
    # hist-filled and knn-filled: down-weighted, not counted as genuine.
    assert weight[0, 1] == 0.3 and genuine[0, 1] == 0.0
    assert weight[1, 0] == 0.3 and genuine[1, 0] == 0.0
    assert obs[1, 0] == 20.0
    assert int(genuine.sum()) == 8 and int((weight > 0).sum()) == _N * _N5MIN
