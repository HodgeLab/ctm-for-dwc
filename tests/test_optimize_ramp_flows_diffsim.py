"""Tests for ``scripts/optimize_ramp_flows_diffsim.py`` (Level 3 diffsim optimizer).

Builds a synthetic bundle whose observed density+flow come from *known*
block-constant ramp flows (via the exact numpy engine), then checks the
optimizer (1) fits density+flow, (2) recovers the held-out ramp flows on
single-ramp cells (unique inverse), and (3) writes wide demand/beta CSVs that
round-trip through the real ``simulate`` -- i.e. are drop-in for
``simulate_ctm_corridor.py`` / ``validate_ctm_corridor.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from optimize_ramp_flows_diffsim import optimize_bundle  # noqa: E402

from ctm_for_dwc.utils.ctm import (  # noqa: E402
    freeway_from_dataframe, scenario_from_dataframes, simulate,
)
from ctm_for_dwc.utils.ctm.model import Scenario  # noqa: E402

# cell 0 plain, cell 1 on-ramp only, cell 2 off-ramp only (single-ramp cells ->
# each ramp flow is uniquely identifiable from mainline density+flow).
_FW = pd.DataFrame({
    "length": [0.5, 0.4, 0.5], "q_max": [4000.0] * 3, "v_f": [65.0] * 3,
    "w": [11.818] * 3, "rho_jam": [400.0] * 3, "rho_crit": [61.538] * 3,
    "on_ramp": [False, True, False], "off_ramp": [False, False, True],
    "on_ramp_capacity": [np.nan, 1500.0, np.nan],
    "off_ramp_capacity": [np.nan, np.nan, np.nan], "lanes": [2] * 3,
})
_SP5, _N5MIN = 15, 6
_T, _N = _SP5 * _N5MIN, 3
_DT_S = 20.0


def _write_synthetic_bundle(tmp_path: Path):
    """Write a bundle from known block-constant ramps; return (dir, truth res)."""
    dt_h = _DT_S / 3600.0
    fwy = freeway_from_dataframe(_FW, dt=dt_h)
    tb = np.linspace(0, 1, _N5MIN)
    demand = np.zeros((_N, _T)); beta = np.zeros((_N, _T))
    demand[1] = np.repeat(500 + 200 * np.sin(2 * np.pi * tb), _SP5)
    beta[2] = np.repeat(0.25 + 0.1 * np.sin(2 * np.pi * tb), _SP5)
    rho0 = np.array([24.0, 26.0, 22.0]); inflow = np.full(_T, 3200.0)

    scn = Scenario.build(fwy, _T, inflow=inflow, demand=demand, beta=beta, rho0=rho0, q0=0.0)
    res = simulate(fwy, scn)
    rho5 = res.density[:, : _N5MIN * _SP5 : _SP5]
    flow5 = res.mainline_flow[:, : _N5MIN * _SP5].reshape(_N, _N5MIN, _SP5).mean(axis=2)

    b = tmp_path / "bundle"; b.mkdir()
    _FW.to_csv(b / "freeway.csv", index=False)
    pd.DataFrame({"cell": range(_N), "rho0": rho0}).to_csv(b / "rho0.csv", index=False)
    pd.DataFrame({"k": range(_T), "inflow": inflow}).to_csv(b / "inflow.csv", index=False)
    pd.DataFrame(
        [{"cell": c, "m_5min": m, "rho_obs": rho5[c, m]} for c in range(_N) for m in range(_N5MIN)]
    ).to_csv(b / "observed_density.csv", index=False)
    pd.DataFrame(
        [{"cell": c, "m_5min": m, "flow_obs": flow5[c, m]} for c in range(_N) for m in range(_N5MIN)]
    ).to_csv(b / "observed_flow.csv", index=False)
    (b / "meta.json").write_text(json.dumps({
        "dt_s": _DT_S, "dt_h": dt_h, "steps_per_5min": _SP5, "n_5min": _N5MIN,
        "T": _T, "n_cells": _N, "gamma": 1.0, "xi": 1.0,
        "on_ramp_cells": [1], "off_ramp_cells": [2],
        "on_ramp_capacity": {"1": 1500.0}, "off_ramp_capacity": {},
        "direct_tiebreak_cells": [0, 1, 2],
    }))
    return b, res, rho0, inflow


def test_optimizer_recovers_ramps_and_output_roundtrips(tmp_path):
    bundle, truth, rho0, inflow = _write_synthetic_bundle(tmp_path)

    report = optimize_bundle(bundle, iters=500)

    # (1) density + flow fit (the objective), reported as RMSE and MAPE.
    assert report["density_rmse"] < 1e-2, report
    assert report["flow_rmse"] < 1.0, report
    assert report["density_mape"] < 1.0 and report["flow_mape"] < 1.0, report
    assert (bundle / "optimize_report.json").exists()

    # (3) written CSVs round-trip through the real simulator ...
    dt_h = _DT_S / 3600.0
    fwy = freeway_from_dataframe(_FW, dt=dt_h)
    demand_df = pd.read_csv(bundle / "demand.csv"); demand_df.columns = demand_df.columns.astype(int)
    beta_df = pd.read_csv(bundle / "beta.csv"); beta_df.columns = beta_df.columns.astype(int)
    assert demand_df.shape == (_T, _N) and beta_df.shape == (_T, _N)
    scn = scenario_from_dataframes(fwy, inflow=inflow, demand=demand_df, beta=beta_df, rho0=rho0, q0=0.0)
    res = simulate(fwy, scn)

    # ... and (2) reproduce the held-out ramp flows (unique inverse here).
    r_rmse = np.sqrt(np.mean((res.on_ramp[1] - truth.on_ramp[1]) ** 2))
    s_rmse = np.sqrt(np.mean((res.off_ramp[2] - truth.off_ramp[2]) ** 2))
    assert r_rmse < 1.0, f"on-ramp flow not recovered: {r_rmse}"   # truth ~500 veh/h
    assert s_rmse < 1.0, f"off-ramp flow not recovered: {s_rmse}"

    # non-ramp cells carry zero demand / zero beta.
    assert np.allclose(demand_df[0], 0.0) and np.allclose(demand_df[2], 0.0)
    assert np.allclose(beta_df[0], 0.0) and np.allclose(beta_df[1], 0.0)


def test_ramp_inputs_are_block_constant(tmp_path):
    """Recovered demand/beta are piecewise-constant over each 5-min block."""
    bundle, *_ = _write_synthetic_bundle(tmp_path)
    optimize_bundle(bundle, iters=200)
    for name, cell in [("demand", 1), ("beta", 2)]:
        col = pd.read_csv(bundle / f"{name}.csv")[str(cell)].to_numpy()
        blocks = col.reshape(_N5MIN, _SP5)
        assert np.allclose(blocks, blocks[:, [0]]), f"{name} varies within a 5-min block"


def test_early_stopping_halts_before_max_iters(tmp_path):
    """Early stopping triggers on loss plateau and restores the best iterate."""
    bundle, *_ = _write_synthetic_bundle(tmp_path)
    report = optimize_bundle(bundle, iters=5000, patience=30, min_delta=1e-2)
    assert report["stopped_early"] is True
    assert report["iters_run"] < 5000
    assert report["best_iter"] <= report["iters_run"]
    assert np.isfinite(report["density_rmse"]) and np.isfinite(report["flow_rmse"])


def test_no_early_stopping_runs_all_iters(tmp_path):
    bundle, *_ = _write_synthetic_bundle(tmp_path)
    report = optimize_bundle(bundle, iters=50, patience=None)
    assert report["stopped_early"] is False
    assert report["iters_run"] == 50


def test_log_fn_receives_progress_rows(tmp_path):
    """The W&B seam: log_fn fires every log_every iters with the metric keys."""
    bundle, *_ = _write_synthetic_bundle(tmp_path)
    rows = []
    optimize_bundle(bundle, iters=10, log_every=1, log_fn=rows.append)
    assert len(rows) == 10
    assert set(rows[0]) == {"iter", "loss", "density_rmse", "flow_rmse",
                            "density_mape", "flow_mape"}
    assert rows[0]["iter"] == 0 and rows[-1]["iter"] == 9
