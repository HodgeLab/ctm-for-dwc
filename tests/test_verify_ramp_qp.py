"""Tests for ``scripts/verify_ramp_qp.py`` (Level 3 round-trip + tightness).

Builds a *self-consistent* bundle directly from a forward CTM run (so the
"fitted" trajectory is, by construction, an exact CTM solution) and checks
that ``round_trip`` reports zero density gap and zero loose operators. A
perturbed bundle then checks the diagnostics actually fire.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from transportation_models.utils.ctm import (
    freeway_from_dataframe,
    scenario_from_dataframes,
    simulate,
)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from verify_ramp_qp import round_trip  # noqa: E402

_FW = pd.DataFrame(
    {
        "length": [0.5, 0.4, 0.5],
        "q_max": [4000.0, 4000.0, 4000.0],
        "v_f": [65.0, 65.0, 65.0],
        "w": [11.818, 11.818, 11.818],
        "rho_jam": [400.0, 400.0, 400.0],
        "rho_crit": [61.538, 61.538, 61.538],
        "on_ramp": [False, True, False],
        "off_ramp": [False, False, True],
        "on_ramp_capacity": [np.nan, 1500.0, np.nan],
        "off_ramp_capacity": [np.nan, np.nan, 1200.0],
        "lanes": [2, 2, 2],
    }
)


def _self_consistent_bundle(out: Path, *, demand_scale: float = 1.0) -> None:
    """Forward-sim a known scenario and write it as a QP 'solution' bundle.

    With ``demand_scale == 1`` the written demand equals the simulated
    admitted flow, so the bundle is an exact CTM fixed point (tight). A
    scale != 1 perturbs the demand so the round-trip must diverge.
    """
    out.mkdir(parents=True, exist_ok=True)
    dt_s = 10.0
    dt_h = dt_s / 3600.0
    sp5, n_5min = 30, 4
    T = sp5 * n_5min
    N = len(_FW)

    freeway = freeway_from_dataframe(_FW, dt=dt_h)
    inflow = np.full(T, 3000.0)
    demand_in = pd.DataFrame({1: np.full(T, 400.0)})
    beta_in = pd.DataFrame({2: np.full(T, 0.10)})
    rho0 = np.array([50.0, 50.0, 50.0])
    res = simulate(
        freeway,
        scenario_from_dataframes(
            freeway, inflow=inflow, demand=demand_in, beta=beta_in,
            rho0=rho0, q0=0.0,
        ),
    )

    cols = [str(i) for i in range(N)]
    # demand = admitted on-ramp flow r (optionally perturbed); beta = input split.
    demand = np.zeros((T, N))
    demand[:, 1] = res.on_ramp[1, :] * demand_scale
    beta = np.zeros((T, N))
    beta[:, 2] = 0.10
    pd.DataFrame(demand, columns=cols).to_csv(out / "demand.csv", index=False)
    pd.DataFrame(beta, columns=cols).to_csv(out / "beta.csv", index=False)
    pd.DataFrame(res.density.T, columns=cols).to_csv(out / "rho_fitted.csv", index=False)
    pd.DataFrame(res.mainline_flow.T, columns=cols).to_csv(
        out / "mainline_flow_fitted.csv", index=False)
    pd.DataFrame(res.off_ramp.T, columns=cols).to_csv(
        out / "off_ramp_flow_fitted.csv", index=False)

    _FW.to_csv(out / "freeway.csv", index=False)
    pd.DataFrame({"cell": range(N), "rho0": rho0}).to_csv(out / "rho0.csv", index=False)
    pd.DataFrame({"k": range(T), "inflow": inflow}).to_csv(out / "inflow.csv", index=False)
    (out / "meta.json").write_text(json.dumps({
        "dt_h": dt_h, "n_cells": N, "gamma": 1.0, "T": T,
    }))


def test_round_trip_tight_on_self_consistent_solution(tmp_path):
    _self_consistent_bundle(tmp_path)
    report = round_trip(tmp_path)
    # An exact CTM trajectory reproduces itself and sits on its min bounds.
    assert report["density_gap_max"] < 1e-6
    assert report["n_loose_operators"] == 0
    assert report["loose_cells"] == []


def test_round_trip_detects_inconsistent_solution(tmp_path):
    # Inflated on-ramp demand breaks the fixed point: the forward sim now
    # admits different flow -> non-zero round-trip gap.
    _self_consistent_bundle(tmp_path, demand_scale=2.0)
    report = round_trip(tmp_path)
    assert report["density_gap_max"] > 1e-3
