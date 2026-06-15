"""Macro-vs-micro validation tests (T7/T8).

The headline regression (CP5): in the uniform-density limit, the adapted mCONV
(`dwpt.compute` over a CTM-equivalent VHT) and the original mCONV (over micro
trajectories) must agree on corridor-aggregate DWPT energy to relative error
< 1e-6. This is fast and runs in the default suite; the slow end-to-end driver
smoke (CP6) is added by T8 and marked ``slow``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from transportation_models.utils.dwpt import compute
from transportation_models.utils.microsim.charging import micro_demand
from transportation_models.utils.microsim.examples import uniform_density_scenario

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))


@pytest.mark.parametrize("n_vehicles", [1, 10, 37])
def test_uniform_limit_macro_micro_equivalence(n_vehicles):
    sc = uniform_density_scenario(n_vehicles=n_vehicles)

    # Original mCONV over micro trajectories.
    E_micro = micro_demand(sc.micro_result, sc.corridor)
    total_micro = E_micro.sum()

    # Adapted mCONV over the CTM-equivalent VHT.
    adapted = compute(sc.vht, sc.corridor, eta_EV=sc.eta_ev)
    total_adapted = adapted.E.sum()

    rel_err = abs(total_micro - total_adapted) / total_adapted
    assert rel_err < 1e-6


def test_uniform_limit_matches_per_position():
    sc = uniform_density_scenario(n_vehicles=10)
    E_micro = micro_demand(sc.micro_result, sc.corridor)
    adapted = compute(sc.vht, sc.corridor, eta_EV=sc.eta_ev)
    # Energy per position, summed over time, must match the micro deposits.
    np.testing.assert_allclose(adapted.E.sum(axis=0), E_micro, rtol=1e-9, atol=1e-12)


@pytest.mark.slow
def test_end_to_end_driver_on_1mi_case():
    """CP6: the driver runs Stage 2 (+ Stage 1 timings) on the real case."""
    case_dir = _REPO / "scripts" / "output" / "1mi_case_study"
    if not (case_dir / "sim_no_ramps" / "result.npz").exists():
        pytest.skip("1mi_case_study not materialized on this machine")

    import pandas as pd
    from run_dwpt_validation import (
        build_and_run_micro,
        run_stage1,
        run_stage2,
    )

    from transportation_models.utils.ctm.results import SimulationResult
    from transportation_models.utils.dwpt.model import PadSpec

    result = SimulationResult.from_npz(case_dir / "sim_no_ramps" / "result.npz")
    cells_df = pd.read_csv(case_dir / "cells.csv")
    pad = PadSpec(
        alpha=3.5, delta=1.0, lambda_gap=0.5, beta=150000.0, beta_prime=150000.0
    )

    run = build_and_run_micro(
        result, pad=pad, dx_grid=0.5, dt=1.0, eta_ev=0.3, seed=0, hours=0.25
    )
    # Micro produced trajectories and the collision guard reported stats.
    assert run.result.guard_stats is not None
    assert np.nanmax(run.result.position) > 0.0

    # Stage 1 timings exist (PeMS scoring skipped without timeseries).
    stage1 = run_stage1(run, result, cells_df, timeseries_dir=None, start=None)
    assert stage1["micro_wall_s"] > 0.0
    assert stage1["ctm_wall_s"] > 0.0

    # Stage 2 emits a finite divergence with positive energy on both sides.
    stage2 = run_stage2(run, result, eta_ev=0.3)
    assert stage2["total_adapted_wh"] > 0.0
    assert stage2["total_micro_wh"] > 0.0
    assert np.isfinite(stage2["relative_divergence"])
