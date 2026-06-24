"""Macro-vs-micro validation tests (T7/T8).

The headline regression (CP5): in the uniform-density limit, the adapted mCONV
(`dwpt.compute` over a CTM-equivalent VHT) and the original mCONV (over micro
trajectories) must agree on corridor-aggregate DWPT energy to relative error
< 1e-6. This is fast and runs in the default suite; the slow end-to-end driver
smoke (CP6) is added by T8 and marked ``slow``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
    E_micro = micro_demand(sc.micro_result, sc.corridor).E
    total_micro = E_micro.sum()

    # Adapted mCONV over the CTM-equivalent VHT.
    adapted = compute(sc.vht, sc.corridor, eta_EV=sc.eta_ev)
    total_adapted = adapted.E.sum()

    rel_err = abs(total_micro - total_adapted) / total_adapted
    assert rel_err < 1e-6


def test_uniform_limit_matches_per_position():
    sc = uniform_density_scenario(n_vehicles=10)
    E_micro = micro_demand(sc.micro_result, sc.corridor).E.sum(axis=0)
    adapted = compute(sc.vht, sc.corridor, eta_EV=sc.eta_ev)
    # Energy per position, summed over time, must match the micro deposits.
    np.testing.assert_allclose(adapted.E.sum(axis=0), E_micro, rtol=1e-9, atol=1e-12)


@pytest.mark.slow
def test_end_to_end_driver_on_1mi_case(tmp_path):
    """CP6: the driver runs Stage 2 (+ Stage 1 timings) on the real case."""
    case_dir = _REPO / "scripts" / "output" / "1mi_case_study"
    if not (case_dir / "sim_no_ramps" / "result.npz").exists():
        pytest.skip("1mi_case_study not materialized on this machine")

    import pandas as pd
    from run_dwpt_validation import (
        _ctm_traffic_metrics,
        _micro_traffic_metrics,
        build_and_run_micro,
        ctm_inflow_rate,
        plot_flow_contours,
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

    dt = 1.0
    n_steps = int(round(0.25 * 3600 / dt))
    inflow = ctm_inflow_rate(result, dt, n_steps)  # no PeMS on CI -> fallback
    n_lanes = int(cells_df["lanes"].max())
    run = build_and_run_micro(
        result, inflow, n_lanes=n_lanes, pad=pad, dx_grid=0.5, dt=dt,
        eta_ev=0.3, seed=0,
    )
    assert run.result.n_lanes == n_lanes
    # Micro produced trajectories and the collision guard reported stats.
    assert run.result.guard_stats is not None
    assert np.nanmax(run.result.position) > 0.0

    # Stage 1 timings exist (PeMS scoring skipped without timeseries).
    stage1 = run_stage1(run, result, cells_df, timeseries_dir=None, start=None)
    assert stage1["micro_wall_s"] > 0.0
    assert stage1["ctm_wall_s"] > 0.0

    # Stage 2 emits finite energy + peak-power divergences with positive sides.
    stage2 = run_stage2(run, result, eta_ev=0.3)
    assert stage2["total_adapted_wh"] > 0.0
    assert stage2["total_micro_wh"] > 0.0
    assert np.isfinite(stage2["energy_divergence"])
    assert stage2["peak_adapted_w"] > 0.0
    assert stage2["peak_micro_w"] > 0.0
    assert np.isfinite(stage2["peak_divergence"])

    # Stage-1 traffic metrics are sane, and the flow-contour plot is written.
    mtm = _micro_traffic_metrics(run)
    ctm = _ctm_traffic_metrics(result, run.n_ctm_steps)
    assert mtm["vht_h"] > 0 and mtm["vmt_mi"] > 0
    assert mtm["n_completed"] <= mtm["n_entered"]
    assert ctm["vht_h"] > 0 and ctm["vmt_mi"] > 0
    plot = plot_flow_contours(tmp_path, run, result)
    assert plot.exists()


# ---- CTM/micro window alignment (--start later than the CTM sim start) -----


def test_ctm_window_offset_basic():
    from run_dwpt_validation import ctm_window_offset

    baked = pd.Timestamp("2023-06-01 00:00")
    # 3 h into a 0.5-h-step sim -> step 6; an on-grid start lands exactly.
    assert ctm_window_offset("2023-06-01 03:00", baked, 0.5, 20) == 6
    # start == baked -> no offset (reproduces today's front-window behavior).
    assert ctm_window_offset(baked, baked, 0.5, 20) == 0


def test_ctm_window_offset_requires_persisted_start():
    from run_dwpt_validation import ctm_window_offset

    with pytest.raises(ValueError, match="no persisted start"):
        ctm_window_offset("2023-06-01 03:00", None, 0.5, 20)


def test_ctm_window_offset_rejects_out_of_range():
    from run_dwpt_validation import ctm_window_offset

    baked = pd.Timestamp("2023-06-01 03:00")
    with pytest.raises(ValueError, match="before the CTM result start"):
        ctm_window_offset("2023-06-01 00:00", baked, 0.5, 20)
    # step 20 is past the last index of a 20-step result.
    with pytest.raises(ValueError, match="at/after the CTM result end"):
        ctm_window_offset("2023-06-01 13:00", baked, 0.5, 20)


@dataclass
class _Cell:
    length: float = 2.0


@dataclass
class _Freeway:
    dt: float
    cells: list

    @property
    def n_cells(self) -> int:
        return len(self.cells)


@dataclass
class _Result:
    """Minimal CTM-result stand-in: only what _ctm_traffic_metrics reads."""
    freeway: _Freeway
    density: np.ndarray       # (n_cells, T+1)
    mainline_flow: np.ndarray  # (n_cells, T)

    @property
    def n_steps(self) -> int:
        return self.mainline_flow.shape[1]


def _ramp_result(t_steps: int = 10, dt_h: float = 0.5):
    # One cell, flow and post-step density both rising linearly with the step
    # index, so a windowed metric over [o, o+k) is exactly sum over that range.
    flow = np.arange(t_steps, dtype=float)[None, :]          # flow[0,t] = t
    density = np.arange(t_steps + 1, dtype=float)[None, :]   # density[0,k] = k
    return _Result(_Freeway(dt_h, [_Cell()]), density, flow)


def test_ctm_traffic_metrics_reads_the_offset_window():
    from run_dwpt_validation import _ctm_traffic_metrics

    res = _ramp_result()           # L=2, dt=0.5 -> L*dt = 1.0
    k = 3
    # offset 0: steps 0,1,2 -> VMT = (0+1+2)*1, VHT = (1+2+3)*1.
    front = _ctm_traffic_metrics(res, k, 0)
    assert front["vmt_mi"] == 3.0 and front["vht_h"] == 6.0
    # offset 4: steps 4,5,6 -> VMT = (4+5+6)*1, VHT = (5+6+7)*1.
    windowed = _ctm_traffic_metrics(res, k, 4)
    assert windowed["vmt_mi"] == 15.0 and windowed["vht_h"] == 18.0


# ---- CTM offset re-run: seeds from the mid-window state, not t=0 -----------


def test_ctm_window_rerun_reproduces_original_window():
    """The offset re-run must reproduce the original CTM trajectory over the
    window -- which holds iff it seeds from the CTM's actual state at the offset
    step (density + queue). Seeding from the t=0 rho0 would fail this.
    """
    from run_dwpt_validation import ctm_window_rerun
    from transportation_models.utils.ctm import examples, simulate

    result = simulate(
        examples.four_cell_freeway(), examples.four_cell_scenario(steps=240)
    )
    o, t = 60, 60
    # The window must start from a genuinely evolved state (rho0 is all zeros),
    # otherwise seeding from rho0 would pass trivially.
    assert not np.allclose(result.density[:, o], result.scenario.rho0)

    rerun = ctm_window_rerun(result, o, t)
    assert rerun.n_steps == t
    np.testing.assert_allclose(
        rerun.density, result.density[:, o : o + t + 1], rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        rerun.mainline_flow, result.mainline_flow[:, o : o + t],
        rtol=0, atol=1e-9,
    )
    # offset 0 reproduces the original front window.
    front = ctm_window_rerun(result, 0, t)
    np.testing.assert_allclose(
        front.density, result.density[:, : t + 1], rtol=0, atol=1e-9
    )


def _write_vds_csv(path, *, start, flows_5min, speeds_mph, station_id):
    n = len(flows_5min)
    pd.DataFrame({
        "timestamp": pd.date_range(start=start, periods=n, freq="5min"),
        "station": station_id,
        "pct_observed": 100.0,
        "total_flow_[veh/5-min]": flows_5min,
        "avg_speed_[mph]": speeds_mph,
    }).to_csv(path, index=False)


@pytest.mark.slow
def test_offset_rerun_scored_against_pems_honors_window(tmp_path):
    """End-to-end: ctm_window_rerun -> compare_against_historical. PeMS is held
    at the corridor's constant equilibrium; the offset window sits in the
    equilibrium tail (matches PeMS ~exactly) while the front window includes the
    rho0=0 transient (mismatches), proving the offset re-run is scored against
    the correctly aligned PeMS window.
    """
    from run_dwpt_validation import ctm_window_rerun
    from transportation_models.utils.ctm import (
        compare_against_historical, examples, simulate,
    )

    result = simulate(
        examples.four_cell_freeway(), examples.four_cell_scenario(steps=240)
    )
    n_cells = result.n_cells
    d_eq = result.density[:, -1]        # constant equilibrium density per cell
    f_eq = result.mainline_flow[:, -1]  # constant equilibrium flow per cell
    vds_ids = list(range(101, 101 + n_cells))
    base = pd.Timestamp("2022-04-12 06:00")
    # 30 5-min bins of constant equilibrium PeMS -> covers front and tail.
    for vid, fe, de in zip(vds_ids, f_eq, d_eq):
        _write_vds_csv(
            tmp_path / f"{vid}.csv", start=base,
            flows_5min=np.full(30, fe / 12.0),   # veh/5min -> fe veh/h
            speeds_mph=np.full(30, fe / de),     # so flow/speed == de
            station_id=vid,
        )
    cells_df = pd.DataFrame({
        "vds_id": vds_ids, "lanes": [1] * n_cells, "vds_lanes": [1] * n_cells,
    })

    t, o = 40, 200  # 4 5-min bins (dt=30s); offset 200 steps = +100 min
    off = compare_against_historical(
        ctm_window_rerun(result, o, t), cells_df, tmp_path,
        start=base + pd.Timedelta(seconds=o * 30),
    )
    front = compare_against_historical(
        ctm_window_rerun(result, 0, t), cells_df, tmp_path, start=base,
    )
    # Aligned equilibrium tail matches PeMS to ~0; the transient front does not.
    assert (off["flow_rmse"] < 1e-6).all()
    assert (off["density_rmse"] < 1e-6).all()
    assert front["flow_rmse"].max() > 100.0
