"""Run the two-stage DWPT macro-vs-micro validation study (P7).

Given a prepared case-study directory (corridor geometry + CTM
``SimulationResult`` + boundary inflow, e.g. ``scripts/output/1mi_case_study``),
this driver:

* builds and runs the single-lane Newbolt microsim over the corridor;
* **Stage 1 (traffic fidelity):** aggregates the micro trajectories to per-cell
  density/flow (Edie) and scores them against PeMS with the same
  ``compare_against_historical`` path the CTM uses, alongside the existing CTM
  ``validation.csv`` and wall-clock compute for each engine;
* **Stage 2 (demand fidelity):** compares the original mCONV over micro
  trajectories against the adapted mCONV over the CTM VHT, on the actual
  corridor field.

Stage 1's PeMS scoring needs the per-VDS timeseries (a runtime dependency);
omit ``--timeseries-dir`` to run Stage 2 only. See
``docs/dwpt_validation_spec.md``.

Run from the repo root::

    python scripts/run_dwpt_validation.py \\
        --case-dir scripts/output/1mi_case_study \\
        --timeseries-dir data/pems/csv_files --start "2022-04-12 06:00"
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from transportation_models.utils.ctm import simulate as ctm_simulate
from transportation_models.utils.ctm.results import SimulationResult
from transportation_models.utils.ctm.validation import compare_against_historical
from transportation_models.utils.dwpt.adapter import (
    corridor_from_ctm,
    mainline_vht,
)
from transportation_models.utils.dwpt.demand import compute
from transportation_models.utils.dwpt.model import PadSpec
from transportation_models.utils.microsim import MPH_TO_MS, MicrosimSpec
from transportation_models.utils.microsim.aggregate import (
    METERS_PER_MILE,
    aggregate_to_cells,
    to_validation_arrays,
)
from transportation_models.utils.microsim.charging import micro_demand
from transportation_models.utils.microsim.seeding import seed_vehicles
from transportation_models.utils.microsim.simulate import simulate

_SECONDS_PER_HOUR = 3600.0


@dataclass
class MicroRun:
    """A completed micro run plus the geometry/timing the stages need."""

    result: object  # MicrosimResult
    corridor: object  # CorridorSpec (snapped)
    cell_edges_m: np.ndarray
    n_ctm_steps: int  # CTM steps spanning the same horizon
    wall_s: float


def build_and_run_micro(
    result: SimulationResult,
    *,
    pad: PadSpec,
    dx_grid: float,
    dt: float,
    eta_ev: float,
    seed: int,
    hours: Optional[float] = None,
    a: float = 1.5,
    b: float = 2.0,
    s_o: float = 25.0,
    vehicle_length: float = 4.69,
) -> MicroRun:
    """Seed and run the microsim over (a prefix of) the CTM horizon."""
    ctm_dt_h = float(result.freeway.dt)
    full_h = result.n_steps * ctm_dt_h
    horizon_h = full_h if hours is None else min(hours, full_h)
    n_ctm_steps = max(1, int(round(horizon_h / ctm_dt_h)))
    n_steps = int(round(horizon_h * _SECONDS_PER_HOUR / dt))

    # Resample admitted boundary inflow (CTM grid) onto micro steps.
    boundary = np.asarray(result.boundary_inflow, dtype=float)
    micro_times_h = np.arange(n_steps) * dt / _SECONDS_PER_HOUR
    ctm_idx = np.clip((micro_times_h / ctm_dt_h).astype(int), 0, boundary.size - 1)
    inflow_rate_per_step = boundary[ctm_idx]

    v_f_mph = float(np.mean([c.v_f for c in result.freeway.cells]))
    spec = MicrosimSpec(
        a=a, b=b, s_o=s_o, vehicle_length=vehicle_length, dt=dt, seed=seed
    )
    vehicles = seed_vehicles(
        inflow_rate_per_step, spec, v_f_ms=v_f_mph * MPH_TO_MS, eta_ev=eta_ev
    )

    corridor = corridor_from_ctm(result, pad, dx_grid)
    cell_edges_m = np.concatenate([[0.0], np.cumsum(corridor.cell_lengths_m)])

    t0 = time.perf_counter()
    micro_result = simulate(
        spec, vehicles, corridor_length_m=corridor.corridor_length_m,
        n_steps=n_steps,
    )
    wall_s = time.perf_counter() - t0

    return MicroRun(
        result=micro_result,
        corridor=corridor,
        cell_edges_m=cell_edges_m,
        n_ctm_steps=n_ctm_steps,
        wall_s=wall_s,
    )


def run_stage1(
    run: MicroRun,
    result: SimulationResult,
    cells_df: pd.DataFrame,
    *,
    timeseries_dir: Optional[Path],
    start: Optional[str],
) -> dict:
    """Stage 1: micro-vs-PeMS scoring (if PeMS available) + compute timings."""
    dt = run.result.dt
    window_steps = max(1, int(round(300.0 / dt)))
    density, flow = aggregate_to_cells(
        run.result, run.cell_edges_m, window_steps=window_steps
    )

    micro_val = None
    if timeseries_dir is not None and start is not None:
        wrapper = to_validation_arrays(density, flow)
        micro_val = compare_against_historical(
            wrapper, cells_df, timeseries_dir, start=pd.Timestamp(start)
        )

    # CTM compute time (re-run from the stored scenario for a comparable wall).
    t0 = time.perf_counter()
    ctm_simulate(result.freeway, result.scenario)
    ctm_wall_s = time.perf_counter() - t0

    return {
        "micro_validation": micro_val,
        "micro_wall_s": run.wall_s,
        "ctm_wall_s": ctm_wall_s,
        "micro_density": density,
        "micro_flow": flow,
    }


def run_stage2(
    run: MicroRun, result: SimulationResult, *, eta_ev: float
) -> dict:
    """Stage 2: original mCONV (micro) vs adapted mCONV (CTM VHT)."""
    vht = mainline_vht(result)[:, : run.n_ctm_steps]
    adapted = compute(vht, run.corridor, eta_EV=eta_ev)
    micro_E = micro_demand(run.result, run.corridor)

    total_adapted = float(adapted.E.sum())
    total_micro = float(micro_E.sum())
    divergence = (
        (total_micro - total_adapted) / total_adapted
        if total_adapted
        else float("nan")
    )
    return {
        "total_adapted_wh": total_adapted,
        "total_micro_wh": total_micro,
        "relative_divergence": divergence,
    }


def _write_outputs(out_dir: Path, stage1: dict, stage2: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if stage1.get("micro_validation") is not None:
        stage1["micro_validation"].to_csv(
            out_dir / "stage1_micro_validation.csv", index=False
        )
    lines = [
        "DWPT macro-vs-micro validation",
        "",
        "Stage 1 (compute, wall-clock):",
        f"  micro : {stage1['micro_wall_s']:.3f} s",
        f"  CTM   : {stage1['ctm_wall_s']:.3f} s",
        "",
        "Stage 2 (corridor-aggregate DWPT energy):",
        f"  adapted (CTM VHT)        : {stage2['total_adapted_wh']:.3f} Wh",
        f"  original (micro traj.)   : {stage2['total_micro_wh']:.3f} Wh",
        f"  relative divergence      : {stage2['relative_divergence']:.4%}",
    ]
    (out_dir / "validation_summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case-dir", required=True, type=Path)
    p.add_argument("--timeseries-dir", type=Path, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--dt", type=float, default=1.0, help="micro time step (s)")
    p.add_argument("--hours", type=float, default=None, help="cap horizon (h)")
    p.add_argument("--eta-ev", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=3.5)
    p.add_argument("--lambda-gap", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=150000.0)
    p.add_argument("--beta-prime", type=float, default=150000.0)
    p.add_argument("--dx-grid", type=float, default=0.5)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    result = SimulationResult.from_npz(
        args.case_dir / "sim_no_ramps" / "result.npz"
    )
    cells_df = pd.read_csv(args.case_dir / "cells.csv")
    pad = PadSpec(
        alpha=args.alpha, delta=args.delta, lambda_gap=args.lambda_gap,
        beta=args.beta, beta_prime=args.beta_prime,
    )

    run = build_and_run_micro(
        result, pad=pad, dx_grid=args.dx_grid, dt=args.dt,
        eta_ev=args.eta_ev, seed=args.seed, hours=args.hours,
    )
    stage1 = run_stage1(
        run, result, cells_df,
        timeseries_dir=args.timeseries_dir, start=args.start,
    )
    stage2 = run_stage2(run, result, eta_ev=args.eta_ev)

    out_dir = args.out_dir or (args.case_dir / "dwpt_validation")
    _write_outputs(out_dir, stage1, stage2)
    gs = run.result.guard_stats
    print(f"Micro guard activations: {gs.n_activations} "
          f"({gs.activation_rate:.3%} of vehicle-steps); "
          f"max accel delta {gs.max_accel_delta_ms2:.2f} m/s^2")
    print(f"Stage 2 relative divergence: {stage2['relative_divergence']:.4%}")
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
