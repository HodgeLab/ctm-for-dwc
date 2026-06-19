"""Run the two-stage DWPT macro-vs-micro validation study (P7).

Given a prepared case-study directory (corridor geometry + CTM
``SimulationResult`` + boundary inflow, e.g. ``scripts/output/1mi_case_study``),
this driver:

* builds and runs the N-lane Newbolt microsim (lane-changing) over the corridor;
* **Stage 1 (traffic fidelity):** aggregates the micro trajectories to per-cell
  density/flow (Edie, summed across lanes) and scores them against PeMS with the
  same ``compare_against_historical`` path the CTM uses, alongside the existing
  CTM ``validation.csv`` and wall-clock compute for each engine;
* **Stage 2 (demand fidelity):** compares the original mCONV over micro
  trajectories against the adapted mCONV over the CTM VHT.

Vehicles are seeded from the **raw PeMS flow** of the upstream boundary VDS (the
same source the CTM inflow derives from) via a Poisson process; pass
``--timeseries-dir`` + ``--start``. Without PeMS, the driver falls back to the
CTM admitted boundary inflow (documented as a fallback, not the intended path).
See ``docs/dwpt_validation_spec.md``.

Run from the repo root::

    python scripts/run_dwpt_validation.py \\
        --case-dir scripts/output/1mi_case_study \\
        --timeseries-dir data/pems/csv_files --start "2022-04-12 06:00"
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
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
_FIVE_MIN_S = 300.0


@dataclass
class MicroRun:
    """A completed micro run plus the geometry/timing the stages need."""

    result: object  # MicrosimResult
    corridor: object  # CorridorSpec (snapped)
    cell_edges_m: np.ndarray
    n_ctm_steps: int  # CTM steps spanning the same horizon
    wall_s: float


def pems_inflow_rate(
    timeseries_dir: Path, vds_id: int, start: str, dt: float, n_steps: int
) -> np.ndarray:
    """Per-micro-step inflow rate (veh/h) from a boundary VDS's raw PeMS flow.

    Reads ``<vds_id>.csv`` (``total_flow_[veh/5-min]``), converts to veh/h, and
    zero-order-holds each 5-min value across the micro steps it spans.
    """
    df = pd.read_csv(
        Path(timeseries_dir) / f"{vds_id}.csv", parse_dates=["timestamp"]
    )
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(seconds=n_steps * dt)
    window = (
        df.loc[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    flow_veh_h = window["total_flow_[veh/5-min]"].to_numpy(dtype=float) * 12.0
    micro_t_s = np.arange(n_steps) * dt
    bin_idx = np.clip(
        (micro_t_s / _FIVE_MIN_S).astype(int), 0, flow_veh_h.size - 1
    )
    return flow_veh_h[bin_idx]


def ctm_inflow_rate(result: SimulationResult, dt: float, n_steps: int) -> np.ndarray:
    """Fallback per-micro-step inflow from the CTM admitted boundary inflow."""
    boundary = np.asarray(result.boundary_inflow, dtype=float)
    ctm_dt_h = float(result.freeway.dt)
    micro_t_h = np.arange(n_steps) * dt / _SECONDS_PER_HOUR
    idx = np.clip((micro_t_h / ctm_dt_h).astype(int), 0, boundary.size - 1)
    return boundary[idx]


def build_and_run_micro(
    result: SimulationResult,
    inflow_rate_per_step: np.ndarray,
    *,
    n_lanes: int,
    pad: PadSpec,
    dx_grid: float,
    dt: float,
    eta_ev: float,
    seed: int,
    a: float = MicrosimSpec.a,
    b: float = MicrosimSpec.b,
    s_o: float = MicrosimSpec.s_o,
    vehicle_length: float = MicrosimSpec.vehicle_length,
    lane_change_prob: float = MicrosimSpec.lane_change_prob,
) -> MicroRun:
    """Seed and run the N-lane microsim over the given inflow horizon."""
    n_steps = int(inflow_rate_per_step.size)
    ctm_dt_h = float(result.freeway.dt)
    horizon_h = n_steps * dt / _SECONDS_PER_HOUR
    n_ctm_steps = min(result.n_steps, max(1, int(round(horizon_h / ctm_dt_h))))

    v_f_mph = float(np.mean([c.v_f for c in result.freeway.cells]))
    spec = MicrosimSpec(
        a=a, b=b, s_o=s_o, vehicle_length=vehicle_length, dt=dt, seed=seed,
        lane_change_prob=lane_change_prob,
    )
    vehicles = seed_vehicles(
        inflow_rate_per_step, spec, v_f_ms=v_f_mph * MPH_TO_MS,
        eta_ev=eta_ev, n_lanes=n_lanes,
    )

    corridor = corridor_from_ctm(result, pad, dx_grid)
    cell_edges_m = np.concatenate([[0.0], np.cumsum(corridor.cell_lengths_m)])

    t0 = time.perf_counter()
    micro_result = simulate(
        spec, vehicles, corridor_length_m=corridor.corridor_length_m,
        n_steps=n_steps, n_lanes=n_lanes,
    )
    wall_s = time.perf_counter() - t0

    return MicroRun(
        result=micro_result, corridor=corridor, cell_edges_m=cell_edges_m,
        n_ctm_steps=n_ctm_steps, wall_s=wall_s,
    )


def _micro_traffic_metrics(run: MicroRun) -> dict:
    """VHT, VMT, travel-time stats, and counts from the micro trajectories."""
    res = run.result
    dt = res.dt
    L = res.corridor_length_m
    active = ~np.isnan(res.velocity)
    vht_h = active.sum() * dt / _SECONDS_PER_HOUR
    vmt_mi = float(np.nansum(res.velocity[active]) * dt / METERS_PER_MILE)

    active_intervals = active.sum(axis=1)
    completed = np.nanmax(res.position, axis=1) >= L - 1e-6
    tt_s = active_intervals[completed] * dt
    return {
        "vht_h": vht_h,
        "vmt_mi": vmt_mi,
        "n_entered": int((active_intervals > 0).sum()),
        "n_completed": int(completed.sum()),
        "mean_tt_s": float(tt_s.mean()) if tt_s.size else float("nan"),
        "max_tt_s": float(tt_s.max()) if tt_s.size else float("nan"),
    }


def _ctm_traffic_metrics(result: SimulationResult, n_ctm_steps: int) -> dict:
    """VHT, VMT, and an implied mean travel time from the CTM result."""
    dt_h = float(result.freeway.dt)
    L_cell = np.array([c.length for c in result.freeway.cells])  # miles
    vht_h = float(mainline_vht(result)[:, :n_ctm_steps].sum())
    flow = result.mainline_flow[:, :n_ctm_steps]  # veh/h
    vmt_mi = float((flow * L_cell[:, None] * dt_h).sum())
    corridor_mi = float(L_cell.sum())
    n_traversals = vmt_mi / corridor_mi if corridor_mi else 0.0
    mean_tt_s = (vht_h / n_traversals * _SECONDS_PER_HOUR) if n_traversals else float("nan")
    return {"vht_h": vht_h, "vmt_mi": vmt_mi, "mean_tt_s": mean_tt_s}


def plot_flow_contours(out_dir: Path, run: MicroRun, result: SimulationResult) -> Path:
    """Space-time flow heatmaps: micro (Edie) vs CTM, saved side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt = run.result.dt
    window_steps = max(1, int(round(_FIVE_MIN_S / dt)))
    _, micro_flow = aggregate_to_cells(
        run.result, run.cell_edges_m, window_steps=window_steps
    )
    ctm_flow = result.mainline_flow[:, : run.n_ctm_steps]

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), constrained_layout=True)
    for ax, data, title in (
        (axes[0], micro_flow, "Micro (Edie) flow [veh/h]"),
        (axes[1], ctm_flow, "CTM flow [veh/h]"),
    ):
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_ylabel("cell")
        fig.colorbar(im, ax=ax)
    axes[1].set_xlabel("time interval")
    out = out_dir / "stage1_flow_contours.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


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
    window_steps = max(1, int(round(_FIVE_MIN_S / dt)))
    density, flow = aggregate_to_cells(
        run.result, run.cell_edges_m, window_steps=window_steps
    )

    micro_val = None
    if timeseries_dir is not None and start is not None:
        wrapper = to_validation_arrays(density, flow)
        micro_val = compare_against_historical(
            wrapper, cells_df, timeseries_dir, start=pd.Timestamp(start)
        )

    # Time the CTM over the SAME horizon as the micro (truncate the scenario)
    # so the compute-cost comparison is fair when --hours caps the run.
    sc = result.scenario
    t = run.n_ctm_steps
    sc_h = replace(
        sc, inflow=sc.inflow[:t], demand=sc.demand[:, :t], beta=sc.beta[:, :t]
    )
    t0 = time.perf_counter()
    ctm_simulate(result.freeway, sc_h)
    ctm_wall_s = time.perf_counter() - t0

    return {
        "micro_validation": micro_val,
        "micro_wall_s": run.wall_s,
        "ctm_wall_s": ctm_wall_s,
        "micro_density": density,
        "micro_flow": flow,
    }


def _rel(micro: float, adapted: float) -> float:
    return (micro - adapted) / adapted if adapted else float("nan")


def run_stage2(
    run: MicroRun, result: SimulationResult, *, eta_ev: float
) -> dict:
    """Stage 2: original mCONV (micro) vs adapted mCONV (CTM VHT).

    Reports corridor energy delivered and peak power, with the micro-vs-adapted
    divergence for each. Peak power is compared at the CTM timestep resolution
    (the micro power series is averaged into CTM bins) for a fair comparison.
    Also times each demand model in isolation (``compute`` vs ``micro_demand``);
    the upstream traffic-sim cost is reported by Stage 1, not here.
    """
    ctm_dt_h = float(result.freeway.dt)
    vht = mainline_vht(result)[:, : run.n_ctm_steps]

    # Time each demand model in isolation (the traffic-sim cost is Stage 1's).
    # Both build a (T, m) DemandResult, so this is a like-for-like comparison.
    t0 = time.perf_counter()
    adapted = compute(vht, run.corridor, eta_EV=eta_ev)
    adapted_wall_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    micro = micro_demand(run.result, run.corridor)
    micro_wall_s = time.perf_counter() - t0

    total_adapted = float(adapted.E.sum())
    total_micro = float(micro.E.sum())

    # Peak power. Adapted: per-CTM-step energy / dt. Micro: instantaneous power
    # averaged into CTM-step bins for a like-for-like resolution.
    power_adapted = adapted.E.sum(axis=1) / ctm_dt_h  # W per CTM step
    peak_adapted = float(power_adapted.max()) if power_adapted.size else 0.0

    power_micro = micro.E.sum(axis=1) / (run.result.dt / _SECONDS_PER_HOUR)  # W per micro step
    spc = max(1, int(round(ctm_dt_h * _SECONDS_PER_HOUR / run.result.dt)))
    nbin = power_micro.size // spc
    if nbin:
        binned = power_micro[: nbin * spc].reshape(nbin, spc).mean(axis=1)
        peak_micro = float(binned.max())
    else:
        peak_micro = float(power_micro.mean()) if power_micro.size else 0.0

    return {
        "total_adapted_wh": total_adapted,
        "total_micro_wh": total_micro,
        "energy_divergence": _rel(total_micro, total_adapted),
        "peak_adapted_w": peak_adapted,
        "peak_micro_w": peak_micro,
        "peak_divergence": _rel(peak_micro, peak_adapted),
        "adapted_wall_s": adapted_wall_s,
        "micro_wall_s": micro_wall_s,
    }


def _format_report(header: dict, stage1: dict, stage2: dict, micro_tm: dict,
                   ctm_tm: dict, flow_plot: Path) -> str:
    """Build the human-readable report (printed and written to file)."""
    gs = header["guard_stats"]
    micro_w, ctm_w = stage1["micro_wall_s"], stage1["ctm_wall_s"]
    speedup = micro_w / ctm_w if ctm_w else float("nan")
    bar = "=" * 64
    L = [
        bar,
        f"DWPT macro-vs-micro validation — {header['ref']}, {header['n_lanes']} lanes",
        f"horizon: {header['horizon_h']:.2f} h "
        f"({header['n_micro_steps']} micro steps @ dt={header['dt']:g}s; "
        f"{header['n_ctm_steps']} CTM steps @ dt={header['ctm_dt_s']:g}s)",
        f"seed: {header['seed']} | eta_EV: {header['eta_ev']:.2f} | "
        f"inflow: {header['inflow_src']}",
        bar,
        "",
        "── Stage 1: traffic fidelity " + "─" * 35,
        "Sim summary:",
        f"  micro: {micro_tm['n_entered']} entered, {micro_tm['n_completed']} "
        f"completed; guard activations {gs.n_activations} "
        f"({gs.activation_rate:.3%} of vehicle-steps)",
        f"  CTM:   {header['n_cells']} cells, {header['n_ctm_steps']} steps",
        "Compute cost (wall-clock):",
        f"  micro: {micro_w:.3f} s",
        f"  CTM:   {ctm_w:.3f} s   (micro / CTM = {speedup:.1f}x)",
        "Vehicle-hours traveled (VHT):",
        f"  micro: {micro_tm['vht_h']:.2f} veh·h",
        f"  CTM:   {ctm_tm['vht_h']:.2f} veh·h   "
        f"(Δ {_rel(micro_tm['vht_h'], ctm_tm['vht_h']):+.1%})",
        "Vehicle-miles traveled (VMT):",
        f"  micro: {micro_tm['vmt_mi']:.1f} veh·mi",
        f"  CTM:   {ctm_tm['vmt_mi']:.1f} veh·mi   "
        f"(Δ {_rel(micro_tm['vmt_mi'], ctm_tm['vmt_mi']):+.1%})",
        "Travel time (corridor traversal):",
        f"  micro: mean {micro_tm['mean_tt_s']:.1f} s, max {micro_tm['max_tt_s']:.1f} s",
        f"  CTM:   mean {ctm_tm['mean_tt_s']:.1f} s (implied VHT/throughput)",
        f"Flow contours -> {flow_plot}",
    ]
    if stage1.get("micro_validation") is not None:
        L.append("Micro-vs-PeMS validation -> stage1_micro_validation.csv")
    L += [
        "",
        "── Stage 2: demand fidelity " + "─" * 36,
        "Sim summary:",
        f"  same corridor/horizon; EV fraction {header['eta_ev']:.2f}",
        "Compute cost (wall-clock, demand model only):",
        f"  adapted (CTM VHT)      : {stage2['adapted_wall_s']:.4f} s",
        f"  original (micro traj.) : {stage2['micro_wall_s']:.4f} s",
        "Corridor energy delivered:",
        f"  adapted (CTM VHT)      : {stage2['total_adapted_wh'] / 1e3:.2f} kWh",
        f"  original (micro traj.) : {stage2['total_micro_wh'] / 1e3:.2f} kWh",
        f"  divergence             : {stage2['energy_divergence']:+.2%}",
        "Corridor peak power (at CTM timestep resolution):",
        f"  adapted                : {stage2['peak_adapted_w'] / 1e3:.1f} kW",
        f"  original (micro traj.) : {stage2['peak_micro_w'] / 1e3:.1f} kW",
        f"  divergence             : {stage2['peak_divergence']:+.2%}",
        bar,
    ]
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case-dir", required=True, type=Path)
    p.add_argument("--timeseries-dir", type=Path, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--dt", type=float, default=1.0, help="micro time step (s)")
    p.add_argument("--hours", type=float, default=None, help="cap horizon (h)")
    p.add_argument("--n-lanes", type=int, default=None, help="override lane count")
    p.add_argument("--eta-ev", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    # Gipps params (paper defaults; CLI-overridable).
    p.add_argument("--a", type=float, default=MicrosimSpec.a)
    p.add_argument("--b", type=float, default=MicrosimSpec.b)
    p.add_argument("--s-o", type=float, default=MicrosimSpec.s_o)
    p.add_argument("--vehicle-length", type=float, default=MicrosimSpec.vehicle_length)
    p.add_argument("--lane-change-prob", type=float, default=MicrosimSpec.lane_change_prob)
    # Pad (study config; CLI-overridable).
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
    n_lanes = args.n_lanes or int(cells_df["lanes"].max())

    # Horizon (steps) and inflow source.
    ctm_dt_h = float(result.freeway.dt)
    full_h = result.n_steps * ctm_dt_h
    horizon_h = full_h if args.hours is None else min(args.hours, full_h)
    n_steps = int(round(horizon_h * _SECONDS_PER_HOUR / args.dt))

    if args.timeseries_dir is not None and args.start is not None:
        boundary_vds = int(cells_df["vds_id"].iloc[0])  # upstream-most detector
        inflow = pems_inflow_rate(
            args.timeseries_dir, boundary_vds, args.start, args.dt, n_steps
        )
        print(f"Seeding from raw PeMS VDS {boundary_vds}.")
    else:
        inflow = ctm_inflow_rate(result, args.dt, n_steps)
        print("WARNING: no --timeseries-dir; seeding from CTM admitted inflow "
              "(fallback, not the intended raw-PeMS path).")

    inflow_src = (
        f"raw PeMS VDS {int(cells_df['vds_id'].iloc[0])}"
        if (args.timeseries_dir is not None and args.start is not None)
        else "CTM admitted inflow (fallback)"
    )

    run = build_and_run_micro(
        result, inflow, n_lanes=n_lanes, pad=pad, dx_grid=args.dx_grid,
        dt=args.dt, eta_ev=args.eta_ev, seed=args.seed,
        a=args.a, b=args.b, s_o=args.s_o, vehicle_length=args.vehicle_length,
        lane_change_prob=args.lane_change_prob,
    )
    stage1 = run_stage1(
        run, result, cells_df,
        timeseries_dir=args.timeseries_dir, start=args.start,
    )
    stage2 = run_stage2(run, result, eta_ev=args.eta_ev)
    micro_tm = _micro_traffic_metrics(run)
    ctm_tm = _ctm_traffic_metrics(result, run.n_ctm_steps)

    out_dir = args.out_dir or (args.case_dir / "dwpt_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    flow_plot = plot_flow_contours(out_dir, run, result)
    if stage1.get("micro_validation") is not None:
        stage1["micro_validation"].to_csv(
            out_dir / "stage1_micro_validation.csv", index=False
        )

    corridor_json = args.case_dir / "corridor.json"
    if corridor_json.exists():
        meta = json.loads(corridor_json.read_text())
        ref = f"{meta.get('ref', 'corridor')} {meta.get('direction', '')}".strip()
    else:
        ref = f"{cells_df.shape[0]}-cell corridor"
    header = {
        "ref": ref,
        "n_lanes": n_lanes, "n_cells": cells_df.shape[0],
        "horizon_h": horizon_h, "n_micro_steps": n_steps,
        "n_ctm_steps": run.n_ctm_steps, "dt": args.dt,
        "ctm_dt_s": ctm_dt_h * _SECONDS_PER_HOUR, "seed": args.seed,
        "eta_ev": args.eta_ev, "inflow_src": inflow_src,
        "guard_stats": run.result.guard_stats,
    }
    report = _format_report(header, stage1, stage2, micro_tm, ctm_tm, flow_plot)
    print(report)
    (out_dir / "validation_summary.txt").write_text(report + "\n")
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
