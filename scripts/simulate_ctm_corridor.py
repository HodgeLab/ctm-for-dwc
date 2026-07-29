"""Step 8: end-to-end CTM simulation on a calibrated corridor.

Wires together the Step 1-6 artifacts and Step 8's PeMS -> Scenario
adapters to actually run :func:`utils.ctm.engine.simulate` on a real
corridor:

1. Build a :class:`Freeway` from the Step-6 ``freeway.csv`` with the
   user-chosen ``--dt-seconds`` (pick this from Step 6's CFL advisory).
2. Extract upstream boundary inflow ``f_0(k)`` from the corridor's
   upstream-most mainline VDS via
   :func:`utils.ctm.scenario.inflow_from_vds`. Window controlled by
   ``--start`` / ``--end``.
3. Extract initial density ``rho_i(0)`` from each cell's mainline VDS
   at ``--start`` via :func:`utils.ctm.scenario.initial_state_from_vds`.
   Requires ``lanes == vds_lanes`` per cell (Step 5 territory).
4. Optionally load wide on-ramp demand and off-ramp split-ratio CSVs
   (``--demand`` / ``--beta``), produced by Step 7
   (``scripts/build_ctm_ramp_scenario.py`` -- the only home of ramp-fill
   logic). Both default to **zero for every cell** if not supplied, i.e.
   ramps are effectively ignored.
5. Run :func:`simulate`, compute metrics, and write per-cell time
   series to ``--out-dir`` as one CSV per quantity (the dict that
   :meth:`SimulationResult.to_dataframes` returns) plus a
   self-contained ``result.npz`` (:meth:`SimulationResult.to_npz`, the
   input the DWPT demand pipeline reloads), plus a
   ``summary.txt`` with horizon totals and the simulation's runtime and
   peak RSS, and four figures matching the ``run_ctm_ctmsim_demo`` style:

      * ``flow_contour.png``      space-time heatmap of mainline flow
      * ``density_contour.png``   space-time heatmap of density
      * ``aggregate_metrics.png`` VHT/VMT/delay/ploss over time
      * ``per_cell_metrics.png``  same four metrics summed per-cell

Run from the repo root::

    python scripts/simulate_ctm_corridor.py \\
        --freeway scripts/output/ctm_corridor/I_210_W/freeway.csv \\
        --cells   scripts/output/ctm_corridor/I_210_W/cells.csv \\
        --timeseries-dir data/pems/csv_files \\
        --dt-seconds 10 \\
        --start "2022-04-12 06:00" \\
        --end   "2022-04-12 09:00"
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from transportation_models.utils.ctm import (
    compute_metrics,
    freeway_from_dataframe,
    inflow_from_vds,
    initial_state_from_vds,
    scenario_from_dataframes,
    simulate,
)
from transportation_models.utils.ctm.plots import (
    downsample_to_plot_period,
    per_period_totals,
    plot_aggregate_metrics,
    plot_flow_density_contour,
    plot_per_cell_metrics,
)


def _peak_rss_mb() -> float:
    """Process peak resident memory [MB] so far (RUSAGE_SELF high-water mark).

    ``ru_maxrss`` is a monotonic high-water mark, reported in bytes on
    macOS and kilobytes on Linux; normalize both to MB.
    """
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 ** 2) if sys.platform == "darwin" else ru / 1024


def _load_wide(
    path: Path | None, *, n_steps: int, label: str,
) -> pd.DataFrame | None:
    """Read a wide T x n_cells CSV; column labels coerced to int cell indices."""
    if path is None:
        return None
    df = pd.read_csv(path)
    if len(df) != n_steps:
        raise SystemExit(
            f"{label} CSV has {len(df)} rows; expected {n_steps} (= sim horizon)."
        )
    df.columns = [int(c) for c in df.columns]
    return df


def _plot_period_steps(dt_s: float, plot_period_s: float) -> int:
    """Sim steps per plotting period; must divide evenly."""
    if plot_period_s < dt_s:
        raise SystemExit(
            f"--plot-period-seconds ({plot_period_s}) must be >= "
            f"--dt-seconds ({dt_s})."
        )
    ratio = plot_period_s / dt_s
    if abs(ratio - round(ratio)) > 1e-6:
        raise SystemExit(
            f"--plot-period-seconds ({plot_period_s}) must be an integer "
            f"multiple of --dt-seconds ({dt_s})."
        )
    return int(round(ratio))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--freeway", required=True, type=Path,
        help="Step-6 freeway.csv (cell-keyed FD + ramp capacities).",
    )
    parser.add_argument(
        "--cells", required=True, type=Path,
        help=("Step-1+2 cells.csv (used to look up vds_id for each cell, "
              "the upstream-most VDS for inflow, and vds_lanes for the "
              "initial-state lane-match precondition)."),
    )
    parser.add_argument(
        "--timeseries-dir", required=True, type=Path,
        help="Directory of Step-3 per-VDS timeseries CSVs (one <vds_id>.csv each).",
    )
    parser.add_argument(
        "--dt-seconds", required=True, type=float,
        help=("Sim time step in seconds. Pick this from Step 6's CFL "
              "advisory (the suggested value is loud-printed by "
              "scripts/assemble_ctm_freeway.py)."),
    )
    parser.add_argument(
        "--start", required=True, type=pd.Timestamp,
        help=("Sim start timestamp (inclusive). Must align to the PeMS "
              "5-min grid when dt <= 5 min, else to the hourly grid."),
    )
    parser.add_argument(
        "--end", required=True, type=pd.Timestamp,
        help="Sim end timestamp (exclusive). Same alignment rule as --start.",
    )
    parser.add_argument(
        "--upstream-vds", type=int, default=None,
        help=("Override the upstream-most mainline VDS id (defaults to "
              "the vds_id of the first row in cells.csv -- the corridor's "
              "upstream end after the Step-1 ordering)."),
    )
    parser.add_argument(
        "--demand", type=Path, default=None,
        help=("Optional wide T x n_cells CSV of on-ramp demand d_i(k) in "
              "veh/h (column labels = integer cell indices), produced by "
              "scripts/build_ctm_ramp_scenario.py. **Defaults to zero for "
              "every cell** -- a ramp-bearing corridor will still simulate, "
              "just with no traffic entering via on-ramps."),
    )
    parser.add_argument(
        "--beta", type=Path, default=None,
        help=("Optional wide T x n_cells CSV of off-ramp split ratio "
              "beta_i(k), unitless in [0, 1), produced by "
              "scripts/build_ctm_ramp_scenario.py. Same default-to-zero "
              "convention as --demand (off-ramps pass all traffic through)."),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help=("Destination for the per-quantity output CSVs + summary.txt. "
              "Defaults to <freeway parent>/sim/."),
    )
    parser.add_argument(
        "--plot-period-seconds", type=float, default=300.0,
        help=("Plotting cadence in seconds for the contour + aggregate-metric "
              "figures (must be an integer multiple of --dt-seconds). "
              "Default 300 (5 min) matches the PeMS native sampling rate."),
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip figure generation (faster for headless / large-batch runs).",
    )
    args = parser.parse_args()

    dt_h = float(args.dt_seconds) / 3600.0
    cells = pd.read_csv(args.cells)
    freeway_df = pd.read_csv(args.freeway)

    print(f"=== Step 8 sim ===", file=sys.stderr)
    print(f"  freeway      : {args.freeway}", file=sys.stderr)
    print(f"  cells        : {args.cells}", file=sys.stderr)
    print(f"  ts dir       : {args.timeseries_dir}", file=sys.stderr)
    print(f"  dt           : {args.dt_seconds:.3f} s ({dt_h:.5f} h)",
          file=sys.stderr)
    print(f"  window       : [{args.start}, {args.end})", file=sys.stderr)

    # 1. Freeway (validates on construction).
    freeway = freeway_from_dataframe(freeway_df, dt=dt_h)
    n_cells = freeway.n_cells

    # 2. Upstream boundary inflow.
    upstream_vds = (
        args.upstream_vds if args.upstream_vds is not None
        else int(cells.iloc[0]["vds_id"])
    )
    upstream_csv = args.timeseries_dir / f"{upstream_vds}.csv"
    print(f"  upstream VDS : {upstream_vds} ({upstream_csv})", file=sys.stderr)
    inflow = inflow_from_vds(
        upstream_csv, dt=dt_h, start=args.start, end=args.end,
    )
    n_steps = inflow.size
    print(f"  horizon      : {n_steps} steps "
          f"({n_steps * dt_h:.2f} h)", file=sys.stderr)

    # 3. Initial density (raises if any cell has lanes != vds_lanes).
    rho0 = initial_state_from_vds(
        cells, args.timeseries_dir, at_time=args.start,
    )

    # 4. Optional demand / beta (Step 7 CSVs). None -> zero for every cell.
    demand_df = _load_wide(args.demand, n_steps=n_steps, label="--demand")
    beta_df = _load_wide(args.beta, n_steps=n_steps, label="--beta")

    if demand_df is None and any(c.on_ramp for c in freeway.cells):
        n_on = sum(1 for c in freeway.cells if c.on_ramp)
        print(f"  demand       : <zero> (no --demand; {n_on} on-ramp "
              "cell(s) will see no inflow -- run "
              "scripts/build_ctm_ramp_scenario.py to derive one)",
              file=sys.stderr)
    if beta_df is None and any(c.off_ramp for c in freeway.cells):
        n_off = sum(1 for c in freeway.cells if c.off_ramp)
        print(f"  beta         : <zero> (no --beta; {n_off} off-ramp "
              "cell(s) will pass all traffic through -- run "
              "scripts/build_ctm_ramp_scenario.py to derive one)",
              file=sys.stderr)

    # 5. Scenario (revalidated inside simulate).
    scenario = scenario_from_dataframes(
        freeway, inflow=inflow, demand=demand_df, beta=beta_df,
        rho0=rho0, q0=0.0,
    )

    # 6. Run. Time and measure the simulation's peak RSS here, before the
    # downstream CSV/plot allocations, so the numbers reflect the sim alone.
    t0 = time.perf_counter()
    result = replace(simulate(freeway, scenario), start=args.start)
    sim_wall_s = time.perf_counter() - t0
    sim_peak_rss_mb = _peak_rss_mb()
    metrics = compute_metrics(result)

    # 7. Write per-quantity CSVs + summary.
    out_dir = (
        args.out_dir if args.out_dir is not None
        else args.freeway.parent / "sim"
    )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in result.to_dataframes().items():
        frame.to_csv(out_dir / f"{name}.csv")
    # Self-contained bundle for the DWPT demand pipeline (utils.dwpt.adapter
    # reloads it via SimulationResult.from_npz).
    result.to_npz(out_dir / "result.npz")

    horizon_h = n_steps * dt_h

    # 8. Figures (mirrors run_ctm_ctmsim_demo.py).
    if not args.no_plots:
        steps_per_period = _plot_period_steps(
            args.dt_seconds, args.plot_period_seconds,
        )
        n_samples = n_steps // steps_per_period
        if n_samples * steps_per_period != n_steps:
            raise SystemExit(
                f"Sim horizon ({n_steps} steps of {args.dt_seconds:.1f}s = "
                f"{horizon_h:.3f}h) must be an integer multiple of the "
                f"plotting period ({args.plot_period_seconds:.1f}s); "
                f"got remainder {n_steps - n_samples * steps_per_period} step(s)."
            )

        kw = dict(steps_per_period=steps_per_period, n_samples=n_samples)
        # Contours are (K, N): time runs down rows, cells across columns.
        flow_KN = downsample_to_plot_period(
            result.mainline_flow, state=False, **kw,
        ).T
        # Drop the initial-state column so the time axis aligns with flow's
        # period-start sampling.
        density_KN = downsample_to_plot_period(
            result.density, state=True, **kw,
        )[:, 1:].T
        # Per-cell postmile boundaries from cells.csv preserve absolute
        # postmiles for pm-cropped corridors (cumsum-of-lengths from the
        # freeway.csv alone would start at 0 and lose that anchor).
        pm_edges = np.concatenate([
            cells["pm_start"].to_numpy(dtype=float),
            [float(cells["pm_end"].iloc[-1])],
        ])

        plot_flow_density_contour(
            flow_KN, title=f"Mainline flow ({args.start} -> {args.end})",
            cbar_label="flow [veh/h]", cmap="viridis",
            horizon_h=horizon_h, pm_edges=pm_edges,
            y_label="time from sim start [h]",
            out_path=out_dir / "flow_contour.png",
        )
        plot_flow_density_contour(
            density_KN, title=f"Density ({args.start} -> {args.end})",
            cbar_label="density [veh/mi]", cmap="magma",
            horizon_h=horizon_h, pm_edges=pm_edges,
            y_label="time from sim start [h]",
            vmax=300.0,
            out_path=out_dir / "density_contour.png",
        )

        time_h = np.arange(n_samples) * steps_per_period * dt_h
        period_s = args.plot_period_seconds
        period_label = (
            f"{period_s:.0f} s" if period_s < 60 else f"{period_s / 60:g} min"
        )
        plot_aggregate_metrics(
            time_h,
            per_period_totals(metrics.vht, **kw),
            per_period_totals(metrics.vmt, **kw),
            per_period_totals(metrics.delay, **kw),
            per_period_totals(metrics.productivity_loss, **kw),
            period_label=period_label, x_label="time from sim start [h]",
            out_path=out_dir / "aggregate_metrics.png",
        )
        plot_per_cell_metrics(
            metrics, title=f"{horizon_h:.2f}-hour totals per cell",
            out_path=out_dir / "per_cell_metrics.png",
        )
    summary_lines = [
        f"=== Step 8 sim summary ===",
        f"  freeway      : {args.freeway}",
        f"  cells        : {n_cells}",
        f"  window       : [{args.start}, {args.end})  ({horizon_h:.3f} h)",
        f"  dt           : {args.dt_seconds:.3f} s",
        f"  steps        : {n_steps}",
        f"  upstream VDS : {upstream_vds}",
        f"  demand       : {'<zero>' if demand_df is None else args.demand}",
        f"  beta         : {'<zero>' if beta_df is None else args.beta}",
        "",
        f"  horizon totals (eqs. 4.10/4.12/4.14/4.16):",
        f"    VHT                = {metrics.vht.sum():12.3f}  veh*h",
        f"    VMT                = {metrics.vmt.sum():12.3f}  veh*mi",
        f"    delay              = {metrics.delay.sum():12.3f}  veh*h",
        f"    productivity loss  = {metrics.productivity_loss.sum():12.3f}  mi*h",
        "",
        f"  corridor travel time (eq. 4.9):",
        f"    mean across horizon = {metrics.travel_time.mean() * 60:7.3f}  min",
        f"    max  across horizon = {metrics.travel_time.max() * 60:7.3f}  min",
        "",
        f"  simulation compute cost:",
        f"    runtime  = {sim_wall_s:8.3f}  s",
        f"    peak RSS = {sim_peak_rss_mb:8.1f}  MB",
    ]
    summary = "\n".join(summary_lines) + "\n"
    (out_dir / "summary.txt").write_text(summary)
    print()
    print(summary)
    print(f"Wrote {out_dir}/")
    for name in sorted(p.name for p in out_dir.iterdir()):
        print(f"  {name}")


if __name__ == "__main__":
    main()
