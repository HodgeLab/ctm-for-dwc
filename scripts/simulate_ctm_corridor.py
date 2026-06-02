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
   (``--demand`` / ``--beta``); both default to **zero for every cell**
   if not supplied. Once Step 7's adapters land, the demo will be able
   to derive these directly from PeMS like inflow/rho0.
5. Run :func:`simulate`, compute metrics, and write all per-cell time
   series to ``--out-dir`` as one CSV per quantity (the dict that
   :meth:`SimulationResult.to_dataframes` returns), plus a
   ``summary.txt`` with horizon totals.

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
import sys
from pathlib import Path

import pandas as pd

from transportation_models.utils.ctm import (
    compute_metrics,
    freeway_from_dataframe,
    inflow_from_vds,
    initial_state_from_vds,
    scenario_from_dataframes,
    simulate,
)


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
              "veh/h (column labels = integer cell indices). **Defaults to "
              "zero for every cell** -- a ramp-bearing corridor will still "
              "simulate, just with no traffic entering via on-ramps. Step 7 "
              "will produce this from PeMS automatically once it lands."),
    )
    parser.add_argument(
        "--beta", type=Path, default=None,
        help=("Optional wide T x n_cells CSV of off-ramp split ratio "
              "beta_i(k), unitless in [0, 1). Same default-to-zero "
              "convention as --demand; Step 7 will produce this too."),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help=("Destination for the per-quantity output CSVs + summary.txt. "
              "Defaults to <freeway parent>/sim/."),
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

    # 4. Optional demand / beta. None -> zero for every cell.
    demand_df = _load_wide(args.demand, n_steps=n_steps, label="--demand")
    beta_df = _load_wide(args.beta, n_steps=n_steps, label="--beta")
    if demand_df is None and any(c.on_ramp for c in freeway.cells):
        n_on = sum(1 for c in freeway.cells if c.on_ramp)
        print(f"  demand       : <zero> (no --demand; {n_on} on-ramp cell(s) "
              "will see no inflow)", file=sys.stderr)
    if beta_df is None and any(c.off_ramp for c in freeway.cells):
        n_off = sum(1 for c in freeway.cells if c.off_ramp)
        print(f"  beta         : <zero> (no --beta; {n_off} off-ramp cell(s) "
              "will pass all traffic through)", file=sys.stderr)

    # 5. Scenario (revalidated inside simulate).
    scenario = scenario_from_dataframes(
        freeway, inflow=inflow, demand=demand_df, beta=beta_df,
        rho0=rho0, q0=0.0,
    )

    # 6. Run.
    result = simulate(freeway, scenario)
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

    horizon_h = n_steps * dt_h
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
