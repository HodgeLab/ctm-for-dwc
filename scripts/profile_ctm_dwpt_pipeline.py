"""Profile the fused CTM sim -> DWPT demand pipeline in a single process.

``scripts/simulate_ctm_corridor.py`` and ``scripts/generate_dwpt_demand.py``
each report their own runtime and peak RSS, but those numbers come from two
separate processes. Summing the runtimes is fine -- the stages are sequential.
Taking ``max()`` of the two peak-RSS figures answers "how much RAM does the
pipeline need when run as two commands", but it *understates* a fused run: in
one process the CTM result arrays stay live while the finer DWPT position grid
is allocated, so the true fused peak lands somewhere between ``max(a, b)`` and
``a + b``. This script runs both stages back-to-back in one process and
measures that peak directly, with no ``.npz`` round-trip between them.

Deliberately writes no per-quantity CSVs, no ``result.npz``, and no figures:
the two production scripts already produce those, and matplotlib figure memory
would pollute the RSS measurement. Outputs are stdout plus a
``pipeline_profile.txt`` summary.

Two caveats on reading the RSS figures. They are process-wide ``ru_maxrss``
marks, so they include the interpreter and the numpy/pandas import baseline --
which dominates for short corridors, hence the "net of startup baseline" line.
And because this script never imports matplotlib, its baseline sits below the
one both production scripts carry; compare fused-vs-split peaks using the
breakdown here, not against the numbers in ``summary.txt`` /
``dwpt_summary.txt``. The headline ``runtime =`` / ``peak RSS =``
lines match the format ``scripts/compare_ctm_case_studies.py`` greps for, and
lead the summary so its first-match search picks up the pipeline totals.

CLI: the CTM stage takes ``simulate_ctm_corridor.py``'s arguments and the DWPT
stage takes ``generate_dwpt_demand.py``'s. Two renames avoid a collision --
``--beta`` is the off-ramp split-ratio CSV in one script and the Rx pad rated
power in the other, so here they are ``--ctm-beta`` and ``--dwpt-beta``
(with ``--beta-prime`` following its partner to ``--dwpt-beta-prime``). Every
other flag keeps the name it has in its source script.

Run from the repo root::

    python scripts/profile_ctm_dwpt_pipeline.py \\
        --freeway scripts/output/1mi_case_study/freeway.csv \\
        --cells   scripts/output/1mi_case_study/cells.csv \\
        --timeseries-dir data/pems/csv_files \\
        --dt-seconds 10 \\
        --start "2022-04-12 06:00" \\
        --end   "2022-04-12 09:00" \\
        --alpha 3.5 --lambda-gap 0.5 --delta 1.0 \\
        --dwpt-beta 150000 --dwpt-beta-prime 150000 \\
        --dx-grid 0.5 --eta-ev 0.3
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from dataclasses import replace
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
from transportation_models.utils.dwpt.adapter import corridor_from_ctm, mainline_vht
from transportation_models.utils.dwpt.demand import compute
from transportation_models.utils.dwpt.model import PadSpec


# Both helpers are verbatim copies from the two scripts this one fuses; kept
# local so neither production script has to change.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    ctm = parser.add_argument_group("CTM stage (scripts/simulate_ctm_corridor.py)")
    ctm.add_argument(
        "--freeway", required=True, type=Path,
        help="Step-6 freeway.csv (cell-keyed FD + ramp capacities).",
    )
    ctm.add_argument(
        "--cells", required=True, type=Path,
        help=("Step-1+2 cells.csv (vds_id per cell, the upstream-most VDS for "
              "inflow, and vds_lanes for the initial-state lane match)."),
    )
    ctm.add_argument(
        "--timeseries-dir", required=True, type=Path,
        help="Directory of Step-3 per-VDS timeseries CSVs (one <vds_id>.csv each).",
    )
    ctm.add_argument(
        "--dt-seconds", required=True, type=float,
        help="Sim time step in seconds (pick from Step 6's CFL advisory).",
    )
    ctm.add_argument(
        "--start", required=True, type=pd.Timestamp,
        help="Sim start timestamp (inclusive).",
    )
    ctm.add_argument(
        "--end", required=True, type=pd.Timestamp,
        help="Sim end timestamp (exclusive).",
    )
    ctm.add_argument(
        "--upstream-vds", type=int, default=None,
        help=("Override the upstream-most mainline VDS id (defaults to the "
              "vds_id of the first row in cells.csv)."),
    )
    ctm.add_argument(
        "--demand", type=Path, default=None,
        help=("Optional wide T x n_cells CSV of on-ramp demand d_i(k) in veh/h "
              "from scripts/build_ctm_ramp_scenario.py. Defaults to zero."),
    )
    ctm.add_argument(
        "--ctm-beta", type=Path, default=None,
        help=("Optional wide T x n_cells CSV of off-ramp split ratio beta_i(k) "
              "from scripts/build_ctm_ramp_scenario.py. Defaults to zero. "
              "This is simulate_ctm_corridor.py's --beta, renamed here."),
    )

    dwpt = parser.add_argument_group("DWPT stage (scripts/generate_dwpt_demand.py)")
    dwpt.add_argument(
        "--alpha", required=True, type=float, help="DWPT Tx pad length (meters)",
    )
    dwpt.add_argument(
        "--lambda-gap", required=True, type=float,
        help="DWPT Tx pad spacing (meters)",
    )
    dwpt.add_argument(
        "--delta", required=True, type=float, help="DWPT Rx pad length (meters)",
    )
    dwpt.add_argument(
        "--dwpt-beta", required=True, type=float,
        help=("DWPT Rx pad rated power (watts). This is "
              "generate_dwpt_demand.py's --beta, renamed here."),
    )
    dwpt.add_argument(
        "--dwpt-beta-prime", required=True, type=float,
        help=("DWPT Tx pad rated power (watts). This is "
              "generate_dwpt_demand.py's --beta-prime, renamed to stay "
              "paired with --dwpt-beta."),
    )
    dwpt.add_argument(
        "--dx-grid", required=True, type=float,
        help="DWPT spatial resolution (meters)",
    )
    dwpt.add_argument(
        "--eta-ev", required=True, type=float, help="DWPT EV fraction [0,1]",
    )

    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help=("Destination for pipeline_profile.txt. Defaults to "
              "<freeway parent>/pipeline_profile/."),
    )
    args = parser.parse_args()

    rss_baseline_mb = _peak_rss_mb()
    dt_h = float(args.dt_seconds) / 3600.0

    print("=== fused CTM -> DWPT pipeline profile ===", file=sys.stderr)
    print(f"  freeway      : {args.freeway}", file=sys.stderr)
    print(f"  ts dir       : {args.timeseries_dir}", file=sys.stderr)
    print(f"  dt           : {args.dt_seconds:.3f} s ({dt_h:.5f} h)", file=sys.stderr)
    print(f"  window       : [{args.start}, {args.end})", file=sys.stderr)

    # ---- Stage 0: input loading ------------------------------------------
    # Timed separately so the two stage timings below stay directly comparable
    # to the numbers the standalone scripts report (both of which exclude
    # their own input loading).
    t0 = time.perf_counter()
    cells = pd.read_csv(args.cells)
    freeway_df = pd.read_csv(args.freeway)
    freeway = freeway_from_dataframe(freeway_df, dt=dt_h)

    upstream_vds = (
        args.upstream_vds if args.upstream_vds is not None
        else int(cells.iloc[0]["vds_id"])
    )
    inflow = inflow_from_vds(
        args.timeseries_dir / f"{upstream_vds}.csv",
        dt=dt_h, start=args.start, end=args.end,
    )
    n_steps = inflow.size
    rho0 = initial_state_from_vds(cells, args.timeseries_dir, at_time=args.start)
    demand_df = _load_wide(args.demand, n_steps=n_steps, label="--demand")
    beta_df = _load_wide(args.ctm_beta, n_steps=n_steps, label="--ctm-beta")
    scenario = scenario_from_dataframes(
        freeway, inflow=inflow, demand=demand_df, beta=beta_df,
        rho0=rho0, q0=0.0,
    )
    setup_wall_s = time.perf_counter() - t0
    rss_after_setup_mb = _peak_rss_mb()
    print(f"  upstream VDS : {upstream_vds}", file=sys.stderr)
    print(f"  horizon      : {n_steps} steps ({n_steps * dt_h:.2f} h)",
          file=sys.stderr)

    # ---- Stage 1: CTM simulation -----------------------------------------
    # Same timed region as simulate_ctm_corridor.py.
    t0 = time.perf_counter()
    sim_result = replace(simulate(freeway, scenario), start=args.start)
    ctm_wall_s = time.perf_counter() - t0
    rss_after_ctm_mb = _peak_rss_mb()

    # ---- Stage 2: DWPT demand --------------------------------------------
    # Same timed region as generate_dwpt_demand.py. sim_result stays live
    # throughout -- that overlap is exactly what the fused peak captures.
    pad = PadSpec(
        alpha=args.alpha, delta=args.delta, lambda_gap=args.lambda_gap,
        beta=args.dwpt_beta, beta_prime=args.dwpt_beta_prime,
    )
    t0 = time.perf_counter()
    corridor = corridor_from_ctm(sim_result, pad, args.dx_grid)
    demand = compute(mainline_vht(sim_result), corridor, args.eta_ev)
    dwpt_wall_s = time.perf_counter() - t0
    rss_after_dwpt_mb = _peak_rss_mb()

    # Untimed, matching both source scripts.
    metrics = compute_metrics(sim_result)

    pipeline_wall_s = setup_wall_s + ctm_wall_s + dwpt_wall_s
    pipeline_peak_rss_mb = rss_after_dwpt_mb
    horizon_h = n_steps * dt_h

    out_dir = (
        args.out_dir if args.out_dir is not None
        else args.freeway.parent / "pipeline_profile"
    )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "=== fused CTM -> DWPT pipeline profile ===",
        f"  freeway      : {args.freeway}",
        f"  cells        : {freeway.n_cells}",
        f"  window       : [{args.start}, {args.end})  ({horizon_h:.3f} h)",
        f"  dt           : {args.dt_seconds:.3f} s",
        f"  steps        : {n_steps}",
        f"  upstream VDS : {upstream_vds}",
        f"  demand       : {'<zero>' if demand_df is None else args.demand}",
        f"  ctm beta     : {'<zero>' if beta_df is None else args.ctm_beta}",
        f"  dx-grid      : {args.dx_grid} m  "
        f"({demand.E.shape[1]} Rx positions x {demand.E.shape[0]} steps)",
        "",
        "  pipeline compute cost (both stages, one process):",
        f"    runtime  = {pipeline_wall_s:8.3f}  s",
        f"    peak RSS = {pipeline_peak_rss_mb:8.1f}  MB",
        "",
        "  runtime breakdown:",
        f"    input loading : {setup_wall_s:8.3f}  s",
        f"    CTM sim       : {ctm_wall_s:8.3f}  s",
        f"    DWPT demand   : {dwpt_wall_s:8.3f}  s",
        "",
        "  RSS high-water marks (monotonic -- each includes all prior stages):",
        f"    at startup      : {rss_baseline_mb:8.1f}  MB",
        f"    after loading   : {rss_after_setup_mb:8.1f}  MB",
        f"    after CTM sim   : {rss_after_ctm_mb:8.1f}  MB",
        f"    after DWPT      : {rss_after_dwpt_mb:8.1f}  MB   <- pipeline peak",
        f"    DWPT stage raised the high-water mark by "
        f"{rss_after_dwpt_mb - rss_after_ctm_mb:.1f} MB",
        f"    peak net of startup baseline: "
        f"{pipeline_peak_rss_mb - rss_baseline_mb:.1f} MB",
        "",
        "  CTM horizon totals (eqs. 4.10/4.12/4.14/4.16):",
        f"    VHT                = {metrics.vht.sum():12.3f}  veh*h",
        f"    VMT                = {metrics.vmt.sum():12.3f}  veh*mi",
        f"    delay              = {metrics.delay.sum():12.3f}  veh*h",
        f"    productivity loss  = {metrics.productivity_loss.sum():12.3f}  mi*h",
        "",
        "  DWPT demand:",
        f"    corridor length     : {corridor.corridor_length_m:.3e} m",
        f"    Energy delivered    : {demand.E.sum():.3e} Wh",
        f"    Peak corridor power : "
        f"{(demand.E.sum(axis=1).max() / sim_result.freeway.dt):.3e} W",
    ]
    summary = "\n".join(summary_lines) + "\n"
    (out_dir / "pipeline_profile.txt").write_text(summary)
    print()
    print(summary)
    print(f"Wrote {out_dir}/pipeline_profile.txt")


if __name__ == "__main__":
    main()
