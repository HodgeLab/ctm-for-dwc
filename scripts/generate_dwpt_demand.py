"""Generate DWPT demand from CTM simulation.

Builds a DWPT corridor from a CTM simulation result and CLI-supplied corridor specs, calculates
DWPT demand at every timestamp and Rx pad position, and writes five plots and a summary of results.
Plots are the corridor power-vs-position profile, the demand space-time heatmap, the
corridor-aggregate load timeseries, the instantaneous load profile at peak, and the
corridor-aggregate load duration curve.

Run from the repo root:
    python scripts/generate_dwpt_demand.py \\
        --ctm-result scripts/output/1mi_case_study/sim_no_ramps/result.npz \\
        --alpha 3.5 \\
        --lambda-gap 0.5 \\
        --delta 1.0 \\
        --beta 150000 \\
        --beta-prime 150000 \\
        --dx-grid 0.5 \\
        --eta-ev 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transportation_models.utils.ctm.results import SimulationResult

from transportation_models.utils.dwpt import plots
from transportation_models.utils.dwpt.adapter import corridor_from_ctm, mainline_vht
from transportation_models.utils.dwpt.demand import compute
from transportation_models.utils.dwpt.model import PadSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ctm-result", required=True, type=Path,
        help="Path to the .npz file with CTM simulation results"
    )
    parser.add_argument(
        "--alpha", required=True, type=float,
        help="DWPT Tx pad length (meters)"
    )
    parser.add_argument(
        "--lambda-gap", required=True, type=float,
        help="DWPT Tx pad spacing (meters)"
    )
    parser.add_argument(
        "--delta", required=True, type=float,
        help="DWPT Rx pad length (meters)"
    )
    parser.add_argument(
        "--beta", required=True, type=float,
        help="DWPT Rx pad rated power (watts)"
    )
    parser.add_argument(
        "--beta-prime", required=True, type=float,
        help="DWPT Tx pad rated power (watts)"
    )
    parser.add_argument(
        "--dx-grid", required=True, type=float,
        help="DWPT spatial resolution (meters)"
    )
    parser.add_argument(
        "--eta-ev", required=True, type=float,
        help="DWPT EV fraction [0,1]"
    )
    parser.add_argument(
        "--segment-m", type=float, default=20.0,
        help=("Segment width (meters) for aggregating the instantaneous "
              "peak load profile. Defaults to 20."),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help=("Output directory for DWPT demand profile. "
              "Defaults to <ctm-result-dir>"),
    )
    parser.add_argument(
        "--drop-first-cell", action=argparse.BooleanOptionalAction, default=True,
        help=("Exclude the upstream-most CTM cell (index 0) from the demand, "
              "whose density is an inflated upstream-boundary artifact. On by "
              "default; pass --no-drop-first-cell to keep it."),
    )
    args = parser.parse_args()

    if not args.ctm_result.exists():
        parser.error(f"--ctm-result not found: {args.ctm_result}")
    ctm_result_path = args.ctm_result.resolve()
    result = SimulationResult.from_npz(ctm_result_path)
    if args.drop_first_cell:
        result = result.without_first_cell()

    out_dir = (
        args.out_dir if args.out_dir is not None
        else args.ctm_result.parent
    )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pad = PadSpec(
        alpha=args.alpha, delta=args.delta, lambda_gap=args.lambda_gap,
        beta=args.beta, beta_prime=args.beta_prime
    )

    corridor = corridor_from_ctm(result, pad, args.dx_grid)
    demand = compute(mainline_vht(result), corridor, args.eta_ev)

    plots.plot_P_y(
        corridor.build_P_y(), corridor.position_m,
        out_path=out_dir / "dwpt_P_y.png",
    )
    plots.plot_demand_heatmap(
        demand, dt_h=result.freeway.dt, out_path=out_dir / "dwpt_demand_heatmap.png",
    )
    plots.plot_aggregate_timeseries(
        demand, dt_h=result.freeway.dt,
        out_path=out_dir / "dwpt_aggregate_timeseries.png",
    )
    plots.plot_peak_load_profile(
        demand, dt_h=result.freeway.dt, segment_m=args.segment_m,
        out_path=out_dir / "dwpt_peak_load_profile.png",
    )
    plots.plot_load_duration_curve(
        demand, dt_h=result.freeway.dt,
        out_path=out_dir / "dwpt_load_duration_curve.png",
    )

    summary_lines = [
        "=== DWPT Demand Generation Summary ===",
        "  CTM result parameters:",
        f"  result              : {ctm_result_path}",
        f"  cells               : {result.n_cells}",
        f"  dt                  : {result.freeway.dt * 3600} s",
        f"  steps               : {result.n_steps}",
        "",
        "  DWPT corridor parameters:",
        f"  corridor length     : {corridor.corridor_length_m:.3e} m",
        f"  alpha               : {args.alpha} m",
        f"  lambda              : {args.lambda_gap} m",
        f"  delta               : {args.delta} m",
        f"  beta                : {args.beta:.3e} W",
        f"  beta_prime          : {args.beta_prime:.3e} W",
        f"  dx-grid             : {args.dx_grid} m",
        f"  eta_EV              : {args.eta_ev}",
        "",
        "  DWPT demand:",
        f"  Energy delivered    : {demand.E.sum():.3e} Wh",
        f"  Peak corridor power : {(demand.E.sum(axis=1).max() / result.freeway.dt):.3e} W"
    ]
    summary = "\n".join(summary_lines) + "\n"
    (out_dir / "dwpt_summary.txt").write_text(summary)
    print()
    print(summary)
    print(f"Wrote {out_dir}/")
    print("  dwpt_P_y.png")
    print("  dwpt_demand_heatmap.png")
    print("  dwpt_aggregate_timeseries.png")
    print("  dwpt_peak_load_profile.png")
    print("  dwpt_load_duration_curve.png")
    print("  dwpt_summary.txt")


if __name__ == "__main__":
    main()