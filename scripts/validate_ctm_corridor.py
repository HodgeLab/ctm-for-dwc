"""Step 9: validate a CTM simulation result against historical PeMS data.

Reads the per-quantity output CSVs that ``scripts/simulate_ctm_corridor.py``
writes (``density.csv`` + ``mainline_flow.csv``), reconstructs the
sim-side density and flow arrays, computes per-cell RMSE and MAPE
against the historical 5-min PeMS samples over the same wallclock
window, and writes a ``validation.csv`` with the per-cell stats.

Stats are summarized to stdout: corridor-wide median / mean / max of
each metric and the top-K worst-fitting cells by density RMSE.

Run from the repo root::

    python scripts/validate_ctm_corridor.py \\
        --sim-dir scripts/output/ctm_corridor/I_210_W/sim \\
        --cells   scripts/output/ctm_corridor/I_210_W/cells.csv \\
        --timeseries-dir data/pems/csv_files \\
        --start "2022-04-12 06:00"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from transportation_models.utils.ctm import compare_against_historical


def _load_sim_arrays(sim_dir: Path) -> SimpleNamespace:
    """Read density.csv + mainline_flow.csv into a minimal result-like wrapper.

    Only the attributes :func:`compare_against_historical` reads are
    populated: ``freeway.n_cells``, ``freeway.dt``, ``density``,
    ``mainline_flow``, and ``n_steps``.
    """
    density_df = pd.read_csv(sim_dir / "density.csv", index_col=0)
    flow_df = pd.read_csv(sim_dir / "mainline_flow.csv", index_col=0)
    # CSVs are wide: time_h on the row index, cell indices on the columns.
    # Validation wants (n_cells, T+1) / (n_cells, T) arrays.
    density = density_df.to_numpy().T
    mainline_flow = flow_df.to_numpy().T
    # Derive sim dt from the time_h index of density (regularly spaced).
    times = density_df.index.to_numpy()
    if times.size < 2:
        raise SystemExit(
            f"density.csv has fewer than 2 rows -- cannot derive dt."
        )
    dt = float(times[1] - times[0])
    n_cells = density.shape[0]
    if mainline_flow.shape[0] != n_cells:
        raise SystemExit(
            f"density.csv has {n_cells} cells but mainline_flow.csv has "
            f"{mainline_flow.shape[0]}; the two CSVs must come from the "
            "same sim run."
        )
    return SimpleNamespace(
        freeway=SimpleNamespace(n_cells=n_cells, dt=dt),
        density=density,
        mainline_flow=mainline_flow,
        n_steps=mainline_flow.shape[1],
    )


def _print_summary(
    stats: pd.DataFrame, *,
    sim_dir: Path, cells_path: Path, ts_dir: Path,
    start: pd.Timestamp, dt: float, n_5min: int,
    out_path: Path, top_k: int,
) -> None:
    """Tabular stdout summary: window, corridor aggregates, worst cells."""
    n_cells = len(stats)
    valid = stats[stats["n_flow_samples"] > 0]
    n_with_vds = len(valid)

    print("=== Step 9: historical validation ===")
    print(f"  sim dir       : {sim_dir}")
    print(f"  cells         : {cells_path}")
    print(f"  ts dir        : {ts_dir}")
    print(f"  sim dt        : {dt * 3600.0:.1f} s")
    end = start + pd.Timedelta(minutes=5 * n_5min)
    print(f"  window        : [{start}, {end})  ({n_5min} 5-min samples)")
    print(f"  cells         : {n_cells} total, {n_with_vds} with assigned VDS")

    if valid.empty:
        print("\n  No cells had an assigned VDS -- nothing to validate.")
        return

    def _fmt(series: pd.Series, fmt: str) -> str:
        return (
            f"median {fmt.format(series.median())}, "
            f"mean {fmt.format(series.mean())}, "
            f"max {fmt.format(series.max())}"
        )

    print()
    print("  Per-cell aggregates over cells with an assigned VDS:")
    print(f"    density RMSE  [veh/mi]: {_fmt(valid['density_rmse'], '{:.2f}')}")
    print(f"    density MAPE  [%]     : {_fmt(valid['density_mape'], '{:.1f}')}")
    print(f"    flow    RMSE  [veh/h] : {_fmt(valid['flow_rmse'], '{:.0f}')}")
    print(f"    flow    MAPE  [%]     : {_fmt(valid['flow_mape'], '{:.1f}')}")

    k = min(top_k, len(valid))
    if k > 0:
        print()
        print(f"  Top {k} worst-fitting cells (by density RMSE):")
        worst = valid.nlargest(k, "density_rmse").reset_index(drop=True)
        # Compact column layout for readability.
        display = worst[[
            "cell", "vds_id",
            "density_rmse", "density_mape",
            "flow_rmse", "flow_mape",
        ]].copy()
        display["density_rmse"] = display["density_rmse"].map("{:.2f}".format)
        display["density_mape"] = display["density_mape"].map("{:.1f}".format)
        display["flow_rmse"] = display["flow_rmse"].map("{:.0f}".format)
        display["flow_mape"] = display["flow_mape"].map("{:.1f}".format)
        print(display.to_string(index=False))

    print()
    print(f"  Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sim-dir", required=True, type=Path,
        help=("Directory containing density.csv + mainline_flow.csv "
              "(output of scripts/simulate_ctm_corridor.py)."),
    )
    parser.add_argument(
        "--cells", required=True, type=Path,
        help=("Step-1+2 cells.csv used for the sim. Must carry vds_id "
              "and vds_lanes (Step 2 attaches both)."),
    )
    parser.add_argument(
        "--timeseries-dir", required=True, type=Path,
        help="Directory of Step-3 per-VDS timeseries CSVs.",
    )
    parser.add_argument(
        "--start", required=True, type=pd.Timestamp,
        help=("Wallclock time of the sim's first step (k=0). Must align "
              "to the PeMS 5-min grid."),
    )
    parser.add_argument(
        "--out", default=None, type=Path,
        help=("Destination for the per-cell validation CSV. "
              "Defaults to <sim-dir>/validation.csv."),
    )
    parser.add_argument(
        "--top-k", default=5, type=int,
        help="How many worst-fitting cells to list in the stdout summary.",
    )
    args = parser.parse_args()

    print(f"loading sim arrays from {args.sim_dir}", file=sys.stderr)
    result = _load_sim_arrays(args.sim_dir)
    cells = pd.read_csv(args.cells)

    stats = compare_against_historical(
        result, cells, args.timeseries_dir, start=args.start,
    )
    n_5min = (
        stats.loc[stats["n_flow_samples"] > 0, "n_flow_samples"].max()
        if (stats["n_flow_samples"] > 0).any() else 0
    )

    out_path = args.out if args.out is not None else args.sim_dir / "validation.csv"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_path, index=False)

    print()
    _print_summary(
        stats, sim_dir=args.sim_dir, cells_path=args.cells,
        ts_dir=args.timeseries_dir, start=args.start,
        dt=result.freeway.dt, n_5min=int(n_5min),
        out_path=out_path, top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
