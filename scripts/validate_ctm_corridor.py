"""Step 9: validate a CTM simulation result against historical PeMS data.

Loads the self-contained ``result.npz`` that
``scripts/simulate_ctm_corridor.py`` writes (which carries the density
and flow arrays, ``dt``, and the sim's wallclock ``start``), computes
per-cell RMSE and MAPE against the historical 5-min PeMS samples over
the same wallclock window, and writes a ``validation.csv`` with the
per-cell stats.

Stats are summarized to stdout: corridor-wide median / mean / max of
each metric and the top-K worst-fitting cells by density RMSE.

Run from the repo root::

    python scripts/validate_ctm_corridor.py \\
        --sim-dir scripts/output/ctm_corridor/I_210_W/sim \\
        --cells   scripts/output/ctm_corridor/I_210_W/cells.csv \\
        --timeseries-dir data/pems/csv_files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from transportation_models.utils.ctm import (
    compare_against_historical,
    compare_corridor_aggregates,
)
from transportation_models.utils.ctm.results import SimulationResult
from transportation_models.utils.ctm.validation import CorridorAggregates


def resolve_start(
    cli_start: pd.Timestamp | None, baked_start: pd.Timestamp | None,
) -> pd.Timestamp:
    """Pick the validation window's start: prefer the baked-in value.

    Falls back to ``--start`` for older npz files that predate
    start-persistence. Raises :class:`ValueError` if neither is available or if
    a given ``--start`` disagrees with the baked-in start.
    """
    if cli_start is None and baked_start is None:
        raise ValueError(
            "result.npz has no persisted start; pass --start (or rebuild the "
            "CTM with scripts/simulate_ctm_corridor.py to bake it in)."
        )
    if (
        cli_start is not None and baked_start is not None
        and cli_start != baked_start
    ):
        raise ValueError(
            f"--start {cli_start} disagrees with the start baked into "
            f"result.npz ({baked_start}); omit --start to use it."
        )
    return cli_start if cli_start is not None else baked_start


def _print_summary(
    stats: pd.DataFrame, *,
    sim_dir: Path, cells_path: Path, ts_dir: Path,
    start: pd.Timestamp, dt: float, n_5min: int,
    out_path: Path, top_k: int,
    aggregates: CorridorAggregates,
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
    print(
        f"  Corridor aggregates (direct + tiebreak VDS, n={aggregates.n_vds}):"
    )
    print(
        f"    VMT [veh*mi]: sim {aggregates.sim_vmt:,.0f}, "
        f"observed {aggregates.obs_vmt:,.0f}, "
        f"diff {aggregates.vmt_pct_diff:+.1f}%"
    )
    print(
        f"    VHT [veh*h] : sim {aggregates.sim_vht:,.0f}, "
        f"observed {aggregates.obs_vht:,.0f}, "
        f"diff {aggregates.vht_pct_diff:+.1f}%"
    )

    print()
    print(f"  Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sim-dir", required=True, type=Path,
        help=("Directory containing result.npz "
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
        "--start", default=None, type=pd.Timestamp,
        help=("Wallclock time of the sim's first step (k=0). Optional: "
              "defaults to the start baked into result.npz. Only needed "
              "for older result.npz files that predate start-persistence; "
              "if given, must match the baked-in start. Must align to the "
              "PeMS 5-min grid."),
    )
    parser.add_argument(
        "--station-metadata", default=Path("data/pems/station_metadata.csv"),
        type=Path,
        help=("PeMS station_metadata.csv (needs ID and Length columns); "
              "supplies each direct VDS's segment length for the corridor "
              "VMT/VHT aggregates. Defaults to data/pems/station_metadata.csv."),
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

    print(f"loading result.npz from {args.sim_dir}", file=sys.stderr)
    result = SimulationResult.from_npz(args.sim_dir / "result.npz")
    cells = pd.read_csv(args.cells)

    try:
        start = resolve_start(args.start, result.start)
    except ValueError as e:
        parser.error(str(e))

    stats = compare_against_historical(
        result, cells, args.timeseries_dir, start=start,
    )
    station_metadata = pd.read_csv(args.station_metadata)
    aggregates = compare_corridor_aggregates(
        result, cells, args.timeseries_dir, station_metadata, start=start,
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
        ts_dir=args.timeseries_dir, start=start,
        dt=result.freeway.dt, n_5min=int(n_5min),
        out_path=out_path, top_k=args.top_k,
        aggregates=aggregates,
    )


if __name__ == "__main__":
    main()
