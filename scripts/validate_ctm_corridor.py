"""Step 9: validate a CTM simulation result against historical PeMS data.

Loads the self-contained ``result.npz`` that
``scripts/simulate_ctm_corridor.py`` writes (which carries the density
and flow arrays, ``dt``, and the sim's wallclock ``start``), computes
per-cell RMSE and MAPE against the historical 5-min PeMS samples over
the same wallclock window, plus the corridor VMT/VHT aggregates, the
GEH statistic, and QQ error-distribution diagnostics.

All artifacts are written to a ``validation/`` subdirectory of the sim
dir (override with ``--out-dir``): ``validation.csv`` (per-cell stats),
``geh_heatmap.png``, ``sim_flow_heatmap.png`` /
``pems_flow_heatmap.png`` and ``sim_density_heatmap.png`` /
``pems_density_heatmap.png`` (simulated vs measured space-time heatmaps,
sharing color limits so they compare directly), and the QQ plots. The same stdout summary -- corridor-wide error, corridor
aggregates, and the top-K worst-fitting cells by density RMSE -- is also
written to ``ctm_validation_summary.txt`` in that directory.

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

import numpy as np
import pandas as pd

from ctm_for_dwc.utils.ctm import (
    compare_against_historical,
    compare_corridor_aggregates,
    compute_flow_geh,
    compute_observed_grid,
    compute_qq_samples,
    corridor_rmse_mape,
)
from ctm_for_dwc.utils.ctm.plots import (
    downsample_to_plot_period,
    plot_flow_density_contour,
    plot_geh_heatmap,
    plot_qq_pooled,
    plot_qq_percell,
)
from ctm_for_dwc.utils.ctm.results import SimulationResult
from ctm_for_dwc.utils.ctm.validation import CorridorAggregates, GEHResult


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


def _fmt_nan(v: float) -> str:
    """Format a float to 1 decimal, or ``-`` when NaN (non-direct cells)."""
    return "-" if pd.isna(v) else f"{v:.1f}"


def _print_summary(
    stats: pd.DataFrame, *,
    sim_dir: Path, cells_path: Path, ts_dir: Path,
    start: pd.Timestamp, dt: float, n_5min: int,
    out_dir: Path, top_k: int,
    aggregates: CorridorAggregates,
    geh: GEHResult | None,
    corridor_errors: dict[str, tuple[int, float, float]],
) -> None:
    """Tabular summary (window, corridor aggregates, worst cells) to stdout
    and ``<out_dir>/ctm_validation_summary.txt``."""

    lines: list[str] = []
    n_cells = len(stats)
    valid = stats[stats["n_flow_samples"] > 0]
    n_with_vds = len(valid)

    lines.append("=== Step 9: historical validation ===")
    lines.append(f"  sim dir       : {sim_dir}")
    lines.append(f"  cells         : {cells_path}")
    lines.append(f"  ts dir        : {ts_dir}")
    lines.append(f"  sim dt        : {dt * 3600.0:.1f} s")
    end = start + pd.Timedelta(minutes=5 * n_5min)
    lines.append(f"  window        : [{start}, {end})  ({n_5min} 5-min samples)")
    lines.append(
        f"  cells         : {n_cells} total, {n_with_vds} scored "
        "(unique direct/tiebreak VDS)"
    )

    if valid.empty:
        lines.append(
            "\n  No direct/tiebreak VDS cells to score -- nothing to validate."
        )
        summary = "\n".join(lines) + "\n"
        print(summary)
        (out_dir / "ctm_validation_summary.txt").write_text(summary)
        return

    lines.append("")
    fn, frmse, fmape = corridor_errors["flow"]
    dn, drmse, dmape = corridor_errors["density"]
    lines.append("  Corridor-pooled error (over scored direct/tiebreak cells):")
    lines.append(f"    flow    RMSE {frmse:.0f} veh/h , MAPE {fmape:.1f}%  (n={fn})")
    lines.append(f"    density RMSE {drmse:.2f} veh/mi, MAPE {dmape:.1f}%  (n={dn})")

    k = min(top_k, len(valid))
    if k > 0:
        lines.append("")
        lines.append(f"  Top {k} worst-fitting cells (by density RMSE):")
        worst = valid.nlargest(k, "density_rmse").reset_index(drop=True)
        # Compact column layout for readability.
        cols = [
            "cell", "vds_id",
            "density_rmse", "density_mape",
            "flow_rmse", "flow_mape",
        ]
        if geh is not None:
            cols += ["flow_geh_median", "flow_geh_pct_under5"]
        display = worst[cols].copy()
        display["density_rmse"] = display["density_rmse"].map("{:.2f}".format)
        display["density_mape"] = display["density_mape"].map("{:.1f}".format)
        display["flow_rmse"] = display["flow_rmse"].map("{:.0f}".format)
        display["flow_mape"] = display["flow_mape"].map("{:.1f}".format)
        if geh is not None:
            display["flow_geh_median"] = display["flow_geh_median"].map(_fmt_nan)
            display["flow_geh_pct_under5"] = (
                display["flow_geh_pct_under5"].map(_fmt_nan)
            )
        lines.append(display.to_string(index=False))

    lines.append("")
    lines.append(
        f"  Corridor aggregates (direct + tiebreak VDS, n={aggregates.n_vds}):"
    )
    lines.append(
        f"    VMT [veh*mi]: sim {aggregates.sim_vmt:,.0f}, "
        f"observed {aggregates.obs_vmt:,.0f}, "
        f"diff {aggregates.vmt_pct_diff:+.1f}%"
    )
    lines.append(
        f"    VHT [veh*h] : sim {aggregates.sim_vht:,.0f}, "
        f"observed {aggregates.obs_vht:,.0f}, "
        f"diff {aggregates.vht_pct_diff:+.1f}%"
    )

    if geh is not None and geh.n_geh_samples > 0:
        lines.append("")
        lines.append(
            f"    flow GEH<5  : {geh.corridor_pct_under5:.0f}% of "
            f"{geh.n_geh_samples} cell-hour samples "
            "(hourly volumes, direct + tiebreak VDS)"
        )

    lines.append("")
    lines.append(f"  Wrote validation artifacts to {out_dir}")

    summary = "\n".join(lines) + "\n"
    print(summary)
    (out_dir / "ctm_validation_summary.txt").write_text(summary)


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
        "--out-dir", default=None, type=Path,
        help=("Directory for all validation artifacts (validation.csv, "
              "geh_heatmap.png, the PeMS measured heatmaps, the QQ plots). "
              "Defaults to <sim-dir>/validation."),
    )
    parser.add_argument(
        "--top-k", default=5, type=int,
        help="How many worst-fitting cells to list in the stdout summary.",
    )
    parser.add_argument(
        "--plot-period-seconds", type=float, default=300.0,
        help=("Plotting period for the simulated flow/density heatmaps; must "
              "match the value passed to simulate_ctm_corridor.py so the "
              "re-rendered sim contours line up with its flow_contour/"
              "density_contour. Defaults to 300s (5 min)."),
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

    # GEH needs whole-hour windows; skip it (rather than fail the whole
    # report) for sims that don't cover an integer number of hours.
    try:
        geh = compute_flow_geh(result, cells, args.timeseries_dir, start=start)
    except ValueError as e:
        print(f"  skipping GEH: {e}", file=sys.stderr)
        geh = None
    if geh is not None:
        stats = stats.merge(
            geh.per_cell[["cell", "flow_geh_median", "flow_geh_pct_under5"]],
            on="cell", how="left",
        )

    n_5min = (
        stats.loc[stats["n_flow_samples"] > 0, "n_flow_samples"].max()
        if (stats["n_flow_samples"] > 0).any() else 0
    )

    out_dir = (
        args.out_dir if args.out_dir is not None else args.sim_dir / "validation"
    )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation.csv"
    stats.to_csv(out_path, index=False)

    if geh is not None and geh.hourly_geh.shape[0] > 0:
        heatmap_path = out_dir / "geh_heatmap.png"
        plot_geh_heatmap(
            geh.hourly_geh,
            row_labels=[
                f"cell {int(c)} (VDS {int(v)})"
                for c, v in zip(geh.per_cell["cell"], geh.per_cell["vds_id"])
            ],
            hour_starts=geh.hour_starts,
            title="Flow GEH by direct/tiebreak cell and hour",
            out_path=heatmap_path,
        )

    # Sim vs measured-PeMS space-time heatmaps, rendered with the same
    # contour plotter (and viridis/plasma colormaps) so the two are directly
    # comparable. Both share color limits taken from the measured PeMS data
    # (the reference). The sim is sampled exactly as simulate_ctm_corridor.py
    # samples its flow_contour/density_contour, so only the color limits
    # differ from that script's contours. Measured coverage shows against the
    # full corridor; cells with no VDS are filled black.
    obs_grid = compute_observed_grid(
        result,
        cells,
        args.timeseries_dir,
        start=start,
    )
    if obs_grid.n_direct_cells:
        end = start + pd.Timedelta(hours=obs_grid.horizon_h)

        # Sim flow/density at the plotting period, matching simulate_ctm_corridor.py.
        dt_s = result.freeway.dt * 3600.0
        ratio = args.plot_period_seconds / dt_s
        steps_per_period = int(round(ratio))
        n_samples, rem = divmod(result.n_steps, steps_per_period) if steps_per_period else (0, 1)
        if steps_per_period < 1 or abs(ratio - steps_per_period) > 1e-6 or n_samples < 1 or rem:
            parser.error(
                f"--plot-period-seconds ({args.plot_period_seconds}) must be a "
                f"positive integer multiple of the sim dt ({dt_s:.1f}s) that "
                f"evenly divides the {result.n_steps}-step horizon."
            )
        kw = dict(steps_per_period=steps_per_period, n_samples=n_samples)
        sim_flow = downsample_to_plot_period(result.mainline_flow, state=False, **kw).T
        sim_density = downsample_to_plot_period(result.density, state=True, **kw)[:, 1:].T

        obs_flow, obs_density = obs_grid.flow, obs_grid.density
        pm_edges = obs_grid.pm_edges
        # Color limits from the measured data so the sim adopts the same
        # scale (sim values outside the measured range clip).
        flow_lim = (float(np.nanmin(obs_flow)), float(np.nanmax(obs_flow)))
        density_lim = (float(np.nanmin(obs_density)), float(np.nanmax(obs_density)))
        for arr, cmap, cbar, lim, missing, fname in (
            (sim_flow, "viridis", "Flow [veh/h]", flow_lim, None,
             "sim_flow_heatmap.png"),
            (obs_flow, "viridis", "Flow [veh/h]", flow_lim, "black",
             "pems_flow_heatmap.png"),
            (sim_density, "plasma", "Density [veh/mi]", density_lim, None,
             "sim_density_heatmap.png"),
            (obs_density, "plasma", "Density [veh/mi]", density_lim, "black",
             "pems_density_heatmap.png"),
        ):
            plot_flow_density_contour(
                arr,
                cbar_label=cbar,
                cmap=cmap,
                horizon_h=obs_grid.horizon_h,
                pm_edges=pm_edges,
                y_label="Simulation time [h]",
                vmin=lim[0],
                vmax=lim[1],
                missing_color=missing,
                out_path=out_dir / fname,
            )

    # QQ error-distribution diagnostics: per quantity, a corridor-pooled
    # two-sample sim-vs-obs QQ plus a per-cell faceted grid.
    qq = compute_qq_samples(result, cells, args.timeseries_dir, start=start)
    corridor_errors = corridor_rmse_mape(qq)
    if qq.per_cell:
        for quantity in ("flow", "density"):
            pooled_path = out_dir / f"{quantity}_qq_pooled.png"
            plot_qq_pooled(qq, quantity=quantity, out_path=pooled_path)
            percell_path = out_dir / f"{quantity}_qq_percell.png"
            plot_qq_percell(qq, quantity=quantity, out_path=percell_path)

    print()
    _print_summary(
        stats, sim_dir=args.sim_dir, cells_path=args.cells,
        ts_dir=args.timeseries_dir, start=start,
        dt=result.freeway.dt, n_5min=int(n_5min),
        out_dir=out_dir, top_k=args.top_k,
        aggregates=aggregates, geh=geh,
        corridor_errors=corridor_errors,
    )


if __name__ == "__main__":
    main()
