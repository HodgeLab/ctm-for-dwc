"""End-to-end demo: load a CTMSIM .mat config, simulate, print, and plot.

Loads one of the bundled I-210W configs from
``src/transportation_models/utils/ctm/ctmsim_configs/`` (24-hour macroscopic
freeway snapshots from CTMSIM v1.1, see :mod:`transportation_models.utils.ctm.io`),
runs our engine for the full 24-hour horizon, prints a per-period summary, and
writes four figures to ``--out-dir`` (default: ``scripts/output/ctm_demo/<day>``):

  * ``flow_contour.png``      space-time heatmap of mainline flow [veh/h]
  * ``density_contour.png``   space-time heatmap of density       [veh/mi]
  * ``aggregate_metrics.png`` VHT, VMT, delay, productivity loss over time
  * ``per_cell_metrics.png``  same four metrics summed per-cell over the day

Run from the repo root:
    python scripts/run_ctm_ctmsim_demo.py                     # default: w060412
    python scripts/run_ctm_ctmsim_demo.py --day w060414
    python scripts/run_ctm_ctmsim_demo.py --day w060412 --out-dir /tmp/ctm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

from transportation_models.utils.ctm import (
    Scenario,
    compare_against_ctmsim,
    compute_metrics,
    ctmsim_demand_at_sim_steps,
    ctmsim_initial_densities,
    freeway_from_ctmsim_mat,
    simulate,
)
from transportation_models.utils.ctm.plots import (
    downsample_to_plot_period,
    per_period_totals,
    plot_aggregate_metrics,
    plot_flow_density_contour,
    plot_per_cell_metrics,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "src/transportation_models/utils/ctm/ctmsim_configs"
RESULTS = REPO / "src/transportation_models/utils/ctm/ctmsim_results"
DEFAULT_OUT = REPO / "scripts/output/ctm_demo"

# CTMSIM I-210W defaults: 5-min plotting period over 10-s sampling -> 30 sim
# steps per plotted sample, 288 samples per 24-hour run.
PLOT_PERIOD_STEPS = 30
PLOT_SAMPLES = 288


def build_scenario(mat_path: Path):
    """Build a 24-hour Scenario from a CTMSIM .mat config.

    Pulls initial densities and the on-ramp demand profile through the I/O
    adapters; pulls the upstream boundary inflow and off-ramp split ratios
    straight from the .mat. For β we prefer ``betaProfile`` (time-varying);
    otherwise (as in the bundled I-210W configs, which use ``frflowProfile``)
    we fall back to the constant per-cell ``FRbeta`` field on each ``celldata``
    struct. Both branches multiply by the per-cell ``FRknob`` calibration
    multiplier so the scenario matches CTMSIM's *effective* β (``FRknob`` is
    degenerate in ``{0, 1}`` across the bundled I-210W days, but applying
    it keeps the adapter correct against future calibrations).
    """
    fwy = freeway_from_ctmsim_mat(mat_path)
    n = fwy.n_cells
    rho0 = ctmsim_initial_densities(mat_path)
    demand_sim = ctmsim_demand_at_sim_steps(mat_path)
    n_steps = demand_sim.shape[1]

    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    inflow_plot = mat["demandProfile"][:, 0]
    inflow_sim = np.repeat(inflow_plot, PLOT_PERIOD_STEPS)[:n_steps]

    fr_knobs = np.array([c.FRknob for c in mat["celldata"]], dtype=float)
    if "betaProfile" in mat:
        # Same K x (N+2) shape as demandProfile: trim mainline/downstream cols.
        beta_plot = mat["betaProfile"][:, 1 : n + 1] * fr_knobs[None, :]
        beta_sim = np.repeat(beta_plot, PLOT_PERIOD_STEPS, axis=0)[:n_steps, :].T
    else:
        beta_const = (
            np.array([c.FRbeta for c in mat["celldata"]], dtype=float)
            * fr_knobs
        )
        beta_sim = np.broadcast_to(beta_const[:, None], (n, n_steps)).copy()

    scn = Scenario(
        rho0=rho0,
        inflow=inflow_sim,
        demand=demand_sim,
        beta=beta_sim,
        q0=np.zeros(n),
    )
    return fwy, scn.validate(fwy)


def per_period_per_cell(per_step: np.ndarray) -> np.ndarray:
    """Sum a (N, T) per-cell series into per-cell, per-period totals (N, K).

    NOTE: pre-existing dead code -- defined but never called within this
    script. Left in place pending an explicit cleanup pass.
    """
    n = per_step.shape[0]
    return per_step.reshape(n, PLOT_SAMPLES, PLOT_PERIOD_STEPS).sum(axis=2)


def _kw():
    return dict(steps_per_period=PLOT_PERIOD_STEPS, n_samples=PLOT_SAMPLES)


def print_summary(fwy, res, metrics, day: str) -> None:
    """Print a per-3-hour snapshot of average density/flow and aggregate metrics."""
    n_steps = res.n_steps
    snap_periods = [0, 36, 72, 108, 144, 180, 216, 252, PLOT_SAMPLES - 1]
    rho_plot = downsample_to_plot_period(res.density, state=True, **_kw())
    flow_plot = downsample_to_plot_period(res.mainline_flow, state=False, **_kw())
    vht = per_period_totals(metrics.vht, **_kw())
    vmt = per_period_totals(metrics.vmt, **_kw())
    delay = per_period_totals(metrics.delay, **_kw())
    ploss = per_period_totals(metrics.productivity_loss, **_kw())

    print(f"\n=== CTMSIM {day} on I-210 West ===")
    print(f"  cells: {fwy.n_cells}  dt: {fwy.dt * 3600:.1f}s  "
          f"horizon: {n_steps * fwy.dt:.1f}h  ({PLOT_SAMPLES} plotting periods)")
    print()
    print(f"  {'time':>6}   {'mean rho':>9}  {'mean f':>9}    "
          f"{'VHT':>8}  {'VMT':>9}  {'delay':>8}  {'ploss':>7}")
    print(f"  {'h':>6}   {'veh/mi':>9}  {'veh/h':>9}    "
          f"{'veh*h':>8}  {'veh*mi':>9}  {'veh*h':>8}  {'mi*h':>7}")
    print(f"  {'-' * 6}   {'-' * 9}  {'-' * 9}    {'-' * 8}  {'-' * 9}  "
          f"{'-' * 8}  {'-' * 7}")
    for p in snap_periods:
        t_h = p * PLOT_PERIOD_STEPS * fwy.dt
        rho_mean = rho_plot[:, p + 1].mean()        # state at end of period p
        f_mean = flow_plot[:, p].mean()
        print(f"  {t_h:6.2f}   {rho_mean:9.2f}  {f_mean:9.1f}    "
              f"{vht[p]:8.3f}  {vmt[p]:9.2f}  {delay[p]:8.3f}  {ploss[p]:7.4f}")
    print()
    print(f"  24-hour totals:  VHT={vht.sum():.1f}  VMT={vmt.sum():.1f}  "
          f"delay={delay.sum():.1f}  ploss={ploss.sum():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--day", default="w060412",
        help="CTMSIM config stem in ctmsim_configs/ (default: w060412)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help=f"Where to write figures (default: {DEFAULT_OUT}/<day>)",
    )
    args = parser.parse_args()

    mat_path = CONFIGS / f"{args.day}.mat"
    if not mat_path.exists():
        available = sorted(p.stem for p in CONFIGS.glob("*.mat"))
        raise SystemExit(f"Config not found: {mat_path}\nAvailable: {available}")
    out_dir = (args.out_dir or DEFAULT_OUT / args.day).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fwy, scn = build_scenario(mat_path)
    res = simulate(fwy, scn)
    metrics = compute_metrics(res)

    print_summary(fwy, res, metrics, args.day)

    # Contours are (K, N): time runs down rows, cells across columns.
    flow_KN = downsample_to_plot_period(
        res.mainline_flow, state=False, **_kw(),
    ).T
    density_KN = downsample_to_plot_period(
        res.density, state=True, **_kw(),
    )[:, 1:].T   # drop initial col
    # Distance from the corridor's upstream end at each cell boundary.
    # `freeway_from_ctmsim_mat` drops the .mat's absolute PMstart/PMend, so
    # cumsum-of-lengths is the natural anchor here.
    lengths = np.array([c.length for c in fwy.cells])
    pm_edges = np.concatenate([[0.0], np.cumsum(lengths)])

    plot_flow_density_contour(
        flow_KN, title=f"Mainline flow — I-210W {args.day}",
        cbar_label="flow [veh/h]", cmap="viridis",
        horizon_h=24.0, pm_edges=pm_edges,
        y_label="time of day [h]",
        out_path=out_dir / "flow_contour.png",
    )
    plot_flow_density_contour(
        density_KN, title=f"Density — I-210W {args.day}",
        cbar_label="density [veh/mi]", cmap="magma",
        horizon_h=24.0, pm_edges=pm_edges,
        y_label="time of day [h]",
        out_path=out_dir / "density_contour.png",
    )

    time_h = np.arange(PLOT_SAMPLES) * PLOT_PERIOD_STEPS * fwy.dt
    plot_aggregate_metrics(
        time_h,
        per_period_totals(metrics.vht, **_kw()),
        per_period_totals(metrics.vmt, **_kw()),
        per_period_totals(metrics.delay, **_kw()),
        per_period_totals(metrics.productivity_loss, **_kw()),
        period_label="5-min", x_label="time of day [h]",
        out_path=out_dir / "aggregate_metrics.png",
    )
    plot_per_cell_metrics(
        metrics, title="24-hour totals per cell",
        out_path=out_dir / "per_cell_metrics.png",
    )

    print(f"\nFigures written to {out_dir}/")
    for name in ("flow_contour.png", "density_contour.png",
                 "aggregate_metrics.png", "per_cell_metrics.png"):
        print(f"  {name}")

    # ---- CTMSIM ground-truth comparison ----------------------------------
    # Compare our trajectory and aggregate metrics against the CTMSIM v1.1
    # reference CSVs bundled under utils/ctm/ctmsim_results/<day>/. The
    # block is skipped (with a note) when the reference directory is
    # missing for the chosen day.
    ref_dir = RESULTS / args.day
    if not ref_dir.exists():
        print(f"\nCTMSIM reference dir not found at {ref_dir}; "
              "skipping ground-truth comparison.")
        return

    report = compare_against_ctmsim(res, ref_dir)
    report.per_cell.to_csv(out_dir / "validation_per_cell.csv", index=False)
    report.per_metric.to_csv(out_dir / "validation_per_metric.csv", index=False)

    print(f"\n=== CTMSIM ground-truth comparison ({args.day}) ===")
    pc = report.per_cell
    print(f"  cells              : {len(pc)}")
    print(f"  density RMSE [veh/mi]: "
          f"median {pc['density_rmse'].median():.4g}, "
          f"mean {pc['density_rmse'].mean():.4g}, "
          f"max {pc['density_rmse'].max():.4g}")
    print(f"  density MAPE [%]     : "
          f"median {pc['density_mape'].median():.4g}, "
          f"mean {pc['density_mape'].mean():.4g}, "
          f"max {pc['density_mape'].max():.4g}")
    print(f"  flow    RMSE [veh/h] : "
          f"median {pc['flow_rmse'].median():.4g}, "
          f"mean {pc['flow_rmse'].mean():.4g}, "
          f"max {pc['flow_rmse'].max():.4g}")
    print(f"  flow    MAPE [%]     : "
          f"median {pc['flow_mape'].median():.4g}, "
          f"mean {pc['flow_mape'].mean():.4g}, "
          f"max {pc['flow_mape'].max():.4g}")
    print()
    print("  Aggregate metrics (per 5-min period):")
    print(f"    {'metric':<20} {'rmse':>12} {'mape [%]':>10}")
    for _, row in report.per_metric.iterrows():
        print(f"    {row['metric']:<20} {row['rmse']:>12.4g} "
              f"{row['mape']:>10.4g}")
    print()
    print(f"  Note: MAPE on delay / productivity_loss is dominated by "
          "near-zero values outside")
    print(f"  the morning peak; RMSE is the more informative number for "
          "those metrics.")
    print()
    print(f"  Wrote validation_per_cell.csv, validation_per_metric.csv")


if __name__ == "__main__":
    main()
