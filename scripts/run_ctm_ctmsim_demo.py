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

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from transportation_models.utils.ctm import (
    Scenario,
    compute_metrics,
    ctmsim_demand_at_sim_steps,
    ctmsim_initial_densities,
    freeway_from_ctmsim_mat,
    simulate,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "src/transportation_models/utils/ctm/ctmsim_configs"
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
    struct -- on these configs that matches CTMSIM's effective β exactly.
    """
    fwy = freeway_from_ctmsim_mat(mat_path)
    n = fwy.n_cells
    rho0 = ctmsim_initial_densities(mat_path)
    demand_sim = ctmsim_demand_at_sim_steps(mat_path)
    n_steps = demand_sim.shape[1]

    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    inflow_plot = mat["demandProfile"][:, 0]
    inflow_sim = np.repeat(inflow_plot, PLOT_PERIOD_STEPS)[:n_steps]

    if "betaProfile" in mat:
        # Same K x (N+2) shape as demandProfile: trim mainline/downstream cols.
        beta_plot = mat["betaProfile"][:, 1 : n + 1]
        beta_sim = np.repeat(beta_plot, PLOT_PERIOD_STEPS, axis=0)[:n_steps, :].T
    else:
        beta_const = np.array([c.FRbeta for c in mat["celldata"]], dtype=float)
        beta_sim = np.broadcast_to(beta_const[:, None], (n, n_steps)).copy()

    scn = Scenario(
        rho0=rho0,
        inflow=inflow_sim,
        demand=demand_sim,
        beta=beta_sim,
        q0=np.zeros(n),
    )
    return fwy, scn.validate(fwy)


def downsample(arr: np.ndarray, *, state: bool) -> np.ndarray:
    """Pick the per-plotting-period columns of a (N, T+1) or (N, T) array.

    For state arrays (``T + 1`` columns) we sample at step boundaries
    ``0, 30, ..., T``; for flow/speed arrays (``T`` columns) we sample at the
    start of each plotting period, ``0, 30, ..., T - 30``.
    """
    if state:
        cols = np.arange(PLOT_SAMPLES + 1) * PLOT_PERIOD_STEPS
    else:
        cols = np.arange(PLOT_SAMPLES) * PLOT_PERIOD_STEPS
    return arr[:, cols]


def per_period_totals(per_step: np.ndarray) -> np.ndarray:
    """Sum a (T,) per-sim-step series into per-plotting-period totals (K,)."""
    return per_step.reshape(PLOT_SAMPLES, PLOT_PERIOD_STEPS).sum(axis=1)


def per_period_per_cell(per_step: np.ndarray) -> np.ndarray:
    """Sum a (N, T) per-cell series into per-cell, per-period totals (N, K)."""
    n = per_step.shape[0]
    return per_step.reshape(n, PLOT_SAMPLES, PLOT_PERIOD_STEPS).sum(axis=2)


def print_summary(fwy, res, metrics, day: str) -> None:
    """Print a per-3-hour snapshot of average density/flow and aggregate metrics."""
    n_steps = res.n_steps
    snap_periods = [0, 36, 72, 108, 144, 180, 216, 252, PLOT_SAMPLES - 1]
    rho_plot = downsample(res.density, state=True)
    flow_plot = downsample(res.mainline_flow, state=False)
    vht = per_period_totals(metrics.vht)
    vmt = per_period_totals(metrics.vmt)
    delay = per_period_totals(metrics.delay)
    ploss = per_period_totals(metrics.productivity_loss)

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


def plot_contour(arr_KN: np.ndarray, *, title: str, cbar_label: str,
                 cmap: str, out_path: Path) -> None:
    """Render a (K, N) space-time array as a heatmap, save to ``out_path``."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    n_cells = arr_KN.shape[1]
    im = ax.imshow(
        arr_KN, aspect="auto", origin="lower", cmap=cmap,
        extent=[-0.5, n_cells - 0.5, 0.0, 24.0],
    )
    ax.set_xlabel("cell index (upstream -> downstream)")
    ax.set_ylabel("time of day [h]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_aggregate_metrics(time_h, vht, vmt, delay, ploss, out_path: Path) -> None:
    """Stack the four aggregate metrics in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    series = [
        (vht,   "VHT [veh*h]",                "tab:blue"),
        (vmt,   "VMT [veh*mi]",               "tab:green"),
        (delay, "delay [veh*h]",              "tab:red"),
        (ploss, "productivity loss [mi*h]",   "tab:purple"),
    ]
    for ax, (y, label, color) in zip(axes.flat, series):
        ax.plot(time_h, y, color=color, lw=1.4)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("time of day [h]")
    fig.suptitle("Aggregate performance metrics per 5-min period")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_cell_metrics(metrics, out_path: Path) -> None:
    """Day-total VHT/VMT/delay/ploss broken down by cell."""
    vht = metrics.vht_per_cell.sum(axis=1)
    vmt = metrics.vmt_per_cell.sum(axis=1)
    delay = metrics.delay_per_cell.sum(axis=1)
    ploss = metrics.productivity_loss_per_cell.sum(axis=1)
    cells = np.arange(vht.size)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    series = [
        (vht,   "VHT [veh*h]",                "tab:blue"),
        (vmt,   "VMT [veh*mi]",               "tab:green"),
        (delay, "delay [veh*h]",              "tab:red"),
        (ploss, "productivity loss [mi*h]",   "tab:purple"),
    ]
    for ax, (y, label, color) in zip(axes.flat, series):
        ax.bar(cells, y, color=color)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")
    for ax in axes[-1, :]:
        ax.set_xlabel("cell index")
    fig.suptitle("24-hour totals per cell")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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
    flow_KN = downsample(res.mainline_flow, state=False).T
    density_KN = downsample(res.density, state=True)[:, 1:].T   # drop initial col

    plot_contour(
        flow_KN, title=f"Mainline flow — I-210W {args.day}",
        cbar_label="flow [veh/h]", cmap="viridis",
        out_path=out_dir / "flow_contour.png",
    )
    plot_contour(
        density_KN, title=f"Density — I-210W {args.day}",
        cbar_label="density [veh/mi]", cmap="magma",
        out_path=out_dir / "density_contour.png",
    )

    time_h = np.arange(PLOT_SAMPLES) * PLOT_PERIOD_STEPS * fwy.dt
    plot_aggregate_metrics(
        time_h,
        per_period_totals(metrics.vht),
        per_period_totals(metrics.vmt),
        per_period_totals(metrics.delay),
        per_period_totals(metrics.productivity_loss),
        out_path=out_dir / "aggregate_metrics.png",
    )
    plot_per_cell_metrics(metrics, out_path=out_dir / "per_cell_metrics.png")

    print(f"\nFigures written to {out_dir}/")
    for name in ("flow_contour.png", "density_contour.png",
                 "aggregate_metrics.png", "per_cell_metrics.png"):
        print(f"  {name}")


if __name__ == "__main__":
    main()
