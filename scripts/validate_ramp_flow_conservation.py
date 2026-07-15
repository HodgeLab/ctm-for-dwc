#!/usr/bin/env python3
"""Benchmark flow-conservation as a ramp gap-fill strategy.

For every type (a)/(b) stretch in the stretches CSV(s), estimates the ramp
flow implied by the mainline pair (``r = q_down - q_up`` on type (a),
``s = q_up - q_down`` on type (b), clipped at zero -- rationale in
``conservation_fill_accuracy``) and scores it against the measured ramp
flow (RMSE / NRMSE / BIAS / NBIAS, one row per stretch). No masking or
seeds: the estimate never consults the ramp's own record, so every
co-observed sample scores.

Alongside the per-stretch table it writes four PNGs next to ``--out``:
NRMSE and NBIAS rank-scatters (per-type mean +/- 1 std bands) and
measured-vs-estimated QQ plots pooled over the on-ramp and off-ramp
stretches.

Run from the repo root::

    python scripts/validate_ramp_flow_conservation.py \\
        --stretches data/pems/stretches/880_N.csv \\
        --timeseries-dir /Volumes/easystore/.../timeseries_data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transportation_models.utils.ctm.ramp_validation import conservation_fill_accuracy
from transportation_models.utils.ramp_flow_estimation.manifest import load_stretches

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STRETCHES = _REPO_ROOT / "data/pems/stretches"

_TYPE_COLORS = {"on": "tab:blue", "off": "tab:orange"}


def _metric_scatter(table: pd.DataFrame, metric: str, out_path: Path) -> None:
    """One point per stretch at its global rank under ``metric``, colored by
    ramp type, with per-type mean line and +/- 1 std band."""
    sub = table.dropna(subset=[metric]).sort_values(metric, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for ramp_type, color in _TYPE_COLORS.items():
        grp = sub[sub["ramp_type"] == ramp_type]
        if grp.empty:
            continue
        mean, std = float(grp[metric].mean()), float(grp[metric].std())
        label = f"{ramp_type}-ramp: {mean:.3f}"
        if np.isfinite(std):
            label += f" $\\pm$ {std:.3f}"
            ax.axhspan(mean - std, mean + std, color=color, alpha=0.12)
        ax.axhline(mean, color=color, lw=1.2)
        ax.scatter(grp.index, grp[metric], s=22, color=color, zorder=3,
                   label=label)
    ax.set_xlabel(f"stretch rank (sorted by {metric.upper()})")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Conservation fill {metric.upper()} per stretch")
    ax.legend(title="mean $\\pm$ 1 std")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _qq_plot(points: pd.DataFrame, ramp_type: str, out_path: Path) -> None:
    """Sorted measured-flow quantiles vs sorted estimate quantiles, pooled
    over every scored sample of the given ramp type."""
    pts = points[points["ramp_type"] == ramp_type]
    measured = np.sort(pts["measured"].to_numpy(dtype=float))
    estimated = np.sort(pts["estimated"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(measured, estimated, s=4, alpha=0.25,
               color=_TYPE_COLORS[ramp_type])
    lim = 1.05 * max(float(measured[-1]), float(estimated[-1]), 1.0)
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("measured ramp flow quantiles [veh/5-min]")
    ax.set_ylabel("conservation-estimate quantiles [veh/5-min]")
    ax.set_title(f"QQ: {ramp_type}-ramp stretches ({len(pts):,} samples, "
                 f"{pts['stretch_id'].nunique()} stretches)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stretches", type=Path, default=_DEFAULT_STRETCHES,
                    help="A stretches.csv or a directory of them.")
    ap.add_argument("--timeseries-dir", type=Path, required=True,
                    help="Directory of per-VDS timeseries CSVs.")
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "scripts/output/ramp_conservation_accuracy.csv",
                    help="Per-stretch table (PNGs land next to it).")
    args = ap.parse_args(argv)

    stretches = load_stretches(args.stretches)
    table, points = conservation_fill_accuracy(stretches, args.timeseries_dir)
    if table.empty:
        print("No scoreable type (a)/(b) stretches in the stretches CSV(s).")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote per-stretch table -> {args.out}")

    for metric in ("nrmse", "nbias"):
        png = args.out.with_name(f"{args.out.stem}_{metric}.png")
        _metric_scatter(table, metric, png)
        print(f"wrote {metric.upper()} scatter -> {png}")
    for ramp_type in ("on", "off"):
        if not (points["ramp_type"] == ramp_type).any():
            print(f"no scored {ramp_type}-ramp samples; skipping its QQ plot")
            continue
        png = args.out.with_name(f"{args.out.stem}_qq_{ramp_type}.png")
        _qq_plot(points, ramp_type, png)
        print(f"wrote {ramp_type}-ramp QQ plot -> {png}")

    print()
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
