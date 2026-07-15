#!/usr/bin/env python3
"""Aggregate compare_zhang_pooling runs: 2x2 tables, averages, MMD trends.

Offline evaluation over a sweep directory of ``result.json`` files (one per
(target, single-source) combination; no W&B involvement). Writes to
``--out-dir``:

* ``combos.csv``  -- one row per combination x arm: NRMSE, R2, NBIAS, the
  arm's own pair MMD, and ``mmd_x`` -- the combination's canonical pair
  distance, taken from the *single/no-DDA* arm (the distance as seen by a
  source-trained backbone before any adaptation, the analogue of the
  paper's Table II).
* ``summary.csv`` -- mean/std of each metric per arm across combinations.
* ``{metric}_vs_mmd.png`` -- per metric (NRMSE, R2, NBIAS): scatter of the
  metric against ``mmd_x`` with one series + least-squares trend line per
  arm, making "how does each arm's accuracy change with pair distance"
  directly readable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ARMS = ("single", "single_dda", "pooled", "pooled_dda")
METRICS = ("nrmse", "r2", "nbias")


def load_combos(results_dir: Path) -> pd.DataFrame:
    """One row per combination x arm from every result.json under the dir."""
    rows = []
    for path in sorted(results_dir.glob("**/result.json")):
        res = json.loads(path.read_text())
        cfg, arms = res.get("config", {}), res.get("arms", {})
        if not all(a in arms for a in ARMS):
            print(f"skipping {path}: does not hold all arms {ARMS}")
            continue
        mmd_x = arms["single"]["pair_mmd"]
        for arm in ARMS:
            rows.append({
                "target": cfg["target_stretch"], "source": cfg["source_stretch"],
                "arm": arm, "mmd_x": mmd_x,
                "pair_mmd": arms[arm]["pair_mmd"],
                **{m: arms[arm]["target_metrics"][m] for m in METRICS},
            })
    return pd.DataFrame(rows)


def summarize(combos: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of each metric per arm across combinations."""
    agg = combos.groupby("arm")[list(METRICS)].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reindex(list(ARMS))


def plot_metric_vs_mmd(combos: pd.DataFrame, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, color in zip(ARMS, plt.rcParams["axes.prop_cycle"].by_key()["color"]):
        part = combos[combos["arm"] == arm].sort_values("mmd_x")
        x, y = part["mmd_x"].to_numpy(), part[metric].to_numpy()
        ax.scatter(x, y, label=arm, color=color, alpha=0.7)
        if len(part) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            ax.plot(x, slope * x + intercept, color=color, linewidth=1)
    ax.set_xlabel("source<->target pair MMD (single backbone, pre-DDA)")
    ax.set_ylabel(f"target {metric.upper()}")
    ax.set_title(f"{metric.upper()} vs pair MMD, per arm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Sweep directory holding per-combination result.json files.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where CSVs/plots go (default: <results-dir>/aggregate).")
    args = ap.parse_args(argv)

    combos = load_combos(args.results_dir)
    if combos.empty:
        print(f"no complete result.json files under {args.results_dir}")
        return 1
    n_combos = len(combos) // len(ARMS)
    out_dir = args.out_dir if args.out_dir is not None \
        else args.results_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    combos.to_csv(out_dir / "combos.csv", index=False)
    summary = summarize(combos)
    summary.to_csv(out_dir / "summary.csv")
    for metric in METRICS:
        plot_metric_vs_mmd(combos, metric, out_dir / f"{metric}_vs_mmd.png")

    print(f"{n_combos} combinations aggregated -> {out_dir}")
    with pd.option_context("display.width", 120):
        print(summary.round(4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
