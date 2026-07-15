#!/usr/bin/env python3
"""Aggregate train_ramp_flow_zhang pair-sweep runs: metrics vs post-DDA MMD.

Offline evaluation over a sweep directory of the pairwise driver's
``result.json`` files (one per source->target pair; no W&B). Writes to
``--out-dir``:

* ``pairs.csv`` -- one row per pair x stage (``dda``, ``dda_mt``): target
  NRMSE, R2, NBIAS (for ``dda_mt`` the mean across survey days, with
  ``*_std`` columns; NaN for ``dda``), plus the pair's backbone and
  post-DDA adaptation-layer MMD distances.
* ``{metric}_vs_mmd.png`` -- per metric: the DDA-only stage as a scatter and
  the DDA+MT stage as mean +/- std error bars across survey days, both
  against the post-DDA pair MMD, with a least-squares trend line per stage.
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

STAGES = ("dda", "dda_mt")
METRICS = ("nrmse", "r2", "nbias")
Y_LIMITS = {"nrmse": (0.0, 1.0), "r2": (-1.0, 1.0), "nbias": (-1.0, 1.0)}


def load_pairs(results_dir: Path) -> pd.DataFrame:
    """One row per pair x stage from every result.json under the dir."""
    rows = []
    for path in sorted(results_dir.glob("**/result.json")):
        res = json.loads(path.read_text())
        cfg = res.get("config", {})
        tm, mmd = res.get("target_metrics", {}), res.get("source_target_mmd", {})
        if not (all(st in tm for st in STAGES) and "dda" in mmd):
            print(f"skipping {path}: not a complete pairwise-driver result")
            continue
        base = {"source": cfg["source_stretch"], "target": cfg["target_stretch"],
                "mmd_backbone": mmd["backbone"], "mmd_dda": mmd["dda"]}
        rows.append({**base, "stage": "dda",
                     **{m: tm["dda"][m] for m in METRICS},
                     **{f"{m}_std": np.nan for m in METRICS}})
        rows.append({**base, "stage": "dda_mt",
                     **{m: tm["dda_mt"]["mean"][m] for m in METRICS},
                     **{f"{m}_std": tm["dda_mt"]["std"][m] for m in METRICS}})
    return pd.DataFrame(rows)


def plot_metric_vs_mmd(pairs: pd.DataFrame, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for stage, color in zip(STAGES, colors):
        part = pairs[pairs["stage"] == stage].sort_values("mmd_dda")
        x, y = part["mmd_dda"].to_numpy(), part[metric].to_numpy()
        if stage == "dda_mt":     # mean +/- std across survey days
            ax.errorbar(x, y, yerr=part[f"{metric}_std"].to_numpy(),
                        fmt="o", color=color, label="DDA+MT (mean +/- std)",
                        alpha=0.7, capsize=3)
        else:
            ax.scatter(x, y, color=color, label="DDA", alpha=0.7)
        if len(part) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            ax.plot(x, slope * x + intercept, color=color, linewidth=1)
    ax.set_xlabel("source<->target pair MMD (adaptation layer, post-DDA)")
    ax.set_ylabel(f"target {metric.upper()}")
    ax.set_ylim(*Y_LIMITS[metric])
    ax.set_title(f"{metric.upper()} vs post-DDA pair MMD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Sweep directory holding per-pair result.json files "
                         "(e.g. scripts/output/zhang_sweep).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where CSV/plots go (default: <results-dir>/aggregate).")
    args = ap.parse_args(argv)

    pairs = load_pairs(args.results_dir)
    if pairs.empty:
        print(f"no complete result.json files under {args.results_dir}")
        return 1
    n_pairs = len(pairs) // len(STAGES)
    out_dir = args.out_dir if args.out_dir is not None \
        else args.results_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs.to_csv(out_dir / "pairs.csv", index=False)
    for metric in METRICS:
        plot_metric_vs_mmd(pairs, metric, out_dir / f"{metric}_vs_mmd.png")

    print(f"{n_pairs} pairs aggregated -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
