#!/usr/bin/env python3
"""Aggregate train_ramp_flow_zhang pair-sweep runs: metrics vs MMD/flow-ratio.

Offline evaluation over a sweep directory of the pairwise driver's
``result.json`` files (one per source->target pair; no W&B). Writes to
``--out-dir``:

* ``pairs.csv`` -- one row per pair x stage (``dda``, ``dda_mt``): target
  NRMSE, R2, NBIAS -- combined and per ramp type (for ``dda_mt`` the mean
  across survey days, with ``*_std`` columns; NaN for ``dda``) -- plus the
  pair's backbone and post-DDA adaptation-layer MMD distances and, when the
  data paths are given, the per-ramp-type flow-amplitude ratios.
* ``{metric}_vs_mmd.png`` -- per metric: the DDA-only stage as a scatter and
  the DDA+MT stage as mean +/- std error bars across survey days, both
  against the post-DDA pair MMD, with an MMD-binned mean +/- std band per
  stage.
* ``{metric}_{on|off}_vs_flowratio.png`` -- the same treatment for the
  per-ramp-type metric against the pair's **flow-amplitude ratio**: the mean
  of ``log(source/target)`` over the rank-matched ``RATIO_TOP_K`` largest
  ramp flows of each stretch (positive = the source's peak demand runs
  hotter). An orthogonal difficulty axis to the (model-dependent) MMD.

The flow ratios need each pair's source/target samples, which are rebuilt
with the build parameters recorded in the pair's ``pair_manifest.json`` --
so they always match the corpus that produced that pair's results (e.g. a
weekday-only sweep stays weekday-only). Pass ``--stretches`` and
``--timeseries-dir`` (machine-specific paths) to enable this; without them
the ratio columns are NaN and the flow-ratio plots are skipped.
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

from transportation_models.utils.ramp_flow_estimation.manifest import (  # noqa: E402
    load_stretches,
)
from transportation_models.utils.ramp_flow_estimation.zhang_features import (  # noqa: E402
    build_stretch_samples,
)

STAGES = ("dda", "dda_mt")
METRICS = ("nrmse", "r2", "nbias")
RAMPS = ("on", "off")
_METRIC_COLS = [*METRICS, *(f"{m}_{r}" for m in METRICS for r in RAMPS)]
Y_LIMITS = {"nrmse": (0.0, 1.0), "r2": (-1.0, 1.0), "nbias": (-1.0, 1.0)}
RATIO_TOP_K = 500


def flow_amplitude_ratio(src_flows, tgt_flows, k: int = RATIO_TOP_K) -> float:
    """Mean log(source/target) over the rank-matched ``k`` largest flows of
    each stretch (fewer if either has fewer samples). Ranks where either
    value is <= 0 are excluded; NaN when none remain (e.g. a structurally
    zero ramp)."""
    src = np.sort(np.asarray(src_flows, dtype=float))[::-1]
    tgt = np.sort(np.asarray(tgt_flows, dtype=float))[::-1]
    k = min(k, len(src), len(tgt))
    a, b = src[:k], tgt[:k]
    ok = (a > 0) & (b > 0)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.log(a[ok] / b[ok])))


def _pair_ratios(manifest_path: Path, stretches_df, timeseries_dir,
                 cache: dict) -> dict:
    """``flow_ratio_on``/``flow_ratio_off`` for one pair, rebuilding samples
    with the build parameters recorded in its pair_manifest.json (per-stretch
    builds cached across pairs)."""
    m = json.loads(manifest_path.read_text())

    def samples(sid):
        key = (sid, m["pct_observed_threshold"], m["ramp_pct_floor"],
               m["window_size"], m.get("weekdays_only", True),
               m["record_start"], m["record_end"])
        if key not in cache:
            cache[key] = build_stretch_samples(
                stretches_df, sid, timeseries_dir,
                pct_threshold=m["pct_observed_threshold"],
                ramp_pct_floor=m["ramp_pct_floor"],
                window_size=m["window_size"],
                weekdays_only=m.get("weekdays_only", True),
                record_start=m["record_start"], record_end=m["record_end"],
            )
        return cache[key]

    src, tgt = samples(m["source_stretch"]), samples(m["target_stretch"])
    return {"flow_ratio_on": flow_amplitude_ratio(src.r, tgt.r),
            "flow_ratio_off": flow_amplitude_ratio(src.s, tgt.s)}


def load_pairs(results_dir: Path, *, stretches_df=None,
               timeseries_dir=None) -> pd.DataFrame:
    """One row per pair x stage from every result.json under the dir; flow
    ratios are computed when the data paths are given."""
    with_ratios = stretches_df is not None and timeseries_dir is not None
    cache: dict = {}
    rows = []
    for path in sorted(results_dir.glob("**/result.json")):
        res = json.loads(path.read_text())
        cfg = res.get("config", {})
        tm, mmd = res.get("target_metrics", {}), res.get("source_target_mmd", {})
        if not (all(st in tm for st in STAGES) and "dda" in mmd):
            print(f"skipping {path}: not a complete pairwise-driver result")
            continue
        ratios = {f"flow_ratio_{r}": np.nan for r in RAMPS}
        if with_ratios:
            manifest_path = path.parent / "pair_manifest.json"
            if manifest_path.exists():
                ratios = _pair_ratios(manifest_path, stretches_df,
                                      timeseries_dir, cache)
            else:
                print(f"no pair_manifest.json next to {path}; ratios are NaN")
        base = {"source": cfg["source_stretch"], "target": cfg["target_stretch"],
                "mmd_backbone": mmd["backbone"], "mmd_dda": mmd["dda"], **ratios}
        rows.append({**base, "stage": "dda",
                     **{c: tm["dda"][c] for c in _METRIC_COLS},
                     **{f"{c}_std": np.nan for c in _METRIC_COLS}})
        rows.append({**base, "stage": "dda_mt",
                     **{c: tm["dda_mt"]["mean"][c] for c in _METRIC_COLS},
                     **{f"{c}_std": tm["dda_mt"]["std"][c] for c in _METRIC_COLS}})
    return pd.DataFrame(rows)


def _binned_stats(x, y, bin_width):
    """Bin ``y`` by ``x`` into fixed-width bins aligned to multiples of
    ``bin_width`` (handles negative x); returns the occupied bins' centers,
    means, and stds (population std; 0 for one-point bins)."""
    lo = np.floor(x.min() / bin_width) * bin_width
    edges = np.arange(lo, x.max() + bin_width, bin_width)
    which = np.digitize(x, edges) - 1
    centers, means, stds = [], [], []
    for b in np.unique(which):
        in_bin = y[which == b]
        centers.append(edges[b] + bin_width / 2)
        means.append(in_bin.mean())
        stds.append(in_bin.std())
    return np.array(centers), np.array(means), np.array(stds)


def plot_metric(pairs: pd.DataFrame, metric_col: str, x_col: str,
                x_label: str, path: Path, *, bin_width: float) -> None:
    """DDA scatter + DDA+MT survey-day error bars against ``x_col``, with an
    x-binned mean +/- std band per stage."""
    base_metric = metric_col.split("_")[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for stage, color in zip(STAGES, colors):
        part = pairs[pairs["stage"] == stage].dropna(subset=[x_col]) \
            .sort_values(x_col)
        if part.empty:
            continue
        x, y = part[x_col].to_numpy(), part[metric_col].to_numpy()
        if stage == "dda_mt":     # mean +/- std across survey days
            ax.errorbar(x, y, yerr=part[f"{metric_col}_std"].to_numpy(),
                        fmt="o", color=color, label="DDA+MT (mean +/- std)",
                        alpha=0.5, capsize=3)
        else:
            ax.scatter(x, y, color=color, label="DDA", alpha=0.5)
        centers, means, stds = _binned_stats(x, y, bin_width)
        ax.plot(centers, means, color=color, linewidth=1.5,
                label=f"{'DDA+MT' if stage == 'dda_mt' else 'DDA'} binned mean")
        ax.fill_between(centers, means - stds, means + stds,
                        color=color, alpha=0.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"target {metric_col.upper()}")
    ax.set_ylim(*Y_LIMITS[base_metric])
    ax.set_title(f"{metric_col.upper()} vs {x_label.split(' (')[0]}")
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
    ap.add_argument("--mmd-bin-width", type=float, default=0.05,
                    help="MMD bin width for the binned mean +/- std bands.")
    ap.add_argument("--ratio-bin-width", type=float, default=0.25,
                    help="Flow-ratio (log-scale) bin width for the binned bands.")
    ap.add_argument("--stretches", type=Path, default=None,
                    help="A stretches.csv or directory of them; with "
                         "--timeseries-dir, enables the flow-ratio computation.")
    ap.add_argument("--timeseries-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    if (args.stretches is None) != (args.timeseries_dir is None):
        print("--stretches and --timeseries-dir must be given together")
        return 1
    stretches_df = None
    if args.stretches is not None:
        stretches_df = load_stretches(args.stretches)
    else:
        print("no --stretches/--timeseries-dir: flow ratios skipped "
              "(NaN columns, no flow-ratio plots)")

    pairs = load_pairs(args.results_dir, stretches_df=stretches_df,
                       timeseries_dir=args.timeseries_dir)
    if pairs.empty:
        print(f"no complete result.json files under {args.results_dir}")
        return 1
    n_pairs = len(pairs) // len(STAGES)
    out_dir = args.out_dir if args.out_dir is not None \
        else args.results_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs.to_csv(out_dir / "pairs.csv", index=False)
    for metric in METRICS:
        plot_metric(pairs, metric, "mmd_dda",
                    "source<->target pair MMD (adaptation layer, post-DDA)",
                    out_dir / f"{metric}_vs_mmd.png",
                    bin_width=args.mmd_bin_width)
        if pairs[[f"flow_ratio_{r}" for r in RAMPS]].notna().any().any():
            for ramp in RAMPS:
                plot_metric(
                    pairs, f"{metric}_{ramp}", f"flow_ratio_{ramp}",
                    f"mean log(source/target) of top-{RATIO_TOP_K} "
                    f"{ramp}-ramp flows",
                    out_dir / f"{metric}_{ramp}_vs_flowratio.png",
                    bin_width=args.ratio_bin_width)

    print(f"{n_pairs} pairs aggregated -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
