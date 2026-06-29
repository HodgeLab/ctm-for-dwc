"""Step 9 (ramp): validate ramp-flow gap-fill accuracy by synthetic hold-out.

Scores how well the Step-7 gap fillers (persistence / historical-average
/ stochastic-historical) reconstruct ramp VDS flow that is *actually
known*, by masking fixed-length blocks of it and comparing the fill to
the held-out truth. Errors are reported stratified by gap length, the
evidence needed to pick a filler per outage duration.

Two surfaces, both driven off a Step-1/2 ``cells.csv`` and the Step-3
per-VDS timeseries directory:

* **on-ramp demand** -- scores the on-ramp flow series directly (the
  ``d_i`` driver).
* **off-ramp split** -- masks + fills the off-ramp flow ``s_i``, then
  reconstructs ``beta_i = s_i / (f_i + s_i)`` against the unmasked
  downstream mainline ``f_i`` and scores both.

Writes per-VDS and corridor-pooled CSVs plus an RMSE-vs-gap-length plot
to ``<cells parent>/ramp_validation/`` (override with ``--out-dir``),
and prints the pooled tables to stdout.

Run from the repo root::

    python scripts/validate_ramp_flow.py \\
        --cells scripts/output/ctm_corridor/I_210_W/cells.csv \\
        --timeseries-dir data/pems/csv_files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from transportation_models.utils.ctm import (
    DEFAULT_GAP_LENGTHS_MIN,
    offramp_split_accuracy,
    onramp_fill_accuracy,
)
from transportation_models.utils.ctm.ramp_validation import RampFillAccuracy


def _parse_gap_lengths(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in text.split(",") if p.strip())


def _plot_rmse_vs_gap(
    pooled: pd.DataFrame, *, value: str, ylabel: str, title: str, out_path: Path,
) -> None:
    """One line per strategy: pooled RMSE vs gap length."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy, g in pooled.groupby("strategy"):
        g = g.sort_values("gap_length_min")
        ax.plot(g["gap_length_min"], g[f"{value}_rmse"], marker="o", label=strategy)
    ax.set_xlabel("synthetic gap length [min]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="filler")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _print_pooled(label: str, pooled: pd.DataFrame, value_cols: list[str]) -> None:
    print(f"\n=== {label} (corridor-pooled) ===")
    if pooled.empty:
        print("  (no scorable ramp VDS found)")
        return
    cols = ["gap_length_min", "strategy"] + value_cols
    view = pooled[cols].sort_values(["gap_length_min", "strategy"])
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(view.to_string(index=False))


def _write(acc: RampFillAccuracy, out_dir: Path, stem: str) -> None:
    acc.per_vds.to_csv(out_dir / f"{stem}_per_vds.csv", index=False)
    acc.pooled.to_csv(out_dir / f"{stem}_pooled.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells", required=True, type=Path,
        help="Step-1/2 cells.csv (on_ramp/off_ramp + *_ramp_vds_id + vds_id).",
    )
    parser.add_argument(
        "--timeseries-dir", required=True, type=Path,
        help="Directory of per-VDS 5-min CSVs (Step-3 output).",
    )
    parser.add_argument(
        "--gap-lengths", default=",".join(str(g) for g in DEFAULT_GAP_LENGTHS_MIN),
        type=_parse_gap_lengths,
        help=("Comma-separated synthetic gap lengths in minutes "
              f"(default: {','.join(str(g) for g in DEFAULT_GAP_LENGTHS_MIN)})."),
    )
    parser.add_argument(
        "--gap-spacing-min", default=120.0, type=float,
        help=("Minutes between consecutive synthetic gaps on the regular "
              "schedule (default 120)."),
    )
    parser.add_argument(
        "--kind", default="both", choices=["onramp", "offramp", "both"],
        help="Which ramp surface to validate (default both).",
    )
    parser.add_argument(
        "--out-dir", default=None, type=Path,
        help="Output directory (default: <cells parent>/ramp_validation).",
    )
    args = parser.parse_args()

    cells = pd.read_csv(args.cells)
    out_dir = (
        args.out_dir if args.out_dir is not None
        else args.cells.resolve().parent / "ramp_validation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"validating ramp gap-fill on {args.cells} "
        f"(gaps={args.gap_lengths} min, spacing={args.gap_spacing_min} min)",
        file=sys.stderr,
    )

    if args.kind in ("onramp", "both"):
        on = onramp_fill_accuracy(
            cells, args.timeseries_dir,
            gap_lengths_min=args.gap_lengths,
            gap_spacing_min=args.gap_spacing_min,
        )
        _write(on, out_dir, "onramp_fill_accuracy")
        _print_pooled(
            "On-ramp flow fill accuracy", on.pooled,
            ["flow_n", "flow_rmse", "flow_mape", "flow_bias"],
        )
        if not on.pooled.empty:
            _plot_rmse_vs_gap(
                on.pooled, value="flow",
                ylabel="pooled flow RMSE [veh/5-min]",
                title="On-ramp flow fill RMSE vs gap length",
                out_path=out_dir / "onramp_flow_rmse.png",
            )

    if args.kind in ("offramp", "both"):
        off = offramp_split_accuracy(
            cells, args.timeseries_dir,
            gap_lengths_min=args.gap_lengths,
            gap_spacing_min=args.gap_spacing_min,
        )
        _write(off, out_dir, "offramp_split_accuracy")
        _print_pooled(
            "Off-ramp flow fill accuracy", off.pooled,
            ["flow_n", "flow_rmse", "flow_mape", "flow_bias"],
        )
        _print_pooled(
            "Off-ramp split (beta) accuracy", off.pooled,
            ["beta_n", "beta_rmse", "beta_mape", "beta_bias"],
        )
        if not off.pooled.empty:
            _plot_rmse_vs_gap(
                off.pooled, value="beta",
                ylabel="pooled split-ratio RMSE [-]",
                title="Off-ramp split (beta) fill RMSE vs gap length",
                out_path=out_dir / "offramp_beta_rmse.png",
            )

    print(f"\nartifacts written to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
