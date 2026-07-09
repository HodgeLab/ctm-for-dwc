#!/usr/bin/env python3
"""Validate historical-average gap filling for ramp VDS timeseries.

For every measured ramp detector in the stretches CSV(s), masks a
seeded-random fraction of its observed samples, fills the masked copy with
``historical_average_fill``, and scores the fill against the held-out truth
(RMSE / NRMSE / BIAS / NBIAS, per detector, classified on/off). Runs one
trial per ``--seeds`` entry and writes the per-VDS **seed-averaged** table
(plus the raw per-trial rows alongside).

Run from the repo root::

    python scripts/validate_ramp_flow.py \\
        --stretches data/pems/stretches \\
        --timeseries-dir /Volumes/easystore/.../timeseries_data \\
        --mask-rate 0.2 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from transportation_models.utils.ctm.ramp_validation import ramp_fill_accuracy
from transportation_models.utils.ramp_flow_estimation.manifest import load_stretches

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STRETCHES = _REPO_ROOT / "data/pems/stretches"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stretches", type=Path, default=_DEFAULT_STRETCHES,
                    help="A stretches.csv or a directory of them.")
    ap.add_argument("--timeseries-dir", type=Path, required=True,
                    help="Directory of per-VDS timeseries CSVs.")
    ap.add_argument("--mask-rate", type=float, default=0.2,
                    help="Fraction of observed samples to hold out per detector.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="One trial per seed; the report averages across them.")
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "scripts/output/ramp_fill_accuracy.csv",
                    help="Seed-averaged table (per-trial rows land next to it).")
    args = ap.parse_args(argv)

    stretches = load_stretches(args.stretches)
    frames = []
    for seed in args.seeds:
        table = ramp_fill_accuracy(
            stretches, args.timeseries_dir,
            mask_rate=args.mask_rate, seed=seed,
        )
        table.insert(0, "seed", seed)
        frames.append(table)
    trials = pd.concat(frames, ignore_index=True)
    if trials.empty:
        print("No measured ramp detectors found in the stretches CSV(s).")
        return 1

    metric_cols = [c for c in trials.columns if c not in ("seed", "vds_id", "ramp_type")]
    averaged = (
        trials.groupby(["vds_id", "ramp_type"], as_index=False)[metric_cols]
        .mean()
        .sort_values("vds_id", ignore_index=True)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    trials_out = args.out.with_name(args.out.stem + "_trials.csv")
    averaged.to_csv(args.out, index=False)
    trials.to_csv(trials_out, index=False)

    print(f"{len(args.seeds)} trial(s) at mask rate {args.mask_rate}; "
          f"averaged over seeds {args.seeds}")
    print(f"wrote averaged table -> {args.out}")
    print(f"wrote per-trial rows -> {trials_out}\n")
    print(averaged.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
