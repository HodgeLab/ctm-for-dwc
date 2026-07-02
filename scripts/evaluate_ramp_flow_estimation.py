#!/usr/bin/env python3
"""Evaluate the Kan ramp-flow estimator on the PeMS-native stretch corpus.

Assembles Kan training samples from every type-(c) stretch in ``--stretches``
(feature matrix Z + alpha targets, via the full-record timeseries and calibrated
station capacities), then runs leave-k-stretches-out cross-validation for each
``--ks`` value (Kan Section V.C Scenarios 1-4) and writes an NRMSE/R^2/BIAS
report.

This is the heavy entrypoint intended for an offline/HPC run: it trains one
model per held-out combination over the full multi-year record. See
docs/ramp_flow_estimation_module.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from transportation_models.utils import constants
from transportation_models.utils.ramp_flow_estimation.features import build_training_data
from transportation_models.utils.ramp_flow_estimation.kan import KanEstimator
from transportation_models.utils.ramp_flow_estimation.validation import run_scenarios

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STRETCHES = _REPO_ROOT / "data/pems/stretches"
_DEFAULT_STATION_META = _REPO_ROOT / "data/pems/calibrated/station_metadata_calibrated.csv"


def _load_stretches(path: Path) -> pd.DataFrame:
    csvs = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stretches", type=Path, default=_DEFAULT_STRETCHES,
                    help="A stretches.csv or a directory of them.")
    ap.add_argument("--timeseries-dir", type=Path,
                    default=constants.CALTRANS_VDS_TIMESERIES_DIRECTORY_PATH)
    ap.add_argument("--station-meta", type=Path, default=_DEFAULT_STATION_META,
                    help="Calibrated station metadata (Station ID, capacity, Lanes).")
    ap.add_argument("--pct-observed-floor", type=float, default=0.0)
    ap.add_argument("--window-size", type=int, default=3)
    ap.add_argument("--model", choices=["rf", "gbm"], default="rf")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="Held-out stretch counts (Kan Scenarios 1-4).")
    ap.add_argument("--max-combos", type=int, default=None,
                    help="Cap held-out combinations per k (sample beyond this).")
    ap.add_argument("--include-conservation", action="store_true",
                    help="Also use conservation-resolvable windows (default: fully-measured only).")
    ap.add_argument("--record-start", type=pd.Timestamp, default=None)
    ap.add_argument("--record-end", type=pd.Timestamp, default=None)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "scripts/output/ramp_flow_evaluation.csv")
    args = ap.parse_args(argv)

    stretches = _load_stretches(args.stretches)
    station_meta = pd.read_csv(args.station_meta)

    data = build_training_data(
        stretches, args.timeseries_dir, station_meta,
        pct_floor=args.pct_observed_floor, window_size=args.window_size,
        require_both_measured=not args.include_conservation,
        record_start=args.record_start, record_end=args.record_end,
    )

    n_samples = len(data.alpha)
    if n_samples == 0:
        print("No training samples assembled (no type-(c) stretch with measured "
              "ramps + calibrated upstream capacity over the record).")
        return 1

    stretch_ids = np.unique(data.stretch_id)
    n_stretch = len(stretch_ids)
    qs = np.quantile(data.alpha, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"corpus: {n_samples} samples across {n_stretch} type-(c) stretches")
    print(f"  alpha quantiles (min/25/50/75/max): "
          f"{', '.join(f'{q:.3f}' for q in qs)}")
    print(f"  samples per stretch: min {np.bincount(np.searchsorted(stretch_ids, data.stretch_id)).min()}, "
          f"max {np.bincount(np.searchsorted(stretch_ids, data.stretch_id)).max()}")

    ks = [k for k in args.ks if k < n_stretch]        # need >=1 stretch to train on
    dropped = [k for k in args.ks if k >= n_stretch]
    if dropped:
        print(f"  skipping k={dropped}: not enough stretches ({n_stretch}) to hold out that many")
    if not ks:
        print("No usable k (need at least 2 stretches). Nothing to evaluate.")
        return 1

    factory = lambda: KanEstimator(
        model=args.model, n_estimators=args.n_estimators, random_state=args.random_state)
    report = run_scenarios(
        data, ks=tuple(ks), estimator_factory=factory,
        max_combos=args.max_combos, random_state=args.random_state)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print(f"\nwrote scenario report -> {args.out}")
    show = ["k", "n_combos", "nrmse_mean", "nrmse_std", "r2_mean",
            "bias_mean", "nrmse_on_mean", "nrmse_off_mean"]
    show = [c for c in show if c in report.columns]
    print(report[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
