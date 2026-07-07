#!/usr/bin/env python3
"""Head-to-head: Kan RF vs. GRU on the split recorded in a split manifest.

Rebuilds the corpus from the manifest written by
``scripts/train_ramp_flow_gru.py`` (digest-verified: fails loudly if the data
or parameters drifted), re-derives the identical stretch-level train/val/test
split, trains the Kan estimator on the train rows (it excludes
degenerate-band rows internally), loads the trained GRU checkpoint (or
retrains with ``--retrain-gru``), and scores both on the held-out test
stretch -- overall and sliced by in-band / alpha-clipped
(conservation-violating) / degenerate-band windows.

Requires a ``require_capacity`` corpus (Kan needs a finite ``c_w`` everywhere).
Plain CLI intended for offline/slurm runs; machine-specific paths can be
overridden without breaking the digest guarantee.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from transportation_models.utils.ramp_flow_estimation.gru import GruEstimator
from transportation_models.utils.ramp_flow_estimation.kan import (
    KanEstimator,
    alpha_targets,
    row_context,
)
from transportation_models.utils.ramp_flow_estimation.manifest import (
    load_manifest,
    rebuild_corpus,
)
from transportation_models.utils.ramp_flow_estimation.validation import (
    flow_metrics,
    slice_ctx,
    train_val_test_split,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, required=True,
                    help="split_manifest.json written by the GRU training script.")
    ap.add_argument("--gru-checkpoint", type=Path, default=None,
                    help="Trained GRU checkpoint (default: gru.pt next to the manifest).")
    ap.add_argument("--retrain-gru", action="store_true",
                    help="Retrain the GRU here instead of loading the checkpoint.")
    # machine-specific path overrides (digest still verifies the data)
    ap.add_argument("--stretches", type=Path, default=None)
    ap.add_argument("--timeseries-dir", type=Path, default=None)
    ap.add_argument("--station-meta", type=Path, default=None)
    # Kan estimator
    ap.add_argument("--model", choices=["rf", "gbm"], default="rf")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--random-state", type=int, default=0)
    # GRU hyperparameters (used only with --retrain-gru)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--target-transform", choices=["zscore", "log1p"], default="zscore")
    ap.add_argument("--gru-seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "scripts/output/ramp_flow_comparison.csv")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    data, stretch_ctx = rebuild_corpus(manifest, stretches=args.stretches,
                                       timeseries_dir=args.timeseries_dir,
                                       station_meta=args.station_meta)
    ctx = row_context(stretch_ctx, data.stretch_id)
    sp = manifest["split"]
    split = train_val_test_split(data, val_stretch=sp["val_stretch"],
                                 test_stretch=sp["test_stretch"], seed=sp["seed"])
    tr, va, te = split["train"], split["val"], split["test"]
    print(f"corpus verified against manifest ({manifest['n_rows']} rows); "
          f"test stretch {sp['test_stretch']} ({te.sum()} rows)")

    if args.model == "rf":
        kan = KanEstimator("rf", n_estimators=args.n_estimators,
                           random_state=args.random_state, n_jobs=args.n_jobs)
    else:
        kan = KanEstimator("gbm", n_estimators=args.n_estimators,
                           random_state=args.random_state)
    kan.fit(data.X[tr], data.r_true[tr], data.s_true[tr], ctx=slice_ctx(ctx, tr))
    r_kan, s_kan = kan.predict_flows(data.X[te], ctx=slice_ctx(ctx, te))

    if args.retrain_gru:
        gru = GruEstimator(
            hidden_size=args.hidden_size, num_layers=args.num_layers, lr=args.lr,
            batch_size=args.batch_size, max_epochs=args.max_epochs,
            patience=args.patience, target_transform=args.target_transform,
            seed=args.gru_seed,
        ).fit(data.X[tr], data.r_true[tr], data.s_true[tr],
              X_val=data.X[va], r_val=data.r_true[va], s_val=data.s_true[va])
    else:
        ckpt = args.gru_checkpoint or args.manifest.parent / "gru.pt"
        gru = GruEstimator.load(ckpt)
    r_gru, s_gru = gru.predict_flows(data.X[te])

    # test-set slices, as masks relative to the test rows
    _, feasible, clipped = alpha_targets(data.X[te], data.r_true[te],
                                         data.s_true[te], slice_ctx(ctx, te))
    slices = {
        "all": np.ones(int(te.sum()), dtype=bool),
        "in_band": feasible & ~clipped,
        "clipped": clipped,
        "degenerate": ~feasible,
    }
    rows = []
    for est_name, (r_hat, s_hat) in (("kan_" + args.model, (r_kan, s_kan)),
                                     ("gru", (r_gru, s_gru))):
        for slice_name, rel in slices.items():
            if not rel.any():
                continue
            m = flow_metrics(data.r_true[te][rel], data.s_true[te][rel],
                             r_hat[rel], s_hat[rel])
            rows.append({"estimator": est_name, "slice": slice_name,
                         "n": int(rel.sum()), **m})
    report = pd.DataFrame(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print(f"wrote comparison -> {args.out}\n")
    show = ["estimator", "slice", "n", "nrmse", "r2", "bias",
            "nrmse_on", "nrmse_off"]
    print(report[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
