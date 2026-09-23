#!/usr/bin/env python3
"""Train the GRU ramp-flow estimator on the PeMS-native stretch corpus.

GRU-only driver: assembles the method-agnostic corpus, makes the stretch-level
train/val/test split, writes the split manifest, trains with per-epoch W&B
logging (respects ``WANDB_MODE=offline``), and checkpoints the model to
``--out-dir``. The test stretch is never touched here; the Kan head-to-head
runs separately via ``scripts/compare_ramp_flow_estimators.py`` against the
same manifest. Use a distinct ``--out-dir`` per configuration (e.g. per
``--target-transform``) so checkpoints/manifests don't overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

from ctm_for_dwc.utils import constants
from ctm_for_dwc.utils.ramp_flow_estimation.features import build_training_data
from ctm_for_dwc.utils.ramp_flow_estimation.gru import GruEstimator
from ctm_for_dwc.utils.ramp_flow_estimation.manifest import (
    load_corpus_cache,
    load_stretches,
    save_corpus_cache,
    write_manifest,
)
from ctm_for_dwc.utils.ramp_flow_estimation.validation import (
    flow_metrics,
    train_val_test_split,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STRETCHES = _REPO_ROOT / "data/pems/stretches"
_DEFAULT_STATION_META = _REPO_ROOT / "data/pems/calibrated/station_metadata_calibrated.csv"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # corpus
    ap.add_argument("--stretches", type=Path, default=_DEFAULT_STRETCHES,
                    help="A stretches.csv or a directory of them.")
    ap.add_argument("--timeseries-dir", type=Path,
                    default=constants.CALTRANS_VDS_TIMESERIES_DIRECTORY_PATH)
    ap.add_argument("--station-meta", type=Path, default=_DEFAULT_STATION_META,
                    help="Calibrated station metadata (Station ID, capacity, Lanes).")
    ap.add_argument("--pct-observed-floor", type=float, default=0.0)
    ap.add_argument("--window-size", type=int, default=3)
    ap.add_argument("--include-conservation", action="store_true",
                    help="Also use conservation-resolvable windows (default: fully-measured only).")
    ap.add_argument("--include-uncalibrated", action="store_true",
                    help="Keep stretches without calibrated upstream capacity "
                         "(GRU-only corpora; Kan cannot run on them).")
    ap.add_argument("--record-start", type=pd.Timestamp, default=None)
    ap.add_argument("--record-end", type=pd.Timestamp, default=None)
    # split
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-stretch", default=None,
                    help="Validation stretch ID, e.g. '880_N:3' (default: seeded random draw).")
    ap.add_argument("--test-stretch", default=None,
                    help="Test stretch ID, e.g. '880_N:7' (default: seeded random draw).")
    # model
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Inter-layer GRU dropout (needs --num-layers >= 2).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta1", type=float, default=0.9,
                    help="AdamW momentum rate (gradient moving-average decay).")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="AdamW decoupled weight decay (0 = plain Adam).")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--no-early-stop", action="store_true",
                    help="Train all --max-epochs and keep the final weights "
                         "(val metrics still logged each epoch).")
    ap.add_argument("--target-transform", choices=["zscore", "log1p", "maxscale"],
                    default="zscore")
    ap.add_argument("--model-seed", type=int, default=0)
    # corpus cache / sweep prep
    ap.add_argument("--corpus-cache", type=Path, default=None,
                    help="Corpus cache (.npz): load if present, else build the "
                         "corpus and save it here. Lets sweep array tasks skip "
                         "the timeseries IO.")
    ap.add_argument("--prepare-only", action="store_true",
                    help="Build the corpus (and cache, with --corpus-cache), "
                         "print the per-stretch census, and exit without training.")
    # outputs
    ap.add_argument("--wandb-project", default="ramp-flow-estimation")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "scripts/output/gru")
    args = ap.parse_args(argv)

    corpus_params = {
        "stretches": str(args.stretches), "station_meta": str(args.station_meta),
        "timeseries_dir": str(args.timeseries_dir),
        "pct_floor": args.pct_observed_floor, "window_size": args.window_size,
        "require_both_measured": not args.include_conservation,
        "require_capacity": not args.include_uncalibrated,
        "record_start": None if args.record_start is None else str(args.record_start),
        "record_end": None if args.record_end is None else str(args.record_end),
    }
    if args.corpus_cache is not None and args.corpus_cache.exists():
        data, stretch_ctx, cached_params = load_corpus_cache(args.corpus_cache)
        if cached_params != corpus_params:
            print(f"corpus cache {args.corpus_cache} was built with different "
                  f"parameters:\n  cached: {cached_params}\n  args:   {corpus_params}")
            return 1
        print(f"loaded corpus cache -> {args.corpus_cache}")
    else:
        data, stretch_ctx = build_training_data(
            load_stretches(args.stretches), args.timeseries_dir,
            pd.read_csv(args.station_meta),
            pct_floor=args.pct_observed_floor, window_size=args.window_size,
            require_both_measured=not args.include_conservation,
            require_capacity=not args.include_uncalibrated,
            record_start=args.record_start, record_end=args.record_end,
        )
        if len(data.r_true) and args.corpus_cache is not None:
            save_corpus_cache(args.corpus_cache, data, stretch_ctx, corpus_params)
            print(f"wrote corpus cache -> {args.corpus_cache}")
    n = len(data.r_true)
    if n == 0:
        print("No training samples assembled (no type-(c) stretch with measured "
              "ramps over the record).")
        return 1
    print(f"corpus: {n} windows across {len(np.unique(data.stretch_id))} stretches")

    if args.prepare_only:
        print("\nper-stretch census (pick --val-stretch/--test-stretch from this):")
        for sid in np.unique(data.stretch_id):
            m = data.stretch_id == sid
            flags = "".join([" [ZERO-ON]" if data.r_true[m].max() == 0 else "",
                             " [ZERO-OFF]" if data.s_true[m].max() == 0 else ""])
            print(f"  {sid}: {m.sum()} rows, "
                  f"mean r {data.r_true[m].mean():.0f}, "
                  f"mean s {data.s_true[m].mean():.0f} veh/hr{flags}")
        return 0

    split = train_val_test_split(data, val_stretch=args.val_stretch,
                                 test_stretch=args.test_stretch, seed=args.split_seed)
    manifest_path = args.out_dir / "split_manifest.json"
    write_manifest(
        manifest_path, corpus_params=corpus_params, split_seed=args.split_seed,
        val_stretch=split["val_stretch"], test_stretch=split["test_stretch"],
        data=data, stretch_ctx=stretch_ctx,
    )
    print(f"split: val stretch {split['val_stretch']} ({split['val'].sum()} rows), "
          f"test stretch {split['test_stretch']} ({split['test'].sum()} rows), "
          f"train {split['train'].sum()} rows")
    print(f"wrote manifest -> {manifest_path}")

    patience = None if args.no_early_stop else args.patience
    config = {
        **corpus_params, "split_seed": args.split_seed,
        "val_stretch": split["val_stretch"], "test_stretch": split["test_stretch"],
        "hidden_size": args.hidden_size, "num_layers": args.num_layers,
        "dropout": args.dropout, "lr": args.lr, "beta1": args.beta1,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "max_epochs": args.max_epochs, "patience": patience,
        "target_transform": args.target_transform, "model_seed": args.model_seed,
    }
    run = wandb.init(project=args.wandb_project, config=config)
    tr, va = split["train"], split["val"]
    est = GruEstimator(
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, lr=args.lr, beta1=args.beta1,
        weight_decay=args.weight_decay, batch_size=args.batch_size,
        max_epochs=args.max_epochs, patience=patience,
        target_transform=args.target_transform, seed=args.model_seed,
    )
    def _log_epoch(epoch, train_loss, val_loss, val_metrics):
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if val_metrics is not None:
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
        wandb.log(row)

    est.fit(
        data.X[tr], data.r_true[tr], data.s_true[tr],
        X_val=data.X[va], r_val=data.r_true[va], s_val=data.s_true[va],
        log_fn=_log_epoch,
    )

    r_hat, s_hat = est.predict_flows(data.X[va])
    val_metrics = flow_metrics(data.r_true[va], data.s_true[va], r_hat, s_hat)
    run.summary.update({f"val_{k}": v for k, v in val_metrics.items()})
    print({k: round(v, 4) for k, v in val_metrics.items()})
    result_path = args.out_dir / "result.json"
    result_path.write_text(json.dumps(
        {"config": config, "val_metrics": val_metrics, "wandb_run_id": run.id},
        indent=2, default=str) + "\n")

    ckpt_path = args.out_dir / "gru.pt"
    est.save(ckpt_path)
    artifact = wandb.Artifact("gru-ramp-flow", type="model")
    artifact.add_file(str(ckpt_path))
    artifact.add_file(str(manifest_path))
    run.log_artifact(artifact)
    run.finish()
    print(f"checkpoint -> {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
