#!/usr/bin/env python3
"""Pooled-vs-DDA 2x2 experiment on the Zhang backbone, one target per run.

Answers "does pooled training subsume Deep Domain Adaptation?" by training
the full 2x2 -- {single-source, pooled} x {no-DDA, +DDA} -- on the Zhang
backbone/corpus and scoring every arm on the same held-out target stretch:

* ``single``      -- backbone trained on --source-stretch only
* ``single_dda``  -- the same backbone after DDA against the target
* ``pooled``      -- backbone trained on ALL --stretch-ids minus the target
* ``pooled_dda``  -- the pooled backbone after DDA against the target

Samples are built once per stretch so every arm sees byte-identical data;
the no-DDA/+DDA arms of each pool share one backbone (evaluated before and
after ``adapt``), and both backbones start from the same --model-seed. Each
arm reports full flow metrics on all target windows plus the
adaptation-layer pair MMD. Model Transfer is deliberately excluded -- the
experiment's question lives in the mainline-features-only regime.

One W&B run per invocation (per-epoch curves under arm prefixes; final
target metrics as a series over ``arm_idx``); ``result.json`` holds the 2x2
table consumed by ``scripts/aggregate_zhang_pooling.py``. Sweep the
(target, source) combinations via ``slurm_scripts/sweep_zhang_pooling.sh``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import wandb

from transportation_models.utils import constants
from transportation_models.utils.ramp_flow_estimation.manifest import load_stretches
from transportation_models.utils.ramp_flow_estimation.validation import flow_metrics
from transportation_models.utils.ramp_flow_estimation.zhang import ZhangEstimator
from transportation_models.utils.ramp_flow_estimation.zhang_features import (
    WINDOW_SIZE,
    build_stretch_samples,
    concat_samples,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STRETCHES = _REPO_ROOT / "data/pems/stretches"

# shared with the pairwise driver (same directory, on sys.path when run as a script)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ramp_flow_zhang import _samples_digest, _split_source_days  # noqa: E402

ARMS = ("single", "single_dda", "pooled", "pooled_dda")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # data
    ap.add_argument("--stretches", type=Path, default=_DEFAULT_STRETCHES,
                    help="A stretches.csv or a directory of them.")
    ap.add_argument("--timeseries-dir", type=Path,
                    default=constants.CALTRANS_VDS_TIMESERIES_DIRECTORY_PATH)
    ap.add_argument("--stretch-ids", nargs="+", required=True,
                    help="Viable stretch IDs; the pooled arms train on all of "
                         "them minus the target.")
    ap.add_argument("--target-stretch", required=True,
                    help="Held-out target every arm is scored on.")
    ap.add_argument("--source-stretch", required=True,
                    help="The single arms' training stretch.")
    ap.add_argument("--pct-observed-threshold", type=float, default=50.0)
    ap.add_argument("--ramp-pct-floor", type=float, default=0.0)
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--weekdays-only", action="store_true",
                    help="Opt back into the paper's workday-only corpus "
                         "(default: all days).")
    ap.add_argument("--record-start", default=None)
    ap.add_argument("--record-end", default=None)
    # validation split (day-level, over each arm's own training stretches)
    ap.add_argument("--source-val-frac", type=float, default=0.2)
    ap.add_argument("--val-seed", type=int, default=0)
    # model (paper hyperparameters as defaults; one seed shared by both backbones)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--adapt-dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--lambda-mmd", type=float, default=0.01)
    ap.add_argument("--dda-epochs", type=int, default=100)
    ap.add_argument("--model-seed", type=int, default=0)
    # outputs
    ap.add_argument("--wandb-project", default="ramp-flow-estimation")
    ap.add_argument("--out-dir", type=Path,
                    default=_REPO_ROOT / "scripts/output/zhang_pooling")
    args = ap.parse_args(argv)

    ids = list(dict.fromkeys(args.stretch_ids))    # keep order, drop dupes
    for name, sid in (("target", args.target_stretch),
                      ("source", args.source_stretch)):
        if sid not in ids:
            print(f"--{name}-stretch {sid!r} is not in --stretch-ids")
            return 1
    if args.target_stretch == args.source_stretch:
        print("--target-stretch and --source-stretch must differ")
        return 1

    stretches = load_stretches(args.stretches)
    build_kw = dict(pct_threshold=args.pct_observed_threshold,
                    window_size=args.window_size,
                    ramp_pct_floor=args.ramp_pct_floor,
                    weekdays_only=args.weekdays_only,
                    record_start=args.record_start, record_end=args.record_end)
    samples, digests = {}, {}
    for sid in ids:
        samples[sid] = build_stretch_samples(stretches, sid,
                                             args.timeseries_dir, **build_kw)
        digests[sid] = _samples_digest(samples[sid])
        if len(samples[sid].r) == 0:
            print(f"stretch {sid} produced no samples; prune it from --stretch-ids")
            return 1
        print(f"{sid}: {len(samples[sid].r)} windows over "
              f"{len(np.unique(samples[sid].date))} days")

    tgt = samples[args.target_stretch]
    pool_ids = [sid for sid in ids if sid != args.target_stretch]
    arm_train = {
        "single": samples[args.source_stretch],
        "pooled": concat_samples([samples[sid] for sid in pool_ids]),
    }
    print(f"target {args.target_stretch}; single source {args.source_stretch} "
          f"({len(arm_train['single'].r)} rows); pooled over {len(pool_ids)} "
          f"stretches ({len(arm_train['pooled'].r)} rows)")

    patience = None if args.no_early_stop else args.patience
    config = {
        "stretches": str(args.stretches), "timeseries_dir": str(args.timeseries_dir),
        "stretch_ids": ids, "target_stretch": args.target_stretch,
        "source_stretch": args.source_stretch, "pool_stretches": pool_ids,
        "pct_observed_threshold": args.pct_observed_threshold,
        "ramp_pct_floor": args.ramp_pct_floor, "window_size": args.window_size,
        "weekdays_only": args.weekdays_only,
        "record_start": args.record_start, "record_end": args.record_end,
        "source_val_frac": args.source_val_frac, "val_seed": args.val_seed,
        "embed_dim": args.embed_dim, "hidden_size": args.hidden_size,
        "adapt_dim": args.adapt_dim, "lr": args.lr, "batch_size": args.batch_size,
        "max_epochs": args.max_epochs, "patience": patience,
        "lambda_mmd": args.lambda_mmd, "dda_epochs": args.dda_epochs,
        "model_seed": args.model_seed, "digests": digests,
    }
    run = wandb.init(project=args.wandb_project, config=config)

    def _stage_log(stage, names):
        def log(epoch, *log_args):
            *losses, val_metrics = log_args
            row = {f"{stage}/epoch": epoch,
                   **{f"{stage}/{n}": v for n, v in zip(names, losses)}}
            if val_metrics is not None:
                row.update({f"{stage}/val_{k}": v for k, v in val_metrics.items()})
            wandb.log(row)
        return log

    def _log_arm(idx, name, metrics, mmd):
        wandb.log({"arm_idx": idx, "arm": name, "target/mmd": mmd,
                   **{f"target/{k}": v for k, v in metrics.items()}})

    arm_results: dict[str, dict] = {}

    def _record(idx, name, est, n_train):
        metrics = flow_metrics(tgt.r, tgt.s, *est.predict_flows(tgt.X))
        mmd = est.pair_mmd(arm_train[name.removesuffix("_dda")].X, tgt.X)
        _log_arm(idx, name, metrics, mmd)
        arm_results[name] = {"target_metrics": metrics, "pair_mmd": mmd,
                             "n_train_rows": int(n_train)}
        print(f"{name:<11} target NRMSE {metrics['nrmse']:.4f} "
              f"R2 {metrics['r2']:.4f} NBIAS {metrics['nbias']:.4f}; "
              f"pair MMD {mmd:.4f}")

    for pool_i, base in enumerate(("single", "pooled")):
        train = arm_train[base]
        val = _split_source_days(train.date, args.source_val_frac, args.val_seed)
        tr = ~val
        est = ZhangEstimator(
            embed_dim=args.embed_dim, hidden_size=args.hidden_size,
            adapt_dim=args.adapt_dim, lr=args.lr, batch_size=args.batch_size,
            max_epochs=args.max_epochs, patience=patience,
            lambda_mmd=args.lambda_mmd, dda_epochs=args.dda_epochs,
            seed=args.model_seed,
        )
        est.fit_source(
            train.X[tr], train.r[tr], train.s[tr],
            X_val=train.X[val], r_val=train.r[val], s_val=train.s[val],
            log_fn=_stage_log(base, ["train_loss", "val_loss"]),
        )
        _record(2 * pool_i, base, est, tr.sum())
        est.adapt(
            train.X[tr], train.r[tr], train.s[tr], tgt.X,
            X_val=train.X[val], r_val=train.r[val], s_val=train.s[val],
            log_fn=_stage_log(f"{base}_dda", ["est_loss", "mmd", "total_loss"]),
        )
        _record(2 * pool_i + 1, f"{base}_dda", est, tr.sum())

    results = {"config": config, "arms": arm_results, "wandb_run_id": run.id}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "result.json"
    result_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    run.summary.update({
        f"{name}_{k}": v
        for name, res in arm_results.items()
        for k, v in ({**res["target_metrics"], "pair_mmd": res["pair_mmd"]}).items()
    })
    run.finish()
    print(f"results -> {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
