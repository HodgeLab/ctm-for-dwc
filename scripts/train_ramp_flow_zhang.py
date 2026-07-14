#!/usr/bin/env python3
"""Train + evaluate the Zhang 2024 estimator on one source->target stretch pair.

Faithful staged pipeline (Zhang et al. 2024, doi 10.1109/TITS.2023.3315693):
the GRU backbone trains on the *source* stretch (early stopping on a seeded
held-out fraction of source days), Deep Domain Adaptation fine-tunes against
the *target* stretch's unlabeled mainline windows for a fixed number of
epochs, and Model Transfer calibrates per-ramp amplitude scalars from one
simulated Traffic-Survey day of the target's real ramp flows -- repeated over
``--n-survey-days`` seeded random days with metrics reported as mean +/- std.
NRMSE/R2 are reported after every stage (backbone -> +DDA -> +DDA+MT) so each
component's contribution on the target is attributable. Per-epoch losses go
to Weights & Biases (``WANDB_MODE=offline`` supported); the checkpoint,
pair manifest (build params + array digests), and result.json land in
``--out-dir`` (use a distinct one per source/target pair).
"""
from __future__ import annotations

import argparse
import hashlib
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
    ZhangSamples,
    build_stretch_samples,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STRETCHES = _REPO_ROOT / "data/pems/stretches"
# a survey day must carry at least this share of the busiest day's windows
_SURVEY_DAY_MIN_FRACTION = 0.5


def _samples_digest(samples: ZhangSamples) -> str:
    h = hashlib.sha256()
    for arr in (samples.X, samples.r, samples.s, samples.t_index):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update("|".join(np.datetime_as_string(samples.date, unit="D")).encode())
    return h.hexdigest()


def _split_source_days(dates, val_frac, seed) -> np.ndarray:
    """Boolean validation mask: a seeded ``val_frac`` of the source's days."""
    days = np.unique(dates)
    if len(days) < 2:
        raise SystemExit("source stretch spans a single day; cannot hold out "
                         "validation days")
    n_val = max(1, round(val_frac * len(days)))
    rng = np.random.default_rng(seed)
    val_days = rng.choice(days, size=n_val, replace=False)
    return np.isin(dates, val_days)


def _survey_days(dates, n_days, seed) -> np.ndarray:
    """Seeded draw of survey days among days with enough windows."""
    days, counts = np.unique(dates, return_counts=True)
    eligible = days[counts >= _SURVEY_DAY_MIN_FRACTION * counts.max()]
    if len(eligible) == 0:
        raise SystemExit("target stretch has no eligible survey days")
    if len(eligible) < n_days:
        print(f"only {len(eligible)} eligible survey days (asked for {n_days}); "
              "using all of them")
    rng = np.random.default_rng(seed)
    return rng.choice(eligible, size=min(n_days, len(eligible)), replace=False)


def _aggregate(per_day: list[dict]) -> dict:
    keys = per_day[0].keys()
    return {
        "mean": {k: float(np.mean([d[k] for d in per_day])) for k in keys},
        "std": {k: float(np.std([d[k] for d in per_day])) for k in keys},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # data
    ap.add_argument("--stretches", type=Path, default=_DEFAULT_STRETCHES,
                    help="A stretches.csv or a directory of them.")
    ap.add_argument("--timeseries-dir", type=Path,
                    default=constants.CALTRANS_VDS_TIMESERIES_DIRECTORY_PATH)
    ap.add_argument("--source-stretch", required=True,
                    help="Source stretch ID (namespaced, e.g. '880_N:3').")
    ap.add_argument("--target-stretch", required=True,
                    help="Target stretch ID; its ramp flows are used only for "
                         "the simulated Traffic Survey and for scoring.")
    ap.add_argument("--pct-observed-threshold", type=float, default=50.0,
                    help="Mainline samples below this pct_observed are treated "
                         "as missing and filled (preprocessing rule 1).")
    ap.add_argument("--ramp-pct-floor", type=float, default=0.0,
                    help="A ramp sample is 'measured' when pct_observed exceeds "
                         "this floor (targets are never filled).")
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--include-weekends", action="store_true",
                    help="Keep weekend windows (the paper, and therefore the "
                         "default, uses workday data only).")
    ap.add_argument("--record-start", default=None)
    ap.add_argument("--record-end", default=None)
    # source validation split (day-level)
    ap.add_argument("--source-val-frac", type=float, default=0.2)
    ap.add_argument("--val-seed", type=int, default=0)
    # model (paper hyperparameters as defaults)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--adapt-dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--no-early-stop", action="store_true",
                    help="Train the backbone all --max-epochs.")
    ap.add_argument("--lambda-mmd", type=float, default=0.01)
    ap.add_argument("--dda-epochs", type=int, default=100)
    ap.add_argument("--model-seed", type=int, default=0)
    # Model Transfer
    ap.add_argument("--n-survey-days", type=int, default=5)
    ap.add_argument("--survey-seed", type=int, default=0)
    # outputs
    ap.add_argument("--wandb-project", default="ramp-flow-estimation")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "scripts/output/zhang")
    args = ap.parse_args(argv)

    stretches = load_stretches(args.stretches)
    build_kw = dict(pct_threshold=args.pct_observed_threshold,
                    window_size=args.window_size,
                    ramp_pct_floor=args.ramp_pct_floor,
                    weekdays_only=not args.include_weekends,
                    record_start=args.record_start, record_end=args.record_end)
    src = build_stretch_samples(stretches, args.source_stretch,
                                args.timeseries_dir, **build_kw)
    tgt = build_stretch_samples(stretches, args.target_stretch,
                                args.timeseries_dir, **build_kw)
    for name, samples in (("source", src), ("target", tgt)):
        if len(samples.r) == 0:
            print(f"{name} stretch produced no samples")
            return 1
    print(f"source {args.source_stretch}: {len(src.r)} windows over "
          f"{len(np.unique(src.date))} days; "
          f"target {args.target_stretch}: {len(tgt.r)} windows over "
          f"{len(np.unique(tgt.date))} days")

    val = _split_source_days(src.date, args.source_val_frac, args.val_seed)
    tr = ~val
    print(f"source split: {tr.sum()} train rows, {val.sum()} val rows "
          f"({args.source_val_frac:.0%} of days, seed {args.val_seed})")

    patience = None if args.no_early_stop else args.patience
    config = {
        "stretches": str(args.stretches), "timeseries_dir": str(args.timeseries_dir),
        "source_stretch": args.source_stretch, "target_stretch": args.target_stretch,
        "pct_observed_threshold": args.pct_observed_threshold,
        "ramp_pct_floor": args.ramp_pct_floor, "window_size": args.window_size,
        "weekdays_only": not args.include_weekends,
        "record_start": args.record_start, "record_end": args.record_end,
        "source_val_frac": args.source_val_frac, "val_seed": args.val_seed,
        "embed_dim": args.embed_dim, "hidden_size": args.hidden_size,
        "adapt_dim": args.adapt_dim, "lr": args.lr, "batch_size": args.batch_size,
        "max_epochs": args.max_epochs, "patience": patience,
        "lambda_mmd": args.lambda_mmd, "dda_epochs": args.dda_epochs,
        "model_seed": args.model_seed, "n_survey_days": args.n_survey_days,
        "survey_seed": args.survey_seed,
        "source_digest": _samples_digest(src), "target_digest": _samples_digest(tgt),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "pair_manifest.json"
    manifest_path.write_text(json.dumps(
        {**config, "n_source_rows": int(len(src.r)),
         "n_target_rows": int(len(tgt.r))}, indent=2) + "\n")
    print(f"wrote pair manifest -> {manifest_path}")

    run = wandb.init(project=args.wandb_project, config=config)
    est = ZhangEstimator(
        embed_dim=args.embed_dim, hidden_size=args.hidden_size,
        adapt_dim=args.adapt_dim, lr=args.lr, batch_size=args.batch_size,
        max_epochs=args.max_epochs, patience=patience,
        lambda_mmd=args.lambda_mmd, dda_epochs=args.dda_epochs,
        seed=args.model_seed,
    )

    # stage 1: backbone on the source
    est.fit_source(
        src.X[tr], src.r[tr], src.s[tr],
        X_val=src.X[val], r_val=src.r[val], s_val=src.s[val],
        log_fn=lambda e, tl, vl: wandb.log(
            {"backbone/epoch": e, "backbone/train_loss": tl, "backbone/val_loss": vl}),
    )
    source_val = flow_metrics(src.r[val], src.s[val],
                              *est.predict_flows(src.X[val]))
    backbone_target = flow_metrics(tgt.r, tgt.s, *est.predict_flows(tgt.X))
    # the paper's per-pair MMD distance (Tables II/IV/VI): Eq. 2 over the
    # Adaptation Layer H_t, i.e. how far apart the stretches look to the
    # source-trained backbone
    mmd_backbone = est.pair_mmd(src.X, tgt.X)
    print(f"backbone  target NRMSE {backbone_target['nrmse']:.4f} "
          f"R2 {backbone_target['r2']:.4f}; "
          f"source<->target adaptation-layer MMD {mmd_backbone:.4f}")

    # stage 2: Deep Domain Adaptation against the target's unlabeled mainline
    est.adapt(
        src.X[tr], src.r[tr], src.s[tr], tgt.X,
        log_fn=lambda e, est_l, mmd_l, tot: wandb.log(
            {"dda/epoch": e, "dda/est_loss": est_l, "dda/mmd": mmd_l,
             "dda/total_loss": tot}),
    )
    dda_target = flow_metrics(tgt.r, tgt.s, *est.predict_flows(tgt.X))
    mmd_dda = est.pair_mmd(src.X, tgt.X)   # should shrink if DDA worked
    print(f"+DDA      target NRMSE {dda_target['nrmse']:.4f} "
          f"R2 {dda_target['r2']:.4f}; "
          f"source<->target adaptation-layer MMD {mmd_dda:.4f}")

    # stage 3: Model Transfer from one simulated Traffic-Survey day, repeated
    survey_days = _survey_days(tgt.date, args.n_survey_days, args.survey_seed)
    per_day = []
    for day in survey_days:
        survey = tgt.date == day
        h_r, h_s = est.calibrate(tgt.X[survey], tgt.r[survey], tgt.s[survey])
        held_out = ~survey                     # paper: T = n_t - n_p
        m = flow_metrics(tgt.r[held_out], tgt.s[held_out],
                         *est.predict_flows(tgt.X[held_out]))
        per_day.append({"survey_day": str(day), "h_r": h_r, "h_s": h_s, **m})
        print(f"+DDA+MT   survey {day}: NRMSE {m['nrmse']:.4f} "
              f"R2 {m['r2']:.4f} (h_r {h_r:.3f}, h_s {h_s:.3f})")
    metric_keys = [k for k in per_day[0] if k != "survey_day"]
    mt_agg = _aggregate([{k: d[k] for k in metric_keys} for d in per_day])
    print(f"+DDA+MT   target NRMSE {mt_agg['mean']['nrmse']:.4f} "
          f"+/- {mt_agg['std']['nrmse']:.4f}, "
          f"R2 {mt_agg['mean']['r2']:.4f} +/- {mt_agg['std']['r2']:.4f} "
          f"over {len(per_day)} survey days")

    results = {
        "config": config,
        "source_target_mmd": {"backbone": mmd_backbone, "dda": mmd_dda},
        "source_val_metrics": source_val,
        "target_metrics": {
            "backbone": backbone_target,
            "dda": dda_target,
            "dda_mt": {"per_survey_day": per_day, **mt_agg},
        },
        "wandb_run_id": run.id,
    }
    result_path = args.out_dir / "result.json"
    result_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    run.summary.update({
        "source_target_mmd_backbone": mmd_backbone,
        "source_target_mmd_dda": mmd_dda,
        **{f"source_val_{k}": v for k, v in source_val.items()},
        **{f"target_backbone_{k}": v for k, v in backbone_target.items()},
        **{f"target_dda_{k}": v for k, v in dda_target.items()},
        **{f"target_dda_mt_{k}_mean": v for k, v in mt_agg["mean"].items()},
        **{f"target_dda_mt_{k}_std": v for k, v in mt_agg["std"].items()},
    })

    # checkpoint carries the post-DDA weights + the *last* survey day's MT
    # scalars; every day's (h_r, h_s) is in result.json
    ckpt_path = args.out_dir / "zhang.pt"
    est.save(ckpt_path)
    artifact = wandb.Artifact("zhang-ramp-flow", type="model")
    artifact.add_file(str(ckpt_path))
    artifact.add_file(str(manifest_path))
    artifact.add_file(str(result_path))
    run.log_artifact(artifact)
    run.finish()
    print(f"results -> {result_path}\ncheckpoint -> {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
