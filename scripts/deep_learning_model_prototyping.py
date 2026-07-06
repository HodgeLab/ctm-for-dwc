#%%
"""Train the GRU ramp-flow estimator (docs/ramp_flow_estimation_module.md).

GRU-only driver: assembles the stretch corpus (keeping infeasible windows,
which carry valid targets for direct-flow estimation), makes the stretch-level
train/val/test split, writes the split manifest, trains with per-epoch W&B
logging, and checkpoints the model. The test stretch is never touched here;
the Kan head-to-head runs separately via
``scripts/compare_ramp_flow_estimators.py`` against the same manifest.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

from transportation_models.utils.ramp_flow_estimation.features import build_training_data
from transportation_models.utils.ramp_flow_estimation.gru import GruEstimator
from transportation_models.utils.ramp_flow_estimation.manifest import (
    load_stretches,
    write_manifest,
)
from transportation_models.utils.ramp_flow_estimation.validation import (
    flow_metrics,
    train_val_test_split,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# corpus
stretches_path = _REPO_ROOT / "data/pems/stretches"
station_meta_path = _REPO_ROOT / "data/pems/calibrated/station_metadata_calibrated.csv"
timeseries_dir = Path("/Volumes/easystore/work/boulder/Caltrans/PeMS/timeseries_data/")
record_start = "2022-01-01 00:00"
record_end = "2022-01-08 00:00"
pct_observed_floor = 0.0
window_size = 3
require_both_measured = True
keep_infeasible = True          # GRU trains on conservation-violating windows too

# split (stretch-level; None -> seeded random draw)
split_seed = 3
val_stretch = None
test_stretch = None

# model
hidden_size = 64
num_layers = 1
lr = 1e-3
batch_size = 256
max_epochs = 200
patience = 20
model_seed = 0

# outputs
wandb_project = "ramp-flow-estimation"
out_dir = _REPO_ROOT / "scripts/output/gru"

#%% corpus
corpus_params = {
    "stretches": str(stretches_path), "station_meta": str(station_meta_path),
    "timeseries_dir": str(timeseries_dir), "pct_floor": pct_observed_floor,
    "window_size": window_size, "require_both_measured": require_both_measured,
    "keep_infeasible": keep_infeasible,
    "record_start": record_start, "record_end": record_end,
}
data = build_training_data(
    load_stretches(stretches_path), timeseries_dir, pd.read_csv(station_meta_path),
    pct_floor=pct_observed_floor, window_size=window_size,
    require_both_measured=require_both_measured, keep_infeasible=keep_infeasible,
    record_start=record_start, record_end=record_end,
)
n = len(data.alpha)
print(f"corpus: {n} windows across {len(np.unique(data.stretch_id))} stretches")
print(f"  degenerate band (no alpha): {100 * np.mean(~data.feasible):.1f}%")
print(f"  alpha clipped (out of band): {100 * np.mean(data.alpha_clipped):.1f}%")

#%% split + manifest
split = train_val_test_split(data, val_stretch=val_stretch,
                             test_stretch=test_stretch, seed=split_seed)
manifest = write_manifest(
    out_dir / "split_manifest.json", corpus_params=corpus_params,
    split_seed=split_seed, val_stretch=split["val_stretch"],
    test_stretch=split["test_stretch"], data=data,
)
print(f"split: val stretch {split['val_stretch']} ({split['val'].sum()} rows), "
      f"test stretch {split['test_stretch']} ({split['test'].sum()} rows), "
      f"train {split['train'].sum()} rows")
print(f"wrote manifest -> {out_dir / 'split_manifest.json'}")

#%% train
run = wandb.init(project=wandb_project, config={
    **corpus_params, "split_seed": split_seed,
    "val_stretch": split["val_stretch"], "test_stretch": split["test_stretch"],
    "hidden_size": hidden_size, "num_layers": num_layers, "lr": lr,
    "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
    "model_seed": model_seed,
})
tr, va = split["train"], split["val"]
est = GruEstimator(
    hidden_size=hidden_size, num_layers=num_layers, lr=lr,
    batch_size=batch_size, max_epochs=max_epochs, patience=patience,
    seed=model_seed,
)
est.fit(
    data.X[tr], data.r_true[tr], data.s_true[tr],
    X_val=data.X[va], r_val=data.r_true[va], s_val=data.s_true[va],
    log_fn=lambda epoch, train_loss, val_loss: wandb.log(
        {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}),
)

#%% val metrics + checkpoint
r_hat, s_hat = est.predict_flows(data.X[va])
val_metrics = flow_metrics(data.r_true[va], data.s_true[va], r_hat, s_hat)
run.summary.update({f"val_{k}": v for k, v in val_metrics.items()})
print({k: round(v, 4) for k, v in val_metrics.items()})

ckpt_path = out_dir / "gru.pt"
est.save(ckpt_path)
artifact = wandb.Artifact("gru-ramp-flow", type="model")
artifact.add_file(str(ckpt_path))
artifact.add_file(str(out_dir / "split_manifest.json"))
run.log_artifact(artifact)
run.finish()
print(f"checkpoint -> {ckpt_path}")
