"""
Sandbox for inspecting the ramp flow estimation module inputs and the GRU estimator
""" 
#%% Imports
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import torch

from transportation_models.utils import constants
from transportation_models.utils.ramp_flow_estimation.features import build_training_data
from transportation_models.utils.ramp_flow_estimation.gru import GruEstimator, Normalizer
from transportation_models.utils.ramp_flow_estimation.manifest import (
    load_stretches,
    write_manifest,
)
from transportation_models.utils.ramp_flow_estimation.validation import (
    flow_metrics,
    train_val_test_split,
)
from torch import nn

#%% Constants
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STRETCHES = _REPO_ROOT / "data/pems/stretches"
_DEFAULT_STATION_META = _REPO_ROOT / "data/pems/calibrated/station_metadata_calibrated.csv"

class ArgParseSpoofer:
    def __init__(self,
        stretches: Path = _DEFAULT_STRETCHES,
        timeseries_dir: Path = constants.CALTRANS_VDS_TIMESERIES_DIRECTORY_PATH,
        station_meta: Path = _DEFAULT_STATION_META,
        pct_observed_floor: float = 0.0,
        window_size: int = 3,
        include_conservation: bool = False,
        include_uncalibrated: bool = False,
        record_start: pd.Timestamp = None,
        record_end: pd.Timestamp = None,
        split_seed: int = 0,
        val_stretch: int = None,
        test_stretch: int = None,
        hidden_size: int = 64,
        num_layers: int = 1,
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 20,
        target_transform: str = "zscore",
        model_seed: int = 0,
        wandb_project: str = "ramp-flow-estimation",
        out_dir: Path = _REPO_ROOT / "scripts/output/gru"
    ):
        self.stretches = stretches
        self.timeseries_dir = timeseries_dir
        self.station_meta = station_meta
        self.pct_observed_floor = pct_observed_floor
        self.window_size = window_size
        self.include_conservation = include_conservation
        self.include_uncalibrated = include_uncalibrated
        self.record_start = record_start
        self.record_end = record_end
        self.split_seed = split_seed
        self.val_stretch = val_stretch
        self.test_stretch = test_stretch
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.target_transform = target_transform
        self.model_seed = model_seed
        self.wandb_project = wandb_project
        self.out_dir = out_dir

args = ArgParseSpoofer(
    record_start="2022-01-01 00:00",
    record_end="2022-01-08 00:00"
)

#%% Spoofing main: data building
corpus_params = {
    "stretches": str(args.stretches), "station_meta": str(args.station_meta),
    "timeseries_dir": str(args.timeseries_dir),
    "pct_floor": args.pct_observed_floor, "window_size": args.window_size,
    "require_both_measured": not args.include_conservation,
    "require_capacity": not args.include_uncalibrated,
    "record_start": None if args.record_start is None else str(args.record_start),
    "record_end": None if args.record_end is None else str(args.record_end),
}
data, stretch_ctx = build_training_data(
    load_stretches(args.stretches), args.timeseries_dir,
    pd.read_csv(args.station_meta),
    pct_floor=args.pct_observed_floor, window_size=args.window_size,
    require_both_measured=not args.include_conservation,
    require_capacity=not args.include_uncalibrated,
    record_start=args.record_start, record_end=args.record_end,
)
n = len(data.r_true)
if n == 0:
    print("No training samples assembled (no type-(c) stretch with measured "
            "ramps over the record).")
    raise RuntimeError()
print(f"corpus: {n} windows across {len(np.unique(data.stretch_id))} stretches")

#%% Spoofing main: data splitting
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

#%% Spoofing GruEstimator fit
tr, va = split["train"], split["val"]
est = GruEstimator(
    hidden_size=args.hidden_size, num_layers=args.num_layers, lr=args.lr,
    batch_size=args.batch_size, max_epochs=args.max_epochs,
    patience=args.patience, target_transform="log1p", 
    seed=args.model_seed,
)

X = data.X[tr]
r = data.r_true[tr]
s = data.s_true[tr]
X_val=data.X[va]
r_val=data.r_true[va]
s_val=data.s_true[va]

torch.manual_seed(est.seed)
x_raw, y_raw = est._sequences(X), est._flows(r,s)
est._norm = Normalizer(est.target_transform).fit(x_raw, y_raw)
xt = est._norm.features(x_raw).to(est.device)
yt = est._norm.targets(y_raw).to(est.device)

#%% Distributions before/after normalization
import matplotlib.pyplot as plt

channel_names = ["up_flow", "up_speed", "up_occ", "down_flow", "down_speed", "down_occ"]
raw_feats = x_raw.reshape(-1, 6).cpu().numpy()      # pool all lags per channel
norm_feats = xt.reshape(-1, 6).cpu().numpy()
raw_targets = y_raw.cpu().numpy()
norm_targets = yt.cpu().numpy()

names = channel_names + ["r", "s"]
raw_cols = [raw_feats[:, i] for i in range(6)] + [raw_targets[:, 0], raw_targets[:, 1]]
norm_cols = [norm_feats[:, i] for i in range(6)] + [norm_targets[:, 0], norm_targets[:, 1]]

fig, axes = plt.subplots(len(names), 2, figsize=(10, 2 * len(names)))
for (ax_raw, ax_norm), name, raw_v, norm_v in zip(axes, names, raw_cols, norm_cols):
    ax_raw.hist(raw_v, bins=80)
    ax_raw.set_ylabel(name)
    ax_norm.hist(norm_v, bins=80, color="tab:orange")
axes[0][0].set_title("raw")
axes[0][1].set_title(f"normalized (targets: {est.target_transform})")
fig.tight_layout()
plt.show()

# %%
