"""One-off: visualize a trained GRU's ramp-flow predictions on its test stretch.

Point ``trial_dir`` at a training/sweep output directory (checkpoint +
split manifest). The corpus comes from ``corpus_cache`` when present (sweep
layout), else it is rebuilt from the manifest -- set ``timeseries_dir`` when
the manifest records another machine's paths (e.g. Alpine's).
"""
#%% Config
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from transportation_models.utils.ramp_flow_estimation.gru import GruEstimator
from transportation_models.utils.ramp_flow_estimation.manifest import (
    corpus_digest,
    load_corpus_cache,
    load_manifest,
    rebuild_corpus,
)
from transportation_models.utils.ramp_flow_estimation.validation import (
    flow_metrics,
    train_val_test_split,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
trial_dir = _REPO_ROOT / "scripts/output/gru_sweep/trial_0"   # the winner's out-dir
corpus_cache = _REPO_ROOT / "scripts/output/gru_sweep/corpus.npz"
timeseries_dir = None       # set when rebuilding on a machine other than the manifest's

#%% Load model, corpus, split; predict on the test stretch
manifest = load_manifest(trial_dir / "split_manifest.json")
if corpus_cache is not None and corpus_cache.exists():
    data, stretch_ctx, _ = load_corpus_cache(corpus_cache)
    if corpus_digest(data, stretch_ctx) != manifest["digest"]:
        raise ValueError("corpus cache does not match the trial's manifest")
else:
    data, stretch_ctx = rebuild_corpus(manifest, timeseries_dir=timeseries_dir)

sp = manifest["split"]
split = train_val_test_split(data, val_stretch=sp["val_stretch"],
                             test_stretch=sp["test_stretch"], seed=sp["seed"])
te = split["test"]

est = GruEstimator.load(trial_dir / "gru.pt")
r_hat, s_hat = est.predict_flows(data.X[te])
r_true, s_true = data.r_true[te], data.s_true[te]

print(f"test stretch {sp['test_stretch']}: {te.sum()} windows, "
      f"target_transform={est.target_transform}")
print({k: round(v, 4) for k, v in
       flow_metrics(r_true, s_true, r_hat, s_hat).items()})

#%% Measured vs. predicted along the test record (gaps between windows collapsed)
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
for ax, y, y_hat, name in ((axes[0], r_true, r_hat, "on-ramp $r$"),
                           (axes[1], s_true, s_hat, "off-ramp $s$")):
    ax.plot(y, lw=0.8, color="k", label="measured")
    ax.plot(y_hat, lw=0.8, color="tab:red", alpha=0.75, label="GRU")
    ax.set_ylabel(f"{name} [veh/hr]")
    ax.legend(loc="upper right")
axes[1].set_xlabel(f"test window index (stretch {sp['test_stretch']}; "
                   "unobserved gaps collapsed)")
fig.suptitle(f"GRU vs. measured ramp flows -- {trial_dir.name}, "
             f"{est.target_transform} targets")
fig.tight_layout()
plt.show()

#%% Parity scatter
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
for ax, y, y_hat, name in ((axes[0], r_true, r_hat, "on-ramp $r$"),
                           (axes[1], s_true, s_hat, "off-ramp $s$")):
    ax.scatter(y, y_hat, s=4, alpha=0.25)
    lim = 1.05 * max(float(np.max(y)), float(np.max(y_hat)), 1.0)
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"measured {name} [veh/hr]")
    ax.set_ylabel(f"GRU {name} [veh/hr]")
fig.suptitle("Parity: points below the diagonal are under-predictions")
fig.tight_layout()
plt.show()
