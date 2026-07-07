"""Split manifest: the contract that separately-launched runs share one corpus.

The GRU training script and the Kan comparison script are separate entrypoints
(the latter typically a slurm job), but must train/evaluate on identical data.
The manifest records how the corpus was built (paths + parameters), how it was
split (seed + resolved val/test stretch IDs), and a digest of the resulting
row-aligned arrays. :func:`rebuild_corpus` re-assembles the corpus from the
recorded parameters and refuses to proceed if the digest no longer matches
(the underlying timeseries/stretch CSVs or a parameter drifted).

Machine-specific paths (e.g. the external-drive timeseries dir vs. an HPC
mount) may be overridden at rebuild time; the digest still guarantees the
resulting rows are byte-identical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import TrainingData, build_training_data


def load_stretches(path) -> pd.DataFrame:
    """A stretches.csv, or a directory of them, concatenated."""
    path = Path(path)
    csvs = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


def corpus_digest(data: TrainingData, stretch_ctx: "pd.DataFrame") -> str:
    """sha256 over the row-aligned corpus arrays and the per-stretch bounds
    context (so e.g. a station-metadata capacity drift is caught too; NaNs and
    infs hash stably)."""
    h = hashlib.sha256()
    for arr in (data.X, data.r_true, data.s_true, data.q_up, data.q_down,
                data.stretch_id):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(np.ascontiguousarray(stretch_ctx.index.to_numpy(dtype=np.int64)).tobytes())
    h.update(np.ascontiguousarray(stretch_ctx.to_numpy(dtype=float)).tobytes())
    return h.hexdigest()


def write_manifest(path, *, corpus_params: dict, split_seed: int,
                   val_stretch: int, test_stretch: int, data: TrainingData,
                   stretch_ctx: "pd.DataFrame") -> dict:
    """Write the manifest JSON; returns the manifest dict.

    ``corpus_params`` must hold exactly the :func:`features.build_training_data`
    inputs: ``stretches``, ``station_meta``, ``timeseries_dir``, ``pct_floor``,
    ``window_size``, ``require_both_measured``, ``require_capacity``,
    ``record_start``, ``record_end`` (paths/timestamps as strings).
    """
    manifest = {
        "corpus": dict(corpus_params),
        "split": {"seed": int(split_seed), "val_stretch": int(val_stretch),
                  "test_stretch": int(test_stretch)},
        "n_rows": int(len(data.r_true)),
        "digest": corpus_digest(data, stretch_ctx),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text())


def rebuild_corpus(manifest: dict, *, stretches=None, timeseries_dir=None,
                   station_meta=None) -> tuple[TrainingData, "pd.DataFrame"]:
    """Re-assemble ``(data, stretch_ctx)`` from the manifest's recorded
    parameters and verify the digest. Keyword overrides substitute
    machine-specific paths only."""
    c = manifest["corpus"]
    stretches_df = load_stretches(stretches if stretches is not None else c["stretches"])
    meta = pd.read_csv(station_meta if station_meta is not None else c["station_meta"])
    data, ctx = build_training_data(
        stretches_df,
        timeseries_dir if timeseries_dir is not None else c["timeseries_dir"],
        meta,
        pct_floor=c["pct_floor"], window_size=c["window_size"],
        require_both_measured=c["require_both_measured"],
        require_capacity=c["require_capacity"],
        record_start=c["record_start"], record_end=c["record_end"],
    )
    if len(data.r_true) != manifest["n_rows"] \
            or corpus_digest(data, ctx) != manifest["digest"]:
        raise ValueError(
            "rebuilt corpus does not match the manifest digest: the timeseries, "
            "stretch/metadata CSVs, or build parameters drifted since the manifest "
            "was written")
    return data, ctx
