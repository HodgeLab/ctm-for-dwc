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

from .features import _CTX_COLUMNS, TrainingData, build_training_data


def load_stretches(path) -> pd.DataFrame:
    """A stretches.csv, or a directory of them, concatenated.

    ``stretch_id`` is namespaced by source file (``"<stem>:<id>"``): the
    builder numbers stretches per output CSV, so raw ids from different files
    collide -- which would silently merge unrelated stretches in splits/CV and
    overwrite one another's bounds context."""
    path = Path(path)
    csvs = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    parts = []
    for c in csvs:
        df = pd.read_csv(c)
        df["stretch_id"] = c.stem + ":" + df["stretch_id"].astype(str)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def corpus_digest(data: TrainingData, stretch_ctx: "pd.DataFrame") -> str:
    """sha256 over the row-aligned corpus arrays and the per-stretch bounds
    context (so e.g. a station-metadata capacity drift is caught too; NaNs and
    infs hash stably)."""
    h = hashlib.sha256()
    for arr in (data.X, data.r_true, data.s_true, data.q_up, data.q_down):
        h.update(np.ascontiguousarray(arr).tobytes())
    # string-valued labels: hash a canonical text form, not dtype-width-dependent bytes
    h.update("|".join(map(str, data.stretch_id)).encode())
    h.update("|".join(map(str, stretch_ctx.index)).encode())
    h.update(np.ascontiguousarray(stretch_ctx.to_numpy(dtype=float)).tobytes())
    return h.hexdigest()


def write_manifest(path, *, corpus_params: dict, split_seed: int,
                   val_stretch, test_stretch, data: TrainingData,
                   stretch_ctx: "pd.DataFrame") -> dict:
    """Write the manifest JSON; returns the manifest dict.

    ``corpus_params`` must hold exactly the :func:`features.build_training_data`
    inputs: ``stretches``, ``station_meta``, ``timeseries_dir``, ``pct_floor``,
    ``window_size``, ``require_both_measured``, ``require_capacity``,
    ``record_start``, ``record_end`` (paths/timestamps as strings).
    """
    manifest = {
        "corpus": dict(corpus_params),
        "split": {"seed": int(split_seed), "val_stretch": val_stretch,
                  "test_stretch": test_stretch},
        "n_rows": int(len(data.r_true)),
        "digest": corpus_digest(data, stretch_ctx),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text())


def save_corpus_cache(path, data: TrainingData, stretch_ctx: "pd.DataFrame",
                      corpus_params: dict) -> None:
    """Cache the assembled corpus (arrays + ctx + build params + digest) so
    that e.g. sweep array tasks skip the multi-year CSV IO. The recorded
    params let loaders verify the cache matches their own build flags."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, X=data.X, r_true=data.r_true, s_true=data.s_true,
        q_up=data.q_up, q_down=data.q_down,
        stretch_id=np.asarray(data.stretch_id).astype(str),
        ctx_index=stretch_ctx.index.to_numpy().astype(str),
        ctx_values=stretch_ctx.to_numpy(dtype=float),
        corpus_params=np.array(json.dumps(corpus_params)),
        digest=np.array(corpus_digest(data, stretch_ctx)),
    )


def load_corpus_cache(path) -> tuple[TrainingData, "pd.DataFrame", dict]:
    """Load ``(data, stretch_ctx, corpus_params)`` from a corpus cache,
    verifying the stored digest against the loaded arrays."""
    z = np.load(Path(path), allow_pickle=False)
    data = TrainingData(X=z["X"], r_true=z["r_true"], s_true=z["s_true"],
                        q_up=z["q_up"], q_down=z["q_down"],
                        stretch_id=z["stretch_id"])
    ctx = pd.DataFrame(z["ctx_values"], columns=_CTX_COLUMNS,
                       index=pd.Index(z["ctx_index"], name="stretch_id"))
    if corpus_digest(data, ctx) != str(z["digest"]):
        raise ValueError(f"corpus cache {path} is corrupt: stored digest does "
                         "not match its arrays")
    return data, ctx, json.loads(str(z["corpus_params"]))


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
