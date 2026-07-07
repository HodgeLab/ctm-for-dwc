"""Method-agnostic validation for ramp-flow estimators.

Estimators are injected via ``estimator_factory`` -- a callable returning a
fresh model exposing the shared interface ``fit(X, r, s, ctx=None)`` and
``predict_flows(X, ctx=None)``. ``ctx`` is an optional dict of row-aligned
context arrays passed through opaquely (e.g. Kan's bounds context from
:func:`kan.row_context`); it is sliced with the same masks as the data.
The default estimator is :class:`kan.KanEstimator`.

Two protocols:

* **Leave-k-stretches-out CV** (``leave_k_out``/``run_scenarios``): hold out
  all of ``k`` stretches' samples, train on the rest, score ``r_hat, s_hat``
  against the known flows. ``k`` reproduces Kan's Section V.C scenarios
  (1 = leave-one-out); for ``k > 1`` the held-out combinations are enumerated
  when tractable, else ``max_combos`` are sampled.
* **Fixed stretch-level split** (``train_val_test_split``): one stretch's rows
  to val, one to test, the rest to train -- used by the deep-learning training
  and comparison scripts.
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pandas as pd


def _nrmse(y, y_hat) -> float:
    denom = float(np.sum(y ** 2))
    return float(np.sqrt(np.sum((y - y_hat) ** 2) / denom)) if denom > 0 else np.nan


def _r2(y, y_hat) -> float:
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - float(np.sum((y - y_hat) ** 2)) / ss_tot if ss_tot > 0 else np.nan


def flow_metrics(r_true, s_true, r_hat, s_hat) -> dict:
    """NRMSE (Kan Eq. 12), R^2, and BIAS (Kan Eq. 13) for the on-ramp, off-ramp,
    and both combined."""
    r_true, s_true = np.asarray(r_true, float), np.asarray(s_true, float)
    r_hat, s_hat = np.asarray(r_hat, float), np.asarray(s_hat, float)
    both_true = np.concatenate([r_true, s_true])
    both_hat = np.concatenate([r_hat, s_hat])
    return {
        "nrmse": _nrmse(both_true, both_hat),
        "r2": _r2(both_true, both_hat),
        "bias": float(np.mean(both_true - both_hat)),
        "nrmse_on": _nrmse(r_true, r_hat),
        "r2_on": _r2(r_true, r_hat),
        "bias_on": float(np.mean(r_true - r_hat)),
        "nrmse_off": _nrmse(s_true, s_hat),
        "r2_off": _r2(s_true, s_hat),
        "bias_off": float(np.mean(s_true - s_hat)),
    }


def train_val_test_split(data, *, val_stretch=None, test_stretch=None, seed=0) -> dict:
    """Stretch-level train/val/test split: one stretch's rows to val, one to
    test, the rest to train. Held-out stretches are drawn with ``seed`` unless
    given explicitly by ID. Returns row masks plus the resolved IDs:
    ``{"train", "val", "test", "val_stretch", "test_stretch"}``."""
    ids = sorted(int(i) for i in np.unique(data.stretch_id))
    if len(ids) < 3:
        raise ValueError(f"need >= 3 stretches for a train/val/test split, got {len(ids)}")
    for name, sid in (("val_stretch", val_stretch), ("test_stretch", test_stretch)):
        if sid is not None and int(sid) not in ids:
            raise ValueError(f"{name}={sid} not in corpus stretch ids")
    if val_stretch is not None and test_stretch is not None \
            and int(val_stretch) == int(test_stretch):
        raise ValueError("val_stretch and test_stretch must differ")

    rng = np.random.default_rng(seed)
    taken = {int(s) for s in (val_stretch, test_stretch) if s is not None}
    free = [i for i in ids if i not in taken]
    if val_stretch is None:
        val_stretch = int(rng.choice(free))
        free.remove(val_stretch)
    if test_stretch is None:
        test_stretch = int(rng.choice(free))

    val = data.stretch_id == int(val_stretch)
    test = data.stretch_id == int(test_stretch)
    return {"train": ~val & ~test, "val": val, "test": test,
            "val_stretch": int(val_stretch), "test_stretch": int(test_stretch)}


def _holdout_combos(ids, k, max_combos, random_state):
    """All k-subsets of ``ids`` (enumerated), or ``max_combos`` distinct random
    ones when the full count exceeds the cap."""
    n = len(ids)
    total = comb(n, k)
    if max_combos is None or total <= max_combos:
        return [tuple(c) for c in itertools.combinations(ids, k)]
    rng = np.random.default_rng(random_state)
    seen: set[tuple] = set()
    while len(seen) < max_combos:
        seen.add(tuple(sorted(rng.choice(ids, size=k, replace=False).tolist())))
    return sorted(seen)


def slice_ctx(ctx, mask):
    """Slice every row-aligned context array with ``mask`` (None passes through)."""
    return None if ctx is None else {k: np.asarray(v)[mask] for k, v in ctx.items()}


def leave_k_out(data, k=1, *, ctx=None, estimator_factory=None, max_combos=None,
                random_state=0) -> dict:
    """Run leave-``k``-stretches-out CV. Returns per-combination metrics plus
    their mean/std across combinations."""
    if estimator_factory is None:
        from .kan import KanEstimator
        estimator_factory = lambda: KanEstimator(model="rf")

    ids = sorted(int(i) for i in np.unique(data.stretch_id))
    if k > len(ids):
        raise ValueError(f"k={k} exceeds number of stretches ({len(ids)})")
    combos = _holdout_combos(ids, k, max_combos, random_state)

    per_combo = []
    for holdout in combos:
        test = np.isin(data.stretch_id, holdout)
        train = ~test
        est = estimator_factory().fit(data.X[train], data.r_true[train],
                                      data.s_true[train], ctx=slice_ctx(ctx, train))
        r_hat, s_hat = est.predict_flows(data.X[test], ctx=slice_ctx(ctx, test))
        m = flow_metrics(data.r_true[test], data.s_true[test], r_hat, s_hat)
        per_combo.append({"holdout": holdout, **m})

    def _agg(values, fn) -> float:
        finite = [v for v in values if not np.isnan(v)]   # avoid all-NaN-slice warnings
        return float(fn(finite)) if finite else float("nan")

    keys = [key for key in per_combo[0] if key != "holdout"] if per_combo else []
    mean = {key: _agg([c[key] for c in per_combo], np.mean) for key in keys}
    std = {key: _agg([c[key] for c in per_combo], np.std) for key in keys}
    return {"k": k, "n_combos": len(combos), "per_combo": per_combo,
            "mean": mean, "std": std}


def run_scenarios(data, ks=(1, 2, 3, 4), *, ctx=None, estimator_factory=None,
                  max_combos=None, random_state=0) -> pd.DataFrame:
    """Leave-``k``-out for each ``k`` in ``ks`` (Kan Scenarios 1-4); one row per
    ``k`` with ``n_combos`` and each metric's ``_mean``/``_std``."""
    rows = []
    for k in ks:
        res = leave_k_out(data, k, ctx=ctx, estimator_factory=estimator_factory,
                          max_combos=max_combos, random_state=random_state)
        row = {"k": k, "n_combos": res["n_combos"]}
        row.update({f"{key}_mean": v for key, v in res["mean"].items()})
        row.update({f"{key}_std": v for key, v in res["std"].items()})
        rows.append(row)
    return pd.DataFrame(rows)
