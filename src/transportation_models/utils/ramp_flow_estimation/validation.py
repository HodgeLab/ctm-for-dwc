"""Method-agnostic validation for ramp-flow estimators.

* **Scoring** (``flow_metrics``): NRMSE, RMSE, R^2, BIAS and NBIAS for a
  predicted ``(r_hat, s_hat)`` pair against the known flows.
* **Fixed stretch-level split** (``train_val_test_split``): one stretch's rows
  to val, one to test, the rest to train -- used by the GRU training script.
"""
from __future__ import annotations

import numpy as np


def _nrmse(y, y_hat) -> float:
    denom = float(np.sum(y ** 2))
    return float(np.sqrt(np.sum((y - y_hat) ** 2) / denom)) if denom > 0 else np.nan


def _rmse(y, y_hat) -> float:
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def _nbias(y, y_hat) -> float:
    denom = float(np.sum(y))
    return float(np.sum(y - y_hat) / denom) if denom > 0 else np.nan


def _r2(y, y_hat) -> float:
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - float(np.sum((y - y_hat) ** 2)) / ss_tot if ss_tot > 0 else np.nan


def flow_metrics(r_true, s_true, r_hat, s_hat) -> dict:
    """NRMSE (Kan Eq. 12), RMSE, R^2, BIAS (Kan Eq. 13), and NBIAS
    (``sum(y - y_hat) / sum(y)``) for the on-ramp, off-ramp, and both combined."""
    r_true, s_true = np.asarray(r_true, float), np.asarray(s_true, float)
    r_hat, s_hat = np.asarray(r_hat, float), np.asarray(s_hat, float)
    both_true = np.concatenate([r_true, s_true])
    both_hat = np.concatenate([r_hat, s_hat])
    out = {}
    for suffix, y, y_hat in (("", both_true, both_hat),
                             ("_on", r_true, r_hat), ("_off", s_true, s_hat)):
        out[f"nrmse{suffix}"] = _nrmse(y, y_hat)
        out[f"rmse{suffix}"] = _rmse(y, y_hat)
        out[f"r2{suffix}"] = _r2(y, y_hat)
        out[f"bias{suffix}"] = float(np.mean(y - y_hat))
        out[f"nbias{suffix}"] = _nbias(y, y_hat)
    return out


def train_val_test_split(data, *, val_stretch=None, test_stretch=None, seed=0) -> dict:
    """Stretch-level train/val/test split: one stretch's rows to val, one to
    test, the rest to train. Held-out stretches are drawn with ``seed`` unless
    given explicitly by ID (an opaque label, e.g. the namespaced ``"<csv>:<n>"``
    from ``manifest.load_stretches``). Returns row masks plus the resolved IDs:
    ``{"train", "val", "test", "val_stretch", "test_stretch"}``."""
    ids = sorted(np.unique(data.stretch_id).tolist())
    if len(ids) < 3:
        raise ValueError(f"need >= 3 stretches for a train/val/test split, got {len(ids)}")
    for name, sid in (("val_stretch", val_stretch), ("test_stretch", test_stretch)):
        if sid is not None and sid not in ids:
            raise ValueError(f"{name}={sid!r} not in corpus stretch ids {ids}")
    if val_stretch is not None and val_stretch == test_stretch:
        raise ValueError("val_stretch and test_stretch must differ")

    rng = np.random.default_rng(seed)
    free = [i for i in ids if i not in (val_stretch, test_stretch)]
    if val_stretch is None:
        val_stretch = free.pop(int(rng.integers(len(free))))
    if test_stretch is None:
        test_stretch = free[int(rng.integers(len(free)))]

    val = data.stretch_id == val_stretch
    test = data.stretch_id == test_stretch
    return {"train": ~val & ~test, "val": val, "test": test,
            "val_stretch": val_stretch, "test_stretch": test_stretch}

