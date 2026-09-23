"""Tests for utils/ramp_flow_estimation/validation.py (CV, split, metrics).

The CV plumbing is tested with a fake estimator (predicts zero flows) so the
tests are fast and independent of any ML model.
"""
from __future__ import annotations

import numpy as np
import pytest

from ctm_for_dwc.utils.ramp_flow_estimation.features import TrainingData
from ctm_for_dwc.utils.ramp_flow_estimation.validation import (
    flow_metrics,
    train_val_test_split,
)


def _data(n_stretch=3, per=10, seed=0, r_value=None):
    rng = np.random.default_rng(seed)
    cat = lambda parts: np.concatenate(parts)
    X, r, s, qu, qd, sid = ([] for _ in range(6))
    for st in range(n_stretch):
        X.append(rng.uniform(0, 1, size=(per, 6)))
        r.append(np.full(per, float(r_value)) if r_value is not None
                 else rng.uniform(100, 500, per))
        s.append(np.full(per, 300.0) if r_value is not None
                 else rng.uniform(100, 500, per))
        qu.append(np.full(per, 1000.0)); qd.append(np.full(per, 1200.0))
        sid.append(np.full(per, st))
    return TrainingData(
        X=np.concatenate(X), r_true=cat(r), s_true=cat(s),
        q_up=cat(qu), q_down=cat(qd), stretch_id=cat(sid).astype(int),
    )


def test_flow_metrics_perfect_prediction():
    r = np.array([100.0, 200.0]); s = np.array([50.0, 80.0])
    m = flow_metrics(r, s, r.copy(), s.copy())
    assert m["nrmse"] == 0.0 and m["bias"] == 0.0
    assert np.isclose(m["r2"], 1.0)


def test_flow_metrics_known_nrmse_on_ramp():
    y = np.array([3.0, 4.0])                       # sum of squares = 25
    m = flow_metrics(y, np.zeros(2), np.zeros(2), np.zeros(2))
    assert np.isclose(m["nrmse_on"], 1.0)          # sqrt(25/25)
    assert np.isclose(m["bias_on"], 3.5)           # mean(y - 0)


def test_flow_metrics_known_rmse_and_nbias():
    r = np.array([100.0, 200.0])
    r_hat = np.array([90.0, 220.0])                # errors +10, -20
    s = np.array([50.0, 50.0])
    s_hat = s.copy()
    m = flow_metrics(r, s, r_hat, s_hat)
    assert np.isclose(m["rmse_on"], np.sqrt((100 + 400) / 2))
    assert np.isclose(m["nbias_on"], (10 - 20) / 300)   # sum(err) / sum(true)
    assert m["rmse_off"] == 0.0 and m["nbias_off"] == 0.0
    assert np.isclose(m["nbias"], -10 / 400)            # combined ramps


def test_flow_metrics_nbias_nan_when_true_flows_sum_to_zero():
    z = np.zeros(3)
    m = flow_metrics(z, z, z + 1.0, z)
    assert np.isnan(m["nbias_on"])


def test_split_partitions_rows_by_stretch():
    data = _data(4)
    sp = train_val_test_split(data, seed=0)
    assert (sp["train"] | sp["val"] | sp["test"]).all()
    assert not (sp["train"] & sp["val"]).any()
    assert not (sp["train"] & sp["test"]).any()
    assert not (sp["val"] & sp["test"]).any()
    # each held-out stretch's rows land entirely in its split
    assert set(data.stretch_id[sp["val"]]) == {sp["val_stretch"]}
    assert set(data.stretch_id[sp["test"]]) == {sp["test_stretch"]}
    assert sp["val_stretch"] != sp["test_stretch"]


def test_split_is_deterministic_per_seed():
    data = _data(5)
    a = train_val_test_split(data, seed=7)
    b = train_val_test_split(data, seed=7)
    assert (a["val_stretch"], a["test_stretch"]) == (b["val_stretch"], b["test_stretch"])


def test_split_honors_explicit_ids():
    sp = train_val_test_split(_data(4), val_stretch=1, test_stretch=3)
    assert sp["val_stretch"] == 1 and sp["test_stretch"] == 3


def test_split_rejects_bad_ids():
    data = _data(3)
    with pytest.raises(ValueError):
        train_val_test_split(data, val_stretch=99)
    with pytest.raises(ValueError):
        train_val_test_split(data, val_stretch=1, test_stretch=1)
    with pytest.raises(ValueError):
        train_val_test_split(_data(2))


