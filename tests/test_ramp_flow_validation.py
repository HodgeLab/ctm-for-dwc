"""Tests for utils/ramp_flow_estimation/validation.py (CV, split, metrics).

The CV plumbing is tested with a fake estimator (predicts zero flows) so the
tests are fast and independent of any ML model.
"""
from __future__ import annotations

import numpy as np
import pytest

from transportation_models.utils.ramp_flow_estimation.features import TrainingData
from transportation_models.utils.ramp_flow_estimation.validation import (
    flow_metrics,
    leave_k_out,
    run_scenarios,
    slice_ctx,
    train_val_test_split,
)


class _FakeEstimator:
    """Fits nothing; always predicts zero ramp flows. Records the ctx it saw."""

    def __init__(self):
        self.fit_ctx = None

    def fit(self, X, r, s, ctx=None):
        self.fit_ctx = ctx
        return self

    def predict_flows(self, X, ctx=None):
        return np.zeros(len(X)), np.zeros(len(X))


def _fake_factory():
    return lambda: _FakeEstimator()


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


def test_leave_one_out_covers_each_stretch_once():
    res = leave_k_out(_data(3), k=1, estimator_factory=_fake_factory())
    assert res["n_combos"] == 3
    assert sorted(c["holdout"] for c in res["per_combo"]) == [(0,), (1,), (2,)]
    assert "mean" in res and "nrmse" in res["mean"]


def test_leave_k_out_enumerates_all_pairs():
    res = leave_k_out(_data(4), k=2, estimator_factory=_fake_factory())
    assert res["n_combos"] == 6                    # C(4, 2)


def test_leave_k_out_samples_when_capped():
    res = leave_k_out(_data(4), k=2, estimator_factory=_fake_factory(),
                      max_combos=3, random_state=0)
    assert res["n_combos"] == 3
    assert all(len(c["holdout"]) == 2 for c in res["per_combo"])
    assert len({c["holdout"] for c in res["per_combo"]}) == 3   # distinct


def test_ctx_sliced_with_the_same_masks():
    data = _data(3)
    seen = []

    class _Spy(_FakeEstimator):
        def fit(self, X, r, s, ctx=None):
            seen.append((len(X), None if ctx is None else len(ctx["c_w"])))
            return self

    ctx = {"c_w": np.arange(len(data.r_true), dtype=float)}
    leave_k_out(data, k=1, ctx=ctx, estimator_factory=lambda: _Spy())
    assert all(n_rows == n_ctx for n_rows, n_ctx in seen)


def test_slice_ctx_none_passes_through():
    assert slice_ctx(None, np.array([True, False])) is None


def test_run_scenarios_one_row_per_k():
    df = run_scenarios(_data(4), ks=(1, 2), estimator_factory=_fake_factory())
    assert list(df["k"]) == [1, 2]
    assert "nrmse_mean" in df.columns and "nrmse_std" in df.columns


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


def test_all_nan_metric_aggregates_to_nan_without_warning(recwarn):
    # constant on-ramp target -> undefined R^2
    res = leave_k_out(_data(3, r_value=500.0), k=1, estimator_factory=_fake_factory())
    assert np.isnan(res["mean"]["r2_on"])
    assert not any(issubclass(w.category, RuntimeWarning) for w in recwarn.list)
