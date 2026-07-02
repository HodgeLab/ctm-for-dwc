"""Tests for utils/ramp_flow_estimation/kan.py (RF/GBM alpha regressor)."""
from __future__ import annotations

import numpy as np
import pytest

from transportation_models.utils.ramp_flow_estimation import bounds
from transportation_models.utils.ramp_flow_estimation.kan import KanEstimator


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 6))
    alpha = np.clip(0.2 + 0.6 * X[:, 0], 0.0, 1.0)   # alpha depends on feature 0
    return X, alpha


def test_random_forest_recovers_a_learnable_alpha_relationship():
    X, alpha = _data()
    est = KanEstimator(model="rf", n_estimators=100, random_state=0).fit(X, alpha)
    pred = est.predict_alpha(X)
    assert np.mean(np.abs(pred - alpha)) < 0.05


def test_predicted_alpha_within_unit_interval():
    X, alpha = _data()
    est = KanEstimator(model="rf", n_estimators=50, random_state=0).fit(X, alpha)
    pred = est.predict_alpha(X)
    assert (pred >= 0.0).all() and (pred <= 1.0).all()


def test_predict_flows_matches_bounds_reconstruction_and_conserves():
    X, alpha = _data()
    est = KanEstimator(model="rf", n_estimators=50, random_state=0).fit(X, alpha)
    q_up = np.full(len(X), 1000.0)
    q_down = np.full(len(X), 1200.0)
    c_w = np.full(len(X), 6000.0)

    r_hat, s_hat = est.predict_flows(X, q_up, q_down, c_w)
    a = est.predict_alpha(X)
    exp_r, exp_s = bounds.reconstruct_pair(a, q_up, q_down, c_w)
    np.testing.assert_allclose(r_hat, exp_r)
    np.testing.assert_allclose(s_hat, exp_s)
    np.testing.assert_allclose(q_up + r_hat - s_hat, q_down)   # conservation


def test_gbm_option_trains_and_predicts_in_range():
    X, alpha = _data()
    est = KanEstimator(model="gbm", n_estimators=50, random_state=0).fit(X, alpha)
    pred = est.predict_alpha(X)
    assert (pred >= 0.0).all() and (pred <= 1.0).all()


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        KanEstimator(model="not-a-model")
