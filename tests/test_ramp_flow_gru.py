"""Tests for utils/ramp_flow_estimation/gru.py.

Small synthetic problems (window 3, 18 feature columns) so training stays in
the sub-second range on CPU; all runs are seeded.
"""
from __future__ import annotations

import numpy as np
import pytest

from transportation_models.utils.ramp_flow_estimation.gru import GruEstimator


def _synthetic(n=400, seed=0):
    """Learnable mapping: r, s are positive linear functions of the features."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 18))
    r = 200.0 + 400.0 * X[:, 12] + 100.0 * X[:, 0]     # up flow at t, t-2
    s = 100.0 + 300.0 * X[:, 15] + 50.0 * X[:, 3]      # down flow at t, t-2
    return X, r, s


def _fast(**over):
    kw = dict(hidden_size=16, lr=1e-2, batch_size=64, max_epochs=60,
              patience=10, seed=0, device="cpu")
    kw.update(over)
    return GruEstimator(**kw)


def test_learns_synthetic_mapping():
    X, r, s = _synthetic()
    r_hat, s_hat = _fast().fit(X, r, s).predict_flows(X)
    ss = lambda y, y_hat: 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)
    assert ss(r, r_hat) > 0.8
    assert ss(s, s_hat) > 0.8


def test_predictions_non_negative():
    X, r, s = _synthetic(n=200)
    est = _fast(max_epochs=5).fit(X, np.zeros_like(r), np.zeros_like(s))
    r_hat, s_hat = est.predict_flows(X)
    assert (r_hat >= 0).all() and (s_hat >= 0).all()


def test_early_stopping_on_noise_val():
    # unlearnable val targets -> val loss plateaus -> stops well before max_epochs
    rng = np.random.default_rng(1)
    X, r, s = _synthetic(n=200)
    Xv = rng.uniform(0, 1, size=(80, 18))
    rv, sv = rng.uniform(0, 1000, 80), rng.uniform(0, 1000, 80)
    epochs = []
    _fast(max_epochs=200, patience=5).fit(
        X, r, s, X_val=Xv, r_val=rv, s_val=sv,
        log_fn=lambda e, tl, vl: epochs.append(e))
    assert len(epochs) < 200


def test_val_loss_passed_to_log_fn():
    X, r, s = _synthetic(n=120)
    logged = []
    _fast(max_epochs=3).fit(X, r, s, X_val=X[:40], r_val=r[:40], s_val=s[:40],
                            log_fn=lambda e, tl, vl: logged.append((e, tl, vl)))
    assert len(logged) == 3
    assert all(np.isfinite(vl) for _, _, vl in logged)


def test_save_load_roundtrip(tmp_path):
    X, r, s = _synthetic(n=150)
    est = _fast(max_epochs=10).fit(X, r, s)
    path = tmp_path / "gru.pt"
    est.save(path)
    loaded = GruEstimator.load(path)
    r_a, s_a = est.predict_flows(X)
    r_b, s_b = loaded.predict_flows(X)
    np.testing.assert_allclose(r_a, r_b, rtol=1e-6)
    np.testing.assert_allclose(s_a, s_b, rtol=1e-6)


def test_bad_feature_width_raises():
    X, r, s = _synthetic(n=50)
    with pytest.raises(ValueError):
        _fast(max_epochs=1).fit(X[:, :17], r, s)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        _fast().predict_flows(np.zeros((3, 18)))
