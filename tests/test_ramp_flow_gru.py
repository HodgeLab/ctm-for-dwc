"""Tests for utils/ramp_flow_estimation/gru.py (Normalizer + GRU estimator).

Small synthetic problems (window 3, 18 feature columns) so training stays in
the sub-second range on CPU; all runs are seeded.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from transportation_models.utils.ramp_flow_estimation.gru import GruEstimator, Normalizer


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


# --------------------------------------------------------------------------- #
# Normalizer
# --------------------------------------------------------------------------- #
def test_normalizer_feature_stats_shared_per_variable():
    # up_flow (ch 0) and down_flow (ch 3) have different distributions but must
    # share one mean/std; same for speed (1, 4) and occupancy (2, 5).
    torch.manual_seed(0)
    x = torch.rand(50, 3, 6)
    x[..., 0] *= 1000.0                       # up flow scale
    x[..., 3] *= 2000.0                       # down flow scale
    norm = Normalizer().fit(x, torch.rand(50, 2))
    xm, xs = norm._x_mean, norm._x_std
    assert xm[0] == xm[3] and xs[0] == xs[3]  # flow group
    assert xm[1] == xm[4] and xs[1] == xs[4]  # speed group
    assert xm[2] == xm[5] and xs[2] == xs[5]  # occupancy group
    # and the group stats actually differ between variables
    assert xm[0] != xm[1]


@pytest.mark.parametrize("transform", ["zscore", "log1p"])
def test_normalizer_target_roundtrip(transform):
    y = torch.tensor([[0.0, 10.0], [500.0, 300.0], [1200.0, 40.0]])
    norm = Normalizer(transform).fit(torch.rand(3, 2, 6), y)
    yt = norm.targets(y)
    assert torch.allclose(yt.mean(dim=0), torch.zeros(2), atol=1e-6)
    torch.testing.assert_close(norm.inverse_targets(yt), y)


def test_normalizer_maxscale_divides_each_target_by_its_train_max():
    y = torch.tensor([[0.0, 10.0], [500.0, 300.0], [1200.0, 40.0]])
    norm = Normalizer("maxscale").fit(torch.rand(3, 2, 6), y)
    yt = norm.targets(y)
    torch.testing.assert_close(yt, y / torch.tensor([1200.0, 300.0]))
    torch.testing.assert_close(yt.max(dim=0).values, torch.ones(2))
    torch.testing.assert_close(norm.inverse_targets(yt), y)


def test_normalizer_inverse_clamps_negative_flows():
    y = torch.tensor([[100.0, 200.0], [300.0, 400.0]])
    norm = Normalizer("zscore").fit(torch.rand(2, 2, 6), y)
    out = norm.inverse_targets(torch.full((2, 2), -100.0))
    assert (out >= 0).all()


def test_normalizer_unknown_transform_raises():
    with pytest.raises(ValueError):
        Normalizer("softplus")


# --------------------------------------------------------------------------- #
# GruEstimator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("transform", ["zscore", "log1p", "maxscale"])
def test_learns_synthetic_mapping(transform):
    X, r, s = _synthetic()
    est = _fast(target_transform=transform).fit(X, r, s)
    r_hat, s_hat = est.predict_flows(X)
    ss = lambda y, y_hat: 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)
    assert ss(r, r_hat) > 0.8
    assert ss(s, s_hat) > 0.8


def test_predictions_non_negative():
    X, r, s = _synthetic(n=200)
    est = _fast(max_epochs=5).fit(X, np.zeros_like(r), np.zeros_like(s))
    r_hat, s_hat = est.predict_flows(X)
    assert (r_hat >= 0).all() and (s_hat >= 0).all()


def test_ctx_is_accepted_and_ignored():
    X, r, s = _synthetic(n=100)
    ctx = {"c_w": np.full(100, 6000.0)}
    est = _fast(max_epochs=2).fit(X, r, s, ctx=ctx)
    a = est.predict_flows(X, ctx=ctx)
    b = est.predict_flows(X)
    np.testing.assert_allclose(a, b)


def test_early_stopping_on_noise_val():
    # unlearnable val targets -> val loss plateaus -> stops well before max_epochs
    rng = np.random.default_rng(1)
    X, r, s = _synthetic(n=200)
    Xv = rng.uniform(0, 1, size=(80, 18))
    rv, sv = rng.uniform(0, 1000, 80), rng.uniform(0, 1000, 80)
    epochs = []
    _fast(max_epochs=200, patience=5).fit(
        X, r, s, X_val=Xv, r_val=rv, s_val=sv,
        log_fn=lambda e, tl, vl, vm: epochs.append(e))
    assert len(epochs) < 200


def test_patience_none_disables_early_stopping():
    # same unlearnable val targets as the early-stopping test, but with
    # patience=None the run must go the full max_epochs (final weights kept)
    rng = np.random.default_rng(1)
    X, r, s = _synthetic(n=200)
    Xv = rng.uniform(0, 1, size=(80, 18))
    rv, sv = rng.uniform(0, 1000, 80), rng.uniform(0, 1000, 80)
    epochs = []
    _fast(max_epochs=15, patience=None).fit(
        X, r, s, X_val=Xv, r_val=rv, s_val=sv,
        log_fn=lambda e, tl, vl, vm: epochs.append(e))
    assert len(epochs) == 15


def test_val_loss_and_metrics_passed_to_log_fn():
    X, r, s = _synthetic(n=120)
    logged = []
    _fast(max_epochs=3).fit(
        X, r, s, X_val=X[:40], r_val=r[:40], s_val=s[:40],
        log_fn=lambda e, tl, vl, vm: logged.append((e, tl, vl, vm)))
    assert len(logged) == 3
    for _, _, vl, vm in logged:
        assert np.isfinite(vl)
        # raw-space per-epoch validation metrics, comparable across transforms
        assert all(np.isfinite(vm[k]) for k in ("rmse", "nrmse", "bias", "nbias"))


def test_no_val_metrics_without_val_set():
    X, r, s = _synthetic(n=80)
    logged = []
    _fast(max_epochs=2).fit(X, r, s,
                            log_fn=lambda e, tl, vl, vm: logged.append(vm))
    assert logged == [None, None]


def test_dropout_requires_stacked_layers():
    with pytest.raises(ValueError, match="num_layers"):
        _fast(dropout=0.25, num_layers=1)
    _fast(dropout=0.25, num_layers=2)          # fine


def test_optimizer_knobs_train(tmp_path):
    X, r, s = _synthetic(n=150)
    est = _fast(max_epochs=5, num_layers=2, dropout=0.25, beta1=0.8,
                weight_decay=1e-4).fit(X, r, s)
    path = tmp_path / "gru.pt"
    est.save(path)
    loaded = GruEstimator.load(path)
    assert (loaded.dropout, loaded.beta1, loaded.weight_decay) == (0.25, 0.8, 1e-4)
    np.testing.assert_allclose(est.predict_flows(X), loaded.predict_flows(X), rtol=1e-6)


@pytest.mark.parametrize("transform", ["zscore", "log1p"])
def test_save_load_roundtrip(tmp_path, transform):
    X, r, s = _synthetic(n=150)
    est = _fast(max_epochs=10, target_transform=transform).fit(X, r, s)
    path = tmp_path / "gru.pt"
    est.save(path)
    loaded = GruEstimator.load(path)
    assert loaded.target_transform == transform
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
