"""Tests for utils/ramp_flow_estimation/zhang.py (normalizer, MMD, and the
staged GRU + DDA + MT estimator). Small synthetic problems (window 5,
10 features/step = 50 columns) keep runs in the sub-second range on CPU."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from transportation_models.utils.ramp_flow_estimation.zhang import (
    ZhangEstimator,
    ZhangNormalizer,
    gaussian_mmd,
)

_NCOLS = 50   # 10 features/step * window 5


def _synthetic(n=400, seed=0, shift=0.0):
    """Learnable mapping: r, s positive linear functions of two features;
    ``shift`` displaces the feature distribution (a 'different stretch')."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, _NCOLS)) + shift
    r = 200.0 + 400.0 * X[:, 40] + 100.0 * X[:, 0]   # q_up at t and t-4
    s = 100.0 + 300.0 * X[:, 41] + 50.0 * X[:, 1]    # q_down at t and t-4
    return X.astype(float), r, s


def _fast(**over):
    kw = dict(embed_dim=8, hidden_size=16, adapt_dim=16, lr=1e-2,
              batch_size=64, max_epochs=60, patience=10, dda_epochs=5,
              seed=0, device="cpu")
    kw.update(over)
    return ZhangEstimator(**kw)


# --------------------------------------------------------------------------- #
# ZhangNormalizer
# --------------------------------------------------------------------------- #
def test_normalizer_per_channel_stats_and_roundtrip():
    torch.manual_seed(0)
    x = torch.rand(50, 5, 10)
    x[..., 0] *= 1000.0                            # flow-scale channel
    y = torch.tensor([[0.0, 10.0], [500.0, 300.0], [1200.0, 40.0]])
    norm = ZhangNormalizer().fit(x, y)
    xn = norm.features(x)
    assert abs(float(xn[..., 0].mean())) < 1e-5    # each channel centered
    assert abs(float(xn[..., 3].mean())) < 1e-5
    assert norm._x_mean[0] != norm._x_mean[3]      # independent per channel
    torch.testing.assert_close(norm.inverse_targets(norm.targets(y)), y)


def test_normalizer_inverse_clamps_negative_flows():
    norm = ZhangNormalizer().fit(torch.rand(3, 5, 10),
                                 torch.tensor([[100.0, 200.0], [300.0, 400.0],
                                               [500.0, 600.0]]))
    assert (norm.inverse_targets(torch.full((2, 2), -100.0)) >= 0).all()


# --------------------------------------------------------------------------- #
# gaussian_mmd
# --------------------------------------------------------------------------- #
def test_mmd_separates_shifted_distributions():
    torch.manual_seed(0)
    a, b = torch.randn(64, 8), torch.randn(64, 8)
    near = gaussian_mmd(a, b)
    far = gaussian_mmd(a, b + 5.0)
    assert 0.0 <= float(near) < float(far)


def test_mmd_is_differentiable():
    a = torch.randn(32, 8, requires_grad=True)
    gaussian_mmd(a, torch.randn(32, 8) + 2.0).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


# --------------------------------------------------------------------------- #
# pair_mmd (the paper's per-pair MMD distance at the Adaptation Layer H_t)
# --------------------------------------------------------------------------- #
def test_pair_mmd_orders_by_distribution_distance():
    X, r, s = _synthetic(n=500)
    X_near, _, _ = _synthetic(n=500, seed=1, shift=0.1)
    X_far, _, _ = _synthetic(n=500, seed=2, shift=3.0)
    est = _fast(max_epochs=10).fit_source(X, r, s)
    near, far = est.pair_mmd(X, X_near), est.pair_mmd(X, X_far)
    assert 0.0 <= near < far


def test_pair_mmd_subsample_is_seeded_and_deterministic():
    X, r, s = _synthetic(n=400)
    X_b, _, _ = _synthetic(n=400, seed=1, shift=0.5)
    est = _fast(max_epochs=5).fit_source(X, r, s)
    kw = dict(max_samples=100, seed=7)
    assert est.pair_mmd(X, X_b, **kw) == est.pair_mmd(X, X_b, **kw)


def test_pair_mmd_shrinks_after_dda():
    # adaptation with a dominant MMD term must reduce the reported distance
    X, r, s = _synthetic(n=300)
    Xt, _, _ = _synthetic(n=200, seed=1, shift=1.0)
    est = _fast(max_epochs=20, dda_epochs=20, lambda_mmd=10.0).fit_source(X, r, s)
    before = est.pair_mmd(X, Xt)
    est.adapt(X, r, s, Xt)
    assert est.pair_mmd(X, Xt) < before


def test_pair_mmd_before_fit_raises():
    with pytest.raises(RuntimeError):
        _fast().pair_mmd(np.zeros((10, _NCOLS)), np.zeros((10, _NCOLS)))


# --------------------------------------------------------------------------- #
# ZhangEstimator: backbone
# --------------------------------------------------------------------------- #
def test_backbone_learns_synthetic_mapping():
    X, r, s = _synthetic()
    est = _fast().fit_source(X, r, s)
    r_hat, s_hat = est.predict_flows(X)
    ss = lambda y, y_hat: 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)
    assert ss(r, r_hat) > 0.8
    assert ss(s, s_hat) > 0.8
    assert est._net.bn is None                     # Fig. 3: no BN pre-DDA


def test_predictions_non_negative():
    X, r, s = _synthetic(n=200)
    est = _fast(max_epochs=5).fit_source(X, np.zeros_like(r), np.zeros_like(s))
    r_hat, s_hat = est.predict_flows(X)
    assert (r_hat >= 0).all() and (s_hat >= 0).all()


def test_early_stopping_on_noise_val():
    rng = np.random.default_rng(1)
    X, r, s = _synthetic(n=200)
    Xv, _, _ = _synthetic(n=80, seed=1)
    rv, sv = rng.uniform(0, 1000, 80), rng.uniform(0, 1000, 80)
    epochs = []
    _fast(max_epochs=200, patience=5).fit_source(
        X, r, s, X_val=Xv, r_val=rv, s_val=sv,
        log_fn=lambda e, tl, vl, vm: epochs.append(e))
    assert len(epochs) < 200


def test_fit_source_logs_val_flow_metrics():
    X, r, s = _synthetic(n=120)
    logged = []
    _fast(max_epochs=3).fit_source(
        X, r, s, X_val=X[:40], r_val=r[:40], s_val=s[:40],
        log_fn=lambda e, tl, vl, vm: logged.append((vl, vm)))
    assert len(logged) == 3
    for vl, vm in logged:
        assert np.isfinite(vl)
        # raw-space per-epoch validation metrics for W&B
        assert all(np.isfinite(vm[k]) for k in ("nrmse", "r2", "rmse", "bias"))


def test_fit_source_no_val_metrics_without_val_set():
    X, r, s = _synthetic(n=80)
    logged = []
    _fast(max_epochs=2).fit_source(X, r, s,
                                   log_fn=lambda e, tl, vl, vm: logged.append(vm))
    assert logged == [None, None]


def test_bad_feature_width_raises():
    X, r, s = _synthetic(n=50)
    with pytest.raises(ValueError, match="features"):
        _fast(max_epochs=1).fit_source(X[:, :35], r, s)   # 35 % 10 != 0


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        _fast().predict_flows(np.zeros((3, _NCOLS)))


# --------------------------------------------------------------------------- #
# ZhangEstimator: DDA
# --------------------------------------------------------------------------- #
def test_adapt_inserts_bn_and_freezes_gru_module():
    X, r, s = _synthetic(n=150)
    Xt, _, _ = _synthetic(n=100, seed=1, shift=0.5)
    est = _fast(max_epochs=5).fit_source(X, r, s)
    frozen_before = {k: v.clone() for k, v in est._net.gru.state_dict().items()}
    emb_before = est._net.embedding.weight.clone()
    est.adapt(X, r, s, Xt)
    assert est._net.bn is not None
    for k, v in est._net.gru.state_dict().items():        # GRU module frozen
        torch.testing.assert_close(v, frozen_before[k])
    torch.testing.assert_close(est._net.embedding.weight, emb_before)


def test_adapt_logs_fixed_epochs_and_val_flow_metrics():
    X, r, s = _synthetic(n=100)
    Xt, _, _ = _synthetic(n=80, seed=1)
    est = _fast(max_epochs=2, dda_epochs=4).fit_source(X, r, s)
    logged = []
    est.adapt(X, r, s, Xt, X_val=X[:30], r_val=r[:30], s_val=s[:30],
              log_fn=lambda e, est_l, mmd_l, tot, vm: logged.append((e, vm)))
    assert [e for e, _ in logged] == [0, 1, 2, 3]
    for _, vm in logged:
        assert np.isfinite(vm["nrmse"]) and np.isfinite(vm["r2"])
    # without a val set the metrics slot is None
    logged.clear()
    est2 = _fast(max_epochs=2, dda_epochs=2).fit_source(X, r, s)
    est2.adapt(X, r, s, Xt,
               log_fn=lambda e, est_l, mmd_l, tot, vm: logged.append(vm))
    assert logged == [None, None]


def test_adapt_before_fit_raises():
    X, r, s = _synthetic(n=50)
    with pytest.raises(RuntimeError, match="fit_source"):
        _fast().adapt(X, r, s, X)


# --------------------------------------------------------------------------- #
# ZhangEstimator: Model Transfer
# --------------------------------------------------------------------------- #
def test_calibrate_recovers_amplitude_scale():
    X, r, s = _synthetic(n=300)
    est = _fast(max_epochs=40).fit_source(X, r, s)
    r_hat, s_hat = est.predict_flows(X[:50])
    # a "target" whose true flows are a scaled copy of the predictions
    h_r, h_s = est.calibrate(X[:50], 2.0 * r_hat, 0.5 * s_hat)
    assert h_r == pytest.approx(2.0)
    assert h_s == pytest.approx(0.5)
    r2, s2 = est.predict_flows(X[:50])
    np.testing.assert_allclose(r2, 2.0 * r_hat, rtol=1e-6)
    np.testing.assert_allclose(s2, 0.5 * s_hat, rtol=1e-6)


def test_calibrate_all_zero_predictions_keeps_unit_scale():
    # Eq. 5 averages y/y_hat over nonzero estimates only; with none, h = 1
    X, r, s = _synthetic(n=100)
    est = _fast(max_epochs=2).fit_source(X, r, s)
    est._predict_raw = lambda X: (np.zeros(len(X)), np.zeros(len(X)))
    h_r, h_s = est.calibrate(X[:20], r[:20], s[:20])
    assert h_r == 1.0 and h_s == 1.0


def test_recalibration_overwrites():
    X, r, s = _synthetic(n=200)
    est = _fast(max_epochs=20).fit_source(X, r, s)
    r_hat, s_hat = est.predict_flows(X[:40])
    est.calibrate(X[:40], 3.0 * r_hat, 3.0 * s_hat)
    h_r, h_s = est.calibrate(X[:40], r_hat, s_hat)   # back to identity
    assert h_r == pytest.approx(1.0)
    assert h_s == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip_full_pipeline(tmp_path):
    X, r, s = _synthetic(n=150)
    Xt, _, _ = _synthetic(n=100, seed=1, shift=0.5)
    est = _fast(max_epochs=5).fit_source(X, r, s)
    est.adapt(X, r, s, Xt)
    est.calibrate(Xt[:30], r[:30], s[:30])
    path = tmp_path / "zhang.pt"
    est.save(path)
    loaded = ZhangEstimator.load(path)
    assert loaded._net.bn is not None
    assert loaded._h_mt == est._h_mt
    np.testing.assert_allclose(est.predict_flows(Xt), loaded.predict_flows(Xt),
                               rtol=1e-6)


def test_save_before_fit_raises(tmp_path):
    with pytest.raises(RuntimeError):
        _fast().save(tmp_path / "zhang.pt")
