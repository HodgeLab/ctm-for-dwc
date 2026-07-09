"""Tests for utils/ramp_flow_estimation/kan.py (alpha regressor + Kan context).

Synthetic single-lag windows (6 feature columns) so the last-lag block *is* the
whole row: X[:, 0] = q_up, X[:, 3] = q_down by the feature layout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ramp_flow_estimation import bounds
from transportation_models.utils.ramp_flow_estimation.kan import (
    KanEstimator,
    alpha_targets,
    row_context,
)

_C_W = 6000.0


def _data(n=200, seed=0):
    """Windows whose true (r, s) is generated *from* a known alpha, so the
    estimator has an exactly recoverable target."""
    rng = np.random.default_rng(seed)
    q_up = rng.uniform(800.0, 1200.0, n)
    q_down = q_up + 200.0                       # on-ramp is determined-first
    X = np.column_stack([
        q_up, rng.uniform(50, 70, n), rng.uniform(5, 15, n),
        q_down, rng.uniform(50, 70, n), rng.uniform(5, 15, n),
    ])
    alpha = np.clip(0.2 + 0.01 * (X[:, 1] - 50.0), 0.0, 1.0)  # depends on up speed
    r, s = bounds.reconstruct_pair(alpha, q_up, q_down, _C_W)
    ctx = {"c_w": np.full(n, _C_W)}
    return X, r, s, alpha, ctx


def test_random_forest_recovers_a_learnable_alpha_relationship():
    X, r, s, alpha, ctx = _data()
    est = KanEstimator(model="rf", n_estimators=100, random_state=0).fit(X, r, s, ctx=ctx)
    pred = est.predict_alpha(X)
    assert np.mean(np.abs(pred - alpha)) < 0.05


def test_predicted_alpha_within_unit_interval():
    X, r, s, _, ctx = _data()
    est = KanEstimator(model="rf", n_estimators=50, random_state=0).fit(X, r, s, ctx=ctx)
    pred = est.predict_alpha(X)
    assert (pred >= 0.0).all() and (pred <= 1.0).all()


def test_predict_flows_matches_bounds_reconstruction_and_conserves():
    X, r, s, _, ctx = _data()
    est = KanEstimator(model="rf", n_estimators=50, random_state=0).fit(X, r, s, ctx=ctx)
    r_hat, s_hat = est.predict_flows(X, ctx=ctx)
    a = est.predict_alpha(X)
    q_up, q_down = X[:, 0], X[:, 3]
    exp_r, exp_s = bounds.reconstruct_pair(a, q_up, q_down, ctx["c_w"])
    np.testing.assert_allclose(r_hat, exp_r)
    np.testing.assert_allclose(s_hat, exp_s)
    np.testing.assert_allclose(q_up + r_hat - s_hat, q_down)   # conservation


def test_fit_excludes_degenerate_band_rows():
    X, r, s, _, ctx = _data()
    ctx = {"c_w": ctx["c_w"].copy()}
    ctx["c_w"][:50] = 100.0                     # c_w < q_up -> degenerate band
    est = KanEstimator(model="rf", n_estimators=20, random_state=0).fit(X, r, s, ctx=ctx)
    assert (est.predict_alpha(X) <= 1.0).all()  # fit succeeded on the rest


def test_fit_requires_ctx_and_predict_requires_finite_c_w():
    X, r, s, _, ctx = _data(n=50)
    with pytest.raises(ValueError, match="ctx"):
        KanEstimator(model="rf", n_estimators=10).fit(X, r, s)
    est = KanEstimator(model="rf", n_estimators=10, random_state=0).fit(X, r, s, ctx=ctx)
    bad = {"c_w": np.full(len(X), np.nan)}
    with pytest.raises(ValueError, match="c_w"):
        est.predict_flows(X, ctx=bad)


def test_alpha_targets_flags_degenerate_and_clipped():
    X, r, s, _, ctx = _data(n=20, seed=1)
    r = r.copy()
    r[:5] = 0.0                                 # below r_min = q_down - q_up -> clipped
    ctx = {"c_w": ctx["c_w"].copy()}
    ctx["c_w"][-5:] = 100.0                     # degenerate band
    alpha, feasible, clipped = alpha_targets(X, r, s, ctx)
    assert clipped[:5].all() and not clipped[5:].any()
    assert not feasible[-5:].any() and feasible[:-5].all()
    np.testing.assert_allclose(alpha[:5], 0.0)


def test_row_context_expands_per_stretch_table():
    tbl = pd.DataFrame({"c_w": [6000.0, 4000.0], "r_demand": [np.inf, 900.0],
                        "s_qmax": [np.inf, np.inf]},
                       index=pd.Index([3, 7], name="stretch_id"))
    ctx = row_context(tbl, np.array([7, 3, 7]))
    np.testing.assert_allclose(ctx["c_w"], [4000.0, 6000.0, 4000.0])
    np.testing.assert_allclose(ctx["r_demand"], [900.0, np.inf, 900.0])


def test_save_load_roundtrip(tmp_path):
    X, r, s, _, ctx = _data(n=80)
    est = KanEstimator(model="rf", n_estimators=20, random_state=0).fit(X, r, s, ctx=ctx)
    path = tmp_path / "kan.joblib"
    est.save(path)
    loaded = KanEstimator.load(path)
    assert loaded.n_feature_lags == 1                # 6-column synthetic windows
    np.testing.assert_allclose(loaded.predict_alpha(X), est.predict_alpha(X))
    np.testing.assert_allclose(loaded.predict_flows(X, ctx=ctx),
                               est.predict_flows(X, ctx=ctx))


def test_gbm_option_trains_and_predicts_in_range():
    X, r, s, _, ctx = _data()
    est = KanEstimator(model="gbm", n_estimators=50, random_state=0).fit(X, r, s, ctx=ctx)
    pred = est.predict_alpha(X)
    assert (pred >= 0.0).all() and (pred <= 1.0).all()


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        KanEstimator(model="not-a-model")
