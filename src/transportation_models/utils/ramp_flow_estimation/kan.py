"""Kan 2021 estimator: tree-ensemble regression of the alpha-between-bounds coefficient.

The corpus (:mod:`features`) is method-agnostic; everything Kan-specific is
derived here. At each window a coefficient ``alpha in [0, 1]`` places the
determined-first ramp within its conservation/capacity band (:mod:`bounds`);
ramp flows are reconstructed from the predicted coefficient (Kan A.18-A.19).
Windows with a degenerate band (congestion, ``c_w - q <= 0``) have no alpha
target and are excluded from *fitting*; windows whose true flow falls outside
the band get a clipped target (:func:`alpha_targets` flags both).

The estimator implements the shared interface ``fit(X, r, s, ctx=None)`` /
``predict_flows(X, ctx=None)``. ``ctx`` holds the row-aligned bounds context
(``c_w``, ``r_demand``, ``s_qmax``; expand the assembler's per-stretch table
with :func:`row_context`); the mainline flows ``q_up``/``q_down`` are recovered
from the raw feature matrix's last-lag block. Kan needs a finite ``c_w`` on
every row -- build the corpus with ``require_capacity=True``.

Kan uses random forest and gradient boosting; both are offered here (RF
default), with Kan's hyperparameters (500 trees, learning rate 0.1) as defaults.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from .bounds import alpha_out_of_band, alpha_target, compute_bounds, reconstruct_pair
from .features import mainline_flows

_CTX_KEYS = ("c_w", "r_demand", "s_qmax")


def row_context(stretch_ctx, stretch_id) -> dict:
    """Expand the assembler's per-stretch bounds-context table to row-aligned
    arrays keyed for the estimator interface's ``ctx``."""
    sub = stretch_ctx.reindex(np.asarray(stretch_id))
    return {key: sub[key].to_numpy(dtype=float) for key in _CTX_KEYS}


def alpha_targets(X, r, s, ctx):
    """Per-row Kan training targets and diagnostics: ``(alpha, feasible,
    clipped)``. ``feasible`` is False on degenerate bands (alpha undefined
    there, including rows with NaN ``c_w``); ``clipped`` marks feasible rows
    whose true flow lies outside the band (conservation/capacity violated in
    the raw data), where alpha saturates at 0 or 1."""
    q_up, q_down = mainline_flows(X)
    b = compute_bounds(q_up, q_down, ctx["c_w"],
                       r_demand=ctx.get("r_demand", np.inf),
                       s_qmax=ctx.get("s_qmax", np.inf))
    alpha, feasible = alpha_target(b, q_up, q_down, r, s)
    clipped = alpha_out_of_band(b, q_up, q_down, r, s) & feasible
    return alpha, feasible, clipped


class KanEstimator:
    """Random-forest / gradient-boosting regressor for the ramp coefficient."""

    def __init__(self, model: str = "rf", *, n_estimators: int = 500,
                 learning_rate: float = 0.1, random_state: int = 0, **kwargs):
        if model == "rf":
            self._model = RandomForestRegressor(
                n_estimators=n_estimators, random_state=random_state, **kwargs)
        elif model == "gbm":
            self._model = GradientBoostingRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate,
                random_state=random_state, **kwargs)
        else:
            raise ValueError(f"unknown model {model!r}; expected 'rf' or 'gbm'")
        self.model = model

    @staticmethod
    def _require_ctx(ctx):
        if ctx is None or any(k not in ctx for k in ("c_w",)):
            raise ValueError("KanEstimator requires the bounds context ctx "
                             "(row-aligned 'c_w', optionally 'r_demand'/'s_qmax')")
        return ctx

    def fit(self, X, r, s, ctx=None) -> "KanEstimator":
        """Derive alpha targets from the bounds context and fit on the rows
        where the band is non-degenerate."""
        ctx = self._require_ctx(ctx)
        X = np.asarray(X, dtype=float)
        alpha, feasible, _ = alpha_targets(X, r, s, ctx)
        if not feasible.any():
            raise ValueError("no feasible (non-degenerate-band) rows to fit on")
        self._model.fit(X[feasible], alpha[feasible])
        return self

    def predict_alpha(self, X) -> np.ndarray:
        """Predicted coefficient, clipped to the valid ``[0, 1]`` interval."""
        return np.clip(self._model.predict(np.asarray(X, dtype=float)), 0.0, 1.0)

    def predict_flows(self, X, ctx=None):
        """Reconstruct ``(r_hat, s_hat)`` from the predicted coefficient and the
        per-row bounds (Kan A.18-A.19)."""
        ctx = self._require_ctx(ctx)
        c_w = np.asarray(ctx["c_w"], dtype=float)
        if not np.isfinite(c_w).all():
            raise ValueError("Kan needs a finite c_w on every row; build the "
                             "corpus with require_capacity=True")
        q_up, q_down = mainline_flows(X)
        alpha = self.predict_alpha(X)
        return reconstruct_pair(alpha, q_up, q_down, c_w,
                                r_demand=ctx.get("r_demand", np.inf),
                                s_qmax=ctx.get("s_qmax", np.inf))

    @property
    def n_feature_lags(self) -> int:
        """Window size the fitted model was trained with (from the RF/GBM's
        expected feature count; the layout is 6 features per lag)."""
        return int(self._model.n_features_in_) // 6

    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        """Checkpoint the fitted estimator (joblib pickle)."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> "KanEstimator":
        est = joblib.load(path)
        if not isinstance(est, cls):
            raise TypeError(f"{path} does not hold a {cls.__name__}")
        return est
