"""Kan 2021 estimator: a tree-ensemble regressor for the ramp-flow coefficient.

The model maps the mainstream feature vector ``Z`` (see :mod:`features`) to the
blend coefficient ``alpha in [0, 1]``; ramp flows are then reconstructed from
``alpha`` and the conservation/capacity bounds (:mod:`bounds`). Kan uses random
forest and gradient boosting; both are offered here (RF default), with Kan's
hyperparameters (500 trees, learning rate 0.1) as defaults.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from .bounds import reconstruct_pair


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

    def fit(self, X, alpha) -> "KanEstimator":
        self._model.fit(np.asarray(X, dtype=float), np.asarray(alpha, dtype=float))
        return self

    def predict_alpha(self, X) -> np.ndarray:
        """Predicted coefficient, clipped to the valid ``[0, 1]`` interval."""
        return np.clip(self._model.predict(np.asarray(X, dtype=float)), 0.0, 1.0)

    def predict_flows(self, X, q_up, q_down, c_w, *,
                      r_demand=np.inf, s_qmax=np.inf):
        """Reconstruct ``(r_hat, s_hat)`` from the predicted coefficient and the
        per-sample bounds (Kan A.18-A.19)."""
        alpha = self.predict_alpha(X)
        return reconstruct_pair(alpha, q_up, q_down, c_w,
                                r_demand=r_demand, s_qmax=s_qmax)
