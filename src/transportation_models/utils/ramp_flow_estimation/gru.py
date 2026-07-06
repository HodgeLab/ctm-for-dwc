"""GRU estimator: direct joint regression of the ramp-flow pair ``(r, s)``.

Motivation (docs/ramp_flow_estimation_module.md): raw PeMS data shows flow is
generally *not* conserved at ramp gores, while Kan's alpha-blend encodes
conservation through ``r_min``/``s_min`` -- out-of-band true flows are
unreachable by construction. This estimator maps the same mainstream feature
matrix ``Z`` (see :mod:`features`) to ``(r, s)`` directly, treating each row as
a ``(window_size, 6)`` sequence. Non-negativity (softplus head) is the only
constraint; no conservation or capacity structure is imposed.

Flows are in veh/hr throughout, matching :func:`features.build_training_data`.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .features import FEATURES_PER_LAG


class _GruNet(nn.Module):
    """GRU -> linear 2-unit head -> softplus (non-negative ``r_hat, s_hat``)."""

    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(FEATURES_PER_LAG, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return nn.functional.softplus(self.head(out[:, -1, :]))


class GruEstimator:
    """Joint ``(r, s)`` regressor over Kan feature windows.

    ``fit`` standardizes features and scales targets by their training std
    (scale-only, preserving non-negativity through the softplus head), trains
    with Adam/MSE, and -- when a validation set is given -- early-stops on val
    loss with best-weight restore. ``log_fn(epoch, train_loss, val_loss)`` is an
    optional per-epoch callback (e.g. for W&B logging).
    """

    def __init__(self, *, hidden_size: int = 64, num_layers: int = 1,
                 lr: float = 1e-3, batch_size: int = 256, max_epochs: int = 200,
                 patience: int = 20, seed: int = 0, device=None):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = None
        self._x_mean = None
        self._x_std = None
        self._y_scale = None

    # ------------------------------------------------------------------ #
    def _sequences(self, X) -> torch.Tensor:
        """Standardize the flat ``(n, 6*T)`` matrix and reshape to ``(n, T, 6)``
        sequences (the per-lag feature layout makes this a pure reshape)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] % FEATURES_PER_LAG:
            raise ValueError(f"expected (n, {FEATURES_PER_LAG}*T) features, got {X.shape}")
        Xs = (X - self._x_mean) / self._x_std
        n, t = X.shape[0], X.shape[1] // FEATURES_PER_LAG
        return torch.as_tensor(Xs.reshape(n, t, FEATURES_PER_LAG), dtype=torch.float32)

    def fit(self, X, r, s, *, X_val=None, r_val=None, s_val=None,
            log_fn=None) -> "GruEstimator":
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.column_stack([np.asarray(r, float), np.asarray(s, float)])
        self._x_mean = X.mean(axis=0)
        self._x_std = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        self._y_scale = np.where(y.std(axis=0) > 0, y.std(axis=0), 1.0)

        xt = self._sequences(X).to(self.device)
        yt = torch.as_tensor(y / self._y_scale, dtype=torch.float32).to(self.device)
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            xv = self._sequences(X_val).to(self.device)
            yv = torch.as_tensor(
                np.column_stack([np.asarray(r_val, float), np.asarray(s_val, float)])
                / self._y_scale, dtype=torch.float32).to(self.device)

        net = _GruNet(self.hidden_size, self.num_layers).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)

        best_loss, best_state, since_best = np.inf, None, 0
        for epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(xt), generator=gen)
            total = 0.0
            for lo in range(0, len(xt), self.batch_size):
                idx = perm[lo:lo + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            train_loss = total / len(xt)

            val_loss = float("nan")
            if has_val:
                net.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(net(xv), yv))
            if log_fn is not None:
                log_fn(epoch, train_loss, val_loss)
            if has_val:
                if val_loss < best_loss:
                    best_loss, since_best = val_loss, 0
                    best_state = copy.deepcopy(net.state_dict())
                else:
                    since_best += 1
                    if since_best >= self.patience:
                        break

        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net.eval()
        return self

    def predict_flows(self, X):
        """Predicted ``(r_hat, s_hat)`` in veh/hr, non-negative by construction."""
        if self._net is None:
            raise RuntimeError("estimator is not fitted")
        with torch.no_grad():
            y = self._net(self._sequences(X).to(self.device)).cpu().numpy()
        y = y * self._y_scale
        return y[:, 0], y[:, 1]

    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        """Checkpoint the fitted model + scalers + architecture hyperparameters."""
        if self._net is None:
            raise RuntimeError("estimator is not fitted")
        torch.save({
            "hidden_size": self.hidden_size, "num_layers": self.num_layers,
            "x_mean": self._x_mean, "x_std": self._x_std, "y_scale": self._y_scale,
            "state_dict": self._net.state_dict(),
        }, path)

    @classmethod
    def load(cls, path, *, device=None) -> "GruEstimator":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        est = cls(hidden_size=ckpt["hidden_size"], num_layers=ckpt["num_layers"],
                  device=device if device is not None else "cpu")
        est._x_mean, est._x_std = ckpt["x_mean"], ckpt["x_std"]
        est._y_scale = ckpt["y_scale"]
        net = _GruNet(est.hidden_size, est.num_layers)
        net.load_state_dict(ckpt["state_dict"])
        est._net = net.to(est.device).eval()
        return est
