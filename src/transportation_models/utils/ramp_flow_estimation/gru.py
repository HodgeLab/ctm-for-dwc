"""GRU estimator: direct joint regression of the ramp-flow pair ``(r, s)``.

Motivation (docs/ramp_flow_estimation_module.md): raw PeMS data shows flow is
generally *not* conserved at ramp gores, while Kan's alpha-blend encodes
conservation through its band -- out-of-band true flows are unreachable by
construction. This estimator maps the feature matrix ``Z`` (see
:mod:`features`) to ``(r, s)`` directly, treating each row as a
``(window_size, 6)`` sequence. No conservation or capacity structure is
imposed; predictions are clamped non-negative on the way back to flow units.

Implements the shared interface ``fit(X, r, s, ctx=None)`` /
``predict_flows(X, ctx=None)``; ``ctx`` (the bounds context used by Kan) is
accepted and ignored. numpy arrays are converted to tensors once at this
boundary; everything inside (normalization, batching, loss) is torch.
Flows are in veh/hr throughout, matching :func:`features.build_training_data`.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .features import FEATURES_PER_LAG, _VARS_PER_STATION
from .validation import flow_metrics

_TARGET_TRANSFORMS = ("zscore", "log1p", "maxscale")


class Normalizer:
    """Fitted normalization for sequence features and flow targets (torch).

    **Features**: one mean/std per physical variable (flow, speed, occupancy),
    shared across the up/downstream stations and all lags -- so every channel
    of one variable stays on a single scale (channel layout
    ``[up_flow, up_speed, up_occ, down_flow, down_speed, down_occ]``).

    **Targets** (``target_transform``):

    * ``"zscore"`` -- center and scale; inverse clamps at zero.
    * ``"log1p"`` -- ``log1p`` then center and scale; inverse ``expm1`` (loss
      then weights relative rather than absolute errors), clamped at zero.
    * ``"maxscale"`` -- divide each target by its training max (r and s scaled
      independently into ``[0, ~1]``); no centering, inverse clamps at zero.
    """

    def __init__(self, target_transform: str = "zscore"):
        if target_transform not in _TARGET_TRANSFORMS:
            raise ValueError(f"unknown target_transform {target_transform!r}; "
                             f"expected one of {_TARGET_TRANSFORMS}")
        self.target_transform = target_transform
        self._x_mean = self._x_std = None    # (channels,)
        self._y_mean = self._y_std = None    # (2,)

    def _pre(self, y: torch.Tensor) -> torch.Tensor:
        return torch.log1p(y) if self.target_transform == "log1p" else y

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "Normalizer":
        """``x``: (n, T, channels) raw sequences; ``y``: (n, 2) raw flows."""
        channels = x.shape[-1]
        x_mean = torch.empty(channels, dtype=x.dtype)
        x_std = torch.ones(channels, dtype=x.dtype)
        for var in range(_VARS_PER_STATION):
            chans = list(range(var, channels, _VARS_PER_STATION))
            vals = x[..., chans]
            x_mean[chans] = vals.mean()
            if (std := vals.std()) > 0:
                x_std[chans] = std
        self._x_mean, self._x_std = x_mean, x_std

        yt = self._pre(y)
        if self.target_transform == "maxscale":
            self._y_mean = torch.zeros(yt.shape[1], dtype=yt.dtype)
            y_max = yt.max(dim=0).values
            self._y_std = torch.where(y_max > 0, y_max, torch.ones_like(y_max))
        else:
            self._y_mean = yt.mean(dim=0)
            y_std = yt.std(dim=0)
            self._y_std = torch.where(y_std > 0, y_std, torch.ones_like(y_std))
        return self

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._x_mean) / self._x_std

    def targets(self, y: torch.Tensor) -> torch.Tensor:
        return (self._pre(y) - self._y_mean) / self._y_std

    def inverse_targets(self, y: torch.Tensor) -> torch.Tensor:
        y = y * self._y_std + self._y_mean
        if self.target_transform == "log1p":
            y = torch.expm1(y)
        return torch.clamp(y, min=0.0)

    def state_dict(self) -> dict:
        return {"target_transform": self.target_transform,
                "x_mean": self._x_mean, "x_std": self._x_std,
                "y_mean": self._y_mean, "y_std": self._y_std}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Normalizer":
        norm = cls(state["target_transform"])
        norm._x_mean, norm._x_std = state["x_mean"], state["x_std"]
        norm._y_mean, norm._y_std = state["y_mean"], state["y_std"]
        return norm


class _GruNet(nn.Module):
    """GRU -> linear 2-unit head, in normalized-target space."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(FEATURES_PER_LAG, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class GruEstimator:
    """Joint ``(r, s)`` regressor over feature windows.

    ``fit`` fits a :class:`Normalizer` on the training data, trains with
    AdamW/MSE in normalized space (``beta1`` is the momentum rate;
    ``weight_decay`` is decoupled, so 0 reproduces plain Adam), and -- when a
    validation set is given -- early-stops on val loss with best-weight restore.
    ``patience=None`` disables early stopping: the model trains all
    ``max_epochs`` and keeps the *final* epoch's weights (val metrics are still
    computed and logged each epoch).
    ``dropout`` is nn.GRU's inter-layer dropout and therefore requires
    ``num_layers >= 2``. ``log_fn(epoch, train_loss, val_loss, val_metrics)``
    is an optional per-epoch callback (e.g. for W&B logging); ``val_metrics``
    is the raw-space :func:`validation.flow_metrics` dict on the validation
    set (comparable across target transforms), or None without one.
    """

    def __init__(self, *, hidden_size: int = 64, num_layers: int = 1,
                 dropout: float = 0.0, lr: float = 1e-3, beta1: float = 0.9,
                 weight_decay: float = 0.0, batch_size: int = 256,
                 max_epochs: int = 200, patience: int | None = 20,
                 target_transform: str = "zscore", seed: int = 0, device=None):
        if dropout > 0.0 and num_layers < 2:
            raise ValueError("dropout is inter-layer (nn.GRU) and has no effect "
                             "with num_layers=1; use num_layers >= 2 or dropout=0")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.beta1 = beta1
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.target_transform = target_transform
        self.seed = seed
        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = None
        self._norm = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _sequences(X) -> torch.Tensor:
        """Convert the flat ``(n, 6*T)`` matrix to raw ``(n, T, 6)`` sequence
        tensors (the per-lag feature layout makes this a pure reshape)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] % FEATURES_PER_LAG:
            raise ValueError(f"expected (n, {FEATURES_PER_LAG}*T) features, got {X.shape}")
        n, t = X.shape[0], X.shape[1] // FEATURES_PER_LAG
        return torch.from_numpy(X.reshape(n, t, FEATURES_PER_LAG))

    @staticmethod
    def _flows(r, s) -> torch.Tensor:
        return torch.from_numpy(
            np.column_stack([np.asarray(r, dtype=np.float32),
                             np.asarray(s, dtype=np.float32)]))

    def fit(self, X, r, s, ctx=None, *, X_val=None, r_val=None, s_val=None,
            log_fn=None) -> "GruEstimator":
        torch.manual_seed(self.seed)
        x_raw, y_raw = self._sequences(X), self._flows(r, s)
        self._norm = Normalizer(self.target_transform).fit(x_raw, y_raw)
        xt = self._norm.features(x_raw).to(self.device)
        yt = self._norm.targets(y_raw).to(self.device)
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            xv = self._norm.features(self._sequences(X_val)).to(self.device)
            yv = self._norm.targets(self._flows(r_val, s_val)).to(self.device)

        net = _GruNet(self.hidden_size, self.num_layers, self.dropout).to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                betas=(self.beta1, 0.999),
                                weight_decay=self.weight_decay)
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

            val_loss, val_metrics = float("nan"), None
            if has_val:
                net.eval()
                with torch.no_grad():
                    val_pred = net(xv)
                    val_loss = float(loss_fn(val_pred, yv))
                    if log_fn is not None:
                        flows = self._norm.inverse_targets(val_pred.cpu()).numpy()
                        val_metrics = flow_metrics(
                            np.asarray(r_val, float), np.asarray(s_val, float),
                            flows[:, 0], flows[:, 1])
            if log_fn is not None:
                log_fn(epoch, train_loss, val_loss, val_metrics)
            if has_val and self.patience is not None:
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

    def predict_flows(self, X, ctx=None):
        """Predicted ``(r_hat, s_hat)`` in veh/hr, clamped non-negative."""
        if self._net is None:
            raise RuntimeError("estimator is not fitted")
        with torch.no_grad():
            xn = self._norm.features(self._sequences(X)).to(self.device)
            y = self._norm.inverse_targets(self._net(xn).cpu())
        y = y.numpy()
        return y[:, 0], y[:, 1]

    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        """Checkpoint the fitted model + normalizer + architecture."""
        if self._net is None:
            raise RuntimeError("estimator is not fitted")
        torch.save({
            "hidden_size": self.hidden_size, "num_layers": self.num_layers,
            "dropout": self.dropout, "beta1": self.beta1,
            "weight_decay": self.weight_decay,
            "normalizer": self._norm.state_dict(),
            "state_dict": self._net.state_dict(),
        }, path)

    @classmethod
    def load(cls, path, *, device=None) -> "GruEstimator":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        norm = Normalizer.from_state_dict(ckpt["normalizer"])
        est = cls(hidden_size=ckpt["hidden_size"], num_layers=ckpt["num_layers"],
                  dropout=ckpt.get("dropout", 0.0),
                  beta1=ckpt.get("beta1", 0.9),
                  weight_decay=ckpt.get("weight_decay", 0.0),
                  target_transform=norm.target_transform,
                  device=device if device is not None else "cpu")
        est._norm = norm
        net = _GruNet(est.hidden_size, est.num_layers, est.dropout)
        net.load_state_dict(ckpt["state_dict"])
        est._net = net.to(est.device).eval()
        return est
