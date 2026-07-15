"""Zhang et al. 2024 estimator: GRU backbone + Deep Domain Adaptation + Model
Transfer (doi 10.1109/TITS.2023.3315693).

Staged interface (one *source* stretch with measured ramps, one *target*
stretch whose ramps are treated as unmeasured):

1. ``fit_source(X, r, s)`` -- train the backbone (Fig. 3: per-step Linear
   embedding -> GRU -> FC -> hidden state ``H_t`` -> separate FC heads for
   ``r`` and ``s``) on the source with Adam/MSE; optional early stopping on a
   held-out source-day validation set.
2. ``adapt(X_src, r_src, s_src, X_tgt)`` -- Deep Domain Adaptation: a fresh
   BatchNorm is inserted before ``H_t`` (the paper adds BN only at this stage;
   Fig. 2 vs Fig. 3), the embedding + GRU are frozen, and the FC/BN/heads
   fine-tune on ``source MSE + lambda_mmd * MMD(H_src, H_tgt)`` for a fixed
   number of epochs (no target labels exist, so no early-stop signal). MMD is
   the RKHS norm of Eq. (2) with a Gaussian RBF kernel, bandwidth set per
   batch by the median heuristic (the paper leaves the kernel unspecified).
3. ``calibrate(X_survey, r_survey, s_survey)`` -- Model Transfer (Eq. 5):
   per-ramp scalars ``h = mean(y / y_hat)`` over the survey samples with
   nonzero predictions; predictions are multiplied by ``(h_r, h_s)``.
   Recalibrating overwrites the previous scalars.

Where the paper conflicts with our data or is silent, decisions are recorded
in docs/ramp_flow_estimation_module.md. Feature normalization (unstated in
the paper) is a per-channel z-score fitted on the source training data;
targets are z-scored and clamped non-negative on the inverse. Flows are
veh/hr throughout, matching :mod:`zhang_features`.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .validation import flow_metrics
from .zhang_features import FEATURES_PER_STEP


class ZhangNormalizer:
    """Per-channel z-score for ``(n, T, C)`` sequences and ``(n, 2)`` targets,
    fitted on the source training data only."""

    def __init__(self):
        self._x_mean = self._x_std = None   # (C,)
        self._y_mean = self._y_std = None   # (2,)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "ZhangNormalizer":
        self._x_mean = x.mean(dim=(0, 1))
        x_std = x.std(dim=(0, 1))
        self._x_std = torch.where(x_std > 0, x_std, torch.ones_like(x_std))
        self._y_mean = y.mean(dim=0)
        y_std = y.std(dim=0)
        self._y_std = torch.where(y_std > 0, y_std, torch.ones_like(y_std))
        return self

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._x_mean) / self._x_std

    def targets(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self._y_mean) / self._y_std

    def inverse_targets(self, y: torch.Tensor) -> torch.Tensor:
        return torch.clamp(y * self._y_std + self._y_mean, min=0.0)

    def state_dict(self) -> dict:
        return {"x_mean": self._x_mean, "x_std": self._x_std,
                "y_mean": self._y_mean, "y_std": self._y_std}

    @classmethod
    def from_state_dict(cls, state: dict) -> "ZhangNormalizer":
        norm = cls()
        norm._x_mean, norm._x_std = state["x_mean"], state["x_std"]
        norm._y_mean, norm._y_std = state["y_mean"], state["y_std"]
        return norm


def gaussian_mmd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """MMD between batches ``a`` (n_a, d) and ``b`` (n_b, d): the RKHS norm of
    Eq. (2) -- square root of the biased squared-MMD estimator -- under a
    Gaussian RBF kernel whose bandwidth is the median off-diagonal pairwise
    squared distance of the pooled batch (detached from the graph)."""
    x = torch.cat([a, b], dim=0)
    d2 = torch.cdist(x, x).pow(2)
    n = x.shape[0]
    off = d2[~torch.eye(n, dtype=torch.bool, device=x.device)]
    bandwidth = off.median().detach().clamp_min(1e-12)
    k = torch.exp(-d2 / bandwidth)
    na = a.shape[0]
    mmd2 = k[:na, :na].mean() + k[na:, na:].mean() - 2.0 * k[:na, na:].mean()
    return torch.sqrt(mmd2.clamp_min(0.0) + 1e-12)


class _ZhangNet(nn.Module):
    """Fig. 3 backbone; ``insert_batchnorm`` adds the Fig. 2 BN layer before
    the adaptation layer ``H_t`` when the DDA stage begins."""

    def __init__(self, embed_dim: int, hidden_size: int, adapt_dim: int):
        super().__init__()
        self.embedding = nn.Linear(FEATURES_PER_STEP, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, adapt_dim)
        self.bn = None
        self.fc_r = nn.Linear(adapt_dim, 1)
        self.fc_s = nn.Linear(adapt_dim, 1)

    def insert_batchnorm(self) -> None:
        if self.bn is None:
            self.bn = nn.BatchNorm1d(self.fc.out_features)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """The adaptation layer ``H_t`` (post-BN when present)."""
        out, _ = self.gru(self.embedding(x))
        h = self.fc(out[:, -1, :])
        return self.bn(h) if self.bn is not None else h

    def heads(self, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.fc_r(h), self.fc_s(h)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.heads(self.hidden(x))


class ZhangEstimator:
    """GRU + DDA + MT over Zhang feature windows (``8 * window_size`` columns,
    :mod:`zhang_features` layout). Paper hyperparameters are the defaults:
    Adam lr 1e-4, batch 64, 100 epochs, GRU hidden 64, adaptation layer 256,
    ``lambda_mmd`` 0.01. ``patience`` (backbone early stopping on the source
    validation set) is ours -- the paper uses early stopping but not its
    signal; ``None`` trains all ``max_epochs``. The DDA stage always runs
    ``dda_epochs`` fixed epochs."""

    def __init__(self, *, embed_dim: int = 64, hidden_size: int = 64,
                 adapt_dim: int = 256, lr: float = 1e-4, batch_size: int = 64,
                 max_epochs: int = 100, patience: int | None = 10,
                 lambda_mmd: float = 0.01, dda_epochs: int = 100,
                 seed: int = 0, device=None):
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.adapt_dim = adapt_dim
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lambda_mmd = lambda_mmd
        self.dda_epochs = dda_epochs
        self.seed = seed
        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = None
        self._norm = None
        self._h_mt = None      # (h_r, h_s) Model Transfer scalars

    # ------------------------------------------------------------------ #
    @staticmethod
    def _sequences(X) -> torch.Tensor:
        """Flat ``(n, 8*T)`` rows -> ``(n, T, 8)`` sequences (pure reshape;
        every step carries its own time features)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] == 0 or X.shape[1] % FEATURES_PER_STEP:
            raise ValueError(f"expected (n, {FEATURES_PER_STEP}*T) features, "
                             f"got {X.shape}")
        n, t = X.shape[0], X.shape[1] // FEATURES_PER_STEP
        return torch.from_numpy(X.reshape(n, t, FEATURES_PER_STEP))

    @staticmethod
    def _flows(r, s) -> torch.Tensor:
        return torch.from_numpy(
            np.column_stack([np.asarray(r, dtype=np.float32),
                             np.asarray(s, dtype=np.float32)]))

    def fit_source(self, X, r, s, *, X_val=None, r_val=None, s_val=None,
                   log_fn=None) -> "ZhangEstimator":
        """Train the backbone on the source stretch. ``log_fn(epoch,
        train_loss, val_loss, val_metrics)`` is an optional per-epoch
        callback; ``val_metrics`` is the raw-space
        :func:`validation.flow_metrics` dict (NRMSE/R2/...) on the validation
        set, or None without one."""
        torch.manual_seed(self.seed)
        x_raw, y_raw = self._sequences(X), self._flows(r, s)
        self._norm = ZhangNormalizer().fit(x_raw, y_raw)
        xt = self._norm.features(x_raw).to(self.device)
        yt = self._norm.targets(y_raw).to(self.device)
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            xv = self._norm.features(self._sequences(X_val)).to(self.device)
            yv = self._norm.targets(self._flows(r_val, s_val)).to(self.device)

        net = _ZhangNet(self.embed_dim, self.hidden_size, self.adapt_dim).to(self.device)
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

    def adapt(self, X_src, r_src, s_src, X_tgt, *, X_val=None, r_val=None,
              s_val=None, log_fn=None) -> "ZhangEstimator":
        """Deep Domain Adaptation fine-tune (fixed ``dda_epochs`` epochs).

        ``log_fn(epoch, est_loss, mmd_loss, total_loss, val_metrics)`` is an
        optional per-epoch callback; ``val_metrics`` is the raw-space
        :func:`validation.flow_metrics` dict on ``X_val`` (e.g. the source
        validation days -- no target labels exist to validate on), or None
        without one. Target batches cycle when the target has fewer rows
        than the source; BN running statistics see both streams."""
        if self._net is None:
            raise RuntimeError("call fit_source before adapt")
        torch.manual_seed(self.seed + 1)
        net = self._net
        net.insert_batchnorm()
        net.to(self.device)
        for module in (net.embedding, net.gru):    # freeze the GRU Module
            for p in module.parameters():
                p.requires_grad_(False)
        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.MSELoss()

        xs = self._norm.features(self._sequences(X_src)).to(self.device)
        ys = self._norm.targets(self._flows(r_src, s_src)).to(self.device)
        xt = self._norm.features(self._sequences(X_tgt)).to(self.device)
        if len(xt) < 2:
            raise ValueError("adapt needs >= 2 target rows (BatchNorm batch stats)")
        gen = torch.Generator().manual_seed(self.seed + 1)

        for epoch in range(self.dda_epochs):
            net.train()
            perm_s = torch.randperm(len(xs), generator=gen)
            perm_t = torch.randperm(len(xt), generator=gen)
            sums = np.zeros(3)
            n_batches = 0
            for k, lo in enumerate(range(0, len(xs), self.batch_size)):
                idx_s = perm_s[lo:lo + self.batch_size]
                if len(idx_s) < 2:      # BN train mode needs batch stats
                    continue
                pos = (k * self.batch_size) % len(xt)
                idx_t = perm_t.roll(-pos)[:self.batch_size]
                opt.zero_grad()
                h_s = net.hidden(xs[idx_s])
                h_t = net.hidden(xt[idx_t])
                est = loss_fn(net.heads(h_s), ys[idx_s])
                mmd = gaussian_mmd(h_s, h_t)
                total = est + self.lambda_mmd * mmd
                total.backward()
                opt.step()
                sums += [est.item(), mmd.item(), total.item()]
                n_batches += 1
            if log_fn is not None:
                val_metrics = None
                if X_val is not None and len(X_val) > 0:
                    net.eval()
                    r_hat, s_hat = self._predict_raw(X_val)
                    val_metrics = flow_metrics(np.asarray(r_val, float),
                                               np.asarray(s_val, float),
                                               r_hat, s_hat)
                log_fn(epoch, *(sums / n_batches), val_metrics)

        self._net = net.eval()
        return self

    def calibrate(self, X_survey, r_survey, s_survey) -> tuple[float, float]:
        """Model Transfer: fit and store ``(h_r, h_s)`` from Traffic Survey
        samples (Eq. 5, mean of true/estimated over nonzero estimates; a ramp
        with no nonzero estimates keeps ``h = 1``). Returns the scalars."""
        r_hat, s_hat = self._predict_raw(X_survey)
        scalars = []
        for y, y_hat in ((np.asarray(r_survey, float), r_hat),
                         (np.asarray(s_survey, float), s_hat)):
            nz = y_hat > 0
            scalars.append(float(np.mean(y[nz] / y_hat[nz])) if nz.any() else 1.0)
        self._h_mt = tuple(scalars)
        return self._h_mt

    def pair_mmd(self, X_a, X_b, *, max_samples: int = 2048, seed: int = 0) -> float:
        """MMD distance between two stretches at the Adaptation Layer -- the
        per-pair diagnostic the paper reports (Tables II/IV/VI). Per Eq. 2 and
        Section III-D the MMD is defined over the Adaptation Layer hidden
        representations ``H_t`` after the GRU module, so this requires a
        trained backbone; the same Gaussian-RBF median-heuristic estimator as
        the DDA loss is used. A seeded subsample of ``max_samples`` rows per
        stretch keeps the O(n^2) kernel matrix tractable. Computed with the
        current network state: after ``fit_source`` it is the pre-adaptation
        distance, after ``adapt`` the post-DDA (BN running-stats) distance."""
        if self._net is None:
            raise RuntimeError("estimator is not fitted")
        rng = np.random.default_rng(seed)

        def hidden(X):
            X = np.asarray(X)
            if len(X) > max_samples:
                X = X[rng.choice(len(X), size=max_samples, replace=False)]
            with torch.no_grad():
                xn = self._norm.features(self._sequences(X)).to(self.device)
                return self._net.hidden(xn)
        return float(gaussian_mmd(hidden(X_a), hidden(X_b)))

    def _predict_raw(self, X):
        if self._net is None:
            raise RuntimeError("estimator is not fitted")
        with torch.no_grad():
            xn = self._norm.features(self._sequences(X)).to(self.device)
            y = self._norm.inverse_targets(self._net(xn).cpu()).numpy()
        return y[:, 0], y[:, 1]

    def predict_flows(self, X):
        """Predicted ``(r_hat, s_hat)`` in veh/hr, non-negative; scaled by the
        Model Transfer scalars when :meth:`calibrate` has run."""
        r_hat, s_hat = self._predict_raw(X)
        if self._h_mt is not None:
            r_hat, s_hat = r_hat * self._h_mt[0], s_hat * self._h_mt[1]
        return r_hat, s_hat

    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        """Checkpoint the fitted model + normalizer + MT scalars."""
        if self._net is None:
            raise RuntimeError("estimator is not fitted")
        torch.save({
            "embed_dim": self.embed_dim, "hidden_size": self.hidden_size,
            "adapt_dim": self.adapt_dim, "has_bn": self._net.bn is not None,
            "h_mt": self._h_mt,
            "normalizer": self._norm.state_dict(),
            "state_dict": self._net.state_dict(),
        }, path)

    @classmethod
    def load(cls, path, *, device=None) -> "ZhangEstimator":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        est = cls(embed_dim=ckpt["embed_dim"], hidden_size=ckpt["hidden_size"],
                  adapt_dim=ckpt["adapt_dim"],
                  device=device if device is not None else "cpu")
        est._norm = ZhangNormalizer.from_state_dict(ckpt["normalizer"])
        est._h_mt = ckpt["h_mt"]
        net = _ZhangNet(est.embed_dim, est.hidden_size, est.adapt_dim)
        if ckpt["has_bn"]:
            net.insert_batchnorm()
        net.load_state_dict(ckpt["state_dict"])
        est._net = net.to(est.device).eval()
        return est
