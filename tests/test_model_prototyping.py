import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, TensorDataset

from transportation_models.scripts.model_prototyping import (
    denormalize_targets_for_metrics,
    simpleGRU,
    TotalLoss,
    PhysicsLoss,
    run_epoch,
)
from transportation_models.utils.data_processing import (
    PeMSDataProcessor,
    TargetNormalization,
)


# ---- Fixtures ---- #

BATCH = 4
INPUT_SIZE = 17
SEQ_LEN = 12
OUTPUT_STEPS = 36
OUTPUT_SIZE = 2


@pytest.fixture
def model():
    return simpleGRU(
        input_size=INPUT_SIZE,
        hidden_size=64,
        output_steps=OUTPUT_STEPS,
        output_size=OUTPUT_SIZE,
    )


@pytest.fixture
def sample_input():
    return torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)


@pytest.fixture
def station_meta():
    """Station meta with shape [batch, output_steps, 6].
    Columns: [Station ID, Lanes, capacity, critical_density, free_flow_speed, congestion_wave_speed]
    """
    meta = torch.zeros(BATCH, OUTPUT_STEPS, 6)
    meta[:, :, 0] = 1  # station id
    meta[:, :, 1] = 3  # lanes
    meta[:, :, 2] = 2000  # capacity
    meta[:, :, 3] = 45  # critical_density
    meta[:, :, 4] = 60  # free_flow_speed
    meta[:, :, 5] = 15  # congestion_wave_speed
    return meta


@pytest.fixture
def targets():
    """Normalized targets [batch, output_steps, 2] with values near 1."""
    return torch.rand(BATCH, OUTPUT_STEPS, OUTPUT_SIZE) * 0.5 + 0.5


@pytest.fixture
def observed_mask():
    """Mask where roughly half the points are observed (1) and half masked (0)."""
    mask = torch.ones(BATCH, OUTPUT_STEPS)
    mask[:, ::2] = 0  # mask every other timestep
    return mask


# ---- simpleGRU tests ---- #


class TestSimpleGRU:
    def test_output_shape(self, model, sample_input):
        output, hidden = model(sample_input)
        assert output.shape == (BATCH, OUTPUT_STEPS, OUTPUT_SIZE)
        assert hidden.shape == (1, BATCH, model.hidden_size)

    def test_output_shape_with_custom_hidden(self, model, sample_input):
        h0 = torch.zeros(1, BATCH, model.hidden_size)
        output, hidden = model(sample_input, hidden=h0)
        assert output.shape == (BATCH, OUTPUT_STEPS, OUTPUT_SIZE)

    def test_single_sample(self, model):
        x = torch.randn(1, SEQ_LEN, INPUT_SIZE)
        output, hidden = model(x)
        assert output.shape == (1, OUTPUT_STEPS, OUTPUT_SIZE)

    def test_different_hidden_sizes(self):
        for hs in [32, 128, 256]:
            m = simpleGRU(input_size=INPUT_SIZE, hidden_size=hs)
            x = torch.randn(2, SEQ_LEN, INPUT_SIZE)
            out, h = m(x)
            assert h.shape == (1, 2, hs)

    def test_gradients_flow(self, model, sample_input):
        output, _ = model(sample_input)
        loss = output.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


# ---- PhysicsLoss.denormalize_targets tests ---- #


class TestDenormalizeTargets:
    def test_basic_denormalization(self, station_meta):
        targets = torch.ones(BATCH, OUTPUT_STEPS, 2)
        result = PhysicsLoss.denormalize_targets(targets, station_meta)
        # flow * capacity=2000, density * critical_density=45
        assert torch.allclose(result[:, :, 0], torch.full_like(result[:, :, 0], 2000.0))
        assert torch.allclose(result[:, :, 1], torch.full_like(result[:, :, 1], 45.0))

    def test_zeros(self, station_meta):
        targets = torch.zeros(BATCH, OUTPUT_STEPS, 2)
        result = PhysicsLoss.denormalize_targets(targets, station_meta)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_output_shape(self, station_meta, targets):
        result = PhysicsLoss.denormalize_targets(targets, station_meta)
        assert result.shape == targets.shape


# ---- PhysicsLoss tests ---- #


class TestPhysicsLoss:
    def test_output_is_scalar(self, targets, station_meta):
        criterion = PhysicsLoss()
        preds = targets.clone()
        mask = torch.ones(BATCH, OUTPUT_STEPS, dtype=torch.bool)
        loss = criterion(preds, targets, station_meta, mask)
        assert loss.dim() == 0  # scalar

    def test_loss_nonnegative(self, targets, station_meta):
        criterion = PhysicsLoss()
        preds = torch.rand_like(targets)
        mask = torch.ones(BATCH, OUTPUT_STEPS, dtype=torch.bool)
        loss = criterion(preds, targets, station_meta, mask)
        assert loss.item() >= 0

    def test_congested_vs_free_flow_slopes(self, station_meta):
        """When predicted density_norm > 1 (congested), slope should use
        congestion_wave_speed; otherwise free_flow_speed. The congestion regime
        in PhysicsLoss is determined by predictions[:, :, 1], so the two cases
        need predictions in the corresponding regimes to exercise both slopes.
        """
        criterion = PhysicsLoss()
        mask = torch.ones(BATCH, OUTPUT_STEPS, dtype=torch.bool)

        # Targets are the same; only the predicted density regime differs.
        targets = torch.ones(BATCH, OUTPUT_STEPS, 2) * 0.8

        # Congested prediction: density_norm > 1
        preds_congested = torch.ones(BATCH, OUTPUT_STEPS, 2) * 1.2
        preds_congested[:, :, 1] = 1.5
        # Free-flow prediction: density_norm < 1
        preds_free = torch.ones(BATCH, OUTPUT_STEPS, 2) * 1.2
        preds_free[:, :, 1] = 0.5

        loss_c = criterion(preds_congested, targets, station_meta, mask)
        loss_f = criterion(preds_free, targets, station_meta, mask)
        # Both should be nonzero and differ because slopes differ
        assert loss_c.item() > 0
        assert loss_f.item() > 0
        assert not torch.isclose(loss_c, loss_f)

    def test_masked_points_only(self, targets, station_meta):
        """Loss should only consider masked (True) points."""
        criterion = PhysicsLoss()
        preds = targets.clone()
        # Mask where no points are masked -> loss based on 0 elements will error or differ
        all_false = torch.zeros(BATCH, OUTPUT_STEPS, dtype=torch.bool)
        all_true = torch.ones(BATCH, OUTPUT_STEPS, dtype=torch.bool)

        loss_all = criterion(preds, targets, station_meta, all_true)
        # With no masked points, mean of empty tensor gives nan
        loss_none = criterion(preds, targets, station_meta, all_false)
        assert torch.isnan(loss_none) or loss_none.item() == 0
        assert not torch.isnan(loss_all)


# ---- TotalLoss tests ---- #


class TestTotalLoss:
    def test_alpha_zero_is_pure_mse(self, targets, station_meta, observed_mask):
        """alpha=0 should give pure MSE loss."""
        criterion = TotalLoss(alpha=0.0)
        preds = targets + 0.1
        loss = criterion(preds, targets, station_meta, observed_mask)
        # Compute expected MSE manually on masked points
        mask = observed_mask == 0
        expected = ((preds[mask] - targets[mask]) ** 2).mean()
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_alpha_one_is_pure_physics(self, targets, station_meta, observed_mask):
        """alpha=1 should give pure physics loss."""
        criterion = TotalLoss(alpha=1.0)
        preds = targets + 0.1
        loss = criterion(preds, targets, station_meta, observed_mask)

        phys_criterion = PhysicsLoss()
        mask = observed_mask == 0
        expected = phys_criterion(preds, targets, station_meta, mask)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_output_is_scalar(self, targets, station_meta, observed_mask):
        criterion = TotalLoss(alpha=0.5)
        preds = targets + 0.1
        loss = criterion(preds, targets, station_meta, observed_mask)
        assert loss.dim() == 0

    def test_alpha_interpolation(self, targets, station_meta, observed_mask):
        """Loss at alpha=0.5 should be between alpha=0 and alpha=1 (or equal to their average)."""
        preds = targets + torch.randn_like(targets) * 0.1
        loss_0 = TotalLoss(alpha=0.0)(preds, targets, station_meta, observed_mask)
        loss_1 = TotalLoss(alpha=1.0)(preds, targets, station_meta, observed_mask)
        loss_half = TotalLoss(alpha=0.5)(preds, targets, station_meta, observed_mask)
        expected = 0.5 * loss_1 + 0.5 * loss_0
        assert torch.isclose(loss_half, expected, atol=1e-5)


# ---- run_epoch tests ---- #


class TestRunEpoch:
    def _make_loader(self, n_samples=16, batch_size=4):
        """Create a small DataLoader with synthetic data.

        run_epoch extracts the observed_mask from the LAST column of meta
        (``meta_b[:, :, -1]``), so meta needs a trailing "observed" column for
        masked-only reductions to see any elements. Without it, loss and all
        metrics reduce over an empty tensor and become NaN, and the training
        step produces no gradients.
        """
        X = torch.randn(n_samples, OUTPUT_STEPS, INPUT_SIZE)
        # Set last feature column as observed mask (mix of 0s and 1s)
        X[:, :, -1] = (torch.rand(n_samples, OUTPUT_STEPS) > 0.5).float()
        y = torch.rand(n_samples, OUTPUT_STEPS, OUTPUT_SIZE) * 0.5 + 0.5
        # 7 meta columns: 6 station-level values + trailing observed mask
        meta = torch.zeros(n_samples, OUTPUT_STEPS, 7)
        meta[:, :, 2] = 2000  # capacity
        meta[:, :, 3] = 45  # critical_density
        meta[:, :, 4] = 60  # free_flow_speed
        meta[:, :, 5] = 15  # congestion_wave_speed
        # Roughly half observed / half masked so both branches (loss reduction
        # and metric accumulation) see non-empty tensors.
        meta[:, :, 6] = (torch.rand(n_samples, OUTPUT_STEPS) > 0.5).float()
        return DataLoader(TensorDataset(X, y, meta), batch_size=batch_size)

    def test_train_returns_expected_types(self, model):
        loader = self._make_loader()
        criterion = TotalLoss(alpha=0.5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        loss, metrics, preds = run_epoch(
            model, loader, torch.device("cpu"), criterion, optimizer, train=True
        )
        assert isinstance(loss, float)
        assert isinstance(metrics, dict)
        assert preds is None  # no predictions collected during training
        assert "flow_rmse" in metrics
        assert "speed_rmse" in metrics

    def test_eval_returns_predictions(self, model):
        loader = self._make_loader(n_samples=8, batch_size=4)
        criterion = TotalLoss(alpha=0.5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        loss, metrics, preds = run_epoch(
            model, loader, torch.device("cpu"), criterion, optimizer, train=False
        )
        assert preds is not None
        pred_tensor, meta_tensor = preds
        assert pred_tensor.shape == (8, OUTPUT_STEPS, OUTPUT_SIZE)
        assert meta_tensor.shape == (8, OUTPUT_STEPS, 7)

    def test_train_updates_weights(self, model):
        loader = self._make_loader()
        criterion = TotalLoss(alpha=0.5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        weights_before = {k: v.clone() for k, v in model.state_dict().items()}
        run_epoch(model, loader, torch.device("cpu"), criterion, optimizer, train=True)
        changed = any(
            not torch.equal(weights_before[k], v)
            for k, v in model.state_dict().items()
        )
        assert changed, "Model weights should change after a training epoch"

    def test_eval_does_not_update_weights(self, model):
        loader = self._make_loader()
        criterion = TotalLoss(alpha=0.5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        weights_before = {k: v.clone() for k, v in model.state_dict().items()}
        run_epoch(model, loader, torch.device("cpu"), criterion, optimizer, train=False)
        for k, v in model.state_dict().items():
            assert torch.equal(weights_before[k], v), f"Weight {k} changed during eval"

    def test_metrics_nonnegative(self, model):
        loader = self._make_loader()
        criterion = TotalLoss(alpha=0.5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        _, metrics, _ = run_epoch(
            model, loader, torch.device("cpu"), criterion, optimizer, train=True
        )
        for key, val in metrics.items():
            assert val >= 0, f"Metric {key} should be non-negative, got {val}"

    def test_input_size_matches_data(self):
        """Model input_size must match feature count in data passed through run_epoch.

        The last column of X is the observed mask, so the full feature width
        (including the mask) must equal the model's input_size.
        """
        n_features = 17  # 16 real features + 1 mask column
        model = simpleGRU(input_size=n_features, hidden_size=64, output_steps=OUTPUT_STEPS)
        loader = self._make_loader()
        criterion = TotalLoss(alpha=0.5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Should succeed — input_size matches data width
        loss, metrics, _ = run_epoch(
            model, loader, torch.device("cpu"), criterion, optimizer, train=False
        )
        assert isinstance(loss, float)

    def test_input_size_mismatch_raises(self):
        """run_epoch should fail if model input_size doesn't match data feature count."""
        # Model expects 16 features but data has 17 (including mask column)
        model = simpleGRU(input_size=16, hidden_size=64, output_steps=OUTPUT_STEPS)
        loader = self._make_loader()
        criterion = TotalLoss(alpha=0.5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        with pytest.raises(RuntimeError, match="input.size\\(-1\\) must be equal to input_size"):
            run_epoch(
                model, loader, torch.device("cpu"), criterion, optimizer, train=False
            )


# ---- denormalize_targets_for_metrics round-trip tests ---- #


def _make_processor(tmp_path, target_normalization):
    """Build a PeMSDataProcessor against a throwaway root/metadata directory.

    normalize_timeseries doesn't touch self.metadata_df, but __init__ loads it
    from disk, so we drop a minimal CSV in place.
    """
    root = tmp_path / "root"
    (root / "metadata").mkdir(parents=True)
    metadata_path = root / "metadata" / "station_metadata.csv"
    pd.DataFrame(
        {
            "Station ID": [1, 2],
            "Lanes": [3, 4],
            "Type": ["ML", "ML"],
            "Abs PM": [0.1, 0.2],
            "Length": [0.5, 0.6],
        }
    ).to_csv(metadata_path, index=False)
    return PeMSDataProcessor(
        root_directory=str(root),
        metadata_filepath=str(metadata_path),
        target_normalization=target_normalization,
    )


def _make_raw_df(flows, densities, capacity=2000.0, critical_density=45.0):
    """Build the minimum df normalize_timeseries expects."""
    return pd.DataFrame(
        {
            "flow_[veh/hr-lane]": list(flows),
            "density_[veh/mi-lane]": list(densities),
            "capacity": [capacity] * len(flows),
            "critical_density": [critical_density] * len(flows),
        }
    )


class TestDenormalizeForMetricsRoundTrip:
    """Verify that denormalize_targets_for_metrics recovers the raw flow and
    density values that PeMSDataProcessor.normalize_timeseries was given as
    input — for every supported TargetNormalization strategy."""

    def test_station_params_roundtrip(self, tmp_path):
        raw_flows = np.array([0.0, 500.0, 1200.0, 1800.0, 2000.0])
        raw_densities = np.array([0.0, 10.0, 25.0, 45.0, 90.0])
        capacity = 2000.0
        critical_density = 45.0

        processor = _make_processor(tmp_path, TargetNormalization.STATION_PARAMS)
        df = _make_raw_df(raw_flows, raw_densities, capacity, critical_density)
        normalized = processor.normalize_timeseries(df.copy())

        # Shape targets as [batch=1, steps=N, 2] to match the metrics helper.
        targets = torch.tensor(
            normalized[["flow", "density"]].to_numpy(dtype=np.float32)
        ).unsqueeze(0)

        # Build matching station_meta: only capacity and critical_density cols
        # are read by the STATION_PARAMS branch (indices 2 and 3).
        station_meta = torch.zeros(1, len(raw_flows), 6)
        station_meta[:, :, 2] = capacity
        station_meta[:, :, 3] = critical_density

        recovered = denormalize_targets_for_metrics(
            targets,
            station_meta,
            target_normalization=TargetNormalization.STATION_PARAMS,
        )

        expected = torch.tensor(
            np.stack([raw_flows, raw_densities], axis=-1), dtype=torch.float32
        ).unsqueeze(0)
        assert torch.allclose(recovered, expected, atol=1e-4)

    def test_station_params_roundtrip_multi_station(self, tmp_path):
        """Round-trip should still work when capacity / critical_density vary
        across rows (i.e. when the long_df spans multiple VDSs)."""
        raw_flows = np.array([600.0, 1500.0, 900.0, 1800.0])
        raw_densities = np.array([12.0, 40.0, 18.0, 55.0])
        capacities = np.array([1800.0, 2000.0, 1800.0, 2000.0])
        crit_densities = np.array([40.0, 50.0, 40.0, 50.0])

        processor = _make_processor(tmp_path, TargetNormalization.STATION_PARAMS)
        df = pd.DataFrame(
            {
                "flow_[veh/hr-lane]": raw_flows,
                "density_[veh/mi-lane]": raw_densities,
                "capacity": capacities,
                "critical_density": crit_densities,
            }
        )
        normalized = processor.normalize_timeseries(df.copy())

        targets = torch.tensor(
            normalized[["flow", "density"]].to_numpy(dtype=np.float32)
        ).unsqueeze(0)
        station_meta = torch.zeros(1, len(raw_flows), 6)
        station_meta[:, :, 2] = torch.tensor(capacities, dtype=torch.float32)
        station_meta[:, :, 3] = torch.tensor(crit_densities, dtype=torch.float32)

        recovered = denormalize_targets_for_metrics(
            targets,
            station_meta,
            target_normalization=TargetNormalization.STATION_PARAMS,
        )
        expected = torch.tensor(
            np.stack([raw_flows, raw_densities], axis=-1), dtype=torch.float32
        ).unsqueeze(0)
        assert torch.allclose(recovered, expected, atol=1e-4)

    def test_min_max_roundtrip(self, tmp_path):
        raw_flows = np.array([0.0, 500.0, 1200.0, 1800.0, 2200.0])
        raw_densities = np.array([5.0, 20.0, 45.0, 80.0, 150.0])

        processor = _make_processor(tmp_path, TargetNormalization.MIN_MAX)
        df = _make_raw_df(raw_flows, raw_densities)
        normalized = processor.normalize_timeseries(df.copy())

        # Scaler stored on the processor by the MIN_MAX branch.
        scaler = processor.target_minmax_scaler
        assert scaler is not None

        # Sanity check: normalized values really are in [0, 1].
        flow_vals = normalized["flow"].to_numpy()
        dens_vals = normalized["density"].to_numpy()
        assert flow_vals.min() == pytest.approx(0.0)
        assert flow_vals.max() == pytest.approx(1.0)
        assert dens_vals.min() == pytest.approx(0.0)
        assert dens_vals.max() == pytest.approx(1.0)

        targets = torch.tensor(
            normalized[["flow", "density"]].to_numpy(dtype=np.float32)
        ).unsqueeze(0)
        # station_meta is unused by the MIN_MAX branch but still required.
        station_meta = torch.zeros(1, len(raw_flows), 6)

        recovered = denormalize_targets_for_metrics(
            targets,
            station_meta,
            target_normalization=TargetNormalization.MIN_MAX,
            target_minmax_scaler=scaler,
        )
        expected = torch.tensor(
            np.stack([raw_flows, raw_densities], axis=-1), dtype=torch.float32
        ).unsqueeze(0)
        assert torch.allclose(recovered, expected, atol=1e-3)

    def test_min_max_roundtrip_preserves_shape(self, tmp_path):
        """inverse_transform flattens to 2D — make sure we reshape back."""
        batch, steps = 3, 7
        raw = np.random.default_rng(0).uniform(
            low=[0.0, 0.0], high=[2000.0, 100.0], size=(batch * steps, 2)
        )
        raw_flows = raw[:, 0]
        raw_densities = raw[:, 1]

        processor = _make_processor(tmp_path, TargetNormalization.MIN_MAX)
        df = _make_raw_df(raw_flows, raw_densities)
        normalized = processor.normalize_timeseries(df.copy())

        targets = torch.tensor(
            normalized[["flow", "density"]].to_numpy(dtype=np.float32)
        ).reshape(batch, steps, 2)
        station_meta = torch.zeros(batch, steps, 6)

        recovered = denormalize_targets_for_metrics(
            targets,
            station_meta,
            target_normalization=TargetNormalization.MIN_MAX,
            target_minmax_scaler=processor.target_minmax_scaler,
        )
        assert recovered.shape == (batch, steps, 2)
        expected = torch.tensor(
            np.stack([raw_flows, raw_densities], axis=-1), dtype=torch.float32
        ).reshape(batch, steps, 2)
        assert torch.allclose(recovered, expected, atol=1e-3)

    def test_min_max_requires_scaler(self):
        targets = torch.rand(1, 4, 2)
        station_meta = torch.zeros(1, 4, 6)
        with pytest.raises(ValueError, match="target_minmax_scaler"):
            denormalize_targets_for_metrics(
                targets,
                station_meta,
                target_normalization=TargetNormalization.MIN_MAX,
                target_minmax_scaler=None,
            )

    def test_whitened_returns_targets_unchanged(self):
        """Per-sample whitening stats aren't in meta, so the helper is a
        no-op for WHITENED and metrics stay in z-score space."""
        targets = torch.randn(2, 5, 2)
        station_meta = torch.zeros(2, 5, 6)
        recovered = denormalize_targets_for_metrics(
            targets,
            station_meta,
            target_normalization=TargetNormalization.WHITENED,
        )
        assert torch.equal(recovered, targets)
