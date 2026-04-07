# ----- Setup -----#

# 3rd party imports
import argparse
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split, TensorDataset

# Local imports
from transportation_models.utils.logs import make_logger

# WandB setup
wandb_entity="transportation-models"
wandb_project_name="PeMS-Imputation"
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Logging setup
logger = make_logger(include_stdout=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Model & function definitions -----#
# Basic GRU model
class simpleGRU(nn.Module):

    def __init__(self, input_size=17, hidden_size=128, output_steps=36, output_size=2):
        super(simpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_steps = output_steps
        self.output_size = output_size

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_steps * output_size)

    def forward(self, x, hidden=None):
        # x: [batch, 12, 16]
        batch_size = x.size(0)

        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_size, device=x.device)

        gru_out, hidden = self.gru(x, hidden)       # [batch, 12, hidden]
        last_step = gru_out[:, -1, :]               # [batch, hidden]

        out = self.fc(last_step)                    # [batch, 6]
        out = out.view(
            batch_size, self.output_steps, self.output_size
        )  # [batch, 36, 2]

        return out, hidden


class TotalLoss(nn.Module):

    def __init__(self, alpha: float) -> None:
        """
        init function for class

        Parameters
        ----------
        alpha : float
            Relative weight of physics-based loss in the total loss.
            Must be between 0 and 1.
            0 indicates complete MSE loss.
            1 indicates complete Physics loss.
        """
        super(TotalLoss, self).__init__()
        self.phys_criterion = PhysicsLoss()
        self.alpha = alpha

    def forward(
        self,
        predictions: torch.tensor,
        targets: torch.tensor,
        station_meta: torch.tensor,
        observed_mask: torch.tensor,
    ) -> torch.tensor:
        """
        Calculate total loss as a weighted combination of MSE loss and physics-based loss.
        Parameter 'alpha' determines weight contribution of physics-based loss.
        Loss is only calculated for masked (unobserved) data points.

        Parameters
        ----------
        predictions : torch.tensor
            Predicted output.
            Shape [batch, output_steps, 2], where dim -1 is [flow_norm, density_norm].
        targets : torch.tensor
            Actual output.
            Shape [batch, output_steps, 2], where dim -1 is [flow_norm, density_norm].
        station_meta : torch.tensor
            Station-specific metadata.
            Shape [batch, output_steps, 6], where dim -1 is
            [Station ID, Lanes, capacity, critical_density, free_flow_speed, congestion_wave_speed].
            These are the raw (unscaled) station-level values used during preprocessing.
        observed_mask : torch.tensor
            Binary mask indicating observed (1) vs masked (0) data points.
            Shape [batch, output_steps].

        Returns
        -------
        torch.tensor
            Total loss. Calculated as:
            alpha * physics_loss + (1 - alpha) * mse_loss
            Scalar value.
        """
        mask = observed_mask == 0  # [batch, output_steps] — True where masked

        physics_loss = self.phys_criterion(predictions, targets, station_meta, mask)
        mse_loss = ((predictions[mask] - targets[mask]) ** 2).mean()
        total_loss = self.alpha * physics_loss + (1 - self.alpha) * mse_loss
        return total_loss


class PhysicsLoss(nn.Module):
    def __init__(self):
        super(PhysicsLoss, self).__init__()

    def forward(
        self,
        predictions: torch.tensor,
        targets: torch.tensor,
        station_meta: torch.tensor,
        mask: torch.tensor,
    ) -> torch.tensor:
        """
        Calculate physics-informed loss according to fundamental diagram.
        Loss is only calculated for masked (unobserved) data points.

        Follow these steps:

            1) Determine the appropriate slope to use:

                If the actual density is greater than 1, use a slope m = -1 * congestion_wave_speed

                If the actual density is less than 1, use a slope m = free_flow_speed

            2) Calculate physical flow and density using denormalize_targets.

            3) Determine the expected flow according to this equation:

                flow_expected = m * (density_phys - critical_density) + capacity

            4) Calculate loss as:

                loss_phys = mean((flow_expected - flow_phys) ** 2)

        Parameters
        ----------
        predictions : torch.tensor
            Predicted output.
            Shape [batch, output_steps, 2], where dim -1 is [flow_norm, density_norm].
        targets : torch.tensor
            Actual output.
            Shape [batch, output_steps, 2], where dim -1 is [flow_norm, density_norm].
        station_meta : torch.tensor
            Station-specific metadata.
            Shape [batch, output_steps, 6], where dim -1 is
            [Station ID, Lanes, capacity, critical_density, free_flow_speed, congestion_wave_speed].
            These are the raw (unscaled) station-level values used during preprocessing.
        mask : torch.tensor
            Boolean mask. True where data points are masked (unobserved).
            Shape [batch, output_steps].

        Returns
        -------
        torch.tensor
            Physics-based loss according to fundamental diagram.
            Scalar value.
        """
        # --- Step 1: Determine appropriate slopes to use ---
        # Determine indices of prediction that correspond to congested traffic conditions
        is_congested = targets[:, :, 1] > 1

        # Calculate appropriate slopes according to congestion mask
        slopes = (
            station_meta[:, :, 4] * ~is_congested
            + station_meta[:, :, 5] * is_congested * -1
        )

        # --- Step 2: Denormalize targets
        predictions_phys = PhysicsLoss.denormalize_targets(predictions, station_meta)

        # --- Step 3: Calculate expected flow ---
        critical_density = station_meta[:, :, 3]
        capacity = station_meta[:, :, 2]
        predicted_flow_phys = predictions_phys[:, :, 0]
        predicted_density_phys = predictions_phys[:, :, 1]
        flow_expected = slopes * (predicted_density_phys - critical_density) + capacity

        # --- Step 4: Calculate physics loss (masked only) ---
        physics_loss = ((flow_expected[mask] - predicted_flow_phys[mask]) ** 2).mean()

        return physics_loss

    # Define a helper function to denormalize the targets
    @staticmethod
    def denormalize_targets(
        targets: torch.Tensor,
        station_meta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reverse the flow and density normalizations applied during preprocessing,
        converting model outputs back to interpretable physical units:
            - flow:            normalized → [veh/hr-lane]
            - density:         normalized → [veh/mi-lane]

        Preprocessing applied (in order):
            flow_norm    = (total_flow_veh_per_5min * 12 / lanes) / capacity
            density_norm = (flow_veh_hr_lane / avg_speed_mph) / critical_density

        Inverse (applied here):
            flow_real    = flow_norm * capacity
            density_real = density_norm * critical_density

        Parameters
        ----------
        targets : torch.Tensor
            Shape [batch, output_steps, 2], where dim -1 is [flow_norm, density_norm].
        station_meta : torch.Tensor
            Shape [batch, output_steps, 6], where dim -1 is
            [Station ID, Lanes, capacity, critical_density, free_flow_speed, congestion_wave_speed].
            These are the raw (unscaled) station-level values used during preprocessing.

        Returns
        -------
        torch.Tensor
            Shape [batch, output_steps, 2], with flow in [veh/hr-lane] and
            density in [veh/mi-lane].
        """
        # Select columns for capacity and critical density
        scale = station_meta[:, :, [2, 3]]  # [batch, output_steps, 2]

        return targets * scale  # [batch, output_steps, 2]


# Define a function to evaluate a single epoch
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train: bool = True,
) -> tuple[float, dict[str, float], tuple[torch.Tensor, torch.Tensor] | None]:
    # Make sure we're on the correct device
    model = model.to(device)

    # Set the model mode – either training or evaluation
    if train:
        model.train()
    else:
        model.eval()

    # Initial values for tracking loss, accuracy, and predictions
    running_loss = 0.0
    running_sq_err = torch.zeros(2)  # one accumulator per output column (flow, speed)
    running_abs_err = torch.zeros(2)
    running_abs_pct_err = torch.zeros(2)
    running_n = 0  # total number of (sample, timestep) pairs seen
    all_preds = [] if not train else None
    all_metas = [] if not train else None

    # Set context based on model mode
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        # Iterate through the batches in the loader
        for xb, yb, meta_b in loader:
            # Transfer to device
            xb, yb, meta_b = xb.to(device), yb.to(device), meta_b.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # -- Forward pass -- #
            # Extract observed mask from last feature column
            observed_mask = xb[:, :, -1]  # [batch, seq_len], 1=observed, 0=masked

            # Get model predictions
            output, hidden = model(xb)

            # Evaluate loss (only on masked data points)
            loss = criterion(output, yb, meta_b, observed_mask)

            # -- Backward pass -- #
            if train:
                # Update gradients
                loss.backward()

                # Adjust learning weights
                optimizer.step()
            else:
                # Collect predictions from validation batches
                all_preds.append(output.cpu())
                all_metas.append(meta_b.cpu())

            # Update running loss
            running_loss += loss.item()

            # --- Physical-space RMSE (masked points only) --- #
            mask = observed_mask == 0  # [batch, seq_len] — True where masked

            # Reverse the per-station normalization
            output_phys = PhysicsLoss.denormalize_targets(
                output, meta_b
            )  # [batch, output_steps, 2]
            yb_phys = PhysicsLoss.denormalize_targets(yb, meta_b)

            # Calculate speed (flow / density)
            output_phys[:, :, 1] = output_phys[:, :, 0] / output_phys[:, :, 1]
            yb_phys[:, :, 1] = yb_phys[:, :, 0] / yb_phys[:, :, 1]

            # Calculate errors only at masked positions
            diff = output_phys - yb_phys  # [batch, output_steps, 2]
            mask_3d = mask.unsqueeze(-1).expand_as(diff)  # [batch, output_steps, 2]
            running_sq_err += (diff ** 2).masked_fill(~mask_3d, 0).sum(dim=(0, 1)).cpu()
            running_abs_err += diff.abs().masked_fill(~mask_3d, 0).sum(dim=(0, 1)).cpu()
            running_abs_pct_err += (diff.abs() / yb_phys.abs().clamp(min=1e-6)).masked_fill(~mask_3d, 0).sum(dim=(0, 1)).cpu()

            running_n += mask.sum().item()  # count of masked timesteps

    # Calculate average batch loss
    running_loss = running_loss / len(loader)

    # Calculate per-column metrics across all samples and timesteps
    mse = (running_sq_err / running_n).tolist()
    rmse = (running_sq_err / running_n).sqrt().tolist()
    mae = (running_abs_err / running_n).tolist()
    mape = (running_abs_pct_err / running_n * 100).tolist()

    metrics = {
        "flow_mae": mae[0],
        "flow_mse": mse[0],
        "flow_rmse": rmse[0],
        "flow_mape": mape[0],
        "speed_mae": mae[1],
        "speed_mse": mse[1],
        "speed_rmse": rmse[1],
        "speed_mape": mape[1],
    }

    # Concatenate all validation predictions and metadata into their own tensors
    if train:
        preds_with_meta = None
    else:
        preds_with_meta = (torch.cat(all_preds, dim=0), torch.cat(all_metas, dim=0))

    # Return loss and metrics for this epoch
    return running_loss, metrics, preds_with_meta


# Define function for training
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    n_epochs: int = 1500,
    early_stopping: bool = True,
    patience: int = 100,
    min_delta: float = 1e-2,
    wandb_config: dict = {},
    wandb_tags: list[str] = [],
    wandb_notes: str = "",
    model_name: str = "baseline",
    results_dir: str = os.getcwd(),
) -> None:
    # Early stopping setup
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    no_improvement_count = 0

    # W&B setup
    wandb_config["epochs"] = n_epochs
    wandb_config["batch_size"] = train_loader.batch_size
    wandb_config["early_stopping"] = early_stopping
    wandb_config["patience"] = patience
    wandb_config["min_delta"] = min_delta

    # Start training
    with wandb.init(
        entity=wandb_entity,
        project=wandb_project_name,
        notes=wandb_notes,
        tags=wandb_tags,
        config=wandb_config,
    ) as run:
        # Create run directory using the W&B run name
        run_dir = os.path.join(results_dir, run.name)
        val_preds_dir = os.path.join(run_dir, "val_predictions")
        os.makedirs(val_preds_dir, exist_ok=True)
        logger.info(f"Run results will be saved to: {run_dir}")

        for epoch in range(n_epochs):
            # Training over epoch
            train_loss, train_metrics, _ = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                train=True,
            )

            # Validation over epoch
            val_loss, val_metrics, val_preds_and_meta = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                train=False,
            )

            # Save validation predictions for this epoch
            if val_preds_dir is not None and val_preds_and_meta is not None:
                preds_filepath = os.path.join(val_preds_dir, f"epoch_{epoch:04d}.pt")
                torch.save(
                    {"preds": val_preds_and_meta[0], "meta": val_preds_and_meta[1]},
                    preds_filepath,
                )

            # Save losses and metrics for this epoch
            run.log(
                {
                    "training_loss": train_loss,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    "validation_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )

            # Early stopping logic
            if early_stopping:
                # Improvement means val_loss got smaller by at least min_delta
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        break

        # Reload best model
        if early_stopping:
            model.load_state_dict(best_model_weights)

        # Save the best model and upload as an artifact
        model_filepath = os.path.join(run_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), model_filepath)
        logger.info(f"Model saved to: {model_filepath}")
        run.log_artifact(model_filepath, name="trained-model", type="model")

    return


# ----- Model training -----#
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Prototyping script for training a neural network to impute missing traffic data"
    )
    parser.add_argument(
        "--PhysWeight",
        type=float,
        default=0.5,
        required=True,
        help="Alpha parameter for loss function: relative weight of physics loss to MSE loss. 0 for pure MSE, 1 for pure physics",
    )
    args = parser.parse_args()

    # Path definitions
    results_dir = "/projects/rost5691/data/Caltrans/PeMS/results"
    data_filepath = "/projects/rost5691/data/Caltrans/PeMS/processed_data/3hr_10pct_imputation/imputation_dataset.pt"

    # Load dataset.
    # Expected keys: "X" (features), "y" (normalized targets), "meta" (per-sample station scalars).
    # "meta" should be a float tensor of shape [N, 6] with these columns:
    # [Station ID, Lanes, capacity, critical_density, free_flow_speed, congestion_wave_speed]
    data = torch.load(data_filepath)
    logger.info("Loaded raw data from filepath: ", data_filepath)
    X, y, meta = data["X"], data["y"], data["meta"]
    dataset = TensorDataset(X.float(), y.float(), meta.float())
    logger.info(f"Loaded dataset with length: {len(dataset)}")

    # Create training, validation, and testing splits
    splits = [0.6, 0.2, 0.2]
    train_set, val_set, test_set = random_split(
        dataset, lengths=splits, generator=torch.Generator().manual_seed(42)
    )
    logger.info(
        f"Dataset split into training, validation, and testing sets with lengths {len(train_set)}, {len(val_set)}, and {len(test_set)}"
    )
    example_x, example_y, example_meta = next(iter(train_set))
    logger.info(f"Samples in train set have shape {example_x.shape}")
    logger.info(f"Targets in train set have shape {example_y.shape}")
    logger.info(f"Station meta in train set has shape {example_meta.shape}")

    # Define hyperparameters
    batch_sizes = [32]
    learning_rates = [0.01]
    momentum_rates = [0.9]
    n_epochs = 20

    for batch_size in batch_sizes:
        # Set up DataLoaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        for lr in learning_rates:
            for alpha in momentum_rates:
                # Define model, loss function, and optimizer
                model = simpleGRU(output_steps=36)
                criterion = TotalLoss(alpha=args.PhysWeight)
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=alpha)

                # Train model
                train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    n_epochs=n_epochs,
                    early_stopping=False,
                    wandb_config={
                        "model": "simpleGRU",
                        "loss_function": "Total",
                        "PhysWeight": args.PhysWeight,
                        "optimizer": "SGD",
                        "learning_rate": lr,
                        "momentum_rate": alpha,
                    },
                    wandb_tags=["prototyping", "imputation"],
                    model_name="simpleGRU",
                    results_dir=results_dir,
                )
