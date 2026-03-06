#%% #----- Setup -----#

# 3rd party imports
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

#%% #----- Data loading -----#

# Load dataset
data_filepath = "/projects/rost5691/data/Caltrans/PeMS/processed_data/hour_lookback_15min_horizon.pt"
data = torch.load(data_filepath)
logger.info("Loaded raw data from filepath: ", data_filepath)
X,y = data['X'], data['y']
dataset = TensorDataset(X.float(),y.float())
logger.info(f"Loaded dataset with length: {len(dataset)}")

# Create training, validation, and testing splits
splits = [0.6, 0.2, 0.2]
train_set, val_set, test_set = random_split(dataset, lengths=splits)
logger.info(f"Dataset split into training, validation, and testing sets with lengths {len(train_set)}, {len(val_set)}, and {len(test_set)}")
example_x, example_y = next(iter(train_set))
logger.info(f"Samples in train set have shape {example_x.shape}")
logger.info(f"Targets in train set have shape {example_y.shape}")

#%% #----- Model & function definitions -----#
# Basic GRU model
class simpleGRU(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, output_steps=3, output_size=2):
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
        out = out.view(batch_size, self.output_steps, self.output_size)  # [batch, 3, 2]

        return out, hidden

# Define a function to save models
def save_model(
    model: nn.Module,
    name: str,
    root_path: str = os.getcwd(),
) -> str:
    # Create a directory for models if it doesn't yet exist
    if not os.path.exists(os.path.join(root_path, "models")):
        os.mkdir(os.path.join(root_path, "models"))

    # Save the model to the directory
    filepath = os.path.join(root_path, "models", f"{name}.pt")
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to: {filepath}")

    # Return the filepath
    return filepath

# Define a function to evaluate a single epoch
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    train: bool = True,
) -> float:
    # Make sure we're on the correct device
    model = model.to(device)

    # Set the model mode – either training or evaluation
    if train:
        model.train()
    else:
        model.eval()

    # Initial values for tracking loss and accuracy over the epoch
    running_loss = 0.0

    # Set context based on model mode
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        # Iterate through the batches in the loader
        for xb, yb in loader:
            # Transfer to device
            xb, yb = xb.to(device), yb.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # -- Forward pass -- #
            # Get model predictions
            output, hidden = model(xb)

            # Evaluate loss
            loss = loss_fn(output, yb)

            # -- Backward pass -- #
            if train:
                # Update gradients
                loss.backward()

                # Adjust learning weights
                optimizer.step()

            # Update running counts for loss and accuracy
            running_loss += loss.item()

    # Calculate average batch loss
    running_loss = running_loss / len(loader)

    # Return loss and accuracy for this epoch
    return running_loss

# Define function for training
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    n_epochs: int = 1500,
    early_stopping: bool = True,
    patience: int = 100,
    min_delta: float = 1e-2,
    wandb_config: dict = {},
    wandb_tags: list[str] = [],
    wandb_notes: str = "",
    model_name: str = "baseline",
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
        for epoch in range(n_epochs):
            # Training over epoch
            train_loss = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                loss_fn=loss_fn,
                optimizer=optimizer,
                train=True,
            )

            # Validation over epoch
            val_loss = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                loss_fn=loss_fn,
                optimizer=optimizer,
                train=False,
            )

            # Save losses for this epoch
            run.log(
                {
                    "training_loss": train_loss,
                    "validation_loss": val_loss,
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

        # Upload the best model as an artifact
        model_filepath = save_model(model=model, name=model_name)
        run.log_artifact(model_filepath, name="trained-model", type="model")

    return

#%% #----- Model training -----#

# Define hyperparameters
batch_sizes = [1024, 4096, 8192]
#batch_sizes = [1024]
learning_rates = [0.01, 0.001]
#learning_rates = [0.01]
momentum_rates = [0.9, 0.75]
#momentum_rates = [0.9]
n_epochs = 20

for batch_size in batch_sizes:
    # Set up DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    for lr in learning_rates:
        for alpha in momentum_rates:
            if (batch_size == 1024) and (lr == 0.01) and (alpha == 0.9):
                continue
            # Define model, loss function, and optimizer
            model = simpleGRU()
            loss_fn = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=alpha)

            # Train model
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                loss_fn=loss_fn,
                optimizer=optimizer,
                n_epochs=n_epochs,
                early_stopping=False,
                wandb_config = {
                    "model": "simpleGRU",
                    "loss_function": "MSE",
                    "optimizer": "SGD",
                    "learning_rate":lr,
                    "momentum_rate":alpha,
                },
                wandb_tags=["prototyping"],
                model_name="simpleGRU"
                )