"""
Script for using a basic ESN to predict traffic flow
"""

# %% Setup
# Imports
import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
import reservoirpy as rpy
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


# Variable definitions
model_type = "EchoStateNetwork_v10"

# root_dir = "/projects/rost5691/data/Caltrans/PeMS"
# root_dir = "/volumes/easystore/work/Boulder/Caltrans/PeMS"
root_dir = "./practice_data"

output_dir = os.path.join(root_dir, "models")
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
output_filepath = os.path.join(output_dir, f"{model_type}_results.csv")

model_filename = f"{model_type}.pkl"
model_filepath = os.path.join(output_dir, model_filename)

input_dir = os.path.join(root_dir, "formatted_data")

log_dir = os.path.join(os.getcwd(), "logs")
now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
log_filename = os.path.join(log_dir, f"ts_pred_{now}.log")
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

summary_filepath = os.path.join(output_dir, f"{model_type}_model_summary.json")

metadata_features = [
    "Station ID",
    "Lanes",
    "Type",
    "County",
    "City",
    "HOV",
    "Abs PM",
    "Length",
    "capacity",
    "free_flow_speed",
    "congestion_wave_speed",
    "jam_density",
    "critical_density",
]
counts_col = "total_flow_[veh/5-min]"

# Set up logging to file and stdout
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# Load training and testing data
X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
logging.info(f"X_train loaded (shape: {X_train.shape}):\n{X_train.head()}")
X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
logging.info(f"X_test loaded (shape: {X_test.shape}):\n{X_test.head()}")
y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))
logging.info(f"y_train loaded (shape: {y_train.shape}):\n{y_train.head()}")
y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv"))
logging.info(f"y_test loaded (shape: {y_test.shape}):\n{y_test.head()}")
y_test_cols = y_test.columns  # Save for constructing y_hat later on

# Convert to numpy arrays for reservoir
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# %% Model Training
# Define parameters
model_params = {
    "seed": 42,
    "reservoir_units": 100,
    "reservoir_lr": 0.2,
    "reservoir_sr": 1.0,
    "ridge_ridge": 1e-7,
    "esn_model_warmup": 10000,
}

# Make everything reproducible!
rpy.set_seed(seed=model_params["seed"])

# Create a reservoir with 100 neurons, a leaking rate of 0.5, and a spectral radius of 0.9
reservoir = rpy.nodes.Reservoir(
    units=model_params["reservoir_units"],
    lr=model_params["reservoir_lr"],
    sr=model_params["reservoir_sr"],
)

# Create a ridge readout to transform the reservoir node activations into a prediction
readout = rpy.nodes.Ridge(ridge=model_params["ridge_ridge"])

# Connect the reservoir to the readout to create an echo state network
esn_model = reservoir >> readout

# Train the ESN
esn_model = esn_model.fit(
    X_train, y_train, warmup=model_params["esn_model_warmup"], workers=-1
)

logging.info(f"{model_type} model training complete")

# %% Model evaluation

# Predict some values
pred = esn_model.run(X_test)

# Calculate the R2 score
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
mape = mean_absolute_percentage_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
logging.info(f"Performance metrics for {model_type}:")
logging.info(f"R2 score: {r2:.3f}")
logging.info(f"Mean absolute error: {mae:.3f}")
logging.info(f"Mean absolute percentage error: {mape:.3f}")
logging.info(f"Mean squared error: {mse:.3f}")

# %% Model persistence
# Prepare a DataFrame of the results
y_hat = pd.DataFrame(pred, columns=y_test_cols)
y_test = pd.DataFrame(y_test, columns=y_test_cols)
results = pd.merge(
    left=y_hat,
    right=y_test,
    left_on=y_hat.index,
    right_on=y_test.index,
    suffixes=["_hat", None],
).drop(columns=["key_0"])

# Save the results
if os.path.exists(output_filepath):
    print(
        f"Attempt to save model results failed. Results already exist at:\n {output_filepath}"
    )
else:
    try:
        results.to_csv(output_filepath, index=False)
        logging.info(f"Results saved to: {output_filepath}")
    except Exception as e:
        logging.error(f"Results could not be saved to {output_filepath}:\n{e}")

# What parameters were used for the model?
preprocessing_params = {"steps": ["OrdinalEncoder", "StandardScaler", "SimpleImputer"]}

# What model was used?
model_params["MODEL TYPE"] = model_type

# How did the model perform?
scores = {
    "R2": r2,
    "MAE": mae,
    "MAPE": mape,
    "MSE": mse,
}

# Where is the model saved?
if os.path.exists(model_filepath):
    print(f"Attempt to save model failed. Filepath already exists:\n {model_filepath}")
else:
    joblib.dump(esn_model, model_filepath)
print(f"Model saved to: {model_filepath}")

# Combine for the summary
summary = {
    "preprocessing": preprocessing_params,
    "regression": model_params,
    "results": scores,
    "model_location": model_filepath,
    "prediction_location": output_filepath,
}

# Write to the log
with open(summary_filepath, "w") as f:
    json.dump(summary, f, indent=4)

print(f"Results saved to: {summary_filepath}")

# %%
results.plot()

# %%
results.iloc[20000 : 20000 + 288 * 3].plot()

# %%
