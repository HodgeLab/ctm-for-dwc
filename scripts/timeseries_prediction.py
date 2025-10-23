"""
Script for using basic scikit-learn regression methods
"""

#####     Setup     ######
# Imports
import os
import json
import joblib
import logging
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)  # Don't forget to redefine model type elsewhere!

# Variable definitions
model_type = "HistGradientBoostingRegressor_basic"

root_dir = "/projects/rost5691/data/Caltrans/PeMS"
# root_dir = "/volumes/easystore/work/Boulder/Caltrans/PeMS"

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

#####     Model training     #####
# Fit the regression
regr = HistGradientBoostingRegressor()
regr.fit(X_train, y_train)

logging.info(f"{model_type} model training complete")

#####     Model evaluation     #####
# Predict some values
pred = regr.predict(X_test)

# Calculate the R2 score
r2_score = regr.score(X_test, y_test)
mae = mean_absolute_error(y_test, pred)
mape = mean_absolute_percentage_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
logging.info(f"Performance metrics for {model_type}:")
logging.info(f"R2 score: {r2_score:.3f}")
logging.info(f"Mean absolute error: {mae:.3f}")
logging.info(f"Mean absolute percentage error: {mape:.3f}")
logging.info(f"Mean squared error: {mse:.3f}")

#####     Model persistence     #####
# Prepare a DataFrame of the results
y_hat = pd.DataFrame(pred, columns=y_test.columns)
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
model_params = regr.get_params()

# What model was used?
model_params["MODEL TYPE"] = model_type

# How did the model perform?
results = {
    "R2": r2_score,
    "MAE": mae,
    "MAPE": mape,
    "MSE": mse,
}

# Where is the model saved?
if os.path.exists(model_filepath):
    print(f"Attempt to save model failed. Filepath already exists:\n {model_filepath}")
else:
    joblib.dump(regr, model_filepath)
print(f"Model saved to: {model_filepath}")

# Combine for the summary
summary = {
    "preprocessing": preprocessing_params,
    "regression": model_params,
    "results": results,
    "model_location": model_filepath,
    "prediction_location": output_filepath,
}

# Write to the log
with open(summary_filepath, "w") as f:
    json.dump(summary, f, indent=4)

print(f"Results saved to: {summary_filepath}")
