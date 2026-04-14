"""
Script for using basic scikit-learn regression methods
"""

#####     Setup     ######
# 3rd party imports
import joblib
import json
import logging
import os
import pandas as pd
from datetime import datetime
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)  # Don't forget to redefine model type elsewhere!
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

# Local imports
from transportation_models.utils.logs import make_logger

# Set up logging to file and stdout
logger = make_logger(include_stdout=True)

# Define constants
MODEL_TYPE = "HistGradientBoostingRegressor_basic"
ROOT_DIR = "/projects/rost5691/data/Caltrans/PeMS"
INPUT_DIR = os.path.join(ROOT_DIR, "formatted_data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "models")
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_results.csv")
MODEL_FILEPATH = os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}.pkl")
SUMMARY_FILEPATH = os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_model_summary.json")

# Load training and testing data
X_train = pd.read_csv(os.path.join(INPUT_DIR, "X_train.csv"))
logging.info(f"X_train loaded (shape: {X_train.shape}):\n{X_train.head()}")
X_test = pd.read_csv(os.path.join(INPUT_DIR, "X_test.csv"))
logging.info(f"X_test loaded (shape: {X_test.shape}):\n{X_test.head()}")
y_train = pd.read_csv(os.path.join(INPUT_DIR, "y_train.csv"))
logging.info(f"y_train loaded (shape: {y_train.shape}):\n{y_train.head()}")
y_test = pd.read_csv(os.path.join(INPUT_DIR, "y_test.csv"))
logging.info(f"y_test loaded (shape: {y_test.shape}):\n{y_test.head()}")

#####     Model training     #####
# Fit the regression
regr = HistGradientBoostingRegressor()
regr.fit(X_train, y_train)

logging.info(f"{MODEL_TYPE} model training complete")

#####     Model evaluation     #####
# Predict some values
pred = regr.predict(X_test)

# Calculate the R2 score
r2_score = regr.score(X_test, y_test)
mae = mean_absolute_error(y_test, pred)
mape = mean_absolute_percentage_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
logging.info(f"Performance metrics for {MODEL_TYPE}:")
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
if os.path.exists(OUTPUT_FILEPATH):
    print(
        f"Attempt to save model results failed. Results already exist at:\n {OUTPUT_FILEPATH}"
    )
else:
    try:
        results.to_csv(OUTPUT_FILEPATH, index=False)
        logging.info(f"Results saved to: {OUTPUT_FILEPATH}")
    except Exception as e:
        logging.error(f"Results could not be saved to {OUTPUT_FILEPATH}:\n{e}")

# What parameters were used for the model?
preprocessing_params = {"steps": ["OrdinalEncoder", "StandardScaler", "SimpleImputer"]}
model_params = regr.get_params()

# What model was used?
model_params["MODEL TYPE"] = MODEL_TYPE

# How did the model perform?
results = {
    "R2": r2_score,
    "MAE": mae,
    "MAPE": mape,
    "MSE": mse,
}

# Where is the model saved?
if os.path.exists(MODEL_FILEPATH):
    print(f"Attempt to save model failed. Filepath already exists:\n {MODEL_FILEPATH}")
else:
    joblib.dump(regr, MODEL_FILEPATH)
print(f"Model saved to: {MODEL_FILEPATH}")

# Combine for the summary
summary = {
    "preprocessing": preprocessing_params,
    "regression": model_params,
    "results": results,
    "model_location": MODEL_FILEPATH,
    "prediction_location": OUTPUT_FILEPATH,
}

# Write to the log
with open(SUMMARY_FILEPATH, "w") as f:
    json.dump(summary, f, indent=4)

print(f"Results saved to: {SUMMARY_FILEPATH}")
