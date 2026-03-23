"""

Script for one-time generation of training and testing datasets for scikit-learn regression

    GOALS
1) Prepare sets of training and testing data for downstream use in timeseries forecasting / imputation methods

    APPROACH
1) Build a "wide-format" DataFrame containing:
    - A unique DateRange index
    - Multiple prediction target columns, one for each Station ID
    - Multiple feature columns, one set of features for each Station ID (e.g., capacity_Station001, capacity_Station002, etc.)
2) Normalize the prediction target column:
    - Flow should be normalized to vehicles per hour per lane
3) Encode categorical data in the feature columns as numeric
4) Use a simple imputation method for filling missing values in the feature and prediction target columns
5) Scale the feature values
6) Split the data into training and testing sets based on the year (2022-2023 for training, 2024 for testing)
7) Save the sets for future use
"""

#####     Setup     ######

# 3rd party imports
import os
import logging
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

# Local imports
from transportation_models.utils.logs import make_logger

# Define constants
ROOT_DIR = "/projects/rost5691/data/Caltrans/PeMS"
# ROOT_DIR = "/volumes/easystore/work/Boulder/Caltrans/PeMS"
DATA_FILEPATH = os.path.join(ROOT_DIR, "data_long.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "formatted_data")
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "data_wide.csv")

# Set up logging to file and stdout
make_logger(include_stdout=True)

# Classify the features in our data
counts_col = "total_flow_[veh/5-min]"
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
temporal_features = ["minute", "hour", "year", "weekday", "timestamp_epoch"]
X_features = temporal_features + metadata_features
categorical_colnames = ["HOV", "Type", "County", "City"]

# Load the long-format data
logging.info(f"Loading long-format dataframe from: {DATA_FILEPATH}")
data = pd.read_csv(DATA_FILEPATH)
logging.info(f"Long-format data loaded successfully")

# Construct the 'timestamp_epoch' column in our data
data["timestamp_epoch"] = pd.to_datetime(data["timestamp"]).apply(
    lambda x: x.timestamp()
)

#####     Primary processing     #####

# Normalize the prediction target column
periods_per_hour = 12
data["normalized_flow_[vphpl]"] = (
    data["total_flow_[veh/5-min]"] * periods_per_hour / data["Lanes"]
)
data["normalized_density_[vpmpl]"] = (
    data["total_flow_[veh/5-min]"]
    * periods_per_hour
    / (data["avg_speed_[mph]"] * data["Lanes"])
)

# Encode categorical data as numeric
logging.info(
    f"Encoding categorical data as numeric for columns: {categorical_colnames}"
)
enc = OrdinalEncoder()
encoded_data = pd.DataFrame(
    enc.fit_transform(data.loc[:, categorical_colnames]), columns=categorical_colnames
)
data.loc[:, categorical_colnames] = encoded_data

# Fill missing values
logging.info(f"Filling missing values with -1")
cols_to_fill = [
    "capacity",
    "free_flow_speed",
    "congestion_wave_speed",
    "jam_density",
    "critical_density",
    "total_flow_[veh/5-min]",
    "avg_speed_[mph]",
    "pct_observed",
    "normalized_flow_[vphpl]",
    "normalized_density_[vpmpl]",
]
data.loc[:, cols_to_fill] = data.loc[:, cols_to_fill].fillna(value=-1.0)

# Save a list of Station IDs
station_ids = data["Station ID"].unique()

# Prepare for scaling by casting all X_feature columns as float
data[X_features] = data[X_features].astype(float)

# Scale the features
logging.info(f"Scaling feature data")
feature_scaler = StandardScaler()
scaled_feature_data = pd.DataFrame(
    feature_scaler.fit_transform(data.loc[:, X_features]), columns=X_features
)
data.loc[:, X_features] = scaled_feature_data

# Save a list of transformed Station IDs
transformed_station_ids = data["Station ID"].unique()

# Build a lookup from transformed Station ID to original Station ID
station_id_lookup = {
    transformed_id: orig_id
    for transformed_id, orig_id in zip(transformed_station_ids, station_ids)
}

#####     Conversion to wide format     #####
logging.info(f"Converting long-format data to wide-format")

# Pivot to wide format
wide = data.pivot(
    index=["timestamp"] + temporal_features,
    columns="Station ID",
    values=(metadata_features + ["normalized_flow_[vphpl]"]),
)

# Flatten the MultiIndex column names
wide.columns = [
    f"{feat}_{station_id_lookup[station]}" for feat, station in wide.columns
]

# Reset index so temporal features are columns
wide = wide.reset_index()

# Set the timestamp to a datetime object
wide["timestamp"] = pd.to_datetime(wide["timestamp"])

# Save the wide-format data
wide.to_csv(OUTPUT_FILEPATH, index=False)

logging.info(f"Wide-format conversion complete, file saved to: {OUTPUT_FILEPATH}")

#####     Split into training/testing sets     #####
logging.info(f"Generating training and testing datasets")

# Use the timestamp to generate training and testing masks
training_mask = wide["timestamp"].apply(lambda ts: ts.year < 2024)
testing_mask = wide["timestamp"].apply(lambda ts: ts.year >= 2024)
training_vals = wide.loc[training_mask, :].sort_values(by="timestamp")
testing_vals = wide.loc[testing_mask, :].sort_values(by="timestamp")

# Redefine features and targets using new column names for the wide format
X_features = [
    col
    for col in wide.columns
    if ("normalized_flow" not in col) and (col != "timestamp")
]
outputs = [col for col in wide.columns if "normalized_flow" in col]

# Build the X,y vectors for the training and testing sets
X_train = training_vals.loc[:, X_features]
X_test = testing_vals.loc[:, X_features]
y_train = training_vals.loc[:, outputs]
y_test = testing_vals.loc[:, outputs]

logging.info(f"Training and testing dataset generation complete")

#####     Saving     #####

X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

logging.info(f"Training and testing files saved to: {OUTPUT_DIR}")
