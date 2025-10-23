"""
Script for one-time generation of long-format PEMS data for scikit-learn regression
"""

# Imports
import os
import sys
import logging
import pandas as pd
from datetime import datetime


# Variable definitions
root_dir = "/projects/rost5691/data/Caltrans/PeMS"
# root_dir = "/volumes/easystore/work/Boulder/Caltrans/PeMS"
timeseries_directory = os.path.join(root_dir, "timeseries_data")
metadata_filepath = os.path.join(root_dir, "metadata/station_metadata_CALIBRATED.csv")
data_filepath = os.path.join(root_dir, "data_long.csv")
log_dir = os.path.join(os.getcwd(), "logs")
now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
log_filename = os.path.join(log_dir, f"data_preparation_{now}.log")
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
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
counts_cols = ["total_flow_[veh/5-min]", "avg_speed_[mph]", "pct_observed"]


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


# Function definitions
def retrieve_mainline_detectors(
    metadata_filepath: str, timeseries_directory: str
) -> list[int]:
    # Load the metadata
    metadata = pd.read_csv(metadata_filepath)

    # Get a set of mainline detectors from the metadata
    detectors_in_metadata = set(
        metadata.loc[metadata["Type"] == "Mainline", "Station ID"].astype(str).to_list()
    )

    # Get a set of detectors from the timeseries directory
    detectors_with_timeseries = set(
        [
            filename.split(".csv")[0]
            for filename in os.listdir(path=timeseries_directory)
            if filename.endswith(".csv")
        ]
    )

    # Cross-reference the two sets to get mainline detectors with metadata and timeseries data
    detectors = list(detectors_in_metadata.intersection(detectors_with_timeseries))

    # Convert to integer IDs
    detectors = [int(detector) for detector in detectors]

    return detectors


def load_timeseries_single_detector(
    detector_timeseries_filepath: str,
    detector_metadata: dict,
    counts_cols: list[str] = [
        "total_flow_[veh/5-min]",
    ],
    full_daterange=pd.date_range(
        start="2022-01-01 00:00:00", end="2024-12-31 23:55:00", freq="5min"
    ),
) -> dict:
    logger.info(f"Loading timeseries for Station ID: {detector_metadata['Station ID']}")
    # Load the data
    df = pd.read_csv(detector_timeseries_filepath, usecols=["timestamp"] + counts_cols)

    # Convert the timestamp to a datetime object
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Reindex to force the data to cover the entire timespan (result will contain NaNs)
    df = (
        df.set_index("timestamp").reindex(full_daterange).reset_index(names="timestamp")
    )

    # Fill missing values with -1
    # NOTE: We can be sure that -1 would never be a real measurement reported in the PeMS data, so
    # it is safe to fill missing values with -1 to indicate that they are missing
    # NOTE #2: You can recover these rows in the following way:
    # df.loc[(df[counts_cols] == -1).all(axis=1), :]
    # NOTE #3: We're removing this step for now because we're including it in the 'prepare_data.py'
    # script instead
    # missing_mask = df.isnull().any(axis=1)
    # df.loc[missing_mask, counts_cols] = -1

    # Assign columns for minute-of-day, hour-of-day, day-of-week
    df["minute"] = df["timestamp"].apply(lambda timestamp: timestamp.minute)
    df["hour"] = df["timestamp"].apply(lambda timestamp: timestamp.hour)
    df["year"] = df["timestamp"].apply(lambda timestamp: timestamp.year)
    df["weekday"] = df["timestamp"].apply(lambda timestamp: timestamp.weekday())

    # Prepare DataFrame for X vector
    df = df.loc[:, ["minute", "hour", "weekday", "year", "timestamp"] + counts_cols]
    for feature, value in detector_metadata.items():
        df[feature] = value

    timeseries_dict = df.to_dict(orient="list")

    return timeseries_dict


def load_timeseries_all_detectors(
    detectors: list[int],
    timeseries_directory: str,
    metadata_filepath: str,
    metadata_features: list[str],
    counts_cols: list[str] = ["total_flow_[veh/5-min]"],
) -> pd.DataFrame:
    # Initialize a list to store the dictionaries
    timeseries_dictionaries = []

    # Load the metadata
    metadata = pd.read_csv(metadata_filepath)

    # Load cleaned timeseries
    for detector in detectors:
        # Construct the complete filepath for the timeseries data
        timeseries_filepath = os.path.join(timeseries_directory, f"{detector}.csv")

        # Extract the metadata
        detector_metadata = (
            metadata.loc[metadata["Station ID"] == detector, metadata_features]
            .reset_index(drop=True)
            .iloc[0]
            .to_dict()
        )
        timeseries_dict = load_timeseries_single_detector(
            detector_timeseries_filepath=timeseries_filepath,
            detector_metadata=detector_metadata,
            counts_cols=counts_cols,
        )
        timeseries_dictionaries.append(timeseries_dict)

    # Build a DataFrame from the dictionaries
    logger.info(
        f"Building consolidated DataFrame with {len(detectors)} unique Station IDs"
    )
    df = pd.DataFrame(timeseries_dictionaries)

    # Explode list-valued columns
    cols_to_explode = (
        counts_cols
        + [
            "minute",
            "hour",
            "year",
            "weekday",
            "timestamp",
        ]
        + metadata_features
    )
    df = df.explode(column=cols_to_explode, ignore_index=True)

    # Return packed dictionary
    return df


logger.info(f"Loading data...")
detectors = retrieve_mainline_detectors(
    metadata_filepath=metadata_filepath, timeseries_directory=timeseries_directory
)
data = load_timeseries_all_detectors(
    detectors=detectors,
    timeseries_directory=timeseries_directory,
    metadata_filepath=metadata_filepath,
    metadata_features=metadata_features,
    counts_cols=counts_cols,
)

logger.info(f"Data loaded (shape: {data.shape}):\n{data.head()}")

try:
    data.to_csv(data_filepath, index=False)
except Exception as e:
    logger.error(f"The following error occurred while saving the DataFrame:\n{e}")
