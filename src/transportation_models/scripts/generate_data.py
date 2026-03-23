"""
Script for one-time generation of long-format PEMS data for scikit-learn regression
"""

# 3rd party imports
import os

# Local imports
from transportation_models.utils.data_processing import (
    retrieve_detectors,
    load_timeseries_all_detectors,
)
from transportation_models.utils.logs import make_logger


# Variable definitions
root_dir = "/projects/rost5691/data/Caltrans/PeMS"
# root_dir = "/volumes/easystore/work/Boulder/Caltrans/PeMS"
timeseries_directory = os.path.join(root_dir, "timeseries_data")
metadata_filepath = os.path.join(root_dir, "metadata/station_metadata_CALIBRATED.csv")
data_filepath = os.path.join(root_dir, "data_long.csv")
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
logger = make_logger(include_stdout=True)


# Main function
def main():
    # Get a list of detectors to load data for
    logger.info(f"Loading data...")
    detectors = retrieve_detectors(
        metadata_filepath=metadata_filepath,
        timeseries_directory=timeseries_directory,
        detector_type="Mainline",
    )

    # Load the timeseries data for each detector
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


if __name__ == "__main__":
    main()
