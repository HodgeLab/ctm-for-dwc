"""
Script for generating a single long-format DataFrame of PeMS timeseries & metadata.
"""

# ----- Setup ----- #

# 3rd party imports
import pandas as pd

# Local imports
from transportation_models.utils.data_processing import PeMSDataProcessor, TargetNormalization
from transportation_models.utils.logs import make_logger

# Logging setup
logger = make_logger(include_stdout=True)

# Create DataProcessor object
DataProcessor = PeMSDataProcessor(
    root_directory="/projects/rost5691/data/Caltrans/PeMS",
    metadata_filepath="/projects/rost5691/data/Caltrans/PeMS/metadata/station_metadata_calibrated.csv",
    save_transforms=False,
    save_directory="/projects/rost5691/data/Caltrans/PeMS/processed_data/whitened_data",
    target_normalization=TargetNormalization.WHITENED
)

# ----- Generation ----- #
# Process raw timeseries and metadata into DataFrame
df = DataProcessor.build_long_df()
logger.info(f"Loaded dataframe:\n{df.head()}")


# Save DataFrame
filepath = "/projects/rost5691/data/Caltrans/PeMS/processed_data/whitened_data_long.csv"
df.to_csv(filepath, index=False)
logger.info(f"Dataframe saved to: {filepath}")

