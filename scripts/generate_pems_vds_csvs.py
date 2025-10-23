"""
Script for one-time extraction of Caltrans PeMS VDS timeseries data into a CSV file.
NOTE: For extracting many CSVs, this script can take a significant amount of time to run.
"""

import pandas as pd
from utils.data_downloading import DataProcessor


# Initialize the object
zipfile_directory = (
    "/Users/robstallman/Library/CloudStorage/OneDrive-UCB-O365/PeMS_Data"
)

detectors = [430136]

data_processor = DataProcessor(zipfile_directory, detectors=detectors)

data_processor.process_gz_files()
