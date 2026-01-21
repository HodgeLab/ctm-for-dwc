"""
Script for downloading timeseries transportation data from Caltrans PeMS database
"""

# %%
# Import 3rd party libraries
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Local imports
from transportation_models.utils.data_downloading import PeMSDownloader

# Load login credentials
load_dotenv("credentials.env")
PEMS_USERNAME = os.environ["PEMS_USERNAME"]
PEMS_PASSWORD = os.environ["PEMS_PASSWORD"]


# Connect to PeMS
pems = PeMSDownloader(username=PEMS_USERNAME, password=PEMS_PASSWORD)

# %%
START_YEAR = 2022
END_YEAR = 2023
DISTRICTS = ["4"]
FILE_TYPES = ["station_5min"]
MONTHS = None  # Use None to download data from all months
SAVE_PATH = os.path.join(os.getcwd(), "zipped_downloads")

# View summary of available files for (start_year, end_year, districts, file_types) query
files = pems.get_files(
    start_year=START_YEAR,
    end_year=END_YEAR,
    districts=DISTRICTS,
    file_types=FILE_TYPES,
    months=MONTHS,
)
for file in files:
    print(file)

# %%
# Download files for (start_year, end_year, districts, file_types) query
pems.download_files(
    start_year=START_YEAR,
    end_year=END_YEAR,
    districts=DISTRICTS,
    file_types=FILE_TYPES,
    months=MONTHS,
    save_path=SAVE_PATH,
)
