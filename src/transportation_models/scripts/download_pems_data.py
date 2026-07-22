"""
Script for downloading timeseries transportation data from Caltrans PeMS database
"""

# Import 3rd party libraries
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Local imports
from transportation_models.utils.data_downloading import PeMSDownloader, PeMSExtractor
from transportation_models.utils.logs import make_logger

# Setup logger
logger = make_logger(include_stdout=True)

# Load login credentials
load_dotenv("credentials.env")
PEMS_USERNAME = os.environ["PEMS_USERNAME"]
PEMS_PASSWORD = os.environ["PEMS_PASSWORD"]

# Connect to PeMS
pems = PeMSDownloader(username=PEMS_USERNAME, password=PEMS_PASSWORD)

START_YEAR = 2022
END_YEAR = 2024
DISTRICTS = ["4"]
FILE_TYPES = ["station_5min"]
MONTHS = None  # Use None to download data from all months
SCRATCH = os.environ.get("SLURM_SCRATCH", os.getcwd())
SAVE_PATH = os.path.join(SCRATCH, "zipped_downloads")

detector_fp = (
    "/projects/rost5691/data/Caltrans/PeMS/metadata/district_4_i880_stations.csv"
)
df = pd.read_csv(detector_fp)

# Download files for (start_year, end_year, districts, file_types) query
pems.download_files(
    start_year=START_YEAR,
    end_year=END_YEAR,
    districts=DISTRICTS,
    file_types=FILE_TYPES,
    months=MONTHS,
    save_path=SAVE_PATH,
)

# Extract zipped files
extractor = PeMSExtractor(zipfile_directory=SAVE_PATH, detectors=df["ID"].to_list())

extractor.process_gz_files()
