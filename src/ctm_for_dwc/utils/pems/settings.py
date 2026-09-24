"""
Settings for interacting with the Caltrans PeMS clearinghouse.

Defines the base URL and query template used by the downloader, plus the list
of district IDs that PeMS publishes data for.
"""

# Import 3rd party libraries
import os

# Set working directory
WORKING_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

# Set data directory
DATA_PATH = os.path.join(WORKING_PATH, "data")

# Set base url
BASE_URL = "http://pems.dot.ca.gov"

# Clearing house url
CLEARING_HOUSE_URL = (
    "{}/?srq=clearinghouse&district_id={}&yy={}&type={}&returnformat=text"
)

# Set available districts
DISTRICTS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
