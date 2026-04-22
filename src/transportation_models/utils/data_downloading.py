"""
Utility script to host methods for downloading transportation data from the Caltrans PeMS database.

Original author: Sebastian D. Goodfellow, Ph.D.
    https://github.com/Seb-Good/caltrans-pems
"""

# 3rd party imports
import os
import re
import sys
import json
import time
import locale
import typing
import logging
import itertools
import mechanize
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import cycle
from datetime import datetime
from bs4 import BeautifulSoup
from http.cookiejar import LWPCookieJar

# Local imports
from transportation_models.utils.pems_settings import (
    DATA_PATH,
    BASE_URL,
    DISTRICTS,
    CLEARING_HOUSE_URL,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


class PeMSDownloader(object):

    def __init__(self, username, password, debug=False):

        # Set parameters
        self.username = username
        self.password = password
        self.debug = debug

        # Set attributes and connect to PeMS
        self.log = self._logger_setup()
        self.browser = self._browser_login()
        self.html_object = self._get_html_object()
        self.label_reference, self.form_data = self._parse_html()

    def get_file_types(self):
        """Collect all available"""
        return list(self.form_data.keys())

    def get_districts(self, file_type):
        """Return the available districts for a file type."""
        # Get form_data district keys
        districts = list(self.form_data[file_type].keys())

        return DISTRICTS if districts[0] == "all" else sorted(districts, key=int)

    def get_files(self, start_year, end_year, districts, file_types, months=None):
        """Return a list of available files for specific query."""
        # Storage for query responses
        files = list()
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

        # Get urls
        urls = self._get_urls(
            start_year=start_year,
            end_year=end_year,
            districts=districts,
            file_types=file_types,
        )

        # Loop through urls and collect available files
        for url in urls:
            response = self._open_url(url=url["url"])
            if response:
                files.extend(
                    self._collect_available_files(
                        file_type=url["file_type"],
                        district=url["district"],
                        year=url["year"],
                        response=response,
                        months=months,
                    )
                )
            else:
                self.log.info(
                    "No data available for filetype: {}, year: {}, "
                    "district: {}".format(
                        url["file_type"], url["year"], url["district"]
                    )
                )

        return files

    def _collect_available_files(self, file_type, district, year, response, months):
        """Return a list of dicts containing file meta data for months."""
        files = list()
        available_months = self._get_available_months(response=response, months=months)
        for month in available_months:
            for file in response["data"][month]:
                files.append(
                    {
                        "file_type": file_type,
                        "district": district,
                        "year": year,
                        "month": month,
                        "file_name": file["file_name"],
                        "file_id": file["file_id"],
                        "megabites": np.round(locale.atof(file["bytes"]) / 10**6, 1),
                        "download_url": file["url"],
                    }
                )
        return files

    @staticmethod
    def _get_available_months(response, months):
        """Return a list of available months that match the user's query."""
        if months is None:
            return list(response["data"].keys())
        return [month for month in months if month in list(response["data"].keys())]

    def _open_url(self, url):
        """Open clearing house url and return json of contents."""
        self.browser.open(url)
        return json.loads(self.browser.response().read())  # type: ignore

    @staticmethod
    def _get_list_of_years(start_year, end_year):
        """Return a list of inclusive years between start_year and end_year."""
        return [str(x) for x in range(start_year, end_year + 1)]

    def _get_urls(self, start_year, end_year, districts, file_types):
        """Return a list of clearing house urls corresponding to user's query."""
        urls = list()
        years = self._get_list_of_years(start_year=start_year, end_year=end_year)
        for file_type, district, year in itertools.product(
            file_types, districts, years
        ):
            urls.append(
                {
                    "file_type": file_type,
                    "district": district,
                    "year": year,
                    "url": CLEARING_HOUSE_URL.format(
                        BASE_URL, district, year, file_type
                    ),
                }
            )
        return urls

    @staticmethod
    def _load_download_lookup(save_path):
        """Import .csv containing information about which files have already been downloaded."""
        if os.path.exists(os.path.join(save_path, "saved_files.csv")):
            return pd.read_csv(os.path.join(save_path, "saved_files.csv"))
        return pd.DataFrame()

    def download_files(
        self, start_year, end_year, districts, file_types, months, save_path=None
    ):
        """Download all text files for user's query."""
        # Create data directory
        save_path = self._create_data_directory(save_path=save_path)

        # Get files to download
        files_to_download = self.get_files(
            start_year=start_year,
            end_year=end_year,
            districts=districts,
            file_types=file_types,
            months=months,
        )
        files_to_download = pd.DataFrame(files_to_download)

        # Get existing files
        files_downloaded = self._load_download_lookup(save_path=save_path)

        # Remove previously downloaded files
        files_to_download = self._check_for_new_files(
            files_to_download=files_to_download, files_downloaded=files_downloaded
        )

        if not files_to_download.empty:
            for index, row in files_to_download.iterrows():
                success = self._download_file(
                    file_name=row["file_name"],
                    file_url=row["download_url"],
                    save_path=save_path,
                )
                if success:
                    new_row_df = pd.DataFrame(row)
                    files_downloaded = pd.concat(
                        [files_downloaded, new_row_df], ignore_index=True
                    )
                time.sleep(5)

            # Save lookup csv
            files_downloaded.to_csv(
                os.path.join(save_path, "saved_files.csv"), index=False
            )
            self.log.info(
                "Downloads complete, {} files, {} megabites".format(
                    files_to_download.shape[0],
                    np.round(files_to_download["megabites"].sum(), 1),
                )
            )
        else:
            self.log.info("No data available to download")

    @staticmethod
    def _check_for_new_files(files_to_download, files_downloaded):
        """Check for files that have already been downloaded and remove them from download list."""
        if files_downloaded.empty:
            return files_to_download
        return files_to_download[~files_to_download.isin(files_downloaded)].dropna()

    @staticmethod
    def _create_data_directory(save_path):
        if save_path is None:
            save_path = DATA_PATH
        os.makedirs(save_path, exist_ok=True)
        return save_path

    def _download_file(self, file_name, file_url, save_path):
        """Download single text file."""
        try:
            self.log.info("Start download, {}".format(file_name))
            self.browser.retrieve(
                "{}{}".format(BASE_URL, file_url), os.path.join(save_path, file_name)
            )
            self.log.info("Download completed")
            return True
        except Exception:
            self.log.info("Error downloading {}}".format(file_name))
            return False

    @staticmethod
    def _logger_setup():
        """Setup logger tool."""
        formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s")
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        log = logging.Logger("scraper")
        log.addHandler(handler)

        return log

    def _browser_login(self):
        """Log into CalTrans PeMS."""
        # Initialize browser
        browser = mechanize.Browser()

        # Set cookie jar
        browser.set_cookiejar(LWPCookieJar())

        # Set browser options
        browser.set_handle_equiv(True)
        browser.set_handle_referer(True)
        browser.set_handle_robots(False)
        browser.set_handle_redirect(mechanize.HTTPRedirectHandler)

        # Follows refresh 0 but doesn't hang on refresh > 0
        browser.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)  # type: ignore

        # Want debugging messages?
        browser.set_debug_http(self.debug)
        browser.set_debug_redirects(self.debug)
        browser.set_debug_responses(self.debug)

        self.log.info("Requesting initial page...")

        # User-Agent
        browser.addheaders = [
            (
                "User-agent",
                "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1",
            )
        ]
        browser.open(BASE_URL + "/?dnode=Clearinghouse")

        self.log.info("Initial page opened.")

        # Set authentication
        browser.select_form(nr=0)
        browser.form["username"] = self.username  # type: ignore
        browser.form["password"] = self.password  # type: ignore

        self.log.info("Logging in...")

        # Submit login
        browser.submit()

        self.log.info("Logged in.")

        return browser

    def _get_html_object(self):
        """Read main page HTML and return bs object."""
        return BeautifulSoup(self.browser.response().read())  # type: ignore

    def _parse_html(self):
        """Convert BeautifulSoup html object to JSON."""
        # Extract script containing valid request parameter values
        script = self.html_object.find("script", text=re.compile("YAHOO\.bts\.Data"))  # type: ignore
        j = re.search(
            r"^\s*YAHOO\.bts\.Data\s*=\s*({.*?})\s*$",
            script.string,  # type: ignore
            flags=re.DOTALL | re.MULTILINE,
        ).group(
            1
        )  # type: ignore

        # Convert to valid JSON
        j = re.sub(r"{\s*(\w)", r'{"\1', j)
        j = re.sub(r",\s*(\w)", r',"\1', j)
        j = re.sub(r"(\w):", r'\1":', j)

        # Convert JSON to dict
        data = json.loads(j)
        assert data["form_data"]["reid_raw"]["all"] == "all"

        return data["labels"], data["form_data"]


class PeMSExtractor(object):
    def __init__(
        self,
        zipfile_directory: str,
        detectors: typing.Optional[list[int]] = None,
        start_date: typing.Optional[str] = None,
        start_time: typing.Optional[str] = None,
        end_date: typing.Optional[str] = None,
        end_time: typing.Optional[str] = None,
    ) -> None:
        # Object attributes
        if not os.path.isdir(zipfile_directory):
            # Ensure that the directory is a directory
            raise RuntimeError(f"Expected a directory from {zipfile_directory}")
        else:
            self.zipfile_directory = zipfile_directory
        self.detectors = detectors

        # These columns are always included in the 5-minute data
        self.base_column_names = [
            "timestamp",
            "station",
            "district",
            "freeway",
            "direction_of_travel",
            "lane_type",
            "station_length",
            "samples",
            "pct_observed",
            "total_flow_[veh/5-min]",
            "avg_occupancy_[%]",
            "avg_speed_[mph]",
        ]

        # Build a starting datetime object – either datetime or None
        if start_date == None:
            self.start_datetime = None
        else:
            self.start_datetime = self._build_datetime(start_date, start_time)

        # Build an ending datetime object – either datetime or None
        if end_date == None:
            self.end_datetime = None
        else:
            self.end_datetime = self._build_datetime(end_date, end_time)

    def _build_datetime(
        self,
        date: str,
        time: typing.Optional[str] = None,
        format: str = "%Y-%m-%d %H:%M:%S",
    ) -> datetime:
        # Use the start of the day as a default
        if time is None:
            time_format = format.split(" ")[-1]
            now = datetime.now()
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            time = datetime.strftime(start_of_day, time_format)
        try:
            datetime_obj = datetime.strptime(f"{date} {time}", format)
        except Exception as e:
            print(
                f"An error occurred while building the datetime from {date} and {time} using format {format}:\n{e}"
            )
        return datetime_obj

    def _build_main_dataframe(
        self,
        detectors: typing.Optional[list[int]] = None,
    ) -> pd.DataFrame:
        # Empty DataFrame to populate and return
        df = pd.DataFrame()

        # Get a list of all the .gz files to pull from
        gz_files = self._get_gz_files()

        # Add each file to the DataFrame
        for gz_file in gz_files:
            print(f"Starting to process file: {gz_file}")
            # Load the new data
            new_df = self._load_single_dataframe(gz_file)

            # Slice the data for the stations
            if detectors is not None:
                # Ensure that the detectors passed are actually in the data
                if not self._ensure_detectors_in_dataframe(new_df, detectors):
                    raise Warning("Some detectors are not in the dataframe!")

                # Filter for detectors in the list
                new_df = new_df.loc[new_df["station"].isin(detectors), :]  # type: ignore

            # Slice the data for the time
            if self.start_datetime is not None:
                new_df[new_df["timestamp"] >= self.start_datetime]

            if self.end_datetime is not None:
                new_df[new_df["timestamp"] <= self.end_datetime]

            # Concatenate the new DataFrame to the main one
            df = pd.concat([df, new_df], ignore_index=True)
            print(f"Completed")

        # Sort the dataframe by date
        df.sort_values(by="timestamp", ascending=True, inplace=True)

        return df

    def _get_gz_files(self) -> list[str]:
        # Empty list of files to populate and return
        files = list()

        # Traverse the directory
        for item in os.listdir(self.zipfile_directory):
            # Construct the complete path to the item
            item_path = os.path.join(self.zipfile_directory, item)

            # Make sure that the item is a .gz file
            if os.path.isfile(item_path) and item_path.endswith(".gz"):
                files.append(item_path)

        return files

    @staticmethod
    def _ensure_detectors_in_dataframe(df: pd.DataFrame, detectors: list[int]):
        # Check that each detector in the list to find is actually in the data
        detectors = set(detectors)  # type: ignore
        detectors_in_data = set(df["station"].unique())
        if detectors_in_data.issubset(detectors_in_data):
            return True
        else:
            detectors_not_in_data = list(detectors.difference(detectors_in_data))  # type: ignore
            print(f"Some detectors are not in this data!\n{detectors_not_in_data}")
            return False

    @staticmethod
    def _infer_dataframe_columns(
        df: pd.DataFrame, base_column_names: list[str]
    ) -> pd.DataFrame:
        # We will infer the maximum number of lanes in the data from the number of columns
        num_leftover_columns = len(df.columns) - len(base_column_names)
        num_fields_per_lane = 5

        # Ensure that the number of leftover columns can divide into the number of fields per lane evenly
        if (num_leftover_columns % num_fields_per_lane) != 0:
            raise RuntimeError(f"Unexpected number of columns!\n {df.head(3)}")

        # Calculate the maximum number of lanes in the data set
        max_lanes = int(num_leftover_columns / num_fields_per_lane)

        # Append a column name for each field for each lane
        for i in range(1, max_lanes + 1):
            base_column_names.append(f"good_samples_lane_{i}")
            base_column_names.append(f"flow_lane_{i}")
            base_column_names.append(f"avg_occupancy_lane_{i}")
            base_column_names.append(f"avg_speed_lane_{i}")
            base_column_names.append(f"observed_lane_{i}")

        # Rename the column names
        df.rename(
            columns={idx: name for idx, name in enumerate(base_column_names)},
            inplace=True,
        )

        return df

    def _load_single_dataframe(self, file: str) -> pd.DataFrame:
        # Load the DataFrame from the file
        try:
            df = pd.read_csv(file, header=None)
        except FileNotFoundError as e:
            print(f"{e}\nSkipping this file")
            df = pd.DataFrame()

        # Infer the column names
        df = self._infer_dataframe_columns(df, self.base_column_names)

        # Make the timestamp column a datetime object
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Limit the dataframe to the time window that we need
        if self.start_datetime is not None:
            df = df.loc[df["timestamp"] >= self.start_datetime, :].copy()

        if self.end_datetime is not None:
            df = df.loc[df["timestamp"] <= self.end_datetime, :].copy()

        return df

    def slice_dataframe_by_detector(
        self, df: pd.DataFrame, detectors: list[int]
    ) -> dict[int, pd.DataFrame]:
        dataframes_by_detector = {}

        for detector in detectors:  # type: ignore
            print(f"Building dataframe for Detector: {detector}")
            # Filter for this detector
            detector_df = df.loc[df["station"] == detector, :].copy()

            # Drop any completely empty columns
            detector_df.dropna(axis=1, how="all")

            dataframes_by_detector[detector] = detector_df

        return dataframes_by_detector

    def make_csv(self, detector_id: int, df: pd.DataFrame, csv_root_dir: Path) -> None:
        # Skip processing if there's no data
        if df.empty:
            print(f"No data for detector: {detector_id}, skipping...")
            return

        # Construct the filepath for this dataframe
        filepath = Path(os.path.join(csv_root_dir, f"{detector_id}.csv"))

        # Logging
        print(f"Saving CSV for {detector_id} to {filepath}")

        # Look for an existing file
        if os.path.isfile(filepath):
            # Load the file
            try:
                existing_df = pd.read_csv(filepath)
            except Exception as e:
                print(f"An error occured while reading data from {filepath}")
                print(e)
                print("Trying again")
                # Catch the weird error where the df fails to load by waiting and trying again
                time.sleep(1)
                try:
                    existing_df = pd.read_csv(filepath)
                except Exception as e2:
                    print(f"Another error occured")
                    print(e2)
                    print("\nPress enter to continue...")
                    import pdb

                    pdb.set_trace()
            # Make the timestamp column a datetime object
            existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])
        else:
            # Pass an empty DF object for concatenation
            existing_df = pd.DataFrame()

        # Concatenate the new DF to the existing one
        complete_df = pd.concat([existing_df, df], ignore_index=True)

        # Drop duplicates
        complete_df.drop_duplicates(inplace=True, ignore_index=True)

        # Sort by date
        complete_df.sort_values(
            by="timestamp", ascending=True, inplace=True, ignore_index=True
        )

        # Save the complete DF
        complete_df.to_csv(filepath, index=False)

    def _validate_detector_list(
        self, df: pd.DataFrame, detectors: typing.Optional[list[int]] = None
    ) -> list[int]:
        if detectors is None:
            # Use all detectors in the dataframe as a default
            detectors = list(df["station"].astype(int).unique())
        else:
            # Make sure that the list that was passed is valid
            detectors_missing_from_dataframe = not self._ensure_detectors_in_dataframe(
                df, detectors
            )
            if detectors_missing_from_dataframe:
                raise Warning("Some detectors are not in the dataframe!")

        return detectors

    def process_gz_files(self) -> None:
        # Get a list of all the .gz files to pull from
        gz_files = self._get_gz_files()

        # For logging
        Nfiles = len(gz_files)
        Nprocessed = 0

        # Define a new directory path for the processed CSV files
        new_directory = Path(os.path.join(self.zipfile_directory, "csv_files"))

        # Build the directory
        Path.mkdir(new_directory, exist_ok=True)

        # Add each file to the DataFrame
        for gz_file in gz_files:
            print(f"Starting to process file: {gz_file}")
            # Load the data
            df = self._load_single_dataframe(gz_file)

            # Skip the rest of the processing if the DF doesn't contain data that we need
            if df.empty:
                continue

            # Get a valid list of detectors
            detectors_to_process = self._validate_detector_list(df, self.detectors)

            # Split the data into a separate DataFrame for each detector
            dataframes_by_detector = self.slice_dataframe_by_detector(
                df, detectors_to_process
            )

            for detector_id, detector_df in dataframes_by_detector.items():
                # Process the data into a CSV
                self.make_csv(detector_id, detector_df, new_directory)

            Nprocessed += 1
            print(f"Completed processing on {Nprocessed}/{Nfiles} files")
