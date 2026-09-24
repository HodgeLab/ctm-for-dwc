"""
Download clearinghouse files from the Caltrans PeMS database (:class:`PeMSDownloader`).

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
from itertools import cycle
from bs4 import BeautifulSoup
from http.cookiejar import LWPCookieJar

# Local imports
from ctm_for_dwc.utils.pems.settings import (
    DATA_PATH,
    BASE_URL,
    DISTRICTS,
    CLEARING_HOUSE_URL,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


class PeMSDownloader(object):
    """
    Authenticate against the Caltrans PeMS website and download clearinghouse
    files matching a user query.

    The constructor logs in with the supplied credentials, fetches the main
    clearinghouse page, and parses the embedded JavaScript form data so that
    subsequent calls to :meth:`get_files` / :meth:`download_files` know which
    file types, districts, and months are available.
    """

    def __init__(self, username: str, password: str, debug: bool = False) -> None:
        """
        Log into PeMS and prime the downloader.

        Parameters
        ----------
        username : str
            PeMS account username.
        password : str
            PeMS account password.
        debug : bool, optional
            If True, enables mechanize HTTP / redirect / response debugging
            output on the underlying browser, by default False.
        """
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
    def _create_data_directory(save_path: typing.Optional[str]) -> str:
        """
        Resolve and create the directory that downloads should be saved to.

        Parameters
        ----------
        save_path : typing.Optional[str]
            Explicit destination directory. If None, falls back to ``DATA_PATH``
            from the PeMS settings module.

        Returns
        -------
        str
            The resolved path (created on disk if necessary).
        """
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
