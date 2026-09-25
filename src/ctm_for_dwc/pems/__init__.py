"""Caltrans PeMS clearinghouse access (Step 3 data, and Step 2 station metadata).

* :class:`PeMSDownloader` (``downloader.py``) -- log in and download
  clearinghouse files (e.g. ``station_5min`` ``.gz`` dumps).
* :class:`PeMSExtractor` (``extractor.py``) -- turn downloaded ``.gz`` dumps
  into one CSV per detector; no network access.
* :func:`download_pems_station_metadata` (``metadata.py``) -- fetch a
  district's ``station_meta`` snapshot.
* :func:`resolve_credentials` (``credentials.py``) -- PeMS username/password
  from arguments, the environment, or ``.env``.
* ``settings.py`` -- clearinghouse URLs and district list.

The downloader and extractor are adapted from Seb-Good/caltrans-pems.
"""

from .credentials import resolve_credentials
from .downloader import PeMSDownloader
from .extractor import PeMSExtractor
from .metadata import (
    DEFAULT_PEMS_DIR,
    PEMS_META_FILE_TYPE,
    download_pems_station_metadata,
)

__all__ = [
    "PeMSDownloader",
    "PeMSExtractor",
    "download_pems_station_metadata",
    "resolve_credentials",
    "DEFAULT_PEMS_DIR",
    "PEMS_META_FILE_TYPE",
]
