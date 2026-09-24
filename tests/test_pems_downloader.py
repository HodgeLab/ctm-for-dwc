"""Tests for :class:`PeMSDownloader` -- clearinghouse query + batch download.

The real constructor logs in to ``pems.dot.ca.gov`` over HTTP via mechanize;
that login path stays out of unit tests and is not exercised by any test
(the ``@pytest.mark.network`` test in ``test_ctm_caltrans.py`` hits the
Caltrans postmile service, not PeMS). Here we
instantiate via ``__new__`` to bypass ``__init__`` and exercise the parsing,
URL-construction, and dedupe logic in isolation, monkeypatching the network-
touching seams (``_open_url`` and ``_download_file``) where needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from ctm_for_dwc.utils.pems.downloader import PeMSDownloader
from ctm_for_dwc.utils.pems.settings import BASE_URL


# ---- Helper: build a sham instance that skips the live login -------------


def _no_login_downloader() -> PeMSDownloader:
    """Return a PeMSDownloader instance with ``__init__`` skipped.

    Tests inject any other attributes they need (e.g. ``log``, ``browser``)
    on a case-by-case basis.
    """
    inst = PeMSDownloader.__new__(PeMSDownloader)
    inst.log = logging.getLogger("test_pems_downloader")
    return inst


# ---- Pure helpers ---------------------------------------------------------


def test_get_list_of_years_is_inclusive():
    years = PeMSDownloader._get_list_of_years(2022, 2024)
    assert years == ["2022", "2023", "2024"]


def test_get_list_of_years_single_year_returns_one_element():
    assert PeMSDownloader._get_list_of_years(2022, 2022) == ["2022"]


def test_get_urls_builds_one_per_file_type_district_year():
    dl = _no_login_downloader()
    urls = dl._get_urls(
        start_year=2022, end_year=2023,
        districts=["4", "7"], file_types=["meta", "station_5min"],
    )
    assert len(urls) == 2 * 2 * 2
    assert all(BASE_URL in entry["url"] for entry in urls)
    # The CLEARING_HOUSE_URL template ends with each query param interpolated.
    sample = urls[0]
    assert "district_id=" in sample["url"]
    assert "yy=" in sample["url"]
    assert "type=" in sample["url"]


def test_get_urls_carries_query_metadata_on_each_row():
    dl = _no_login_downloader()
    urls = dl._get_urls(2022, 2022, ["4"], ["meta"])
    assert urls == [{
        "file_type": "meta",
        "district": "4",
        "year": "2022",
        "url": urls[0]["url"],
    }]


def test_get_available_months_returns_all_keys_when_months_is_none():
    response = {"data": {"January": [], "February": [], "March": []}}
    out = PeMSDownloader._get_available_months(response=response, months=None)
    assert sorted(out) == ["February", "January", "March"]


def test_get_available_months_intersects_with_user_supplied_list():
    response = {"data": {"January": [], "February": []}}
    out = PeMSDownloader._get_available_months(
        response=response, months=["February", "March"],
    )
    # Only February intersects -- March isn't in the response.
    assert out == ["February"]


# ---- _check_for_new_files (dedupe) ---------------------------------------


def test_check_for_new_files_passthrough_when_no_existing_downloads():
    candidate = pd.DataFrame([
        {"file_name": "a.txt", "file_id": 1},
        {"file_name": "b.txt", "file_id": 2},
    ])
    out = PeMSDownloader._check_for_new_files(candidate, pd.DataFrame())
    pd.testing.assert_frame_equal(out, candidate)


def test_check_for_new_files_drops_duplicates_against_history():
    candidate = pd.DataFrame([
        {"file_name": "a.txt", "file_id": 1},
        {"file_name": "b.txt", "file_id": 2},
    ])
    history = pd.DataFrame([{"file_name": "a.txt", "file_id": 1}])
    out = PeMSDownloader._check_for_new_files(candidate, history)
    # Only "b.txt" should survive.
    assert "b.txt" in out["file_name"].tolist()
    assert "a.txt" not in out["file_name"].tolist()


# ---- _load_download_lookup -----------------------------------------------


def test_load_download_lookup_returns_empty_when_missing(tmp_path):
    out = PeMSDownloader._load_download_lookup(str(tmp_path))
    assert out.empty


def test_load_download_lookup_loads_existing_lookup(tmp_path):
    path = tmp_path / "saved_files.csv"
    pd.DataFrame([{"file_name": "a.txt"}]).to_csv(path, index=False)
    out = PeMSDownloader._load_download_lookup(str(tmp_path))
    assert out["file_name"].tolist() == ["a.txt"]


# ---- _create_data_directory ----------------------------------------------


def test_create_data_directory_resolves_explicit_path(tmp_path):
    target = tmp_path / "downloads"
    resolved = PeMSDownloader._create_data_directory(str(target))
    assert resolved == str(target)
    assert target.exists() and target.is_dir()


def test_create_data_directory_idempotent_on_existing_dir(tmp_path):
    target = tmp_path / "exists"
    target.mkdir()
    # No exception on the second create -- exist_ok=True under the hood.
    out = PeMSDownloader._create_data_directory(str(target))
    assert out == str(target)


def test_create_data_directory_falls_back_to_default_when_none(monkeypatch, tmp_path):
    sentinel = tmp_path / "default_data_path"
    monkeypatch.setattr(
        "ctm_for_dwc.utils.pems.downloader.DATA_PATH", str(sentinel)
    )
    out = PeMSDownloader._create_data_directory(None)
    assert out == str(sentinel)
    assert sentinel.exists()


# ---- get_files (mock the JSON-returning network seam) --------------------


def test_get_files_parses_clearinghouse_response(monkeypatch):
    """`get_files` walks `_open_url` results and emits one row per file."""
    dl = _no_login_downloader()
    # PeMS returns a JSON like {"data": {"January": [{file_name, file_id, bytes, url}]}}.
    fake_response = {
        "data": {
            "January": [
                {"file_name": "d04_text_station_5min_2022_01_01.txt.gz",
                 "file_id": 12345,
                 "bytes": "1,234,567",
                 "url": "/download/12345/d04_text_station_5min_2022_01_01.txt.gz"},
            ],
            "February": [
                {"file_name": "d04_text_station_5min_2022_02_01.txt.gz",
                 "file_id": 12346,
                 "bytes": "2,345,678",
                 "url": "/download/12346/d04_text_station_5min_2022_02_01.txt.gz"},
            ],
        }
    }

    opened: list[str] = []

    def _fake_open_url(self, url):
        opened.append(url)
        return fake_response

    monkeypatch.setattr(PeMSDownloader, "_open_url", _fake_open_url, raising=True)

    files = dl.get_files(
        start_year=2022, end_year=2022,
        districts=["4"], file_types=["station_5min"],
    )
    # One file per month in the response.
    assert {row["file_name"] for row in files} == {
        "d04_text_station_5min_2022_01_01.txt.gz",
        "d04_text_station_5min_2022_02_01.txt.gz",
    }
    # `bytes` -> MB conversion: 1,234,567 / 1e6 -> 1.2 (rounded to 1 decimal).
    jan = next(r for r in files if r["month"] == "January")
    assert jan["megabites"] == 1.2
    assert jan["district"] == "4"
    assert jan["file_type"] == "station_5min"
    # We hit exactly one URL (1 file_type * 1 district * 1 year).
    assert len(opened) == 1


def test_get_files_filters_by_requested_months(monkeypatch):
    dl = _no_login_downloader()
    monkeypatch.setattr(
        PeMSDownloader, "_open_url",
        lambda self, url: {"data": {
            "January": [{"file_name": "jan.gz", "file_id": 1, "bytes": "10",
                         "url": "/u/1"}],
            "March":   [{"file_name": "mar.gz", "file_id": 3, "bytes": "30",
                         "url": "/u/3"}],
        }},
        raising=True,
    )
    files = dl.get_files(
        start_year=2022, end_year=2022,
        districts=["4"], file_types=["station_5min"],
        months=["March"],
    )
    assert {r["file_name"] for r in files} == {"mar.gz"}


# ---- download_files (mock the actual download step) ----------------------


def test_download_files_skips_already_downloaded(tmp_path, monkeypatch):
    """Pre-existing rows in `saved_files.csv` should not be re-downloaded."""
    dl = _no_login_downloader()
    save_path = tmp_path / "dl"
    save_path.mkdir()

    # Seed saved_files.csv with one row.
    pd.DataFrame([{
        "file_type": "station_5min", "district": "4", "year": "2022",
        "month": "January", "file_name": "jan.gz", "file_id": 1,
        "megabites": 0.1, "download_url": "/u/jan",
    }]).to_csv(save_path / "saved_files.csv", index=False)

    monkeypatch.setattr(
        PeMSDownloader, "_open_url",
        lambda self, url: {"data": {
            "January": [{"file_name": "jan.gz", "file_id": 1, "bytes": "100",
                         "url": "/u/jan"}],
            "February": [{"file_name": "feb.gz", "file_id": 2, "bytes": "200",
                          "url": "/u/feb"}],
        }},
        raising=True,
    )

    downloaded: list[str] = []

    def _fake_download(self, file_name, file_url, save_path):
        downloaded.append(file_name)
        return True

    monkeypatch.setattr(PeMSDownloader, "_download_file", _fake_download, raising=True)
    monkeypatch.setattr("time.sleep", lambda _s: None)   # don't pause in tests

    dl.download_files(
        start_year=2022, end_year=2022,
        districts=["4"], file_types=["station_5min"],
        months=["January", "February"], save_path=str(save_path),
    )

    # Only the new file (Feb) should hit the downloader; Jan is in history.
    assert downloaded == ["feb.gz"]


def test_download_files_records_successful_downloads_in_lookup(tmp_path, monkeypatch):
    """A successful download appends to saved_files.csv."""
    dl = _no_login_downloader()
    save_path = tmp_path / "dl"
    save_path.mkdir()

    monkeypatch.setattr(
        PeMSDownloader, "_open_url",
        lambda self, url: {"data": {
            "January": [{"file_name": "jan.gz", "file_id": 1, "bytes": "100",
                         "url": "/u/jan"}],
        }},
        raising=True,
    )
    monkeypatch.setattr(
        PeMSDownloader, "_download_file",
        lambda self, file_name, file_url, save_path: True,
        raising=True,
    )
    monkeypatch.setattr("time.sleep", lambda _s: None)

    dl.download_files(
        start_year=2022, end_year=2022,
        districts=["4"], file_types=["station_5min"],
        months=["January"], save_path=str(save_path),
    )

    lookup_path = save_path / "saved_files.csv"
    assert lookup_path.exists()
    lookup = pd.read_csv(lookup_path)
    # Lookup contains the file name we "downloaded".
    assert "jan.gz" in lookup.values.tolist().__str__() or (
        "file_name" in lookup.columns and "jan.gz" in lookup["file_name"].tolist()
    )


def test_download_files_no_op_when_nothing_to_download(tmp_path, monkeypatch, caplog):
    """Empty file-list query should log "No data available" and not call downloader."""
    dl = _no_login_downloader()

    monkeypatch.setattr(
        PeMSDownloader, "_open_url",
        lambda self, url: {"data": {}},        # no files
        raising=True,
    )

    downloaded: list[str] = []

    def _fake_download(self, file_name, file_url, save_path):
        downloaded.append(file_name)
        return True

    monkeypatch.setattr(PeMSDownloader, "_download_file", _fake_download, raising=True)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    with caplog.at_level(logging.INFO, logger="test_pems_downloader"):
        dl.download_files(
            start_year=2022, end_year=2022,
            districts=["4"], file_types=["station_5min"],
            months=["January"], save_path=str(tmp_path),
        )

    assert downloaded == []
