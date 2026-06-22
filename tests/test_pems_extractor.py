"""Tests for :class:`PeMSExtractor` -- the parser for raw PeMS ``.gz`` dumps.

PeMS publishes 5-minute station data as headerless gzipped CSVs where the
column layout is ``12 station-level columns + 5 columns per lane`` (lane
count varies by station). Real exports are hundreds of MB per file; these
tests build tiny in-memory ``.gz`` files (≈ 10 rows × 2 lanes, < 2 KB)
written to ``tmp_path`` so the test suite stays lightweight.

The synthetic-fixture pattern matches ``test_data_processing.py``.
"""

from __future__ import annotations

import gzip
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from transportation_models.utils.data_downloading import PeMSExtractor


# ---- Synthetic gzipped-PeMS fixture builder ------------------------------


def _write_pems_gz(
    path: Path,
    *,
    timestamps: list[str],
    station: int,
    lanes: int = 2,
    flow: int = 100,
) -> None:
    """Write a tiny headerless gzipped PeMS-format CSV to ``path``.

    The file has ``len(timestamps)`` rows, one VDS station id, and ``lanes``
    lanes' worth of per-lane columns. Values are deterministic placeholders
    so test assertions can target specific rows.
    """
    n_rows = len(timestamps)
    # 12 station-level columns + 5 per lane.
    station_block = pd.DataFrame({
        "timestamp": timestamps,
        "station": [station] * n_rows,
        "district": [4] * n_rows,
        "freeway": [880] * n_rows,
        "direction": ["N"] * n_rows,
        "lane_type": ["ML"] * n_rows,
        "station_length": [0.5] * n_rows,
        "samples": [100] * n_rows,
        "pct_observed": [100] * n_rows,
        "total_flow": [flow] * n_rows,
        "avg_occupancy": [0.1] * n_rows,
        "avg_speed": [60.0] * n_rows,
    })
    lane_cols: dict[str, list] = {}
    for lane in range(1, lanes + 1):
        lane_cols[f"good_lane_{lane}"] = [100] * n_rows
        lane_cols[f"flow_lane_{lane}"] = [flow // lanes] * n_rows
        lane_cols[f"occ_lane_{lane}"] = [0.1] * n_rows
        lane_cols[f"speed_lane_{lane}"] = [60.0] * n_rows
        lane_cols[f"observed_lane_{lane}"] = [1] * n_rows
    df = pd.concat([station_block, pd.DataFrame(lane_cols)], axis=1)

    with gzip.open(path, "wt") as f:
        df.to_csv(f, header=False, index=False)


def _ts_range(start: str, n: int) -> list[str]:
    """``n`` consecutive 5-minute timestamps starting at ``start``.

    Emitted in the real PeMS clearinghouse format (``MM/DD/YYYY HH:MM:SS``)
    so fixtures match what the extractor parses in production.
    """
    return [
        (pd.Timestamp(start) + pd.Timedelta(minutes=5 * i)).strftime("%m/%d/%Y %H:%M:%S")
        for i in range(n)
    ]


# ---- Constructor ----------------------------------------------------------


def test_constructor_rejects_non_directory(tmp_path):
    bogus = tmp_path / "does-not-exist"
    with pytest.raises(RuntimeError, match="Expected a directory"):
        PeMSExtractor(str(bogus))


def test_constructor_accepts_existing_directory(tmp_path):
    ext = PeMSExtractor(str(tmp_path))
    assert ext.zipfile_directory == str(tmp_path)
    # No datetime bounds when start/end aren't supplied.
    assert ext.start_datetime is None
    assert ext.end_datetime is None


def test_constructor_builds_datetime_bounds(tmp_path):
    ext = PeMSExtractor(
        str(tmp_path),
        start_date="2022-01-15", start_time="08:00:00",
        end_date="2022-01-15", end_time="12:00:00",
    )
    assert ext.start_datetime == datetime(2022, 1, 15, 8, 0, 0)
    assert ext.end_datetime == datetime(2022, 1, 15, 12, 0, 0)


# ---- _build_datetime ------------------------------------------------------


def test_build_datetime_defaults_to_start_of_day(tmp_path):
    ext = PeMSExtractor(str(tmp_path))
    dt = ext._build_datetime("2022-01-15")
    assert dt.year == 2022 and dt.month == 1 and dt.day == 15
    assert dt.hour == 0 and dt.minute == 0 and dt.second == 0


def test_build_datetime_combines_date_and_time(tmp_path):
    ext = PeMSExtractor(str(tmp_path))
    dt = ext._build_datetime("2022-01-15", "14:30:00")
    assert dt == datetime(2022, 1, 15, 14, 30, 0)


# ---- _infer_dataframe_columns --------------------------------------------


def test_infer_columns_at_2_lanes(tmp_path):
    """12 station + 5*2 lane columns = 22 -> infers 2 lanes."""
    df = pd.DataFrame([[0] * 22])
    ext = PeMSExtractor(str(tmp_path))
    out = ext._infer_dataframe_columns(df, list(ext.base_column_names))
    assert list(out.columns)[:5] == [
        "timestamp", "station", "district", "freeway", "direction_of_travel",
    ]
    assert "flow_lane_2" in out.columns
    assert "observed_lane_2" in out.columns
    assert "flow_lane_3" not in out.columns


def test_infer_columns_at_4_lanes(tmp_path):
    """12 station + 5*4 = 32 columns -> infers 4 lanes."""
    df = pd.DataFrame([[0] * 32])
    ext = PeMSExtractor(str(tmp_path))
    out = ext._infer_dataframe_columns(df, list(ext.base_column_names))
    assert "flow_lane_4" in out.columns
    assert "flow_lane_5" not in out.columns


def test_infer_columns_raises_when_leftover_isnt_multiple_of_5(tmp_path):
    df = pd.DataFrame([[0] * 14])  # 12 + 2 -> 2 leftover, not a clean lane block
    ext = PeMSExtractor(str(tmp_path))
    with pytest.raises(RuntimeError, match="Unexpected number of columns"):
        ext._infer_dataframe_columns(df, list(ext.base_column_names))


# ---- _load_single_dataframe ----------------------------------------------


def test_load_single_parses_timestamps_as_datetimes(tmp_path):
    gz = tmp_path / "fragment.gz"
    _write_pems_gz(gz, timestamps=_ts_range("2022-01-15 08:00:00", 3), station=400839)
    ext = PeMSExtractor(str(tmp_path))
    df = ext._load_single_dataframe(str(gz))
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["station"].unique().tolist() == [400839]


def test_load_single_clips_to_time_window(tmp_path):
    """Rows outside [start, end] are dropped at file-load time."""
    gz = tmp_path / "fragment.gz"
    _write_pems_gz(gz, timestamps=_ts_range("2022-01-15 08:00:00", 6), station=400839)
    ext = PeMSExtractor(
        str(tmp_path),
        start_date="2022-01-15", start_time="08:10:00",
        end_date="2022-01-15", end_time="08:20:00",
    )
    df = ext._load_single_dataframe(str(gz))
    # Should keep 08:10, 08:15, 08:20 -> 3 rows.
    assert len(df) == 3
    assert df["timestamp"].min() == pd.Timestamp("2022-01-15 08:10:00")
    assert df["timestamp"].max() == pd.Timestamp("2022-01-15 08:20:00")


# ---- _get_gz_files --------------------------------------------------------


def test_get_gz_files_lists_only_gz_at_top_level(tmp_path):
    (tmp_path / "a.gz").write_bytes(b"")
    (tmp_path / "b.gz").write_bytes(b"")
    (tmp_path / "skip.txt").write_text("ignored")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.gz").write_bytes(b"")     # not at top level
    ext = PeMSExtractor(str(tmp_path))
    found = sorted(Path(p).name for p in ext._get_gz_files())
    assert found == ["a.gz", "b.gz"]


# ---- _ensure_detectors_in_dataframe + _validate_detector_list -----------


def test_validate_detector_list_defaults_to_all_stations(tmp_path):
    df = pd.DataFrame({"station": [1, 2, 2, 3]})
    ext = PeMSExtractor(str(tmp_path))
    assert sorted(ext._validate_detector_list(df)) == [1, 2, 3]


def test_validate_detector_list_returns_explicit_list_when_present(tmp_path):
    df = pd.DataFrame({"station": [1, 2, 3]})
    ext = PeMSExtractor(str(tmp_path))
    assert ext._validate_detector_list(df, detectors=[2, 3]) == [2, 3]


# ---- slice_dataframe_by_detector -----------------------------------------


def test_slice_partitions_dataframe_by_detector(tmp_path):
    df = pd.DataFrame({
        "station": [1, 1, 2, 3],
        "flow": [100, 110, 200, 300],
    })
    ext = PeMSExtractor(str(tmp_path))
    parts = ext.slice_dataframe_by_detector(df, [1, 2, 3])
    assert set(parts) == {1, 2, 3}
    assert parts[1]["flow"].tolist() == [100, 110]
    assert parts[2]["flow"].tolist() == [200]
    assert parts[3]["flow"].tolist() == [300]


def test_slice_skips_unrequested_detectors(tmp_path):
    df = pd.DataFrame({"station": [1, 2, 3], "flow": [100, 200, 300]})
    ext = PeMSExtractor(str(tmp_path))
    parts = ext.slice_dataframe_by_detector(df, [2])
    assert set(parts) == {2}
    assert parts[2]["flow"].tolist() == [200]


# ---- make_csv -------------------------------------------------------------


def test_make_csv_writes_new_file(tmp_path):
    ext = PeMSExtractor(str(tmp_path))
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2022-01-15 08:00:00", "2022-01-15 08:05:00"]),
        "station": [123, 123],
        "flow": [100, 110],
    })
    ext.make_csv(123, df, tmp_path)
    out = tmp_path / "123.csv"
    assert out.exists()
    reread = pd.read_csv(out)
    assert reread["flow"].tolist() == [100, 110]


def test_make_csv_appends_and_dedupes_when_target_exists(tmp_path):
    ext = PeMSExtractor(str(tmp_path))
    # Initial write.
    first = pd.DataFrame({
        "timestamp": pd.to_datetime(["2022-01-15 08:00:00", "2022-01-15 08:05:00"]),
        "station": [123, 123], "flow": [100, 110],
    })
    ext.make_csv(123, first, tmp_path)
    # Second write with one overlapping + one new row.
    second = pd.DataFrame({
        "timestamp": pd.to_datetime(["2022-01-15 08:05:00", "2022-01-15 08:10:00"]),
        "station": [123, 123], "flow": [110, 120],
    })
    ext.make_csv(123, second, tmp_path)
    merged = pd.read_csv(tmp_path / "123.csv")
    # Dedup + sort by timestamp.
    assert merged["flow"].tolist() == [100, 110, 120]
    assert list(merged["timestamp"]) == sorted(merged["timestamp"])


def test_make_csv_skips_empty_dataframe(tmp_path):
    ext = PeMSExtractor(str(tmp_path))
    ext.make_csv(456, pd.DataFrame(), tmp_path)
    assert not (tmp_path / "456.csv").exists()


# ---- process_gz_files (end-to-end) ---------------------------------------


def test_process_gz_files_writes_per_detector_csv(tmp_path):
    """Smoke-test the canonical workflow on two small synthetic dumps."""
    _write_pems_gz(
        tmp_path / "d4_2022_01_15.gz",
        timestamps=_ts_range("2022-01-15 08:00:00", 4), station=400839,
    )
    _write_pems_gz(
        tmp_path / "d4_2022_01_16.gz",
        timestamps=_ts_range("2022-01-16 08:00:00", 4), station=400839,
    )
    ext = PeMSExtractor(str(tmp_path), detectors=[400839])
    ext.process_gz_files()

    out = tmp_path / "csv_files" / "400839.csv"
    assert out.exists()
    df = pd.read_csv(out)
    # 4 rows per day, two days, no duplicates.
    assert len(df) == 8
    assert df["station"].unique().tolist() == [400839]
    # Output is sorted by timestamp ascending.
    timestamps = pd.to_datetime(df["timestamp"])
    assert (timestamps.diff().dropna() > pd.Timedelta(0)).all()


def test_process_gz_files_filters_unwanted_detectors(tmp_path):
    """A .gz containing multiple stations is filtered down to the requested one."""
    # Mixed-station file: 2 rows for 400839 followed by 2 rows for 400840.
    timestamps_a = _ts_range("2022-01-15 08:00:00", 2)
    timestamps_b = _ts_range("2022-01-15 09:00:00", 2)
    file_a = tmp_path / "stationA.gz"
    file_b = tmp_path / "stationB.gz"
    _write_pems_gz(file_a, timestamps=timestamps_a, station=400839)
    _write_pems_gz(file_b, timestamps=timestamps_b, station=400840)

    ext = PeMSExtractor(str(tmp_path), detectors=[400839])
    ext.process_gz_files()

    assert (tmp_path / "csv_files" / "400839.csv").exists()
    # The other station shouldn't get a CSV (it wasn't in the detector filter).
    assert not (tmp_path / "csv_files" / "400840.csv").exists()
