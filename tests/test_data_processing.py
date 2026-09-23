"""Tests for PeMSDataProcessor's sequential-grid preprocessing invariant.

These tests verify that the preprocessing pipeline produces a contiguous
5-minute grid per station, with missing/bad data represented as NaN rather
than dropped rows. That invariant is what makes the `sequence.isna().any().any()`
checks in `prepare_sequence_dataset` / `prepare_imputation_dataset` sufficient
to filter out non-contiguous windows.
"""

import numpy as np
import pandas as pd
import pytest

from ctm_for_dwc.utils.data_processing import PeMSDataProcessor


# Matches load_data_by_id's reindex: one year of 5-minute rows.
YEAR = 2022
ROWS_PER_STATION = 12 * 24 * 365


def _write_mock_dataset(
    root,
    n_stations=1,
    gap_rows=None,
    low_pct_rows=None,
    seed=0,
):
    """Create metadata + per-station timeseries CSVs under ``root``.

    - ``gap_rows``: dict[station_idx, list[int]] — row indices to physically
      drop from the CSV (simulating missing PeMS records).
    - ``low_pct_rows``: dict[station_idx, list[int]] — row indices whose
      ``pct_observed`` is set to 50 (below the default 100 threshold).
    """
    rng = np.random.default_rng(seed)
    metadata_dir = root / "metadata"
    ts_dir = root / "timeseries_data"
    metadata_dir.mkdir(parents=True)
    ts_dir.mkdir(parents=True)

    metadata = pd.DataFrame(
        {
            "Station ID": list(range(n_stations)),
            "Lanes": [3] * n_stations,
            "Type": ["Mainline"] * n_stations,
            "Abs PM": [float(i) for i in range(n_stations)],
            "Length": [0.5] * n_stations,
            "capacity": [2000.0] * n_stations,
            "free_flow_speed": [60.0] * n_stations,
            "congestion_wave_speed": [15.0] * n_stations,
            "critical_density": [45.0] * n_stations,
        }
    )
    metadata_path = metadata_dir / "station_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    timestamps = pd.date_range(
        start=f"{YEAR}-01-01", periods=ROWS_PER_STATION, freq="5min"
    )
    for i in range(n_stations):
        rows = pd.DataFrame(
            {
                "timestamp": timestamps,
                "station": i,
                "pct_observed": 100.0,
                "total_flow_[veh/5-min]": rng.uniform(10, 150, ROWS_PER_STATION),
                "avg_speed_[mph]": rng.uniform(30, 70, ROWS_PER_STATION),
            }
        )
        if low_pct_rows and i in low_pct_rows:
            rows.loc[low_pct_rows[i], "pct_observed"] = 50.0
        if gap_rows and i in gap_rows:
            rows = rows.drop(index=gap_rows[i]).reset_index(drop=True)
        rows.to_csv(ts_dir / f"{i}.csv", index=False)

    return metadata_path


def _make_processor(root, metadata_path):
    return PeMSDataProcessor(
        root_directory=str(root),
        metadata_filepath=str(metadata_path),
    )


# ---- Change 1: load_data_by_id preserves reindexed gap rows ---- #


class TestLoadDataByIdPreservesGrid:
    def test_gap_rows_kept_with_nan_raw_counts(self, tmp_path):
        gap = [100, 101, 102, 500]
        metadata_path = _write_mock_dataset(
            tmp_path, n_stations=1, gap_rows={0: gap}
        )
        proc = _make_processor(tmp_path, metadata_path)
        df = proc.load_data_by_id(station_id="0")

        # Full 5-minute grid, strictly contiguous.
        assert len(df) == ROWS_PER_STATION
        diffs = df["timestamp"].diff().dropna()
        assert (diffs == pd.Timedelta("5min")).all()

        # Gap rows are NaN in raw-count / pct columns.
        assert df.loc[gap, "total_flow_[veh/5-min]"].isna().all()
        assert df.loc[gap, "avg_speed_[mph]"].isna().all()
        assert df.loc[gap, "pct_observed"].isna().all()

        # Metadata survives the inner merge on reindexed rows.
        assert (df.loc[gap, "Station ID"] == 0).all()
        assert df.loc[gap, "capacity"].notna().all()
        assert df.loc[gap, "Lanes"].notna().all()


# ---- Change 2: build_long_df NaN-fills sub-threshold rows ---- #


class TestBuildLongDfNullsLowPct:
    def test_low_pct_rows_nanned_not_dropped(self, tmp_path):
        low_pct = [50, 51, 52]
        metadata_path = _write_mock_dataset(
            tmp_path, n_stations=1, low_pct_rows={0: low_pct}
        )
        proc = _make_processor(tmp_path, metadata_path)
        df = proc.build_long_df(detectors=["0"])

        # No rows dropped.
        assert len(df) == ROWS_PER_STATION

        # Sub-threshold rows have NaN targets; healthy rows do not.
        assert df.loc[low_pct, "flow"].isna().all()
        assert df.loc[low_pct, "density"].isna().all()
        healthy = list(range(200, 210))
        assert df.loc[healthy, "flow"].notna().all()
        assert df.loc[healthy, "density"].notna().all()

        # Grid remains contiguous.
        diffs = df["timestamp"].diff().dropna()
        assert (diffs == pd.Timedelta("5min")).all()


# ---- End-to-end: preprocess_data + prepare_imputation_dataset ---- #


class TestPreprocessDataContiguousGrid:
    def test_grid_size_and_nan_positions(self, tmp_path):
        gap_rows = {0: [100, 101], 1: [500]}
        low_pct_rows = {0: [300], 1: [700, 701]}
        metadata_path = _write_mock_dataset(
            tmp_path,
            n_stations=2,
            gap_rows=gap_rows,
            low_pct_rows=low_pct_rows,
        )
        proc = _make_processor(tmp_path, metadata_path)
        df = proc.preprocess_data(detectors=["0", "1"])

        for sid in [0, 1]:
            station_df = df.loc[df["Station ID"] == sid, :].reset_index(drop=True)
            assert len(station_df) == ROWS_PER_STATION

            expected_nan = set(gap_rows.get(sid, []) + low_pct_rows.get(sid, []))
            for idx in expected_nan:
                assert np.isnan(station_df.loc[idx, "flow"])
                assert np.isnan(station_df.loc[idx, "density"])

            # A row that is neither a gap nor sub-threshold stays non-NaN.
            healthy = 1000
            assert not np.isnan(station_df.loc[healthy, "flow"])
            assert not np.isnan(station_df.loc[healthy, "density"])


class TestPrepareImputationDatasetSkipsBadWindows:
    """Non-overlapping windows of seq_len: any window containing a NaN row
    should be excluded by the isna().any().any() gate, and no other."""

    def test_gap_excludes_exactly_one_window(self, tmp_path):
        seq_len = 36
        gap_idx = seq_len + 5  # inside window 1 (rows [36, 72)), not on a boundary
        metadata_path = _write_mock_dataset(
            tmp_path, n_stations=1, gap_rows={0: [gap_idx]}
        )
        proc = _make_processor(tmp_path, metadata_path)
        df = proc.preprocess_data(detectors=["0"])
        dataset = proc.prepare_imputation_dataset(df, seq_len=seq_len)

        max_windows = ROWS_PER_STATION // seq_len
        assert len(dataset) == max_windows - 1

    def test_low_pct_excludes_exactly_one_window(self, tmp_path):
        seq_len = 36
        bad_idx = 5 * seq_len + 10
        metadata_path = _write_mock_dataset(
            tmp_path, n_stations=1, low_pct_rows={0: [bad_idx]}
        )
        proc = _make_processor(tmp_path, metadata_path)
        df = proc.preprocess_data(detectors=["0"])
        dataset = proc.prepare_imputation_dataset(df, seq_len=seq_len)

        max_windows = ROWS_PER_STATION // seq_len
        assert len(dataset) == max_windows - 1
