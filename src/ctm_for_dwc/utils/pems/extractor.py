"""
Turn raw 5-minute PeMS ``.gz`` dumps into per-detector CSVs (:class:`PeMSExtractor`).
"""

# 3rd party imports
import os
import time
import typing
import pandas as pd
from pathlib import Path
from datetime import datetime


class PeMSExtractor(object):
    """
    Read raw 5-minute PeMS ``.gz`` files off disk and emit per-detector CSVs.

    The extractor walks a directory of gzipped PeMS dumps, infers the lane-count
    for each file from its column count, optionally filters by detector list
    and/or time window, and writes one CSV per detector (appending to any
    existing file for that detector).
    """

    def __init__(
        self,
        zipfile_directory: str,
        detectors: typing.Optional[list[int]] = None,
        start_date: typing.Optional[str] = None,
        start_time: typing.Optional[str] = None,
        end_date: typing.Optional[str] = None,
        end_time: typing.Optional[str] = None,
    ) -> None:
        """
        Configure the extractor and build the start/end datetime bounds.

        Parameters
        ----------
        zipfile_directory : str
            Directory containing ``.gz`` PeMS files to process. Must exist.
        detectors : typing.Optional[list[int]], optional
            VDS station IDs to keep. If None, all detectors present in the data
            are processed, by default None.
        start_date : typing.Optional[str], optional
            Lower-bound date, formatted ``YYYY-MM-DD``. If None, no lower
            bound is applied, by default None.
        start_time : typing.Optional[str], optional
            Lower-bound time, formatted ``HH:MM:SS``. Defaults to start-of-day
            if ``start_date`` is provided without a time.
        end_date : typing.Optional[str], optional
            Upper-bound date, formatted ``YYYY-MM-DD``. If None, no upper
            bound is applied, by default None.
        end_time : typing.Optional[str], optional
            Upper-bound time, formatted ``HH:MM:SS``. Defaults to start-of-day
            if ``end_date`` is provided without a time.

        Raises
        ------
        RuntimeError
            If ``zipfile_directory`` does not exist or is not a directory.
        """
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
        """
        Combine a date and (optional) time string into a ``datetime``.

        Parameters
        ----------
        date : str
            Date portion, matching the date segment of ``format``.
        time : typing.Optional[str], optional
            Time portion. If None, the start of the day is substituted, by
            default None.
        format : str, optional
            ``strptime`` format string applied to ``"{date} {time}"``, by
            default ``"%Y-%m-%d %H:%M:%S"``.

        Returns
        -------
        datetime
            Parsed datetime.
        """
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
        """
        Load every ``.gz`` in the configured directory into one DataFrame.

        Applies the detector filter (if any) and the time window bounds stored
        on the instance, then concatenates and sorts by timestamp.

        Parameters
        ----------
        detectors : typing.Optional[list[int]], optional
            VDS station IDs to keep. If None, all detectors are kept, by
            default None.

        Returns
        -------
        pd.DataFrame
            Combined dataframe sorted ascending by ``timestamp``.
        """
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
        """
        List all ``.gz`` files in the configured directory.

        Returns
        -------
        list[str]
            Absolute paths to each ``.gz`` file at the top level of
            ``self.zipfile_directory``.
        """
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
    def _ensure_detectors_in_dataframe(df: pd.DataFrame, detectors: list[int]) -> bool:
        """
        Check that every detector in ``detectors`` appears in ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a ``station`` column of detector IDs.
        detectors : list[int]
            Detector IDs to verify.

        Returns
        -------
        bool
            True if all detectors are present, False otherwise (missing IDs
            are printed to stdout as a side effect).
        """
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
        """
        Rename a headerless PeMS DataFrame using inferred lane column names.

        The PeMS 5-minute format emits a fixed block of station-level columns
        followed by five lane-level columns per lane. The number of lanes is
        derived from the total column count; this method appends a column name
        per lane and applies the mapping.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame loaded with integer column labels.
        base_column_names : list[str]
            The station-level column names (mutated in place with the
            per-lane names appended).

        Returns
        -------
        pd.DataFrame
            The input DataFrame with string column names applied.

        Raises
        ------
        RuntimeError
            If the number of leftover columns is not a multiple of 5.
        """
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
        """
        Load a single gzipped PeMS file into a DataFrame.

        Applies column-name inference, converts ``timestamp`` to datetime, and
        clips to ``self.start_datetime`` / ``self.end_datetime`` if set.

        Parameters
        ----------
        file : str
            Path to a ``.gz`` PeMS file.

        Returns
        -------
        pd.DataFrame
            Parsed DataFrame (empty if the file could not be read).
        """
        # Load the DataFrame from the file
        try:
            df = pd.read_csv(file, header=None)
        except FileNotFoundError as e:
            print(f"{e}\nSkipping this file")
            df = pd.DataFrame()

        # Infer the column names
        df = self._infer_dataframe_columns(df, self.base_column_names)

        # Make the timestamp column a datetime object. Specifying the PeMS
        # format avoids slow per-element inference on large files.
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%m/%d/%Y %H:%M:%S"
        )

        # Limit the dataframe to the time window that we need
        if self.start_datetime is not None:
            df = df.loc[df["timestamp"] >= self.start_datetime, :].copy()

        if self.end_datetime is not None:
            df = df.loc[df["timestamp"] <= self.end_datetime, :].copy()

        return df

    def slice_dataframe_by_detector(
        self, df: pd.DataFrame, detectors: list[int]
    ) -> dict[int, pd.DataFrame]:
        """
        Partition a PeMS DataFrame into one sub-DataFrame per detector.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a ``station`` column.
        detectors : list[int]
            Detector IDs to partition on.

        Returns
        -------
        dict[int, pd.DataFrame]
            Mapping from detector ID to the rows of ``df`` where
            ``station == detector_id``.
        """
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
        """
        Write (or append to) a per-detector CSV under ``csv_root_dir``.

        If ``{detector_id}.csv`` already exists, it is loaded, concatenated
        with ``df``, deduplicated, and re-sorted by timestamp before being
        written back.

        Parameters
        ----------
        detector_id : int
            VDS ID used to name the output file.
        df : pd.DataFrame
            Rows to write.
        csv_root_dir : Path
            Directory the CSV is written into.
        """
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
        """
        Resolve the effective detector list for a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a ``station`` column.
        detectors : typing.Optional[list[int]], optional
            User-supplied detector list. If None, the unique detectors in
            ``df`` are returned, by default None.

        Returns
        -------
        list[int]
            Validated detector list.

        Raises
        ------
        Warning
            If any of the provided detectors are missing from ``df``.
        """
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

    def _append_detector_csv(
        self,
        detector_id: int,
        df: pd.DataFrame,
        csv_root_dir: Path,
        written: set,
    ) -> None:
        """
        Append a detector's rows to its CSV without reading the file back.

        The first time a detector is seen this run, any stale CSV is removed
        and a header is written; subsequent calls append headerless. Dedup and
        sort are deferred to :meth:`_finalize_csv` so this stays O(rows-written)
        instead of re-reading and re-writing a growing file on every call.

        Parameters
        ----------
        detector_id : int
            VDS ID used to name the output file.
        df : pd.DataFrame
            Rows to append.
        csv_root_dir : Path
            Directory the CSV is written into.
        written : set
            Detector IDs already touched this run; mutated in place.
        """
        if df.empty:
            return

        filepath = Path(os.path.join(csv_root_dir, f"{detector_id}.csv"))

        # First touch this run: clear any stale file and write a header.
        first_write = detector_id not in written
        if first_write and os.path.isfile(filepath):
            os.remove(filepath)

        # Pin the datetime format: without date_format, to_csv renders an
        # all-midnight slice as date-only ("2022-08-26"), which mixes formats
        # across appended slices and breaks the read-back in _finalize_csv.
        df.to_csv(
            filepath,
            mode="a",
            header=first_write,
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        written.add(detector_id)

    def _finalize_csv(self, detector_id: int, csv_root_dir: Path) -> None:
        """
        Dedup and sort a detector's accumulated CSV, once, in place.

        Run after all ``.gz`` files have been streamed in by
        :meth:`_append_detector_csv`. Reads the file a single time, drops
        duplicate rows, sorts ascending by ``timestamp``, and rewrites.

        Parameters
        ----------
        detector_id : int
            VDS ID whose CSV should be finalized.
        csv_root_dir : Path
            Directory containing the CSV.
        """
        filepath = Path(os.path.join(csv_root_dir, f"{detector_id}.csv"))

        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%Y-%m-%d %H:%M:%S"
        )
        df.drop_duplicates(inplace=True, ignore_index=True)
        df.sort_values(
            by="timestamp", ascending=True, inplace=True, ignore_index=True
        )
        df.to_csv(filepath, index=False)

    def process_gz_files(self) -> None:
        """
        Convert every ``.gz`` in the configured directory into per-detector CSVs.

        CSVs are written to a ``csv_files/`` subdirectory created inside
        ``self.zipfile_directory``. Each ``.gz`` is streamed in once and its
        rows appended to the relevant per-detector CSVs
        (:meth:`_append_detector_csv`); each detector CSV is then deduped and
        sorted a single time at the end (:meth:`_finalize_csv`).
        """
        # Get a list of all the .gz files to pull from
        gz_files = self._get_gz_files()

        # For logging
        Nfiles = len(gz_files)
        Nprocessed = 0

        # Define a new directory path for the processed CSV files
        new_directory = Path(os.path.join(self.zipfile_directory, "csv_files"))

        # Build the directory
        Path.mkdir(new_directory, exist_ok=True)

        # Detector IDs touched this run (drives header / stale-file handling)
        written: set = set()

        # Phase 1: stream each file in once, appending rows per detector
        for gz_file in gz_files:
            start = time.time()
            print(f"Starting to process file: {gz_file}")
            # Load the data
            df = self._load_single_dataframe(gz_file)

            # Skip the rest of the processing if the DF doesn't contain data that we need
            if df.empty:
                continue

            # Get a valid list of detectors
            detectors_to_process = self._validate_detector_list(df, self.detectors)

            # Keep only the detectors we care about, then split in a single pass
            df = df.loc[df["station"].isin(detectors_to_process), :]
            for detector_id, detector_df in df.groupby("station"):
                self._append_detector_csv(
                    int(detector_id), detector_df, new_directory, written
                )

            Nprocessed += 1
            print(
                f"Completed processing on {Nprocessed}/{Nfiles} files "
                f"({time.time() - start:.1f}s)"
            )

        # Phase 2: dedup + sort each detector CSV exactly once
        for detector_id in sorted(written):
            print(f"Finalizing CSV for detector: {detector_id}")
            self._finalize_csv(detector_id, new_directory)
