"""
Prepare PeMS transportation data for downstream deep-learning models.

Houses :class:`PeMSDataProcessor`, which wraps the standard preprocessing
pipeline (timeseries + metadata load, unit standardization, flow/density
normalization, fundamental-diagram calibration, timestamp deconstruction,
feature encoding/scaling, and sequence/imputation dataset construction).
"""

#####     Setup     #####

# 3rd party imports
import logging
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch
import typing
from enum import Enum
from joblib import dump, load
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
    OrdinalEncoder,
    MinMaxScaler,
)
from torch.utils.data import (
    TensorDataset,
)


class TargetNormalization(str, Enum):
    """How to normalize the flow/density target columns in PeMSDataProcessor."""

    # Normalize flow by station capacity and density by station critical density
    # (uses the calibrated fundamental diagram parameters for each VDS).
    STATION_PARAMS = "station_params"

    # Whiten flow and density by their per-(weekday, 5min_block) mean and stddev.
    WHITENED = "whitened"

    # Min/max scale flow and density using min/max computed across all detectors
    # in the long_df.
    MIN_MAX = "min_max"

# Local imports
import transportation_models.utils.constants as constants
import transportation_models.utils.logs as logs
import transportation_models.utils.validation as validation

# Default logger
logger = logs.make_logger(log_prefix="data_processing")
class PeMSDataProcessor:
    """
    Class for preprocessing PeMS data into suitable tensors for downstream machine learning tasks.
    
    General workflow:
        - Load timeseries and metadata for PeMS vehicle detection stations (VDSs) into memory
        - Apply transformations, including:
            - Normalize 5-minute, station-wide counts to hourly, per-lane counts
            - Normalize flow by station capacity, density by station critical density
            - Deconstruct timestamp into separate columns for year, month, day, day-of-week, hour, minute, 5min_block
            - Apply some scalars
            - Optional additional transformations (e.g., whitening)
    """

    def __init__(
        self,
        root_directory: str = constants.ROOT_PATH,
        metadata_filepath: typing.Optional[str] = None,
        imputation_threshold: float = 100.0,
        save_transforms: bool = False,
        save_directory: typing.Optional[str] = None,
        target_normalization: TargetNormalization = TargetNormalization.STATION_PARAMS,
    ) -> None:
        """
        Configure paths, encoders, scalers, and processing parameters.

        Seeds ``random`` and ``numpy`` with 42 for reproducibility and loads
        the station metadata CSV so that ``metadata_cols`` reflects whatever
        calibration parameters are actually present in that file.

        Parameters
        ----------
        root_directory : str, optional
            PeMS root directory containing ``metadata/`` and
            ``timeseries_data/`` subdirectories, by default
            ``constants.ROOT_PATH``.
        metadata_filepath : typing.Optional[str], optional
            Explicit path to the station metadata CSV. If None, defaults to
            ``{root_directory}/metadata/station_metadata.csv``, by default
            None.
        imputation_threshold : float, optional
            Minimum value of ``pct_observed`` for a row to be kept during
            processing, by default 100.0.
        save_transforms : bool, optional
            If True, fitted encoders/scalers are written to disk during the
            pipeline, by default False.
        save_directory : typing.Optional[str], optional
            Directory that pipeline outputs (preprocessed CSV, torch dataset,
            fitted transforms, FD plots) are written to when no explicit path
            is passed to the individual methods. Created if missing. If None,
            falls back to ``{root_directory}/processed_data``, by default
            None.
        target_normalization : TargetNormalization, optional
            Strategy used by :meth:`normalize_timeseries` to normalize the
            flow and density target columns, by default
            ``TargetNormalization.STATION_PARAMS``.
        """
        # Set seeds for reproducibility
        random.seed(42)
        np.random.seed(42)  # For reproducibility

        # Define paths
        self.root_directory = root_directory
        self.metadata_path = metadata_filepath or os.path.join(
            root_directory, "metadata", "station_metadata.csv"
        )
        self.timeseries_directory = os.path.join(root_directory, "timeseries_data")
        self.processed_data_directory = os.path.join(root_directory, "processed_data")
        if not os.path.exists(self.processed_data_directory):
            os.mkdir(self.processed_data_directory)

        # Resolve the active save directory.
        if save_directory is not None:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory, exist_ok=True)
            self._save_dir: typing.Optional[str] = save_directory
        else:
            self._save_dir = None

        self.save_transforms = save_transforms

        # Initialized encoders and scalers
        self.encoder = OrdinalEncoder()
        self.scaler = MinMaxScaler()

        # Map days of week to integer values
        self.day_of_week_mapping = {
            0: "Mon",
            1: "Tue",
            2: "Wed",
            3: "Thu",
            4: "Fri",
            5: "Sat",
            6: "Sun",
        }

        # Set parameters
        self.imputation_threshold = imputation_threshold
        self.target_normalization = TargetNormalization(target_normalization)

        # Load files
        self.timeseries_cols = ['timestamp', 'station', 'pct_observed', 'total_flow_[veh/5-min]', 'avg_speed_[mph]']
        self.metadata_base_cols = [
            'Station ID', 'Lanes', 'Type', 'Abs PM', 'Length'
        ]
        self.calibration_params = ['capacity', 'free_flow_speed', 'congestion_wave_speed', 'critical_density']
        self.metadata_df = pd.read_csv(
            self.metadata_path,
        )
        self.metadata_cols = self.metadata_base_cols + [
            col for col in self.calibration_params if col in self.metadata_df.columns
        ]

    @staticmethod
    def generate_mock_data(n_stations: int = 1) -> pd.DataFrame:
        """
        Generate a synthetic long-form DataFrame for testing.

        Produces one year of 5-minute rows per station with random metadata
        (capacity, densities, speeds, postmile, length) and random flow and
        density timeseries drawn uniformly within physically reasonable
        limits.

        Parameters
        ----------
        n_stations : int, optional
            Number of synthetic stations to generate, by default 1.

        Returns
        -------
        pd.DataFrame
            Long-form DataFrame with timeseries and metadata columns for each
            station.
        """
        # Generate timestamps
        n_periods = 365 * 24 * 12  # 365 days * 24 hours * 12 (5-min intervals)
        start_time = pd.Timestamp("2022-01-01 00:00:00")  # January 1
        timestamps = pd.date_range(start=start_time, periods=n_periods, freq="5min")

        long_df = pd.DataFrame()

        for i in range(n_stations):
            # Station metadata
            station_id = i                                        # PeMS key
            lanes = np.random.randint(0,6)                        # Number of lanes
            capacity = np.random.uniform(1800, 2400)              # Segment capacity [veh/hr-lane]
            free_flow_speed = np.random.uniform(45,65)            # miles/hr
            congestion_wave_speed = np.random.uniform(10,25)      # miles/hr
            jam_density = np.random.uniform(150,200)              # Segment jam density [veh/mi-lane]
            critical_density = np.random.uniform(25,35)           # Segment critical density [veh/mi-lane]
            abs_pm = float(i)                                     # Absolute postmile
            length = np.random.uniform(0.1,1.0)                   # Segment length [mi]

            # Station timeseries
            flow = np.random.uniform(0,capacity,n_periods)
            density = np.random.uniform(0,jam_density, n_periods)

            # Build dataframe for this station
            df = pd.DataFrame({
                'time_of_day': (timestamps.minute / 60) + (timestamps.hour),
                'weekday': timestamps.weekday,
                'year': timestamps.year,
                'timestamp': timestamps,
                'normalized_flow_[veh/hr-lane]': flow,
                'normalized_density_[veh/mi-lane]': density,
                'Station ID': station_id,
                'Lanes': lanes,
                'Abs PM': abs_pm,
                'Length': length,
                'capacity': capacity,
                'free_flow_speed': free_flow_speed,
                'congestion_wave_speed': congestion_wave_speed,
                'jam_density': jam_density,
                'critical_density': critical_density
            })

            # Concatenate onto long dataframe
            long_df = pd.concat((long_df, df))

        return long_df

    def retrieve_detectors(
        self, 
        detector_type: typing.Optional[str] = None
    ) -> list[str]:
        """
        Returns a list of VDSs that appear in both the metadata file and timeseries directory

        Parameters
        ----------
        detector_type : typing.Optional[str], optional
            Filter for a specific kind of detector (e.g., Mainline, On Ramp, Off Ramp), by default None

        Returns
        -------
        list[str]
            List of VDSs
        """
        # Load the metadata
        metadata = pd.read_csv(self.metadata_path)

        # Get a set of detectors matching the specified type from the metadata
        if detector_type is None:
            mask = pd.Series(True, index=metadata.index)
        else:
            mask = metadata['Type'] == detector_type
        detectors_in_metadata = set(
            metadata.loc[mask, "Station ID"].astype(str).to_list()
        )

        # Get a set of detectors from the timeseries directory
        detectors_with_timeseries = set(
            [
                filename.split(".csv")[0]
                for filename in os.listdir(path=self.timeseries_directory)
                if filename.endswith(".csv")
            ]
        )

        # Cross-reference the two sets to get detectors matching the specified type with metadata and timeseries data
        detectors = list(detectors_in_metadata.intersection(detectors_with_timeseries))

        return detectors

    def load_data_by_id(
        self,
        station_id: str
        ) -> pd.DataFrame:
        """
        Load the timeseries CSV for ``station_id`` and merge station metadata.

        Reindexes the timeseries to a complete 5-minute grid spanning the
        year range observed in the file (filling missing rows with NaN), then
        inner-joins with ``self.metadata_df`` on ``Station ID``.

        Parameters
        ----------
        station_id : str
            VDS station ID; must correspond to ``{station_id}.csv`` in the
            timeseries directory.

        Returns
        -------
        pd.DataFrame
            Merged timeseries + metadata DataFrame.
        """
        # Build the full filepath based on the station ID
        filename = f"{station_id}.csv"
        filepath = os.path.join(self.timeseries_directory, filename)

        # Load the timeseries dataframe
        timeseries_df = pd.read_csv(filepath, usecols=self.timeseries_cols)

        # Ensure that the timestamp column exists and is a datetime object
        timeseries_df = validation.validate_timestamp_dtype(timeseries_df)

        # Ensure complete indices
        years_in_df = timeseries_df['timestamp'].dt.year.unique()
        range_start = f"{str(years_in_df.min())}-01-01 00:00"
        range_end = f"{str(years_in_df.max())}-12-31 23:59"
        full_date_range = pd.date_range(start=range_start, end=range_end, freq='5min')
        df_reindexed = timeseries_df.set_index('timestamp').reindex(full_date_range)
        df_reindexed.reset_index(names='timestamp', inplace=True)

        # Broadcast station onto reindexed gap rows so they survive the inner merge below.
        df_reindexed['station'] = int(station_id)

        # Merge the metadata into the timeseries
        metadata_to_merge = self.metadata_df.loc[:, self.metadata_cols]
        df = df_reindexed.rename(columns={'station': 'Station ID'}).merge(right=metadata_to_merge, how='inner', on='Station ID')

        return df

    def standardize_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the raw counts (flow, speed) in the timeseries data.
        Standardization includes:
            - Calculating per-hour, per-lane flow from 5-minute, station-wide flow. Result has units [veh/hr*lanes]
            - Calculating per-hour, per-lane density from flow and average station speed. Result has units [veh/mi*lanes]

        Parameters
        ----------
        df : pd.DataFrame
            The timeseries data

        Returns
        -------
        pd.DataFrame
            Standardized version of timeseries data.
            Changes are:
                - ADDED column 'flow_[veh/hr-lane]'
                - ADDED column 'density_[veh/mi-lane]'
                - DROPPED columns ['total_flow_[veh/5-min]', 'avg_speed_[mph]']
        """
        # Number of 5-minute periods per hour (used for normalization)
        periods_per_hour = 12

        # Scale the flow: [veh/5-min] * [5-min/hr] * [1/lanes] = [veh/hr*lanes]
        df["flow_[veh/hr-lane]"] = (
            df["total_flow_[veh/5-min]"] * periods_per_hour
        ) / df["Lanes"]

        # Scale the density: [veh/5-min] * [5-min/h] * [1/lanes] / [mi/h] = [veh/mi*lanes]
        df["density_[veh/mi-lane]"] = df["flow_[veh/hr-lane]"] / df["avg_speed_[mph]"]

        # Drop old columns
        df = df.drop(
            columns=[
                "total_flow_[veh/5-min]",
                "avg_speed_[mph]",
            ]
        )

        return df

    def get_daily_aggregation(
        self,
        df: pd.DataFrame,
        colname: str,
        aggregation_type: str = "mean",
    ) -> pd.DataFrame:
        """
        Aggregate a timeseries column by (weekday, 5-minute-of-day).

        The resulting DataFrame has one row per 5-minute block in the day and
        one column per weekday (named with the short day-of-week label from
        ``self.day_of_week_mapping``). The index is a timestamp anchored at
        1970-01-01 whose time-of-day matches the block.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a ``timestamp`` column and ``colname``.
        colname : str
            Column in ``df`` to aggregate.
        aggregation_type : str, optional
            Either ``"mean"`` or ``"std_dev"``, by default ``"mean"``.

        Returns
        -------
        pd.DataFrame
            Aggregated DataFrame indexed by within-day timestamp.

        Raises
        ------
        ValueError
            If ``aggregation_type`` is not one of the accepted values.
        """
        # Pick out relevant columns from the df
        sub_df = df.loc[:, ["timestamp", colname]]

        # Generate some derived temporal qualities from the timestamp
        sub_df["weekday"] = sub_df["timestamp"].dt.weekday
        sub_df["5min_block"] = (sub_df["timestamp"].dt.minute / 60) + (
            sub_df["timestamp"].dt.hour
        )
        sub_df.drop(columns=["timestamp"], inplace=True)

        # Calculate the aggregation type for the metric for each (t,weekday) index
        if aggregation_type == "mean":
            aggregation_df = (
                sub_df.groupby(["5min_block", "weekday"]).mean().unstack("weekday")
            )
        elif aggregation_type == "std_dev":
            aggregation_df = (
                sub_df.groupby(["5min_block", "weekday"]).std().unstack("weekday")
            )
        else:
            raise ValueError(
                f"Unknown aggregation type provided: {aggregation_type}. Please use one of ['mean','std_dev']"
            )

        # Drop the metric index and map the weekday integers onto days of the week
        aggregation_df.columns = aggregation_df.columns.get_level_values("weekday").map(
            self.day_of_week_mapping
        )

        # Translate the time information in '5min_block' back into a timestamp & use this as the index
        aggregation_df.reset_index(inplace=True)
        aggregation_df["timestamp"] = (
            pd.to_timedelta(aggregation_df["5min_block"], unit="h")
            .apply(lambda x: (pd.Timestamp("1970-01-01") + x))
            .dt.round(freq="5min")
        )
        aggregation_df.set_index("timestamp", inplace=True)

        return aggregation_df

    def get_agg_value(self, row: pd.Series, agg_df: pd.DataFrame) -> float:
        """
        Look up an aggregated statistic (e.g., mean, std_dev) for a given time block (expressed as a (day_of_week, 5min_block) combination)

        Parameters
        ----------
        row : pd.Series
            A row in the full timeseries dataframe for a given VDS
        agg_df : pd.DataFrame
            A DataFrame containing the desired aggregation statistics for the VDS

        Returns
        -------
        agg: float
            The value in agg_df corresponding to the (day_of_week, 5min_block) combination in row
        """
        day_of_week = self.day_of_week_mapping.get(row["weekday"], np.nan)
        time_block = row["5min_block"]
        agg = agg_df.loc[agg_df["5min_block"] == time_block, day_of_week].values[0]
        return agg

    def whiten_timeseries(self, timeseries_df: pd.DataFrame) -> pd.DataFrame:
        """
        Whiten flow and density against their (weekday, 5-minute) statistics.

        For each row, subtracts the per-(weekday, 5-minute-block) mean and
        divides by the per-(weekday, 5-minute-block) standard deviation for
        flow and density. Adds the intermediate mean/stddev columns and the
        resulting ``whitened_flow`` / ``whitened_density`` columns; these are
        used downstream for outlier filtering and for
        ``TargetNormalization.WHITENED``.

        Parameters
        ----------
        timeseries_df : pd.DataFrame
            DataFrame with ``timestamp``, ``flow_[veh/hr-lane]``, and
            ``density_[veh/mi-lane]`` columns.

        Returns
        -------
        pd.DataFrame
            Copy of the input with the derived ``weekday``, ``5min_block``,
            ``mean_*``, ``stddev_*``, and ``whitened_*`` columns added.
        """
        # Make a local copy to avoid writing over the input dataframe
        df = timeseries_df.copy()

        # Generate some derived temporal qualities from the timestamp
        df["weekday"] = df["timestamp"].dt.weekday
        df["5min_block"] = (df["timestamp"].dt.minute / 60) + (df["timestamp"].dt.hour)

        # Get the mean and standard deviation for flow and density
        mean_flow_df = self.get_daily_aggregation(
            df,
            colname="flow_[veh/hr-lane]",
            aggregation_type="mean",
        )
        stddev_flow_df = self.get_daily_aggregation(
            df,
            colname="flow_[veh/hr-lane]",
            aggregation_type="std_dev",
        )
        mean_density_df = self.get_daily_aggregation(
            df,
            colname="density_[veh/mi-lane]",
            aggregation_type="mean",
        )
        stddev_density_df = self.get_daily_aggregation(
            df,
            colname="density_[veh/mi-lane]",
            aggregation_type="std_dev",
        )

        # Add columns for mean and standard deviation
        df["mean_flow"] = df.apply(
            lambda row: self.get_agg_value(row, agg_df=mean_flow_df), axis=1
        )
        df["stddev_flow"] = df.apply(
            lambda row: self.get_agg_value(row, agg_df=stddev_flow_df), axis=1
        )
        df["mean_density"] = df.apply(
            lambda row: self.get_agg_value(row, agg_df=mean_density_df), axis=1
        )
        df["stddev_density"] = df.apply(
            lambda row: self.get_agg_value(row, agg_df=stddev_density_df), axis=1
        )

        # Whiten the flow and density timeseries
        df["whitened_flow"] = (df["flow_[veh/hr-lane]"] - df["mean_flow"]) / df[
            "stddev_flow"
        ]
        df["whitened_density"] = (
            df["density_[veh/mi-lane]"] - df["mean_density"]
        ) / df["stddev_density"]

        # Return the result
        return df

    def filter_flow_outliers(
        self,
        timeseries_df: pd.DataFrame,
        flow_col: str = "whitened_flow",
        iqr_multiplier: float = 1.5,
    ) -> pd.DataFrame:
        """
        Drop rows whose ``flow_col`` falls outside an IQR-based fence.

        Computes Q1/Q3 of ``flow_col`` and keeps rows in
        ``[Q1 - k*IQR, Q3 + k*IQR]``, where ``k = iqr_multiplier``.

        Parameters
        ----------
        timeseries_df : pd.DataFrame
            DataFrame containing ``flow_col``.
        flow_col : str, optional
            Column used to identify outliers, by default ``"whitened_flow"``.
        iqr_multiplier : float, optional
            Fence multiplier; larger values are more permissive, by default
            1.5.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with the index reset.
        """
        q1 = timeseries_df[flow_col].quantile(0.25)
        q3 = timeseries_df[flow_col].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - iqr_multiplier * iqr
        upper_fence = q3 + iqr_multiplier * iqr
        mask = timeseries_df[flow_col].between(lower_fence, upper_fence)
        n_removed = (~mask).sum()
        if n_removed > 0:
            logger.debug(
                f"Outlier filter removed {n_removed} rows "
                f"(flow outside [{lower_fence:.1f}, {upper_fence:.1f}] veh/hr-lane)"
            )
        return timeseries_df.loc[mask].reset_index(drop=True)

    def normalize_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the flow and density target columns in the timeseries data.

        Normalization strategy is controlled by ``self.target_normalization``:

        - ``TargetNormalization.STATION_PARAMS`` (default): divide flow by station
          capacity and density by station critical density. Flow values range
          [0,1]; density values range [0,1] in free-flow and (1,inf) in
          congestion.
        - ``TargetNormalization.WHITENED``: whiten flow and density against their
          per-(weekday, 5min_block) mean and stddev.
        - ``TargetNormalization.MIN_MAX``: min/max scale flow and density using
          the global min and max observed across all detectors in ``df``. If
          ``self.save_transforms`` is set, the fitted scaler is saved to disk
          as ``target_minmax_scaler.joblib``.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the standardized ``flow_[veh/hr-lane]`` and
            ``density_[veh/mi-lane]`` columns.

        Returns
        -------
        pd.DataFrame
            Modified version of the passed DataFrame.
            Modifications include:
                - ADDED column 'flow'
                - ADDED column 'density'
                - DROPPED columns ['flow_[veh/hr-lane]', 'density_[veh/mi-lane]']
        """
        if self.target_normalization == TargetNormalization.STATION_PARAMS:
            # Normalize flow by station capacity
            df["flow"] = df["flow_[veh/hr-lane]"] / df["capacity"]

            # Normalize density by critical density
            df["density"] = df["density_[veh/mi-lane]"] / df["critical_density"]

            # Drop unnecessary columns
            df = df.drop(columns=["flow_[veh/hr-lane]", "density_[veh/mi-lane]"])
        elif self.target_normalization == TargetNormalization.WHITENED:
            # Whiten per station so the (weekday, 5-minute) mean/stddev are
            # specific to each station rather than pooled across all of them.
            df = df.groupby("Station ID", group_keys=False).apply(
                self.whiten_timeseries
            )

            # Set flow, density to the whitened values
            df["flow"] = df["whitened_flow"]
            df["density"] = df["whitened_density"]

            # Drop unnecessary columns
            df = df.drop(
                columns=[
                    "flow_[veh/hr-lane]",
                    "mean_flow",
                    "stddev_flow",
                    "whitened_flow",
                    "density_[veh/mi-lane]",
                    "mean_density",
                    "stddev_density",
                    "whitened_density",
                    "weekday",
                    "5min_block",
                ]
            )
        elif self.target_normalization == TargetNormalization.MIN_MAX:
            # Fit a MinMaxScaler across all detectors using the (flow, density)
            # columns, then apply it. The scaler is stored on the instance so
            # downstream code can invert the transform if needed.
            flow_density = df[["flow_[veh/hr-lane]", "density_[veh/mi-lane]"]].to_numpy()
            self.target_minmax_scaler = MinMaxScaler()
            scaled = self.target_minmax_scaler.fit_transform(flow_density)
            df["flow"] = scaled[:, 0]
            df["density"] = scaled[:, 1]
            logger.debug(
                "Applied global MinMax scaling to flow/density: "
                f"flow min/max = {self.target_minmax_scaler.data_min_[0]:.2f}/"
                f"{self.target_minmax_scaler.data_max_[0]:.2f}, "
                f"density min/max = {self.target_minmax_scaler.data_min_[1]:.2f}/"
                f"{self.target_minmax_scaler.data_max_[1]:.2f}"
            )

            # Optionally persist the fitted scaler alongside other transforms.
            if self.save_transforms:
                _out_dir = (
                    self._save_dir
                    if self._save_dir is not None
                    else self.processed_data_directory
                )
                target_scaler_filepath = os.path.join(
                    _out_dir, "target_minmax_scaler.joblib"
                )
                dump(self.target_minmax_scaler, target_scaler_filepath)
                logger.info(
                    f"Saved target min/max scaler to: {target_scaler_filepath}"
                )

            # Drop unnecessary columns
            df = df.drop(columns=["flow_[veh/hr-lane]", "density_[veh/mi-lane]"])
        else:
            raise ValueError(
                f"Unsupported target_normalization: {self.target_normalization!r}"
            )

        return df

    def build_long_df(self, detectors: typing.Optional[list[str]] = None):
        """
        Generate a single DataFrame containing data for multiple VDSs.
        DataFrame contains a mix of timeseries and metadata for each VDS 
        (e.g., free_flow_speed (metadata) and flow (timeseries)).
        Timeseries data is normalized according to hourly, per-lane counts and the relevant limit
        (station capacity for flow, station critical density for density)

        Parameters
        ----------
        detectors : typing.Optional[list[str]], optional
            A list of VDS station IDs. Useful for avoiding processing of every VDS available, by default None

        Returns
        -------
        pd.DataFrame
            The long-form DataFrame
        """
        # Get a list of detectors to work on
        if detectors is None:
            detectors = self.retrieve_detectors(detector_type='Mainline')

        # Load dataframe for each detector
        detector_dfs = []
        N_detectors = len(detectors)
        for idx, detector in enumerate(detectors):
            logger.info(f"Loading dataframe for detector {idx + 1}/{N_detectors}: VDS {detector}")
            detector_dfs.append(self.load_data_by_id(detector))

        # Stack dataframes into one
        df = pd.concat(detector_dfs)

        # NaN out raw counts below the imputation threshold at the source, so no
        # downstream transform operates on sub-threshold data and the per-station
        # 5-minute grid stays contiguous (required by the sequence/imputation
        # dataset builders). NaN propagates through the arithmetic in both
        # standardize_timeseries and normalize_timeseries.
        mask = df["pct_observed"] < self.imputation_threshold
        n_nulled = mask.sum()
        logger.debug(
            f"{((n_nulled / len(df)) * 100):.2f} percent of"
            f" datapoints NaN'd in dataframe of length {len(df)}:"
            f" pct_observed < {self.imputation_threshold}"
        )
        df.loc[mask, ["total_flow_[veh/5-min]", "avg_speed_[mph]"]] = np.nan

        # Standardize and normalize counts
        df = self.standardize_timeseries(df)
        df = self.normalize_timeseries(df)

        return df

    def deconstruct_timestamps(
        self,
        df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Transform the 'timestamp' column into derived time-related columns, e.g., 'hour'.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with timeseries data

        Returns
        -------
        pd.DataFrame
            Modified version of the passed DataFrame. Modifications are:
                - DROPPED column 'timestamp'
                - ADDED column 'year'
                - ADDED column 'month'
                - ADDED column 'day'
                - ADDED column 'dayofweek'
                - ADDED column 'hour'
                - ADDED column 'minute'
                - ADDED column '5min_block'
        """
        # Validate timestamp column
        df = validation.validate_timestamp_dtype(df)

        # Assign derived time-related columns
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['5min_block'] = df['minute']/60 + df['hour']

        # Drop timestamp column
        df = df.drop(columns=['timestamp'])

        return df

    def encode_and_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ordinal-encode categorical features and min/max-scale numeric ones.

        Duplicates ``Station ID``, ``free_flow_speed``, and
        ``congestion_wave_speed`` so that each can serve as both an unscaled
        metadata column and a scaled/encoded feature. If
        ``self.save_transforms`` is set, the fitted encoder and scaler are
        written to ``encoder.joblib`` and ``scaler.joblib`` in the resolved
        save directory.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame already processed through :meth:`deconstruct_timestamps`.

        Returns
        -------
        pd.DataFrame
            DataFrame with encoded categorical features, scaled numeric
            features, and added ``*_scaled`` / ``*_encoded`` columns.
        """
        # Duplicate Station ID, free flow speed, and congestion wave speed
        # These columns are both metadata and useful features
        df["Station ID (encoded)"] = df.loc[:, "Station ID"]
        df["free_flow_speed_scaled"] = df.loc[:, "free_flow_speed"]
        df["congestion_wave_speed_scaled"] = df.loc[:, "congestion_wave_speed"]

        categorical_features = ["Station ID (encoded)", "Type"]
        df.loc[:, categorical_features] = self.encoder.fit_transform(df[categorical_features])
        df[categorical_features] = df[categorical_features].astype('float')     # Cast as float after encoding

        minmax_features = [
            "pct_observed",
            "Abs PM",
            "Length",
            "free_flow_speed_scaled",
            "congestion_wave_speed_scaled",
            "year",
            "month",
            "day",
            "dayofweek",
            "hour",
            "minute",
            "5min_block",
        ]
        df[minmax_features] = df[minmax_features].astype('float')   # Cast as float ahead of scaling
        df.loc[:, minmax_features] = self.scaler.fit_transform(df[minmax_features])

        if self.save_transforms:
            # Write to self._save_dir if set, otherwise fall back to processed_data_directory.
            _out_dir = self._save_dir if self._save_dir is not None else self.processed_data_directory
            encoder_filepath = os.path.join(_out_dir, "encoder.joblib")
            scaler_filepath = os.path.join(_out_dir, "scaler.joblib")

            dump(self.encoder, encoder_filepath)
            logger.info(f"Saved encoder to: {encoder_filepath}")

            dump(self.scaler, scaler_filepath)
            logger.info(f"Saved scaler to: {scaler_filepath}")

        return df

    def estimate_free_flow_speed(
        self,
        timeseries_df: pd.DataFrame,
        critical_density_estimate: float,
        test_points: list[float],
    ) -> tuple[float, LinearRegression]:
        """
        Fit a no-intercept line to the free-flow regime of an FD.

        Keeps rows with density ≤ ``critical_density_estimate`` and fits
        ``flow = v_f * density`` (no intercept); the slope is the free-flow
        speed estimate.

        Parameters
        ----------
        timeseries_df : pd.DataFrame
            DataFrame with ``flow_[veh/hr-lane]`` and ``density_[veh/mi-lane]``
            columns.
        critical_density_estimate : float
            Density boundary below which points are considered free-flow.
        test_points : list[float]
            Density values at which to log the fitted model's predictions.

        Returns
        -------
        tuple[float, LinearRegression]
            Free-flow speed (the fitted slope) and the underlying sklearn
            model.
        """

        # 1. Filter points with density <= critical_density_estimate
        free_flow_points = timeseries_df.loc[
            timeseries_df["density_[veh/mi-lane]"] <= critical_density_estimate, :
        ].copy()

        # 2. Fit regression
        X = free_flow_points["density_[veh/mi-lane]"].copy().to_numpy().reshape(-1, 1)
        y = free_flow_points["flow_[veh/hr-lane]"].copy().to_numpy()

        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        free_flow_speed = model.coef_[0]

        # Print the results
        logger.debug(f"Free flow speed model fit score: {model.score(X, y)}")
        for test_point in test_points:
            test_point_arr = np.array(test_point).reshape(-1, 1)
            logger.debug(
                f"Model predicts flow = {model.predict(test_point_arr)} for density = {test_point_arr}"
            )

        return free_flow_speed, model

    def estimate_congestion_wave_speed(
        self,
        timeseries_df: pd.DataFrame,
        critical_density_estimate: float,
        test_points: list[float],
        bin_size: int = 10,
        iqr_multiplier: float = 1.0,
    ) -> tuple[float, float, LinearRegression, pd.DataFrame]:
        """
        Fit the congestion regime of an FD via binned peak-flow regression.

        Restricts to rows with density > ``critical_density_estimate``, sorts
        by density, and partitions into non-overlapping bins of ``bin_size``
        rows. Within each bin, drops whitened-flow outliers (above
        ``Q3 + iqr_multiplier * IQR``) and records the bin's mean density and
        max flow. Fits a straight line ``flow = w * density + b`` to those
        bin points; ``w`` is the congestion-wave slope and ``-b/w`` is the
        jam density.

        Parameters
        ----------
        timeseries_df : pd.DataFrame
            DataFrame with ``flow_[veh/hr-lane]``, ``density_[veh/mi-lane]``,
            and ``whitened_flow`` columns.
        critical_density_estimate : float
            Density boundary above which points are considered congested.
        test_points : list[float]
            Density values at which to log the fitted model's predictions.
        bin_size : int, optional
            Number of points per non-overlapping density bin, by default 10.
        iqr_multiplier : float, optional
            IQR-fence multiplier applied per bin to reject high-flow
            outliers, by default 1.0.

        Returns
        -------
        tuple[float, float, LinearRegression, pd.DataFrame]
            ``(wave_speed, jam_density, model, bin_df)`` where ``bin_df``
            has columns ``BinDensity`` and ``BinFlow``.
        """

        # 1. Filter points with density > critical_density_estimate (congested regime)
        congested = timeseries_df.loc[
            timeseries_df["density_[veh/mi-lane]"] > critical_density_estimate, :
        ].copy()
        congested = congested.sort_values(by="density_[veh/mi-lane]").reset_index(
            drop=True
        )

        # 2. Partition into non-overlapping bins of bin_size
        num_bins = len(congested) // bin_size
        bin_densities = []
        bin_flows = []

        for i in range(num_bins):
            bin_data = congested.iloc[i * bin_size : (i + 1) * bin_size]
            flows = bin_data["flow_[veh/hr-lane]"].values
            densities = bin_data["density_[veh/mi-lane]"].values
            whitened_flows = bin_data["whitened_flow"].values

            # Compute IQR on whitened_flow
            Q1 = np.percentile(whitened_flows, 25)
            Q3 = np.percentile(whitened_flows, 75)
            IQR = Q3 - Q1
            outlier_threshold = Q3 + iqr_multiplier * IQR

            # Build a valid mask from whitened_flow, apply to both flows and densities
            valid_mask = whitened_flows <= outlier_threshold

            valid_flows = flows[valid_mask]
            valid_densities = densities[valid_mask]

            if len(valid_flows) == 0:
                continue

            BinFlow = np.max(valid_flows)
            BinDensity = np.mean(valid_densities)

            bin_densities.append(BinDensity)
            bin_flows.append(BinFlow)

        # Convert to DataFrame
        bin_df = pd.DataFrame({"BinDensity": bin_densities, "BinFlow": bin_flows})

        # 3. Fit unconstrained regression
        X = bin_df["BinDensity"].values.reshape(-1, 1)
        y = bin_df["BinFlow"].values

        model = LinearRegression()
        model.fit(X, y)
        wave_speed = model.coef_[0]

        # 4. Compute x-intercept (jam density): flow = 0 => density = -intercept / wave_speed
        jam_density = -model.intercept_ / wave_speed

        # Print the results
        logger.debug(f"Congestion wave model fit score: {model.score(X, y)}")
        for test_point in test_points:
            test_point_arr = np.array(test_point).reshape(-1, 1)
            logger.debug(
                f"Model predicts flow = {model.predict(test_point_arr)} for density = {test_point_arr}"
            )

        return wave_speed, jam_density, model, bin_df

    def find_fundamental_diagram_peak(
        self,
        free_flow_model: LinearRegression,
        congestion_model: LinearRegression,
    ) -> tuple[float, float]:
        """
        Find the intersection of the free flow and congestion fit lines.
        Free flow model:  flow = ff_slope * density                (no intercept)
        Congestion model: flow = cw_slope * density + cw_intercept
        Solving: ff_slope * density = cw_slope * density + cw_intercept
                density = cw_intercept / (ff_slope - cw_slope)
        """
        ff_slope = free_flow_model.coef_[0]
        cw_slope = congestion_model.coef_[0]
        cw_intercept = congestion_model.intercept_

        critical_density = cw_intercept / (ff_slope - cw_slope)
        max_capacity = ff_slope * critical_density

        logger.debug(f"Critical density: {critical_density:.4f} veh/mi-lane")
        logger.debug(f"Max capacity: {max_capacity:.4f} veh/hr-lane")

        return critical_density, max_capacity

    @staticmethod
    def plot_fundamental_diagram(
        timeseries_df: pd.DataFrame,
        bin_df: pd.DataFrame,
        free_flow_model: LinearRegression,
        congestion_model: LinearRegression,
        params: dict[str, float],
        save_path: typing.Optional[Path] = None,
    ) -> None:
        """
        Render a fundamental-diagram plot for a calibrated detector.

        Overlays the raw (density, flow) scatter, the binned congestion
        points, the fitted free-flow and congestion lines, horizontal /
        vertical reference lines at capacity / critical / jam densities, and
        a text box summarizing the calibrated parameters.

        Parameters
        ----------
        timeseries_df : pd.DataFrame
            DataFrame with ``flow_[veh/hr-lane]`` and
            ``density_[veh/mi-lane]`` columns.
        bin_df : pd.DataFrame
            Binned congestion points with ``BinDensity`` and ``BinFlow``.
        free_flow_model : LinearRegression
            Fitted free-flow-regime model.
        congestion_model : LinearRegression
            Fitted congestion-regime model.
        params : dict[str, float]
            Calibration parameters; must contain ``Station ID``,
            ``capacity``, ``free_flow_speed``, ``congestion_wave_speed``,
            ``jam_density``, and ``critical_density``.
        save_path : typing.Optional[Path], optional
            If provided, save the figure here instead of calling
            ``plt.show()``.

        Returns
        -------
        None
            Plots as a side effect.
        """
        # Generate some prediction points using the models
        free_flow_xvals = np.linspace(
            0,
            params["critical_density"],
        ).reshape(-1, 1)
        free_flow_pred = free_flow_model.predict(free_flow_xvals)

        congestion_xvals = np.linspace(
            bin_df["BinDensity"].min(), bin_df["BinDensity"].max(), 100
        )
        congestion_slope = congestion_model.coef_[0]
        congestion_pred = (
            congestion_slope * (congestion_xvals - params["critical_density"])
            + params["capacity"]
        )

        plt.rcParams.update(
            {
                "font.size": 32,
                "axes.titlesize": 32,
                "axes.labelsize": 28,
                "xtick.labelsize": 22,
                "ytick.labelsize": 22,
                "legend.fontsize": 20,
            }
        )

        fig, ax = plt.subplots(figsize=(20, 13))

        # Plot the data points
        plt.scatter(
            timeseries_df["density_[veh/mi-lane]"],
            timeseries_df["flow_[veh/hr-lane]"],
            label="Raw (data)",
            color="tab:blue",
            marker=".",
            s=5,
            alpha=0.3,
        )
        plt.plot(
            bin_df["BinDensity"],
            bin_df["BinFlow"],
            color="gray",
            alpha=0.2,
            marker="o",
            markersize=5,
            linestyle="-",
            label="Congestion Bins",
        )

        # Plot the regression lines
        plt.plot(
            free_flow_xvals,
            free_flow_pred,
            color="tab:orange",
            linewidth=3,
            label="Free flow (fit)",
        )
        plt.plot(
            congestion_xvals,
            congestion_pred,
            color="tab:green",
            linewidth=3,
            label="Congestion (fit)",
        )

        # Plot a horizontal line for capacity
        plt.axhline(
            y=params["capacity"], color="gray", linestyle="--", label="Capacity"
        )

        # Plot vertical lines for critical and jam densities
        plt.axvline(
            x=params["critical_density"],
            color="gray",
            linestyle="-.",
            label="Critical Density",
        )
        plt.axvline(
            x=params["jam_density"], color="gray", linestyle=":", label="Jam Density"
        )

        # Set limits
        ax.set_ylim(-25, 2000)
        ax.set_xlim(-10, 200)

        # Add plot features
        plt.xlabel("Density (Vehicles Per Mile Per Lane)")
        plt.ylabel("Flow (Vehicles Per Hour Per Lane)")
        plt.title(f"Fundamental Diagram for Detector: {params['Station ID']}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        params_as_text = "\n".join(
            [
                rf"$q_{{\max}}$ = {params['capacity']:.2f} vphpl",
                rf"$\rho_{{\mathrm{{jam}}}}$ = {params['jam_density']:.2f} vpmpl",
                rf"$\rho_{{\mathrm{{crit}}}}$ = {params['critical_density']:.2f} vpmpl",
                rf"$v_f$ = {params['free_flow_speed']:.2f} mph",
                rf"$w$ = {params['congestion_wave_speed']:.2f} mph",
            ]
        )
        plt.annotate(
            text=params_as_text,
            xy=(0.77, 0.25),
            xycoords="figure fraction",
            fontsize=24,
            bbox=dict(boxstyle="square,pad=0.5", fc="lightgray", ec="black", lw=2),
        )

        if save_path is None:
            # Show the plot
            plt.show()
        else:
            try:
                plt.savefig(save_path)
                logger.info(f"Figure for detector {params['Station ID']} saved to: {save_path}")
            except Exception as e:
                logger.error(
                    f"An exception was raised while saving the figure for detector {params['Station ID']}: {e}"
                )
        # Close the figure
        plt.close()

    def extract_fundamental_diagram_params_from_timeseries(
        self,
        timeseries_df: pd.DataFrame,
        detector_id: str,
        make_plots: bool = False,
        save_path: typing.Optional[Path] = None,
    ) -> dict[str, float]:
        """
        Derive the full set of FD parameters for a single detector.

        Seeds the critical-density estimate from the row with the highest
        observed flow, fits both regimes (free-flow and congestion), and
        recovers the capacity and critical density from their intersection.

        Parameters
        ----------
        timeseries_df : pd.DataFrame
            Whitened and outlier-filtered DataFrame for a single detector.
        detector_id : str
            VDS station ID associated with the data.
        make_plots : bool, optional
            If True, call :meth:`plot_fundamental_diagram`, by default False.
        save_path : typing.Optional[Path], optional
            Passed through to the plotter; ignored when ``make_plots`` is
            False, by default None.

        Returns
        -------
        dict[str, float]
            Dict with keys ``Station ID``, ``capacity``, ``free_flow_speed``,
            ``congestion_wave_speed``, ``jam_density``, ``critical_density``.
        """
        # Estimate critical density based on peak observed flow
        capacity_idx = timeseries_df["flow_[veh/hr-lane]"].idxmax()
        critical_density_estimate = timeseries_df.loc[
            capacity_idx, "density_[veh/mi-lane]"
        ]

        # Fit free flow speed and congestion wave speed with linear model
        free_flow_speed, free_flow_speed_model = self.estimate_free_flow_speed(
            timeseries_df=timeseries_df,
            critical_density_estimate=critical_density_estimate,
            test_points=[0.0, critical_density_estimate],
        )
        congestion_slope, jam_density, congestion_wave_speed_model, bin_df = (
            self.estimate_congestion_wave_speed(
                timeseries_df=timeseries_df,
                critical_density_estimate=critical_density_estimate,
                test_points=[critical_density_estimate],
            )
        )

        # Extract critical density and capacity from speed models
        critical_density, capacity = self.find_fundamental_diagram_peak(
            free_flow_model=free_flow_speed_model,
            congestion_model=congestion_wave_speed_model,
        )

        # Pack parameters dictionary
        params = {
            "Station ID": int(detector_id),
            "capacity": capacity,
            "free_flow_speed": free_flow_speed,
            "congestion_wave_speed": -1 * congestion_slope,
            "jam_density": jam_density,
            "critical_density": critical_density,
        }

        if make_plots:
            self.plot_fundamental_diagram(
                timeseries_df=timeseries_df,
                bin_df=bin_df,
                free_flow_model=free_flow_speed_model,
                congestion_model=congestion_wave_speed_model,
                params=params,
                save_path=save_path,
            )

        return params

    def calibrate_fundamental_diagrams(
        self,
        detectors: typing.Optional[list[str]] = None,
        filter_colname: str = "whitened_flow",
        iqr_multiplier: float = 1.0,
        make_plots: bool = False,
        saved_plot_dir: typing.Optional[Path] = None,
        save_params: bool = False,
        output_metadata_path: typing.Optional[str] = None,
    ) -> None:
        """
        Run FD calibration for one or more detectors and optionally persist it.

        For each detector, loads its timeseries, applies standardization,
        thresholds on ``pct_observed``, whitens, filters outliers, and fits
        the FD via
        :meth:`extract_fundamental_diagram_params_from_timeseries`. If
        ``save_params`` is True, the calibrated parameters are written back
        into ``self.metadata_df`` and the metadata is saved to CSV.

        Parameters
        ----------
        detectors : typing.Optional[list[str]], optional
            Subset of detectors to process. Defaults to all mainline
            detectors returned by :meth:`retrieve_detectors`.
        filter_colname : str, optional
            Column passed to :meth:`filter_flow_outliers`, by default
            ``"whitened_flow"``.
        iqr_multiplier : float, optional
            Outlier-filter IQR multiplier, by default 1.0.
        make_plots : bool, optional
            If True, render and save an FD plot per detector, by default
            False.
        saved_plot_dir : typing.Optional[Path], optional
            Destination directory for FD plots. If not provided and
            ``self._save_dir`` is set, defaults to a
            ``fundamental_diagram_plots`` subdirectory there.
        save_params : bool, optional
            If True, persist calibrated parameters back to metadata CSV, by
            default False.
        output_metadata_path : typing.Optional[str], optional
            Explicit output path for the calibrated metadata CSV; otherwise
            defaults to ``station_metadata_calibrated.csv`` inside the
            resolved save directory.
        """
        # Get a list of detectors to work on
        if detectors is None:
            detectors = self.retrieve_detectors(detector_type="Mainline")

        # If self._save_dir is set and no explicit saved_plot_dir was given,
        # auto-create a 'fundamental_diagram_plots' subdirectory within self._save_dir.
        if saved_plot_dir is None and self._save_dir is not None and make_plots:
            saved_plot_dir = Path(os.path.join(self._save_dir, "fundamental_diagram_plots"))

        # Ensure that save_dir is a directory and exists
        if saved_plot_dir is not None:
            if not os.path.exists(saved_plot_dir):
                os.mkdir(saved_plot_dir)
            assert saved_plot_dir.is_dir()

        # Calibrate each detector
        N_detectors = len(detectors)
        for idx, detector in enumerate(detectors):
            logger.info(f"Calibrating detector {idx + 1}/{N_detectors}: VDS {detector}")
            # Prepare a filename for the calibrated diagram plot
            if saved_plot_dir:
                save_path = Path(os.path.join(saved_plot_dir), f"{detector}.png")
            else:
                save_path = None

            # Load the detector data
            df = self.load_data_by_id(detector)

            # Standardize the timeseries
            df_standardized = self.standardize_timeseries(df)

            # Filter for measurements exceeding the imputation threshold
            len_original = len(df_standardized)
            df_standardized = df_standardized.loc[
                df_standardized["pct_observed"] >= self.imputation_threshold, :
            ].reset_index(drop=True)
            len_filtered = len(df_standardized)
            logger.debug(
                f"{(((len_original - len_filtered) / len_original) * 100):.2f} percent of"
                f" datapoints removed from dataframe of length {len_original}:"
                f" pct_observed < {self.imputation_threshold}"
            )
            if df_standardized.empty:
                logger.debug(f"Calibration failed for VDS: {detector}. Too many missing data points.")
                continue

            # Whiten the timeseries
            df_whitened = self.whiten_timeseries(df_standardized)

            # Filter outliers based on IQR of whitened flow
            df_filtered = self.filter_flow_outliers(
                df_whitened, flow_col=filter_colname, iqr_multiplier=iqr_multiplier
            )

            # Extract parameters
            params = self.extract_fundamental_diagram_params_from_timeseries(
                timeseries_df=df_filtered,
                detector_id=detector,
                make_plots=make_plots,
                save_path=save_path,
            )
            params_str = "\n".join(f"  {k}: {v:.3f}" for k, v in params.items())
            logger.info(f"Calibrated parameters:\n{params_str}")

            if save_params:
                # Update metadata_df with calibrated parameters
                for key, value in params.items():
                    self.metadata_df.loc[self.metadata_df["Station ID"] == int(detector), key] = value

        if save_params:
            # Resolve output path: explicit arg > self._save_dir > processed_data_directory fallback.
            _default_dir = self._save_dir if self._save_dir is not None else self.processed_data_directory
            output_path = output_metadata_path or os.path.join(_default_dir, "station_metadata_calibrated.csv")

            # Save metadata_df to a CSV file
            self.metadata_df.to_csv(output_path, index=False)
            logger.info(
                f"Updated version of metadata saved to: {output_path}"
            )

            # Refresh metadata_cols now that calibration params are present
            self.metadata_cols = self.metadata_base_cols + [
                col for col in self.calibration_params if col in self.metadata_df.columns
            ]

    def preprocess_data(
        self,
        detectors: typing.Optional[list[str]] = None,
        save_name: typing.Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run the full feature-building pipeline and (optionally) save the CSV.

        Calls :meth:`build_long_df`, :meth:`deconstruct_timestamps`, and
        :meth:`encode_and_scale` in sequence. Saves the resulting DataFrame
        if a name is given explicitly or if ``self._save_dir`` is set (in
        which case the file is named ``preprocessed_data.csv``).

        Parameters
        ----------
        detectors : typing.Optional[list[str]], optional
            Subset of detectors; forwarded to :meth:`build_long_df`, by
            default None.
        save_name : typing.Optional[str], optional
            Filename for the output CSV. If None and no save directory is
            set on the instance, the DataFrame is not saved.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame.
        """
        logger.info("Running preprocessing steps on PeMS data...")
        # Execute processing steps
        df = self.build_long_df(detectors=detectors)
        df = self.deconstruct_timestamps(df)
        df = self.encode_and_scale(df)

        # Determine where to save:
        #   1. If an explicit save_name was provided, honour it.
        #   2. Else if self._save_dir is set, auto-save with a default filename.
        #   3. Otherwise do not save.
        _resolved_save_name = save_name
        if _resolved_save_name is None and self._save_dir is not None:
            _resolved_save_name = "preprocessed_data.csv"

        if _resolved_save_name is not None:
            _out_dir = self._save_dir if self._save_dir is not None else self.processed_data_directory
            filepath = os.path.join(_out_dir, _resolved_save_name)
            df.to_csv(filepath, index=False)
            logger.info(f"Dataframe saved to: {filepath}")

        return df

    def prepare_sequence_dataset(
        self,
        df: pd.DataFrame,
        seq_len: int = 12,
        pred_horizon: int = 3,
        save_name: typing.Optional[str] = None,
    ) -> TensorDataset:
        """
        Build a rolling-window forecasting dataset from a preprocessed DF.

        For each station, slides a window of ``seq_len`` input rows followed
        by ``pred_horizon`` target rows. Sequences containing any NaN in
        features or targets are skipped. The returned ``TensorDataset`` is
        of shape ``(X, y, meta)``; optionally persists a dict to disk via
        :meth:`_build_dataset_save_dict`.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame produced by :meth:`preprocess_data`.
        seq_len : int, optional
            Number of rows used as input per sample, by default 12.
        pred_horizon : int, optional
            Number of rows predicted per sample, by default 3.
        save_name : typing.Optional[str], optional
            Filename for the saved ``.pt`` dataset. If None and
            ``self._save_dir`` is set, defaults to ``sequence_dataset.pt``.

        Returns
        -------
        TensorDataset
            Dataset yielding ``(features, targets, metadata)`` tuples.
        """
        # Columns to include in meta tensor
        meta_columns = [
            "Station ID",
            "Lanes",
            "capacity",
            "critical_density",
            "free_flow_speed",
            "congestion_wave_speed",
            "5min_block",
            "dayofweek",
        ]

        # Columns that appear only in meta, not in features
        meta_only_columns = [
            "Station ID",
            "Lanes",
            "capacity",
            "critical_density",
            "free_flow_speed",
            "congestion_wave_speed",
        ]

        # Columns to include in feature tensor
        feature_columns = [col for col in df.columns if col not in meta_only_columns]

        # Columns to include in target tensor
        target_columns = ["flow", "density"]

        # Empty lists for meta, features, and targets
        meta_list, X_list, y_list = [], [], []

        N_stations = len(df['Station ID'].unique())
        for idx, station_id in enumerate(df['Station ID'].unique()):
            logger.info(f"Preparing sequences for station {idx+1}/{N_stations}")
            station_df = df.loc[df['Station ID'] == station_id, :]

            for i in range(len(station_df) - (seq_len + pred_horizon) + 1):
                # Define indices
                target_idx_start = i + seq_len
                target_idx_end = target_idx_start + pred_horizon - 1

                # Select metadata, features, and targets
                metadata = station_df.loc[target_idx_start:target_idx_end, meta_columns]
                features = station_df.loc[i : target_idx_start - 1, feature_columns]
                targets = station_df.loc[
                    target_idx_start:target_idx_end, target_columns
                ]

                # Don't use any sequences with missing data
                if features.isna().any().any() or targets.isna().any().any():
                    continue

                # Convert to tensors
                metadata = torch.from_numpy(metadata.values).to(dtype=torch.float32)
                features = torch.from_numpy(features.values).to(dtype=torch.float32)
                targets = torch.from_numpy(targets.values).to(dtype=torch.float32)

                # Append to list
                meta_list.append(metadata)
                X_list.append(features)
                y_list.append(targets)

        # Convert lists to tensors
        meta = torch.stack(meta_list)
        X = torch.stack(X_list)
        y = torch.stack(y_list)

        # Create dataset
        dataset = TensorDataset(X, y, meta)

        # Determine where to save:
        #   1. Explicit save_name takes priority.
        #   2. Else auto-save to self._save_dir with a default filename if set.
        _resolved_save_name = save_name
        if _resolved_save_name is None and self._save_dir is not None:
            _resolved_save_name = "sequence_dataset.pt"

        if _resolved_save_name is not None:
            _out_dir = self._save_dir if self._save_dir is not None else self.processed_data_directory
            filepath = os.path.join(_out_dir, _resolved_save_name)
            save_dict = self._build_dataset_save_dict(X=X, y=y, meta=meta)
            torch.save(save_dict, filepath)
            logger.info(f"Saving sequence dataset to {filepath}")

        # Return
        return dataset

    def prepare_imputation_dataset(
        self,
        df: pd.DataFrame,
        seq_len: int = 36,
        missing_rate: float = 0.1,
        save_name: typing.Optional[str] = None,
    ) -> TensorDataset:
        """
        Build a masked-sequence imputation dataset from a preprocessed DF.

        For each station, partitions the timeseries into non-overlapping
        windows of length ``seq_len``. Sequences with any NaN are skipped.
        For each retained sequence, a random ``missing_rate`` fraction of
        rows is masked (indicated by a new ``observed`` metadata column).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame produced by :meth:`preprocess_data`.
        seq_len : int, optional
            Length of each non-overlapping window, by default 36.
        missing_rate : float, optional
            Fraction of rows to mask within each window, by default 0.1.
        save_name : typing.Optional[str], optional
            Filename for the saved ``.pt`` dataset. If None and
            ``self._save_dir`` is set, defaults to
            ``imputation_dataset.pt``.

        Returns
        -------
        TensorDataset
            Dataset yielding ``(features, targets, metadata)`` tuples; the
            last channel of the metadata tensor is the ``observed`` mask.
        """
        # ----- Column definitions -----
        # Each tensor's columns are listed explicitly here so that the
        # mapping from DataFrame to tensor is easy to audit and update.
        #
        # X (features)  — model input, shape [seq_len, 16]
        # y (targets)   — ground-truth values the model must reconstruct,
        #                  shape [seq_len, 2]
        # meta          — per-timestep context carried alongside each sample
        #                  for the loss function and metric reporting,
        #                  shape [seq_len, 9]
        #
        # "observed" is generated per-sequence below (not present on df).
        # "flow" and "density" appear in both features and targets; in the
        # feature tensor masked positions are replaced with a -1 sentinel,
        # while the target tensor always holds the true values.

        feature_columns = [
            "Type",
            "Abs PM",
            "Length",
            "pct_observed",
            "flow",
            "density",
            "Station ID (encoded)",
            "free_flow_speed_scaled",
            "congestion_wave_speed_scaled",
            "year",
            "month",
            "day",
            "dayofweek",
            "hour",
            "minute",
            "5min_block",
        ]

        target_columns = [
            "flow",
            "density",
        ]

        meta_columns = [
            "Station ID",
            "Lanes",
            "capacity",
            "critical_density",
            "free_flow_speed",
            "congestion_wave_speed",
            "5min_block",
            "dayofweek",
            "observed",
        ]

        # Calculate number of entries to mask for imputation
        missing_counts = min(math.ceil(missing_rate * seq_len), seq_len)

        # Prepare lists for features and targets
        meta_list, X_list, y_list = [], [], []

        N_stations = len(df["Station ID"].unique())
        for idx, station_id in enumerate(df["Station ID"].unique()):
            logger.info(f"Preparing sequences for station {idx+1}/{N_stations}")
            station_df = df.loc[df["Station ID"] == station_id, :].reset_index(drop=True)

            for i in range(0, len(station_df) - (seq_len) + 1, seq_len):
                # Locate the sequence
                sequence = station_df.loc[i : i + seq_len - 1, :].copy()

                # Don't use any sequences with missing data
                if sequence.isna().any().any():
                    continue

                # Generate a set of random indices to mask
                mask_ids = random.sample(range(i, i + seq_len), missing_counts)

                # Capture true targets before masking
                targets = sequence.loc[:, target_columns]

                # Add masking indicator and replace masked targets with sentinel
                sequence['observed'] = (~sequence.index.isin(mask_ids)).astype(int)
                sequence.loc[mask_ids, ["flow", "density"]] = -1

                # Select metadata and features (after masking)
                metadata = sequence.loc[:, meta_columns]
                features = sequence.loc[:, feature_columns]

                # Convert to tensor
                metadata = torch.from_numpy(metadata.values).to(dtype=torch.float32)
                features = torch.from_numpy(features.values).to(dtype=torch.float32)
                targets = torch.from_numpy(targets.values).to(dtype=torch.float32)

                # Append to list
                meta_list.append(metadata)
                X_list.append(features)
                y_list.append(targets)

        # Convert lists to tensors
        meta = torch.stack(meta_list)
        X = torch.stack(X_list)
        y = torch.stack(y_list)

        # Create dataset
        dataset = TensorDataset(X, y, meta)

        # Determine where to save:
        #   1. Explicit save_name takes priority.
        #   2. Else auto-save to self._save_dir with a default filename if set.
        _resolved_save_name = save_name
        if _resolved_save_name is None and self._save_dir is not None:
            _resolved_save_name = "imputation_dataset.pt"

        if _resolved_save_name is not None:
            _out_dir = self._save_dir if self._save_dir is not None else self.processed_data_directory
            filepath = os.path.join(_out_dir, _resolved_save_name)
            save_dict = self._build_dataset_save_dict(X=X, y=y, meta=meta)
            torch.save(save_dict, filepath)
            logger.info(f"Saving imputation dataset to {filepath}")

        # Return
        return dataset

    def _build_dataset_save_dict(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        meta: torch.Tensor,
    ) -> dict:
        """
        Build the dict persisted to disk for a prepared dataset.

        Always includes X/y/meta and the ``target_normalization`` mode used to
        produce the targets. When MIN_MAX scaling was applied, also embeds the
        global min/max used by the fitted scaler so that downstream consumers
        can invert the transform without loading the joblib file.
        """
        save_dict: dict = {
            "X": X,
            "y": y,
            "meta": meta,
            "target_normalization": self.target_normalization.value,
        }
        if self.target_normalization == TargetNormalization.MIN_MAX:
            scaler = getattr(self, "target_minmax_scaler", None)
            if scaler is None:
                raise RuntimeError(
                    "target_normalization is MIN_MAX but no fitted scaler is "
                    "available — did normalize_timeseries run?"
                )
            # Embed the fitted scaler itself so downstream consumers can call
            # inverse_transform rather than reimplementing the math (which
            # would silently drift for non-default feature_range values).
            # Scaler columns are [flow_[veh/hr-lane], density_[veh/mi-lane]].
            save_dict["target_minmax_scaler"] = scaler
        return save_dict


# Example usage
if __name__ == "__main__":
    # Pass save_directory to automatically route all outputs there.
    DataProcessor = PeMSDataProcessor(
        root_directory="/projects/rost5691/data/Caltrans/PeMS",
        metadata_filepath="/projects/rost5691/data/Caltrans/PeMS/metadata/station_metadata_calibrated.csv",
        save_transforms=True,
        save_directory="/projects/rost5691/data/Caltrans/PeMS/processed_data/3hr_10pct_imputation",
    )

    # Process raw timeseries and metadata into DataFrame
    # Output saved automatically as 'preprocessed_data.csv' inside save_directory.
    df = DataProcessor.preprocess_data()
    logger.info(f"Loaded dataframe:\n{df.head()}")

    # Generate Dataset from DataFrame
    # Output saved automatically as 'imputation_dataset.pt' inside save_directory.
    dataset = DataProcessor.prepare_imputation_dataset(df)
