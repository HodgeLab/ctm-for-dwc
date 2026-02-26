"""
Script for preparing transportation data for deep learning prediction
"""

#####     Setup     #####

# 3rd party imports
import logging
import numpy as np
import os
import pandas as pd
import torch
import typing
from sklearn.preprocessing import (
    OrdinalEncoder,
    MinMaxScaler,
)
from torch.utils.data import (
    TensorDataset,
)

# Local imports
import transportation_models.utils.constants as constants
import transportation_models.utils.logs as logs
import transportation_models.utils.validation as validation

# Default logger
logger = logging.getLogger(__name__)

def load_timeseries_single_detector(
    detector_timeseries_filepath: str,
    detector_metadata: dict,
    counts_cols: list[str] = [
        "total_flow_[veh/5-min]",
    ],
    full_daterange=pd.date_range(
        start="2022-01-01 00:00:00", end="2024-12-31 23:55:00", freq="5min"
    ),
) -> dict:
    logger.info(f"Loading timeseries for Station ID: {detector_metadata['Station ID']}")
    # Load the data
    df = pd.read_csv(detector_timeseries_filepath, usecols=["timestamp"] + counts_cols)

    # Convert the timestamp to a datetime object
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Reindex to force the data to cover the entire timespan (result will contain NaNs)
    df = (
        df.set_index("timestamp").reindex(full_daterange).reset_index(names="timestamp")
    )

    # Assign columns for minute-of-day, hour-of-day, day-of-week
    df["minute"] = df["timestamp"].apply(lambda timestamp: timestamp.minute)
    df["hour"] = df["timestamp"].apply(lambda timestamp: timestamp.hour)
    df["year"] = df["timestamp"].apply(lambda timestamp: timestamp.year)
    df["weekday"] = df["timestamp"].apply(lambda timestamp: timestamp.weekday())

    # Prepare DataFrame for X vector
    df = df.loc[:, ["minute", "hour", "weekday", "year", "timestamp"] + counts_cols]
    for feature, value in detector_metadata.items():
        df[feature] = value

    timeseries_dict = df.to_dict(orient="list")

    return timeseries_dict

def load_timeseries_all_detectors(
    detectors: list[str],
    timeseries_directory: str = constants.CALTRANS_VDS_TIMESERIES_DIRECTORY_PATH,
    metadata_filepath: str = constants.CALTRANS_VDS_METADATA_FILEPATH,
    metadata_features: list[str] = [],
    counts_cols: list[str] = ["total_flow_[veh/5-min]"],
) -> pd.DataFrame:
    # Initialize a list to store the dictionaries
    timeseries_dictionaries = []

    # Load the metadata
    metadata = pd.read_csv(metadata_filepath)

    # Load cleaned timeseries
    for detector in detectors:
        # Construct the complete filepath for the timeseries data
        timeseries_filepath = os.path.join(timeseries_directory, f"{detector}.csv")

        # Extract the metadata
        detector_metadata = (
            metadata.loc[metadata["Station ID"] == int(detector), metadata_features]
            .reset_index(drop=True)
            .iloc[0]
            .to_dict()
        )
        timeseries_dict = load_timeseries_single_detector(
            detector_timeseries_filepath=timeseries_filepath,
            detector_metadata=detector_metadata,
            counts_cols=counts_cols,
        )
        timeseries_dictionaries.append(timeseries_dict)

    # Build a DataFrame from the dictionaries
    logger.info(
        f"Building consolidated DataFrame with {len(detectors)} unique Station IDs"
    )
    df = pd.DataFrame(timeseries_dictionaries)

    # Explode list-valued columns
    cols_to_explode = (
        counts_cols
        + [
            "minute",
            "hour",
            "year",
            "weekday",
            "timestamp",
        ]
        + metadata_features
    )
    df = df.explode(column=cols_to_explode, ignore_index=True)

    # Return packed dictionary
    return df


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
    ) -> None:
        np.random.seed(42)  # For reproducibility
        
        # Define paths
        self.root_directory = root_directory
        self.metadata_path = os.path.join(root_directory, "metadata", "station_metadata_CALIBRATED.csv")
        self.timeseries_directory = os.path.join(root_directory, "timeseries_data")
        self.processed_data_directory = os.path.join(root_directory, "processed_data")
        if not os.path.exists(self.processed_data_directory):
            os.mkdir(self.processed_data_directory)

        # Load files
        metadata_cols = [
            'Station ID', 'Lanes', 'Type', 'Abs PM', 'Length', 'capacity', 'free_flow_speed', 'congestion_wave_speed', 'critical_density'
        ]
        self.metadata_df = pd.read_csv(
            self.metadata_path,
            usecols=metadata_cols)

        # Initialized encoders and scalers
        self.encoder = OrdinalEncoder()
        self.scaler = MinMaxScaler()

    @staticmethod
    def generate_mock_data(n_stations: int = 1):
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

        # Build the full filepath based on the station ID
        filename = f"{station_id}.csv"
        filepath = os.path.join(self.timeseries_directory, filename)

        # Load the timeseries dataframe
        timeseries_df = pd.read_csv(filepath, usecols=['timestamp', 'station', 'pct_observed', 'total_flow_[veh/5-min]', 'avg_speed_[mph]'])

        # Ensure that the timestamp column exists and is a datetime object
        timeseries_df = validation.validate_timestamp_dtype(timeseries_df)

        # Ensure complete indices
        years_in_df = timeseries_df['timestamp'].dt.year.unique()
        range_start = f"{str(years_in_df.min())}-01-01 00:00"
        range_end = f"{str(years_in_df.max())}-12-31 23:59"
        full_date_range = pd.date_range(start=range_start, end=range_end, freq='5min')
        df_reindexed = timeseries_df.set_index('timestamp').reindex(full_date_range)
        df_reindexed.reset_index(names='timestamp', inplace=True)

        # Merge the metadata into the timeseries
        df = df_reindexed.rename(columns={'station': 'Station ID'}).merge(right=self.metadata_df, how='inner', on='Station ID')

        return df

    def normalize_counts(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize the raw counts (flow, speed) in the timeseries data. 
        Normalization steps include:
            - Calculating per-hour, per-lane flow from 5-minute, station-wide flow. Result has units [veh/hr*lanes]
            - Calculating per-hour, per-lane density from flow and average station speed. Result has units [veh/mi*lanes]
            - Normalizing flow by station capacity. Resulting values range [0,1].
            - Normalizing density by critical density. Resulting values range [0,1] if traffic is in free-flow, and (1,inf) if traffic is congested.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 

        Returns
        -------
        pd.DataFrame
            Modified version of passed DataFrame.
            Modifications include:
                - ADDED column 'flow'
                - ADDED column 'density'
                - DROPPED columns ['total_flow_[veh/5-min]', 'avg_speed_[mph]', 'Lanes', 'capacity', 'critical_density']
                 
        """
        # Number of 5-minute periods per hour (used for normalization)
        periods_per_hour = 12

        # Scale the flow: [veh/5-min] * [5-min/hr] * [1/lanes] = [veh/hr*lanes]
        df["flow"] = (
            df['total_flow_[veh/5-min]'] * periods_per_hour
        ) / df['Lanes']

        # Scale the density: [veh/5-min] * [5-min/h] * [1/lanes] / [mi/h] = [veh/mi*lanes]
        df["density"] = df['flow'] / df['avg_speed_[mph]']

        # Normalize flow by station capacity
        df['flow'] = df['flow'] / df['capacity']

        # Normalize density by critical density
        df['density'] = df['density'] / df['critical_density']

        # Drop unnecessary columns
        df = df.drop(columns=[
            'total_flow_[veh/5-min]',
            'avg_speed_[mph]',
            'Lanes',
            'capacity',
            'critical_density'
        ])

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
        detector_dfs = [self.load_data_by_id(detector) for detector in detectors]

        # Stack dataframes into one
        df = pd.concat(detector_dfs)

        # Normalize counts
        df = self.normalize_counts(df)

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

    def encode_and_scale(
        self,
        df: pd.DataFrame
        ) -> pd.DataFrame:
        
        categorical_features = ['Station ID', 'Type']
        df.loc[:, categorical_features] = self.encoder.fit_transform(df[categorical_features])
        df[categorical_features] = df[categorical_features].astype('float')     # Cast as float after encoding

        minmax_features = ['pct_observed', 'Abs PM', 'Length', 'free_flow_speed', 'congestion_wave_speed', 'year', 'month', 'day', 'dayofweek', 'hour', 'minute', '5min_block']
        df[minmax_features] = df[minmax_features].astype('float')   # Cast as float ahead of scaling
        df.loc[:, minmax_features] = self.scaler.fit_transform(df[minmax_features])

        return df

    def process_data(self, detectors: typing.Optional[list[str]], save_name: typing.Optional[str] = None) -> pd.DataFrame:
        # Execute processing steps
        df = self.build_long_df(detectors=detectors)
        df = self.deconstruct_timestamps(df)
        df = self.encode_and_scale(df)
        
        # Optionally save DataFrame
        if save_name is not None:
            filepath = os.path.join(self.processed_data_directory, save_name)
            df.to_csv(filepath)
        
        return df

    def prepare_sequence_dataset(self, df: pd.DataFrame, seq_len: int = 12, pred_horizon: int = 3, save_name: typing.Optional[str] = None):
        X_list, y_list = [], []

        for station_id in df['Station ID'].unique():
            station_df = df.loc[df['Station ID'] == station_id, :]

            for i in range(len(station_df) - (seq_len + pred_horizon) + 1):
                # Select features and targets
                features = station_df.loc[i:i + seq_len-1, :]
                targets = station_df.loc[i + seq_len:i+seq_len+pred_horizon-1, ['flow', 'density']]

                # Don't use any sequences with missing data
                if features.isna().any().any() or targets.isna().any().any():
                    continue
            
                # Convert to tensor
                features = torch.from_numpy(features.values)
                targets = torch.from_numpy(targets.values)

                # Append to list
                X_list.append(features)
                y_list.append(targets)
        
        # Convert lists to tensors
        X = torch.stack(X_list)
        y = torch.stack(y_list)

        # Create dataset
        dataset = TensorDataset(X, y)

        # Optionally save Dataset
        if save_name is not None:
            filepath = os.path.join(self.processed_data_directory, save_name)
            torch.save(torch.save({'X': X, 'y': y}, filepath))

        # Return
        return dataset
        
# Example usage
if __name__ == "__main__":
    # Initialize processor
    DataProcessor = PeMSDataProcessor(root_directory="/projects/rost5691/data/Caltrans/PeMS", process=True)

    # Process raw timeseries and metadata into DataFrame
    df = DataProcessor.process_data(save_name="data_long.csv")

    # Generate Dataset from DataFrame
    dataset = DataProcessor.prepare_sequence_dataset(df, save_name="hour_lookback_15min_horizon.pt")
