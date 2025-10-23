"""
Script for preparing transportation data for deep learning prediction
"""

#####     Setup     ######

# Imports
import os
import torch
import logging
import numpy as np
import pandas as pd
import networkx as nx
from utils.logs import make_logger
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data, Batch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.loader import DataLoader as GNNDataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler


class TransportationDataProcessor:
    def __init__(
        self,
        root_directory: str = "/projects/rost5691/data/Caltrans/PeMS",
        static_filename: str = "station_metadata_CALIBRATED.csv",
        dynamic_filename: str = "data_long.csv",
        logger: logging.Logger = logging.getLogger(),
    ) -> None:

        np.random.seed(42)  # For reproducibility

        # Load data
        # static_data_filepath = os.path.join(root_directory, static_filename)
        # dynamic_data_filepath = os.path.join(root_directory, dynamic_filename)
        # self.static_data = pd.read_csv(static_data_filepath)
        # self.dynamic_data = pd.read_csv(dynamic_data_filepath)
        self.static_data = self._create_static_links_data()
        self.dynamic_data = self._create_dynamic_links_data(
            self.static_data["Station ID"]
        )

        # Construct a graph and extract the adjacency matrix
        self.graph = self._build_graph_from_static()
        self.adjacency_matrix = nx.adjacency_matrix(self.graph)

        # Define other variables
        self.categorical_features = ["Station ID", "Type"]
        self.columns_to_fill = [
            "capacity",
            "free_flow_speed",
            "congestion_wave_speed",
            "jam_density",
            "critical_density",
        ]
        self.static_scaler = StandardScaler()
        self.dynamic_scaler = StandardScaler()
        self.encoder = OrdinalEncoder()
        self.imputer = SimpleImputer(strategy="constant", fill_value=-1)

    def _create_static_links_data(self, num_links=5):
        """
        Create a dummy static dataset for roadway links.
        """
        data = {
            "link_id": [f"L{i+1}" for i in range(num_links)],
            "from_node_id": [f"N{i+1}" for i in range(num_links)],
            "to_node_id": [f"N{i+2}" for i in range(num_links)],
            "Station ID": [f"ST{i+1}" for i in range(num_links)],
            "Type": np.random.choice(["mainline", "onramp", "offramp"], size=num_links),
            "Lanes": np.random.randint(2, 6, size=num_links),
            "Abs PM": np.round(np.random.uniform(0, 20, size=num_links), 2),
            "Length": np.round(np.random.uniform(0.3, 1.5, size=num_links), 2),
            "capacity": np.random.randint(1800, 2400, size=num_links),
            "free_flow_speed": np.random.randint(55, 70, size=num_links),
            "congestion_wave_speed": np.random.randint(10, 20, size=num_links),
            "jam_density": np.random.randint(160, 220, size=num_links),
            "critical_density": np.random.randint(25, 35, size=num_links),
        }

        return pd.DataFrame(data)

    def _create_dynamic_links_data(
        self, station_ids, start_time="2022-01-01 00:00", periods=2016, freq="5min"
    ):
        """
        Create a dummy dynamic dataset for time-varying traffic data.
        Each station_id gets its own time series of flow, speed, and observation percentage.
        """
        timestamps = pd.date_range(start=start_time, periods=periods, freq=freq)
        records = []

        for station in station_ids:
            for t in timestamps:
                records.append(
                    {
                        "timestamp": t,
                        "Station ID": station,
                        "total_flow_[veh/5-min]": np.random.poisson(lam=60),
                        "avg_speed_[mph]": np.random.normal(loc=60, scale=10),
                        "pct_observed": np.random.uniform(80, 100),
                    }
                )

        return pd.DataFrame(records)

    def _build_graph_from_static(self):
        """
        Build a directed graph from the static links DataFrame.
        Each row represents a directed edge from 'from_node_id' → 'to_node_id'.
        Edge attributes are all remaining columns.
        """
        G = nx.DiGraph()

        for _, row in self.static_data.iterrows():
            G.add_edge(
                row["from_node_id"],
                row["to_node_id"],
                **{
                    col: row[col]
                    for col in self.static_data.columns
                    if col not in ["from_node_id", "to_node_id"]
                },
            )

        return G

    def preprocess_data(self):
        """
        Implement the following preprocessing steps to improve prediction quality:
            1) Encode categorical static data
            2) Normalize the dynamic features according to the relevant limit
            3) Scale the normalized static and dynamic features
            4) Fill missing values in static and dynamic data with -1
        """

        # Encode
        categorical_features = self.static_data.drop(self.categorical_features, axis=1)
        self.static_data.loc[:, categorical_features.columns] = (
            self.encoder.fit_transform(categorical_features)
        )

        # Normalize


if __name__ == "__main__":
    # Set up logging
    logger = make_logger(include_stdout=True, log_prefix="test")

    processor = TransportationDataProcessor(logger=logger)

    print("Static data:\n", processor.static_data.head())
    print("Dynamic data:\n", processor.dynamic_data.head())
