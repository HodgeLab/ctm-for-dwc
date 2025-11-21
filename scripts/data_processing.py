"""
Script for preparing transportation data for deep learning prediction
"""

#####     Setup     ######

# Imports
import torch
import logging
import numpy as np
import pandas as pd
import networkx as nx
from utils.logs import make_logger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GNNDataLoader
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    MinMaxScaler,
)


class TransportationDataProcessor:

    def __init__(
        self,
        root_directory: str = "/projects/rost5691/data/Caltrans/PeMS",
        static_filename: str = "station_metadata_CALIBRATED.csv",
        dynamic_filename: str = "data_long.csv",
        logger: logging.Logger = logging.getLogger(),
        n_links: int = 5,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:

        np.random.seed(42)  # For reproducibility
        self.n_links = n_links
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

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

        # Define encoders
        self.station_id_encoder = OrdinalEncoder()
        self.link_id_encoder = OrdinalEncoder()
        self.node_id_encoder = OrdinalEncoder()
        self.one_hot_encoder = OneHotEncoder()

        # Define scalers
        self.dynamic_minmax_scaler = MinMaxScaler()
        self.static_minmax_scaler = MinMaxScaler()

    def _create_static_links_data(self) -> pd.DataFrame:
        """
        Create a dummy static dataset for roadway links.
        """
        data = {
            "link_id": [f"L{i+1}" for i in range(self.n_links)],
            "from_node_id": [f"N{i+1}" for i in range(self.n_links)],
            "to_node_id": [f"N{i+2}" for i in range(self.n_links)],
            "Station ID": [f"ST{i+1}" for i in range(self.n_links)],
            "Type": np.random.choice(
                ["mainline", "onramp", "offramp"], size=self.n_links
            ),
            "Lanes": np.random.randint(2, 6, size=self.n_links),
            "Abs PM": np.round(np.random.uniform(0, 20, size=self.n_links), 2),
            "Length": np.round(np.random.uniform(0.3, 1.5, size=self.n_links), 2),
            "capacity": np.random.randint(1800, 2400, size=self.n_links),
            "free_flow_speed": np.random.randint(55, 70, size=self.n_links),
            "congestion_wave_speed": np.random.randint(10, 20, size=self.n_links),
            "jam_density": np.random.randint(160, 220, size=self.n_links),
            "critical_density": np.random.randint(25, 35, size=self.n_links),
        }

        return pd.DataFrame(data)

    def _create_dynamic_links_data(
        self, station_ids, start_time="2022-01-01 00:00", periods=2016, freq="5min"
    ) -> pd.DataFrame:
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
            1) Normalize the dynamic features according to the relevant limit
            2) Encode categorical static data
            3) Fill missing values in dynamic data with -1
            4) Scale static and dynamic features
        """

        ## Normalize
        # Temporarily merge the static and dymamic data to enable vectorization
        data = pd.merge(left=self.static_data, right=self.dynamic_data, on="Station ID")

        # Calculate normalized metrics
        periods_per_hour = 12
        self.dynamic_data["normalized_flow_[vphpl]"] = (
            data["total_flow_[veh/5-min]"] * periods_per_hour / data["Lanes"]
        ) / data["capacity"]
        self.dynamic_data["normalized_density_[vpmpl]"] = (
            data["total_flow_[veh/5-min]"]
            * periods_per_hour
            / (data["avg_speed_[mph]"] * data["Lanes"])
        ) / data["critical_density"]

        # Drop the underlying metric
        self.dynamic_data.drop(
            columns=["total_flow_[veh/5-min]", "avg_speed_[mph]"], inplace=True
        )
        self.static_data.drop(
            columns=["Lanes", "capacity", "critical_density"], inplace=True
        )

        ## Encode
        # TODO: Figure out how best to use the encoded version
        # First encode the nominal data using a OneHotEncoder
        # nominal_data = self.static_data["Type"].to_numpy().reshape(-1, 1)
        # self.static_data["Type"] = self.one_hot_encoder.fit_transform(nominal_data)
        self.static_data.drop(columns=["Type"], inplace=True)

        # Use an OrdinalEncoder for the ordered data
        station_ids_static = self.static_data["Station ID"].to_numpy().reshape(-1, 1)
        station_ids_dynamic = self.dynamic_data["Station ID"].to_numpy().reshape(-1, 1)
        link_ids = self.static_data["link_id"].to_numpy().reshape(-1, 1)
        node_ids = np.array(
            list(
                set(self.static_data["from_node_id"])
                | set(self.static_data["to_node_id"])
            )
        ).reshape(-1, 1)
        node_ids_from = self.static_data["from_node_id"].to_numpy().reshape(-1, 1)
        node_ids_to = self.static_data["to_node_id"].to_numpy().reshape(-1, 1)

        self.static_data.loc[:, "Station ID"] = self.station_id_encoder.fit_transform(
            station_ids_static
        )
        self.dynamic_data.loc[:, "Station ID"] = self.station_id_encoder.transform(
            station_ids_dynamic
        )
        self.static_data.loc[:, "link_id"] = self.link_id_encoder.fit_transform(
            link_ids
        )
        self.node_id_encoder.fit(node_ids)
        self.static_data.loc[:, "from_node_id"] = self.node_id_encoder.transform(
            node_ids_from
        )
        self.static_data.loc[:, "to_node_id"] = self.node_id_encoder.transform(
            node_ids_to
        )

        ## Fill
        missing_mask = self.dynamic_data.isnull().any(axis=1)
        self.dynamic_data.loc[
            missing_mask, ["normalized_flow_[vphpl]", "normalized_density_[vpmpl]"]
        ] = -1

        ## Scale the remaining data using a min-max scaler
        dynamic_features_to_scale = self.dynamic_data.loc[:, ["pct_observed"]]
        self.dynamic_data.loc[:, dynamic_features_to_scale.columns] = (
            self.dynamic_minmax_scaler.fit_transform(dynamic_features_to_scale)
        )
        static_features_to_scale = self.static_data.loc[
            :,
            [
                "Abs PM",
                "Length",
                "free_flow_speed",
                "congestion_wave_speed",
                "jam_density",
            ],
        ]
        self.static_data.loc[:, static_features_to_scale.columns] = (
            self.static_minmax_scaler.fit_transform(static_features_to_scale)
        )

    def prepare_gnn_data(self) -> dict[str, list[Data]]:
        """
        Uses the static and dynamic data to construct a PyTorch Geometric dataset.
        Each dataset is for a single timestamp, and contains both static and dynamic features.
        The static features include:
            - Abs PM
            - Length
            - free_flow_speed
            - congestion_wave_speed
            - jam_density
        The dynamic features include:
            - pct_observed
            - normalized_density_[vpmpl]

        The target value in the dataset is assumed to be the normalized flow:
            - normalized_flow_[vphpl]

        Returns
        -------
        dict[str, list[Data]]
            A set of training, validation, and testing datasets.
            Dictionary keys are one of: ['train', 'val', 'test'], and are used to access different
            datasets.
            Dictionary values are lists of torch_geometric.data.Data objects. The list is like a
            torch_geometric.data.Dataset, but this is not actually implemented. However, the list
            can be directly used with a torch_geometric.loader.DataLoader object to obtain
            mini-batches of data.
        """

        gnn_data = {"train": [], "val": [], "test": []}

        timestamps = sorted(self.dynamic_data["timestamp"].unique())
        n_timestamps = len(timestamps)

        train_end = int(n_timestamps * self.train_ratio)
        val_end = int(n_timestamps * (self.train_ratio + self.val_ratio))

        split_indices = {
            "train": range(0, train_end),
            "val": range(train_end + 1, val_end),
            "test": range(val_end + 1, n_timestamps),
        }

        for split, indices in split_indices.items():
            for t_idx in indices:
                current_time = timestamps[t_idx]

                node_features = []
                node_targets = []

                for link_id in self.static_data["link_id"].unique():
                    # Extract static features
                    station_id = self.static_data.loc[
                        self.static_data["link_id"] == link_id, "Station ID"
                    ].values[0]
                    static_feat = (
                        self.static_data[self.static_data["link_id"] == link_id]
                        .drop(
                            columns=[
                                "link_id",
                                "from_node_id",
                                "to_node_id",
                                "Station ID",
                            ]
                        )
                        .values[0]
                    )

                    # Extract dynamic features
                    dynamic_data = self.dynamic_data.loc[
                        (self.dynamic_data["timestamp"] == current_time)
                        & (self.dynamic_data["Station ID"] == station_id),
                        :,
                    ].drop(
                        columns=[
                            "timestamp",
                            "Station ID",
                        ]
                    )
                    dynamic_feat = dynamic_data.drop(
                        columns=[
                            "normalized_flow_[vphpl]",
                        ]
                    ).values[0]

                    # Append static features to list
                    node_feat = np.concatenate([static_feat, dynamic_feat])
                    node_features.append(node_feat)

                    # Extract target values
                    target = dynamic_data["normalized_flow_[vphpl]"].values[0]

                    node_targets.append(target)

                # Create edge indices from adjacency matrix
                edge_indices = []
                edge_weights = []
                # Complete graph based on adjacency matrix
                for i in range(self.n_links):
                    for j in range(self.n_links):
                        if self.adjacency_matrix[i, j] > 0:
                            edge_indices.append([i, j])
                            edge_weights.append(self.adjacency_matrix[i, j])

                # Create PyTorch Geometric Data object
                graph_data = Data(
                    x=torch.FloatTensor(np.array(node_features)),
                    edge_index=torch.LongTensor(edge_indices).t().contiguous(),
                    edge_attr=torch.FloatTensor(edge_weights).unsqueeze(1),
                    y=torch.FloatTensor(np.array(node_targets)),
                    timestamp=current_time,
                )

                gnn_data[split].append(graph_data)

        return gnn_data


if __name__ == "__main__":
    # Set up logging
    logger = make_logger(include_stdout=True, log_prefix="test")

    # Define the processor object
    processor = TransportationDataProcessor(logger=logger)

    # Run the preprocessing steps
    processor.preprocess_data()

    print("Static data:\n", processor.static_data.head())
    print("Dynamic data:\n", processor.dynamic_data.head())

    # Prepare the GNN data
    gnn_data = processor.prepare_gnn_data()
    print("\n=== GNN Data Info ===")
    for split, graphs in gnn_data.items():
        if graphs:
            print(f"{split}: {len(graphs)} graphs")
            print(
                f"  Sample graph - Nodes: {graphs[0].x.shape[0]}, "
                f"Features: {graphs[0].x.shape[1]}, "
                f"Edges: {graphs[0].edge_index.shape[1]}"
            )

    # Load graphs into DataLoader object
    gnn_train_loader = GNNDataLoader(
        gnn_data["train"], batch_size=32, shuffle=True, drop_last=True
    )
