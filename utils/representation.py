"""
Tools for building a representation of a freeway which is compatible with the CTM
"""

# 3rd party imports
import os
import pdb
import typing
import logging
import numpy as np
import pandas as pd
import networkx as nx
import osm2gmns as og
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import xml.etree.ElementTree as ET
from scipy import stats
from shapely import wkt
from pathlib import Path
from datetime import datetime
from shapely.ops import split, linemerge
from shapely.geometry import LineString, Point, MultiLineString, MultiPoint

# Set up logging
logger = logging.getLogger(__name__)


class Freeway:
    def __init__(
        self,
        links_filepath: Path,
        nodes_filepath: Path,
        detector_metadata_filepath: Path,
        detector_timeseries_data_directory: Path,
        postmile_filepath: typing.Optional[Path] = None,
        bounding_filepath: typing.Optional[Path] = None,
        timeseries_reference_station: int = 407219,
        freeway_name: str = "I880",
        direction: str = "N",
        links_crs: str = "EPSG:4326",
        nodes_crs: str = "EPSG:4326",
        vds_crs: str = "EPSG:4326",
        postmile_crs: str = "EPSG:4326",
        boundary_crs: typing.Optional[str] = None,
        common_crs: str = "EPSG:4087",
        postmile_start: typing.Optional[float] = None,
        postmile_end: typing.Optional[float] = None,
        additional_offramp_node_ids: list[int] = [],
        additional_onramp_node_ids: list[int] = [],
        additional_mainline_node_ids: list[int] = [],
        offramp_node_ids_to_exclude: list[int] = [],
        onramp_node_ids_to_exclude: list[int] = [],
        mainline_node_ids_to_exclude: list[int] = [],
        link_to_station_override: dict[typing.Any, int] = {},
        *args,
        **kwargs,
    ) -> None:
        # Load the data
        self.links = self._load_gdf_from_csv(
            csv_filepath=links_filepath, data_crs=links_crs, common_crs=common_crs
        )
        self.nodes = self._load_gdf_from_csv(
            csv_filepath=nodes_filepath, data_crs=nodes_crs, common_crs=common_crs
        )
        self.detectors = self._load_gdf_from_csv(
            csv_filepath=detector_metadata_filepath,
            data_crs=vds_crs,
            common_crs=common_crs,
        )
        if postmile_filepath is not None:
            self.postmiles = self._load_gdf_from_shp(
                shp_filepath=postmile_filepath,
                data_crs=postmile_crs,
                common_crs=common_crs,
            )
        else:
            self.postmiles = None
        self.boundary = self._create_boundary(
            boundary_filepath=bounding_filepath,
            data_crs=boundary_crs,
            common_crs=common_crs,
        )

        # Assign parameters
        self.timeseries_reference_station = timeseries_reference_station
        self.freeway_name = freeway_name
        self.direction = self._validate_direction(direction=direction)
        self.detector_timeseries_data_directory = detector_timeseries_data_directory
        self.common_crs = common_crs
        self.postmile_start = postmile_start
        self.postmile_end = postmile_end
        self.additional_offramp_node_ids = additional_offramp_node_ids
        self.additional_onramp_node_ids = additional_onramp_node_ids
        self.additional_mainline_node_ids = additional_mainline_node_ids
        self.offramp_node_ids_to_exclude = offramp_node_ids_to_exclude
        self.onramp_node_ids_to_exclude = onramp_node_ids_to_exclude
        self.mainline_node_ids_to_exclude = mainline_node_ids_to_exclude
        self.link_to_station_override = link_to_station_override

        # Prepare the input data to convert to a graph
        self._prepare_roadway_data()

        # Build the graph
        self.graph = self._build_freeway_graph()

    def __repr__(self) -> None:
        """
        Representation of the freeway
        """
        logging.info("I am a freeway!")

    @staticmethod
    def _load_gdf_from_csv(
        csv_filepath: Path, data_crs: str, common_crs: str
    ) -> gpd.GeoDataFrame:
        # Load the file as a DataFrame
        df = pd.read_csv(csv_filepath)

        # Build a valid geometry object
        if "geometry" in df.columns:
            df["geometry"] = df["geometry"].apply(wkt.loads)  # type: ignore
        elif "x_coord" in df.columns and "y_coord" in df.columns:
            df["geometry"] = df.apply(
                lambda row: Point(row["x_coord"], row["y_coord"]), axis=1  # type: ignore
            )
        else:
            raise ValueError(f"Could not find columns to use for geometry definition")

        # Create the GeoDataFrame
        gdf = gpd.GeoDataFrame(data=df, geometry="geometry", crs=data_crs)

        # Reproject the GDF to a common crs
        if data_crs != common_crs:
            gdf.to_crs(common_crs, inplace=True)

        # Return the GDF
        return gdf

    @staticmethod
    def _load_gdf_from_shp(
        shp_filepath: Path, data_crs: str, common_crs: str
    ) -> gpd.GeoDataFrame:
        # Ensure that the filepath exists
        assert shp_filepath.exists()

        # Load the file into a GDF
        gdf = gpd.read_file(shp_filepath)

        # Assign the crs
        if gdf.crs is None:
            gdf = gdf.set_crs(crs=data_crs)

        # Transform to the common crs
        gdf = gdf.to_crs(common_crs)

        # Return the gdf
        return gdf

    def _classify_node_purpose(
        self,
        node_id: int,
        onramp_node_ids: np.ndarray,
        offramp_node_ids: np.ndarray,
        mainline_node_ids: np.ndarray,
    ) -> str:
        if node_id in onramp_node_ids:
            return "merge"
        elif node_id in offramp_node_ids:
            return "diverge"
        elif node_id in mainline_node_ids:
            return "mainline"
        else:
            return ""

    def _assign_node_purpose(
        self,
    ) -> None:
        """
        Modifies self.nodes directly by adding a 'purpose' column
        """
        # Obtain segments of the data
        offramp_node_ids = self.links.loc[
            self.links["link_type"] == "Off-ramp", "from_node_id"
        ].unique()
        onramp_node_ids = self.links.loc[
            self.links["link_type"] == "On-ramp", "to_node_id"
        ].unique()
        mainline_node_ids = pd.unique(
            self.links.loc[
                self.links["link_type"] == "Mainline", ["from_node_id", "to_node_id"]
            ].values.ravel()
        )

        # Add any additional IDs
        offramp_node_ids = np.append(offramp_node_ids, self.additional_offramp_node_ids)
        onramp_node_ids = np.append(onramp_node_ids, self.additional_onramp_node_ids)
        mainline_node_ids = np.append(
            mainline_node_ids, self.additional_mainline_node_ids
        )

        # Remove any specified IDs
        offramp_node_ids = np.array(
            [
                node_id
                for node_id in offramp_node_ids
                if node_id not in self.offramp_node_ids_to_exclude
            ]
        )
        onramp_node_ids = np.array(
            [
                node_id
                for node_id in onramp_node_ids
                if node_id not in self.onramp_node_ids_to_exclude
            ]
        )
        mainline_node_ids = np.array(
            [
                node_id
                for node_id in mainline_node_ids
                if node_id not in self.mainline_node_ids_to_exclude
            ]
        )

        # Assign a purpose to the node
        self.nodes["purpose"] = self.nodes["node_id"].apply(
            self._classify_node_purpose,
            args=(onramp_node_ids, offramp_node_ids, mainline_node_ids),
        )

    def _validate_direction(self, direction: str):
        # Check the direction parameter
        direction = direction.upper()
        if direction in ["N", "S", "E", "W"]:
            return direction
        else:
            logger.error(f"'direction' parameter must be one of ['N', 'S', 'E', 'W']")
            raise ValueError

    def _classify_node_direction(
        self, node_id: int, directed_node_ids: np.ndarray
    ) -> str:
        if node_id in directed_node_ids:
            return self.direction
        else:
            return ""

    def _assign_node_direction(self) -> None:
        """
        Modifies self.nodes directly by adding a 'direction' column
        """
        # Obtain segments of the data
        directed_node_ids = pd.unique(
            self.links.loc[
                self.links["direction"] == self.direction,
                ["from_node_id", "to_node_id"],
            ].values.ravel()
        )

        # Assign a direction to the node
        self.nodes["direction"] = self.nodes["node_id"].apply(
            self._classify_node_direction, args=(directed_node_ids,)
        )

    def _classify_vds_direction(self, freeway_name: str) -> str:
        if f"{self.freeway_name}-{self.direction}" in freeway_name:
            return self.direction
        else:
            return ""

    def _assign_vds_direction(self) -> None:
        """
        Modifies self.detectors directly by adding a 'Direction' column
        """
        self.detectors["Direction"] = self.detectors["Fwy"].apply(
            self._classify_vds_direction
        )

    def _filter_data_by_direction(self):
        # Assign directions to the nodes and detector stations (links are already assigned)
        self._assign_node_direction()
        self._assign_vds_direction()

        # Create masks that filter each GDF by the direction
        directed_link_mask = self.links["direction"] == self.direction
        directed_node_mask = self.nodes["direction"] == self.direction
        directed_vds_mask = self.detectors["Direction"] == self.direction
        if self.postmiles is not None:
            directed_postmiles_mask = (
                self.postmiles["Direction"].str.strip("B") == self.direction
            )

        # Filter the data by these masks
        self.links = self.links.loc[directed_link_mask, :].copy().reset_index(drop=True)
        self.nodes = self.nodes.loc[directed_node_mask, :].copy().reset_index(drop=True)
        self.detectors = (
            self.detectors.loc[directed_vds_mask, :].copy().reset_index(drop=True)
        )
        if self.postmiles is not None:
            self.postmiles = (
                self.postmiles.loc[directed_postmiles_mask, :]
                .copy()
                .reset_index(drop=True)
            )

    def _filter_postmiles_by_route(self):
        if self.postmiles is not None:
            mask = self.postmiles["Route"] == int(self.freeway_name.strip("I"))
            self.postmiles = self.postmiles.loc[mask, :].copy().reset_index(drop=True)

    def _create_boundary(
        self,
        boundary_filepath: typing.Optional[Path] = None,
        data_crs: typing.Optional[str] = None,
        common_crs: str = "",
    ):
        # Make sure that the CRS for the data exists if the filepath exists
        if boundary_filepath is not None:
            assert data_crs is not None

        # Load the boundary if it exists
        if str(boundary_filepath).endswith(".csv"):
            boundary_gdf = self._load_gdf_from_csv(
                csv_filepath=boundary_filepath,  # type: ignore
                data_crs=data_crs,  # type: ignore
                common_crs=common_crs,
            )
        elif str(boundary_filepath).endswith(".shp"):
            boundary_gdf = self._load_gdf_from_shp(
                shp_filepath=boundary_filepath, data_crs=data_crs, common_crs=common_crs  # type: ignore
            )
        else:
            return None

        # Obtain the convex hull of the points in the boundary data
        points = boundary_gdf.geometry.values
        multipoint = MultiPoint(points)  # type: ignore
        convex_hull = multipoint.convex_hull

        return convex_hull

    def _filter_data_by_boundary(
        self, gdf_to_filter: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        # Protect against option to pass no bounding shapefile
        if self.boundary is None:
            logger.info(f"No boundary to filter by!")
            return gdf_to_filter

        # Filter the data for elements contained within the boundary
        elements_in_boundary = gdf_to_filter.loc[
            gdf_to_filter.geometry.within(self.boundary), :
        ]

        return elements_in_boundary

    def _filter_important_nodes(self):
        """
        Modifies self.nodes directly by filtering for:
        1. Nodes that are from/to nodes of ramp links (On-ramp or Off-ramp)
        2. Nodes where the number of lanes differs between connected mainline links
        """
        # --- Step 1: Identify nodes connected to ramps ---
        ramp_links = self.links[self.links["link_type"].isin(["On-ramp", "Off-ramp"])]
        ramp_node_ids = set(ramp_links["from_node_id"]) | set(ramp_links["to_node_id"])

        # --- Step 2: Identify nodes where lane count changes between mainline links ---
        mainline_links = self.links[self.links["link_type"] == "Mainline"].copy()

        # Ensure 'lanes' is numeric
        mainline_links["lanes"] = pd.to_numeric(
            mainline_links["lanes"], errors="coerce"
        )

        lane_change_nodes = set()
        for node_id in self.nodes["node_id"]:
            connected_links = mainline_links[
                (mainline_links["from_node_id"] == node_id)
                | (mainline_links["to_node_id"] == node_id)
            ]
            if len(connected_links) >= 2:
                if connected_links["lanes"].nunique() > 1:
                    lane_change_nodes.add(node_id)

        # --- Step 3: Combine the two sets of nodes and filter the node gdf ---
        important_node_ids = ramp_node_ids.union(lane_change_nodes)
        self.nodes = self.nodes[self.nodes["node_id"].isin(important_node_ids)].copy()

    def _collapse_mainline_links(self):
        """
        Collapse adjacent mainline links between retained nodes into single segments.
        Retains geometry, lane count (as average or mode), and other attributes.
        Leaves ramp segments unchanged.
        """
        # Separate mainline and non-mainline links
        mainline_links = self.links[self.links["link_type"] == "Mainline"].copy()
        ramp_links = self.links[self.links["link_type"] != "Mainline"].copy()

        # Clean up expected fields
        mainline_links["lanes"] = pd.to_numeric(
            mainline_links["lanes"], errors="coerce"
        )
        mainline_links["facility_type"] = mainline_links["facility_type"].fillna(
            "Unknown"
        )
        mainline_links["allowed_uses"] = mainline_links["allowed_uses"].fillna(
            "Unknown"
        )
        mainline_links["direction"] = mainline_links["direction"].fillna("Unknown")

        # Build graph for mainline
        G = nx.DiGraph()
        for idx, row in mainline_links.iterrows():
            G.add_edge(
                row["from_node_id"],
                row["to_node_id"],
                index=idx,
                link_id=row["link_id"],
            )

        retained_node_ids = set(self.nodes["node_id"])

        collapsed_links = []
        visited_edges = set()

        valid_start_nodes = [
            node
            for node in retained_node_ids
            if node in G and (G.out_degree(node) == 1 and G.in_degree(node) <= 1)
        ]

        for start_node in valid_start_nodes:
            for succ in G.successors(start_node):
                path = [start_node, succ]
                while (
                    succ not in retained_node_ids
                    and G.out_degree(succ) == 1
                    and G.in_degree(succ) == 1
                ):
                    next_node = next(G.successors(succ))
                    path.append(next_node)
                    succ = next_node

                # Skip if the path has already been collapsed
                path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                if any((u, v) in visited_edges for (u, v) in path_edges):
                    continue
                visited_edges.update(path_edges)

                # Combine geometries and attributes
                geometries = []
                osm_way_ids = []
                lanes = []
                facility_types = []
                allowed_uses = []
                directions = []

                for u, v in path_edges:
                    edge_data = G.get_edge_data(u, v)
                    edge_idx = edge_data["index"]
                    link = mainline_links.loc[edge_idx]
                    geometries.append(link.geometry)
                    osm_way_ids.append(link["osm_way_id"])
                    lanes.append(link["lanes"])
                    facility_types.append(link["facility_type"])
                    allowed_uses.append(link["allowed_uses"])
                    directions.append(link["direction"])

                from_node_id = next(n for n in path if n in retained_node_ids)
                to_node_id = next(n for n in reversed(path) if n in retained_node_ids)

                collapsed_geometry = linemerge(geometries)
                length_m = collapsed_geometry.length
                length_mi = length_m / 1609.344

                # Convert some lists to Series
                lanes = pd.Series(lanes)
                facility_types = pd.Series(facility_types)
                allowed_uses = pd.Series(allowed_uses)
                directions = pd.Series(directions)
                osm_way_ids = pd.Series(osm_way_ids)

                collapsed_links.append(
                    {
                        "from_node_id": from_node_id,
                        "to_node_id": to_node_id,
                        "link_type": "Mainline",
                        "geometry": collapsed_geometry,
                        "lanes": (
                            lanes.mode().iloc[0] if not lanes.mode().empty else None
                        ),
                        "link_id": f"collapsed_{from_node_id}_{to_node_id}",
                        "name": "Nimitz Freeway",
                        "osm_way_id": list(osm_way_ids.dropna().unique()),
                        "directed": 1,
                        "dir_flag": True,
                        "length_m": length_m,
                        "length_mi": length_mi,
                        "facility_type": (
                            facility_types.mode().iloc[0]
                            if not facility_types.empty
                            else None
                        ),
                        "allowed_uses": (
                            allowed_uses.mode().iloc[0]
                            if not allowed_uses.empty
                            else None
                        ),
                        "direction": (
                            directions.mode().iloc[0] if not directions.empty else None
                        ),
                    }
                )

        collapsed_links_gdf = gpd.GeoDataFrame(collapsed_links, crs=self.common_crs)

        # Add length columns to non-mainline links
        ramp_links["length_m"] = ramp_links.geometry.length
        ramp_links["length_mi"] = ramp_links["length_m"] / 1609.344

        # Drop unnecessary columns from non-mainline links
        ramp_links = ramp_links.drop(
            columns=["free_speed", "capacity", "notes", "length"], errors="ignore"
        )

        # Concatenate collapsed mainline and untouched ramp links
        self.links = gpd.GeoDataFrame(
            pd.concat([collapsed_links_gdf, ramp_links], ignore_index=True),
            crs=self.common_crs,
        )

    def _attach_nearest_detectors(
        self,
        buffer_m: float = 5.0,
        detector_type: str = "",
    ):
        # Validate detector_type and define associated link_type
        if detector_type == "Mainline":
            link_type = "Mainline"
        elif detector_type == "Off Ramp":
            link_type = "Off-ramp"
        elif detector_type == "On Ramp":
            link_type = "On-ramp"
        else:
            raise ValueError(
                f"Expected 'detector_type' to be one of ['Mainline', 'Off Ramp', 'On Ramp'], but it is {detector_type}"
            )

        # Filter for type
        vds_gdf = self.detectors.loc[self.detectors["Type"] == detector_type, :].copy()
        leftover_vds = self.detectors.loc[
            self.detectors["Type"] != detector_type, :
        ].copy()
        links_gdf = self.links.loc[self.links["link_type"] == link_type, :].copy()
        leftover_links = self.links.loc[
            self.links["link_type"] != link_type, :
        ].copy()  # For stitching back together

        # Prepare result columns
        links_gdf["nearest_detectors"] = [[] for _ in range(len(links_gdf))]
        links_gdf["Station ID"] = None
        links_gdf["Abs PM"] = np.nan

        # Prepare lane columns
        vds_gdf["Lanes"] = pd.to_numeric(vds_gdf["Lanes"], errors="coerce")
        links_gdf["lanes"] = pd.to_numeric(links_gdf["lanes"], errors="coerce")

        # Prepare PM column
        vds_gdf["Abs PM"] = pd.to_numeric(vds_gdf["Abs PM"], errors="coerce")

        # Build spatial index for faster lookup
        vds_sindex = vds_gdf.sindex

        # Iterate through links
        for idx, link_row in links_gdf.iterrows():
            # Get link attributes
            link_geom = link_row.geometry
            buffer_geom = link_geom.buffer(buffer_m)
            link_id = link_row["link_id"]
            link_lanes = link_row.get("lanes", None)

            # Manual override
            if link_id in self.link_to_station_override:
                assigned_station_id = self.link_to_station_override[link_id]
                links_gdf.at[idx, "nearest_detectors"] = [assigned_station_id]
                links_gdf.at[idx, "Station ID"] = assigned_station_id
                links_gdf.at[idx, "Abs PM"] = vds_gdf.loc[
                    vds_gdf["Station ID"] == assigned_station_id, "Abs PM"
                ].iloc[  # type: ignore
                    0
                ]

                # Also set detector geometry to midpoint of link
                midpoint = link_geom.interpolate(link_geom.length / 2)
                vds_gdf.loc[vds_gdf["Station ID"] == assigned_station_id, "geometry"] = Point(  # type: ignore
                    midpoint.x, midpoint.y
                )
                continue

            # Get possible matches using spatial index
            possible_matches_index = list(vds_sindex.intersection(buffer_geom.bounds))
            possible_matches = vds_gdf.iloc[possible_matches_index]  # type: ignore

            # Filter detectors actually within buffer
            within_buffer = possible_matches[
                possible_matches.geometry.within(buffer_geom)
            ]

            # Filter by matching lane count if applicable
            if link_lanes is not None and "lanes" in within_buffer.columns:
                within_buffer = within_buffer.loc[
                    within_buffer["lanes"] == link_lanes, :
                ]

            if not within_buffer.empty:
                # Compute distances to the link geometry
                within_buffer = within_buffer.copy()
                within_buffer["distance"] = within_buffer.geometry.apply(
                    lambda pt: pt.distance(link_geom)
                )

                # Sort by distance
                sorted_detectors = within_buffer.sort_values(by="distance")

                # Extract sorted list of Station IDs
                detector_ids = sorted_detectors["Station ID"].tolist()
                closest_station_id = detector_ids[0]

                # Assign to result columns
                links_gdf.at[idx, "nearest_detectors"] = detector_ids
                links_gdf.at[idx, "Station ID"] = closest_station_id
                links_gdf.at[idx, "Abs PM"] = vds_gdf.loc[
                    vds_gdf["Station ID"] == closest_station_id, "Abs PM"
                ].iloc[  # type: ignore
                    0
                ]

                # Update detector geometry to midpoint of link
                midpoint = link_geom.interpolate(link_geom.length / 2)
                vds_gdf.loc[vds_gdf["Station ID"] == closest_station_id, "geometry"] = (  # type: ignore
                    Point(midpoint.x, midpoint.y)
                )

        # Assign the modified GDFs back to the original objects
        self.links = gpd.GeoDataFrame(
            pd.concat([links_gdf, leftover_links], ignore_index=True),
            crs=self.common_crs,
        )
        self.detectors = gpd.GeoDataFrame(
            pd.concat([vds_gdf, leftover_vds], ignore_index=True), crs=self.common_crs
        )

    def _assign_virtual_detectors(self):
        """
        Assigns virtual detectors to links without a real detector.
        """
        # Identify links without detectors
        missing_detector_mask = self.links["Station ID"].isna()
        missing_links = self.links.loc[missing_detector_mask, :].copy()
        non_missing_links = self.links.loc[~missing_detector_mask, :].copy()

        # Filter for links within the boundary
        missing_links = self._filter_data_by_boundary(gdf_to_filter=missing_links)

        if missing_links.empty:
            logger.info(
                "No links with missing vehicle detection stations found - no virtual detectors created."
            )
            return

        # Generate virtual detector records
        virtual_detectors = []
        next_station_id = 0

        # Mapping between lane type and detector type
        link_to_detector_type_map = {
            "Mainline": "Mainline",
            "Off-ramp": "Off ramp",
            "On-ramp": "On ramp",
        }

        for idx, link in missing_links.iterrows():
            midpoint = link.geometry.interpolate(0.5, normalized=True)
            link_id = link["link_id"]
            station_id = f"V{str(link_id).zfill(5)}"
            lanes = link["lanes"] if "lanes" in link else None
            link_type = link["link_type"] if "link_type" in link else None
            detector_type = (
                link_to_detector_type_map.get(link_type, None)
                if link_type is not None
                else None
            )
            new_detector_record = {
                "Station ID": station_id,
                "Name": f"Virtual Detector {station_id}",
                "Lanes": lanes,
                "Type": detector_type,
                "Sensor Type": "virtual",
                "Fwy": f"{self.freeway_name}-{self.direction}",
                "geometry": midpoint,
                # Fill in other expected columns with NaN
                **{
                    col: np.nan
                    for col in self.detectors.columns
                    if col
                    not in [
                        "Station ID",
                        "Name",
                        "Lanes",
                        "Type",
                        "Sensor Type",
                        "Fwy",
                        "geometry",
                    ]
                },
            }

            # Add the new record to the list
            virtual_detectors.append(new_detector_record)

            # Update the link to reference the virtual detector
            missing_links.at[idx, "nearest_detectors"] = [station_id]
            missing_links.at[idx, "Station ID"] = station_id

            # Prepare the next station ID
            next_station_id += 1

        # Create a new GeoDF of virtual detectors
        virtual_vds_gdf = gpd.GeoDataFrame(virtual_detectors, crs=self.common_crs)

        # Combine virtual and real detectors
        self.detectors = gpd.GeoDataFrame(
            pd.concat([self.detectors, virtual_vds_gdf], ignore_index=True),
            crs=self.common_crs,
        )

        # Combine missing links and non-missing links
        self.links = gpd.GeoDataFrame(
            pd.concat([missing_links, non_missing_links], ignore_index=True),
            crs=self.common_crs,
        )

    def _create_mock_virtual_detector_timeseries(self):
        """
        Creates mock timeseries CSV files for each virtual detector, matching the
        structure and time index of a reference detector CSV.
        """
        # Load the reference CSV
        assert self.detector_timeseries_data_directory.is_dir()
        reference_csv_path = os.path.join(
            str(self.detector_timeseries_data_directory),
            f"{self.timeseries_reference_station}.csv",
        )
        assert Path(reference_csv_path).is_file()
        ref_df = pd.read_csv(reference_csv_path)

        # Identify the column names (excluding timestamp column)
        timestamp_col = ref_df.columns[0]
        data_cols = ref_df.columns[1:]

        # Filter virtual detectors
        virtual_detectors = self.detectors.loc[
            self.detectors["Sensor Type"] == "virtual", :
        ]

        for _, row in virtual_detectors.iterrows():
            station_id = row["Station ID"]

            mock_csv_path = os.path.join(
                self.detector_timeseries_data_directory, f"{station_id}.csv"
            )

            if Path(mock_csv_path).is_file():
                logger.info(
                    f"Timeseries data for {station_id} already exists at {mock_csv_path} – skipping."
                )
                continue

            logger.info(
                f"Creating mock timeseries data for virtual vehicle detector station: {station_id}"
            )

            # Copy time index
            mock_df = pd.DataFrame()
            mock_df[timestamp_col] = ref_df[timestamp_col]

            # Use NaNs for the other columns, matching the data type
            for col in data_cols:
                mock_df[col] = np.nan

            # Save to CSV
            mock_df.to_csv(mock_csv_path, index=False)
            logger.info(f"Mock data saved to: {mock_csv_path}")

    def _identify_node_postmile(
        self, node_row: gpd.GeoSeries, search_radius_m: float = 1000.0
    ) -> float:
        """
        Identify the postmile for a node, adjusted for its position relative to the nearest postmile.

        Parameters
        ----------
        node_row : gpd.GeoSeries
            Row from nodes GeoDataFrame containing 'geometry'.
        search_radius : float, optional
            Buffer distance in degrees (small number, e.g., ~1 km) to find candidate postmiles.

        Returns
        -------
        float
            Adjusted postmile for the node.
        """
        node_geom = node_row.geometry
        node_id = node_row["node_id"]

        # Without a postmiles GeoDF, we cannot do anything
        if self.postmiles is None:
            return np.nan

        # Find candidate postmiles near the node
        candidate_postmiles = self.postmiles[
            self.postmiles.geometry.distance(node_geom) <= search_radius_m
        ]
        if candidate_postmiles.empty:
            logger.warning(
                f"No postmile markers within {search_radius_m} meters of Node {node_id} were found"
            )
            return np.nan

        # Find the nearest postmile point
        nearest_pm_idx = candidate_postmiles.distance(node_geom).idxmin()
        nearest_pm = candidate_postmiles.loc[nearest_pm_idx]
        nearest_pm_geom = nearest_pm.geometry
        base_milepost = nearest_pm["PM"]

        # Find the link that this node belongs to (assuming node is an endpoint)
        touching_links_mask = (self.links["link_type"] == "Mainline") & (
            (self.links["from_node_id"] == node_id)
            | (self.links["to_node_id"] == node_id)
        )
        touching_links = self.links.loc[touching_links_mask, :]
        if touching_links.empty:
            # Can't refine without a roadway segment
            return base_milepost  # type: ignore

        # Find the touching link that's closest to the postmile
        nearest_link = touching_links.loc[
            touching_links.distance(nearest_pm_geom).idxmin()
        ]
        link_geom = nearest_link.geometry

        # Project both the postmile point and the node onto the link
        node_proj = link_geom.project(node_geom)
        pm_proj = link_geom.project(nearest_pm_geom)

        # Distance along the link between the milepost and the node (in miles)
        dist_along_link = abs(node_proj - pm_proj) / 1609.34

        # Determine if the node is ahead (+) or behind (-) relative to the postmile
        if ((self.direction == "N") and (nearest_link["from_node_id"] == node_id)) or ((self.direction == "S") and (nearest_link["to_node_id"] == node_id)):  # type: ignore
            adjusted_milepost = base_milepost - dist_along_link
        elif ((self.direction == "N") and (nearest_link["to_node_id"] == node_id)) or ((self.direction == "S") and (nearest_link["from_node_id"] == node_id)):  # type: ignore
            adjusted_milepost = base_milepost + dist_along_link
        else:
            adjusted_milepost = base_milepost

        return round(adjusted_milepost, 3)  # type: ignore

    def _assign_postmile_to_node(self):
        """
        Assign adjusted postmiles to all nodes in the GeoDF
        """
        classification_mask = self.nodes["purpose"].isin(
            ["merge", "diverge", "mainline"]
        )
        nodes_to_classify = self.nodes.loc[classification_mask, :].copy()
        nodes_to_classify["postmile"] = nodes_to_classify.apply(
            self._identify_node_postmile,
            axis=1,
        )
        self.nodes.loc[classification_mask, "postmile"] = nodes_to_classify["postmile"]

    def _filter_by_postmile(self):
        """
        Filter the nodes and links by a postmile range
        """
        # Start by assigning unfiltered copies of the original links and nodes to use as default
        filtered_nodes = self.nodes.copy()
        filtered_links = self.links.copy()

        # Filter nodes
        if self.postmile_start is not None:
            filtered_nodes = filtered_nodes.loc[
                (self.nodes["postmile"] >= self.postmile_start),
                :,
            ]
        if self.postmile_end is not None:
            filtered_nodes = filtered_nodes.loc[
                (self.nodes["postmile"] <= self.postmile_end),
                :,
            ]

        # Filter links based on nodes:
        # 1) Mainline links with both endpoints in the set of valid nodes
        # 2) On-ramp links with "to" node in the set of valid nodes
        # 3) Off-ramp links with "from" node in the set of valid nodes
        valid_node_ids = set(filtered_nodes["node_id"])
        link_mask = (
            (
                (self.links["link_type"] == "Mainline")
                & (
                    (self.links["from_node_id"].isin(valid_node_ids))
                    & (self.links["to_node_id"].isin(valid_node_ids))
                )
            )
            | (
                (self.links["link_type"] == "On-ramp")
                & (self.links["to_node_id"].isin(valid_node_ids))
            )
            | (
                (self.links["link_type"] == "Off-ramp")
                & (self.links["from_node_id"].isin(valid_node_ids))
            )
        )
        filtered_links = self.links.loc[
            link_mask,
            :,
        ].copy()

        # Go back and include the ramp nodes
        revalidated_node_ids = pd.unique(
            filtered_links.loc[:, ["from_node_id", "to_node_id"]].values.ravel()
        )
        filtered_nodes = self.nodes.loc[
            self.nodes["node_id"].isin(revalidated_node_ids)
        ].copy()

        # Assign the filtered version of both back to the original
        self.nodes = filtered_nodes
        self.links = filtered_links

    def _prepare_roadway_data(self):
        """
        Modifies the input data so that it matches the following criteria:
        1)

        * IMPORTANT: Functions should be called in this order, as they modify the links, nodes, and
        detector GeoDFs in-place, and may depend on a certain modification being done by an earlier
        function.
        """

        # Add a 'purpose' column to the nodes
        self._assign_node_purpose()

        # Filter the postmiles based on the freeway name
        self._filter_postmiles_by_route()

        # Filter the links, nodes, and detectors based on the direction
        self._filter_data_by_direction()

        # Add a 'postmile' column to the nodes
        self._assign_postmile_to_node()

        # Eliminate unnecessary nodes
        self._filter_important_nodes()

        # Combine mainline links now that the nodes have been eliminated
        self._collapse_mainline_links()

        # Assign detectors to the links: 1 detector per link
        self._attach_nearest_detectors(buffer_m=10, detector_type="Mainline")
        self._attach_nearest_detectors(buffer_m=30, detector_type="Off Ramp")
        self._attach_nearest_detectors(buffer_m=30, detector_type="On Ramp")
        self._assign_virtual_detectors()

        # Create some mock timeseries data for each virtual detector
        self._create_mock_virtual_detector_timeseries()

        # Filter for postmile boundaries
        self._filter_by_postmile()

    def _build_freeway_graph(self) -> nx.DiGraph:
        """
        Build a roadway graph from node, link, and detector GeoDataFrames.
        Node attributes come from node_gdf.
        Edge attributes merge link_gdf attributes with detector attributes from detector_gdf.
        Attributes are categorized as 'static' or 'dynamic'.
        """

        # Initialize an empty directed graph
        G = nx.DiGraph()

        # --- Filter links: keep only those with a Station ID ---
        filtered_links = self.links.dropna(subset=["Station ID"]).copy()

        # --- Filter nodes: keep only those referenced in filtered links ---
        valid_node_ids = set(filtered_links["from_node_id"]).union(
            filtered_links["to_node_id"]
        )
        filtered_nodes = self.nodes.loc[
            self.nodes["node_id"].isin(valid_node_ids), :
        ].copy()

        # --- Add nodes ---
        for _, node in filtered_nodes.iterrows():
            node_attrs = node.to_dict()
            G.add_node(
                node["node_id"],  # node key
                static={k: v for k, v in node_attrs.items()},
                dynamic={},  # no dynamic attrs for nodes yet
            )

        # --- Prepare detector attributes ---
        detector_attrs = self.detectors.set_index("Station ID").to_dict(orient="index")

        # --- Add edges ---
        for _, link in filtered_links.iterrows():
            edge_attrs = {
                col: link[col]
                for col in self.links.columns
                if col not in ["from_node_id", "to_node_id"]
            }

            # Merge detector attributes if Station ID is available
            station_id = link.get("Station ID", None)
            if station_id in detector_attrs:
                # Avoid overwriting existing keys in edge_attrs
                for k, v in detector_attrs[station_id].items():
                    if k == "geometry":
                        edge_attrs[f"{k}_detector"] = v  # Save the detector geometry
                    elif k not in edge_attrs:
                        edge_attrs[k] = v

            # Separate into static/dynamic attributes
            static_attrs = {k: v for k, v in edge_attrs.items() if k != "Status"}
            dynamic_attrs = {k: v for k, v in edge_attrs.items() if k == "Status"}

            G.add_edge(
                link["from_node_id"],
                link["to_node_id"],
                static=static_attrs,
                dynamic=dynamic_attrs,
            )

        return G

    def visualize_roadway_graph(
        self,
        figsize=(12, 8),
        node_color="#ccebc5",
        layout="transit",
        ramp_offset=0.2,
    ):
        # Figure out an ordering of nodes along the corridor
        # This assumes the graph is basically a path with small branches for ramps
        ordered_nodes = list(nx.topological_sort(self.graph))

        pos = {}

        # Choose a layout function
        if layout == "transit":
            i = 0
            for node in ordered_nodes:
                # Get the node purpose
                attrs = self.graph.nodes[node]
                node_purpose = (
                    attrs.get("static", {}).get("purpose")
                    if isinstance(attrs.get("static"), dict)
                    else attrs.get("purpose")
                )
                if node_purpose in ["merge", "diverge", "mainline"]:
                    x = i
                    y = 0.0  # Mainline default
                    pos[node] = (x, y)
                    i += 1

            # Offset ramp nodes if not on mainline path
            for edge in self.graph.edges:
                u, v = edge
                # Get the edge type
                attrs = self.graph.edges[edge]
                edge_type = (
                    attrs.get("static", {}).get("link_type")
                    if isinstance(attrs.get("static"), dict)
                    else attrs.get("Type")
                )
                if edge_type == "Off-ramp":
                    # Get the x location of the source node
                    x = pos[u][0] + 0.5
                    # Off-ramps go down
                    y = 0.0 - ramp_offset
                    pos[v] = (x, y)
                elif edge_type == "On-ramp":
                    # Get the x location of the destination node
                    x = pos[v][0] - 0.5
                    # On-ramps come from above
                    y = 0.0 + ramp_offset
                    pos[u] = (x, y)
                else:
                    continue

        elif layout == "spring":
            pos = nx.spring_layout(self.graph, k=0.3, iterations=50)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        elif layout == "shell":
            pos = nx.shell_layout(self.graph)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        else:
            # Default to geographic layout
            pos = {
                node_id: (geom.x, geom.y)  # type: ignore
                for node_id, geom in zip(self.nodes["node_id"], self.nodes.geometry)
            }

        fig, ax = plt.subplots(figsize=figsize)

        # Extract node colors based on detector Status (based on Set3 colormap in matplotlib)
        # node_purpose_colors = {
        #     "merge": "#8dd3c7",  # green-ish
        #     "diverge": "#fb8072",  # red-ish
        #     "mainline": "#ffffb3",  # yellow-ish
        #     "Unknown": "#d9d9d9",  # gray-ish
        #     None: "#d9d9d9",  # gray-ish
        # }
        # node_colors = [
        #     node_purpose_colors.get(
        #         self.graph.nodes[n].get("static", {}).get("purpose"), None
        #     )
        #     for n in self.graph.nodes
        # ]

        # Extract edge colors based on sensor type
        sensor_type_map = {
            "loops": "#bc80bd",  # purple-ish
            "virtual": "#fdb462",  # orange
            None: "#d9d9d9",  # gray
        }
        edge_colors = [
            sensor_type_map.get(
                self.graph.edges[e].get("static", {}).get("Sensor Type"), None
            )
            for e in self.graph.edges
        ]

        # Proxy artists for edge colors
        loop_edge_line = mlines.Line2D(
            [],
            [],
            color=sensor_type_map["loops"],
            marker="",
            linestyle="-",
            linewidth=2,
            label="PeMS Detector",
        )
        virtual_edge_line = mlines.Line2D(
            [],
            [],
            color=sensor_type_map["virtual"],
            marker="",
            linestyle="-",
            linewidth=2,
            label="Missing Detector",
        )
        unknown_edge_line = mlines.Line2D(
            [],
            [],
            color=sensor_type_map[None],
            marker="",
            linestyle="-",
            linewidth=2,
            label="Unknown",
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=700,
            node_color=node_color,  # type: ignore
            edgecolors="black",
            ax=ax,
        )

        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos, edge_color=edge_colors, arrows=True, arrowstyle="-|>", arrowsize=12, width=2.5, ax=ax  # type: ignore
        )

        # Node labels
        nx.draw_networkx_labels(
            self.graph, pos, labels={n: n for n in self.graph.nodes}, font_size=8, ax=ax
        )

        # Edge labels (Station ID)
        edge_labels = {
            e: self.graph.edges[e].get("static", {}).get("Station ID", "")
            for e in self.graph.edges
        }
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5, ax=ax
        )

        # Add legend to the axes
        ax.legend(
            handles=[
                loop_edge_line,
                virtual_edge_line,
                unknown_edge_line,
            ],
            loc="upper right",
            fontsize=8,
            frameon=True,
        )

        plt.title(
            f"{self.freeway_name}-{self.direction} Segment\n{self.graph}", fontsize=14
        )
        plt.axis("off")
        plt.show()


def limit_dataframe_by_date(
    df: pd.DataFrame,
    start: typing.Union[str, pd.DatetimeIndex],
    end: typing.Union[str, pd.DatetimeIndex],
) -> pd.DataFrame:
    """
    Limit a DataFrame with a DatetimeIndex to the given start and end datetime range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.
    start : str or pd.Timestamp
        Starting datetime (inclusive).
    end : str or pd.Timestamp
        Ending datetime (inclusive).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    # Convert to Timestamp in case start/end are strings
    start = pd.to_datetime(start)  # type: ignore
    end = pd.to_datetime(end)  # type: ignore

    # Slice by datetime range
    return df.loc[start:end]


def load_link_flows_from_dir(
    links_gdf: gpd.GeoDataFrame,
    timeseries_dir: Path,
    timestamp_col: typing.Optional[str] = "timestamp",
    flow_col_name: str = "total_flow_[veh/5-min]",
    start_datetime: typing.Union[str, pd.DatetimeIndex, None] = None,
    end_datetime: typing.Union[str, pd.DatetimeIndex, None] = None,
) -> dict:
    """
    Load flow timeseries for each link/station from CSV files in timeseries_dir.

    Parameters
    ----------
    links_gdf : gpd.GeoDataFrame
        Must contain 'link_id' and 'Station ID' columns.
    timeseries_dir : Path
        Directory where CSVs are stored with filenames like "{station_id}.csv"
    timestamp_col : typing.Optional[str], optional
        If None, the first column of the CSV is treated as the timestamp column. By default "timestamp"
    flow_col_name : str, optional
        Specify different flow columns for lanes, or use the total. By default "total_flow_[veh/5-min]"

    Returns
    -------
    dict
        Maps link_id -> pd.Series (indexed by pd.DatetimeIndex) with flow values.
    """
    assert timeseries_dir.is_dir
    link_flow_series = {}

    for _, link in links_gdf.iterrows():
        station_id = link.get("Station ID")
        link_id = link.get("link_id")
        if pd.isna(station_id) or pd.isna(link_id):
            logger.warning(
                f"Could not identify a Station ID or Link ID for link:\n{link}"
            )
            continue

        csv_path = timeseries_dir / f"{station_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Timeseries file not found for station {station_id}: {csv_path}"
            )

        df = pd.read_csv(csv_path)
        if timestamp_col is None:
            tcol = df.columns[0]
        else:
            tcol = timestamp_col

        # identify flow column
        if flow_col_name not in df.columns:
            raise ValueError(
                f"Could not find flow column {flow_col_name} in {csv_path}"
            )

        series = pd.to_datetime(df[tcol])
        series = (
            pd.Series(df[flow_col_name].values, index=series, name=flow_col_name)
            .sort_index()
            .fillna(0.0)
        )
        # name by link_id (keep link_id as string)
        link_flow_series[link_id] = series

    # align timestamps: reindex each series to TimeseriesIndex based on the min/max timestamp and inferred frequency
    min_timestamp = min([ts for s in link_flow_series.values() for ts in s.index])
    max_timestamp = max([ts for s in link_flow_series.values() for ts in s.index])
    next_year_start = datetime(max_timestamp.year + 1, 1, 1, 0, 0, 0)
    delta = next_year_start - max_timestamp
    minutes_delta = int(delta.total_seconds() // 60)
    full_index = pd.date_range(
        start=min_timestamp, end=max_timestamp, freq=f"{minutes_delta}min"
    )
    for k in list(link_flow_series.keys()):
        link_flow_series[k] = link_flow_series[k].reindex(full_index)

    # Filter by time
    if start_datetime is None:
        start_datetime = all_times[0]  # type: ignore
    if end_datetime is None:
        end_datetime = all_times[-1]  # type: ignore
    for k in list(link_flow_series.keys()):
        link_flow_series[k] = limit_dataframe_by_date(
            link_flow_series[k], start=start_datetime, end=end_datetime  # type: ignore
        )

    return link_flow_series


def compute_split_ratios_for_node(
    node_id: typing.Hashable,
    G: nx.DiGraph,
    merged_df: pd.DataFrame,
    eps: float = 1e-9,
    imputation_col_name: str = "pct_observed",
    imputation_threshold: float = 100.0,
    flow_col_name: str = "total_flow_[veh/5-min]",
    freq="5min",
):
    """
    Compute split ratios for a single node.

    Parameters
    ----------
    node_id : hashable
        Node ID in the graph G.
    G : networkx.DiGraph
        Graph where edges are links. Edges must have attribute 'link_id' or the graph edge keys must be link ids.
    merged_df : pd.DataFrame
        DataFrame containing timeseries data merged with metadata for the links. Must contain
        columns:  ['link_id', 'from_node_id', 'to_node_id', 'total_flow_[veh/5-min]']
    eps : float
        Tolerance for zero flow checks.

    Returns
    -------
    pd.DataFrame
        Index: timestamps. Columns: "{from_link}->{to_link}". Values: split ratios (floats).
    """
    incoming_links = []
    outgoing_links = []

    # collect incoming and outgoing link ids from the graph edges (link_id is a static attribute)
    for u, v in G.in_edges(node_id):
        attrs = G.edges[u, v]
        # Access the relevant attributes
        lid = (
            attrs.get("static", {}).get("link_id")
            if isinstance(attrs.get("static"), dict)
            else attrs.get("link_id")
        )
        incoming_links.append(lid)

    for u, v in G.out_edges(node_id):
        attrs = G.edges[u, v]
        lid = (
            attrs.get("static", {}).get("link_id")
            if isinstance(attrs.get("static"), dict)
            else attrs.get("link_id")
        )
        outgoing_links.append(lid)

    # Filter None or NaN link ids
    incoming_links = [
        lid for lid in incoming_links if lid in merged_df["link_id"].unique()
    ]
    outgoing_links = [
        lid for lid in outgoing_links if lid in merged_df["link_id"].unique()
    ]

    # If no flows available, return empty df
    if (len(incoming_links) == 0) and (len(outgoing_links) == 0):
        logging.info(f"No flows available for node {node_id}")
        return pd.DataFrame()

    # Create a mapping of link id -> link type
    link_id_to_type = dict(zip(merged_df["link_id"], merged_df["link_type"]))

    # get common index (timestamp)
    if freq == "5min":
        ts_index = pd.date_range(
            start="2022-01-01 00:00:00",
            end="2024-12-31 23:55:00",
            freq=freq,
            name="timestamp",
        )
    elif freq == "15min":
        ts_index = pd.date_range(
            start="2022-01-01 00:00:00",
            end="2024-12-31 23:45:00",
            freq=freq,
            name="timestamp",
        )
    else:
        raise ValueError(
            f"Expected parameter 'freq' to be one of ['5min', '15min'], but {freq} was passed"
        )

    # prepare DataFrame to fill
    col_names = []
    for i in incoming_links:
        if link_id_to_type[i] != "Mainline":
            continue
        for j in outgoing_links:
            if link_id_to_type[j] == "Off-ramp":
                col_names.append("beta")
            elif link_id_to_type[j] == "Mainline":
                col_names.append("1-beta")
            else:
                logger.warning(
                    f"Unexpected connection type between incoming link {i} (type: {incoming_links[i]}) and outgoing link {j} (type: {outgoing_links[j]})"
                )
    splits_df = pd.DataFrame(index=ts_index, columns=col_names, dtype=float)

    if len(incoming_links) == 1 and len(outgoing_links) > 1:
        # Diverging / Off-ramp Node: calculate the split ratio
        logger.info(f"Calculating split ratio for diverging node {node_id}")

        # Get the inbound link data
        in_id = incoming_links[0]
        in_df = merged_df.loc[merged_df["link_id"] == in_id, :]
        # Make sure to put the timeseries data on a common index
        in_series = in_df.loc[:, flow_col_name].copy().reindex(ts_index)

        # Debugging
        in_station = in_df.loc[:, "Station ID"].unique()[0]
        logger.info(f"Node ID {node_id} uses station {in_station} for the input")

        # Check against excessive outgoing links
        if len(outgoing_links) > 2:
            raise RuntimeError(
                f"Invalid node type detected: {len(outgoing_links)} outgoing links"
            )
        for out_id in outgoing_links:
            # Get the outbound link data
            out_type = link_id_to_type[out_id]
            out_df = merged_df.loc[merged_df["link_id"] == out_id, :]
            # Make sure to put the timeseries data on a common index
            out_series = out_df.loc[:, flow_col_name].copy().reindex(ts_index)

            # Debugging
            out_station = out_df.loc[:, "Station ID"].unique()[0]
            logger.info(
                f"Node ID {node_id} uses station {out_station} for the {out_type}"
            )

            # Identify good indices:
            # 1) Non-zero inbound flow
            # 2) Inbound flow AND outbound flow are not missing
            # 3) Inbound flow AND outbound flow meet the criteria for including PeMS-imputated data
            good_mask = (
                (in_series.abs() >= eps)
                & (in_series.notna())
                & (out_series.notna())
                & (in_df[imputation_col_name] >= imputation_threshold)
                & (out_df[imputation_col_name] >= imputation_threshold)
            )
            if not good_mask.any():
                in_detector = in_df["Station ID"].unique()[0]
                out_detector = out_df["Station ID"].unique()[0]
                logger.warning(
                    f"No valid data for split ratio calculation at node {node_id} for outgoing link {out_id} (Inbound detector {in_detector}, outbound detector {out_detector}"
                )
                continue

            num_original_values = min(len(in_series), len(out_series))
            num_remaining_values = good_mask.sum()
            logger.info(
                f"After applying filtering criteria at node {node_id} for outgoing_link {out_id}, {num_remaining_values} values remain from the original {num_original_values} values"
            )

            # Calculate split ratio for only good indices
            split_ratio = (out_series[good_mask] / in_series[good_mask]).replace([np.inf, -np.inf], np.nan)  # type: ignore
            if out_type == "Off-ramp":
                split_key = "beta"
            elif out_type == "Mainline":
                split_key = "1-beta"
            else:
                logger.warning(
                    f"Unexpected connection type between incoming link {lid} (type: {incoming_links[lid]}) and outgoing link {out_id} (type: {out_type})"
                )
                continue

            splits_df.loc[good_mask, split_key] = split_ratio

        # Again, identify good indices:
        # 1) Off-ramp AND mainline splits are not missing
        num_original_values = max(
            splits_df["beta"].notna().sum(), splits_df["1-beta"].notna().sum()
        )
        good_mask = (splits_df["beta"].notna()) & (splits_df["1-beta"].notna())
        splits_df.loc[~good_mask, :] = np.nan
        logger.info(
            f"Applying final filtering criteria for node {node_id} left {good_mask.sum()} valid split ratios from {num_original_values} calculated partial splits"
        )
    elif (len(outgoing_links) == 1 and len(incoming_links) >= 1) or (
        len(incoming_links) == 0 and len(outgoing_links) >= 1
    ):
        # Merging / On-ramp node: we don't care about split ratios
        logger.info(
            f"Node {node_id} is an on-ramp node, skipping split ratio calculation"
        )
    elif len(incoming_links) == 1 and len(outgoing_links) == 1:
        # Mainline node: we don't care about split ratios
        logger.info(
            f"Node {node_id} is a mainline node, skipping split ratio calculation"
        )
    elif len(incoming_links) >= 1 and len(outgoing_links) == 0:
        # Destination node for off-ramp: we don't care about split ratios
        logger.info(
            f"Node {node_id} is the destination node for an off-ramp, skipping split ratio calculation"
        )
    else:
        # General many-to-many: for now, we do not attempt to infer. Return NaN and warn.
        logger.warning(
            f"Node {node_id} is a many-to-many node (in={len(incoming_links)}, out={len(outgoing_links)}). Returning NaNs for split ratios."
        )
    return splits_df  # Empty unless the node was a diverging node


def compute_all_split_ratios(
    G: nx.DiGraph,
    merged_df: pd.DataFrame,
    node_ids: typing.Optional[list] = None,
    flow_col_name: str = "total_flow_[veh/5-min]",
    freq: str = "5min",
):
    """
    Compute split ratios for many nodes.

    Parameters
    ----------
    G : networkx.DiGraph
    link_flow_series : dict mapping link_id -> pd.Series
    node_ids : iterable or None
        If None, compute for all nodes in G.

    Returns
    -------
    dict : node_id -> pandas.DataFrame of split ratios for off-ramp nodes
    """
    results = {}
    if node_ids is None:
        node_ids = list(G.nodes)

    for n in node_ids:
        df = compute_split_ratios_for_node(
            n, G, merged_df, flow_col_name=flow_col_name, freq=freq
        )
        results[n] = df

    # Filter for items with a populated DF
    results = {k: v for k, v in results.items() if not v.dropna().empty}

    return results


def build_link_timeseries_df(
    links_df,
    timeseries_dir: Path,
    cols_to_load: list[str] = [
        "total_flow_[veh/5-min]",
        "avg_speed_[mph]",
        "pct_observed",
    ],
    start_datetime: typing.Union[str, pd.DatetimeIndex, None] = None,
    end_datetime: typing.Union[str, pd.DatetimeIndex, None] = None,
):
    # Validate the timeseries directory that was passed
    assert timeseries_dir.is_dir

    # Allocate an empty list to pack with DataFrames for each link
    combined_list = []

    for _, link in links_df.iterrows():
        station_id = link.get("Station ID")
        link_id = link.get("link_id")
        if pd.isna(station_id) or pd.isna(link_id):
            logger.warning(
                f"Could not identify a Station ID or Link ID for link:\n{link}"
            )
            continue

        csv_path = timeseries_dir / f"{station_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Timeseries file not found for station {station_id}: {csv_path}"
            )

        # Make sure there is a timestamp column
        if "timestamp" not in cols_to_load:
            cols_to_load.append("timestamp")
        df = pd.read_csv(csv_path, usecols=cols_to_load)

        # Set the timestamp as the index and make it a DateTime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Filter by time
        if start_datetime is None:
            start_datetime = df.index[0]
        if end_datetime is None:
            end_datetime = df.index[-1]
        df = limit_dataframe_by_date(df=df, start=start_datetime, end=end_datetime)  # type: ignore

        # Attach the link metadata to each row of its timeseries
        for col in links_df.columns:
            df[col] = link[col]

        combined_list.append(df)

    # Concatenate all link-specific DataFrames into one
    if combined_list:
        combined_df = pd.concat(combined_list, ignore_index=False)
    else:
        combined_df = pd.DataFrame()

    return combined_df


def prepare_directory(directory_path: Path) -> bool:
    try:
        # Try to make the directory assuming it does not already exist
        directory_path.mkdir(parents=True)
        logger.info(f"Directory '{directory_path}' created successfully.")
        return True
    except FileExistsError:
        # If it already exists, provide the option to overwrite or keep
        logger.info(f"Directory '{directory_path}' already exists.")
        response = input(
            f"Directory '{directory_path}' already exists. Would you like to overwrite the directory? (Y/n)"
        )
        if response.lower() == "y":
            directory_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Overwriting files in directory {directory_path}.")
            return True
        elif response.lower() == "n":
            logger.info(f"Using existing files in directory {directory_path}.")
            return False
        else:
            raise ValueError(
                f"Unexpected response. Got {response}, expecting 'y' or 'n'."
            )
    except Exception as e:
        # Catch other error
        logger.error(
            f"An error occurred while preparing output directory {directory_path}:\n{e}"
        )
        raise e


def find_file_os_walk(directory: str, filename: str) -> typing.Union[str, None]:
    # Make sure the directory really is a directory
    assert Path(directory).is_dir

    # Look for the filename
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def download_osm_file_by_relation_id(
    relation_id: int, output_dir: str, osm_filename: str
) -> Path:
    # Prepare a directory for the output files
    write_new_directory = prepare_directory(Path(output_dir))

    # Construct the filepath for the osm file
    osm_filepath = os.path.join(output_dir, osm_filename)

    if write_new_directory:
        logger.info(
            f"Initiating download of OSM Relation ID {relation_id} to {osm_filepath}"
        )
        # Download the osm file
        try:
            og.downloadOSMData(area_id=relation_id, output_filepath=osm_filepath)
        except Exception as e:
            logger.error(
                f"An exception occurred while downloading OSM data for Relation ID {relation_id} to {osm_filepath}."
            )
            raise e
    else:
        logger.info(f"Using existing files in {osm_filepath}")

    # Ensure that the file has downloaded successfully
    osm_filepath = Path(osm_filepath)
    assert osm_filepath.is_file()

    # Return the Path object to the file
    return osm_filepath


def download_gmns_network_from_osm_file(
    osm_filepath: Path,
    link_types: list[str] = ["motorway"],
    mode_types: list[str] = ["auto"],
    output_dir: str = os.path.join(".", "gmns_files"),
) -> None:
    # Validate the link types
    VALID_LINK_TYPES = [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "service",
        "cycleway",
        "footway",
        "track",
        "unclassified",
        "railway",
        "aeroway",
    ]
    validated_link_types = [
        link_type for link_type in link_types if link_type in VALID_LINK_TYPES
    ]
    if len(validated_link_types) == 0:
        raise ValueError(
            f"Link types {link_types} are not valid. Select from:\n{VALID_LINK_TYPES}"
        )

    # Validate the mode types
    VALID_MODE_TYPES = ["auto", "bike", "walk", "railway", "aeroway"]
    validated_mode_types = [
        mode_type for mode_type in mode_types if mode_type in VALID_MODE_TYPES
    ]
    if len(validated_mode_types) == 0:
        raise ValueError(
            f"Mode types {mode_types} are not valid. Select from:\n{VALID_MODE_TYPES}"
        )

    # Load the network
    try:
        net = og.getNetFromFile(
            str(osm_filepath),
            mode_types=validated_mode_types,  # type: ignore
            link_types=validated_link_types,
        )
    except Exception as e:
        logger.error(
            f"An exception occurred while loading the GMNS network from {str(osm_filepath)}"
        )
        raise e

    # Write link and node CSVs for the network
    try:
        og.outputNetToCSV(net, output_folder=output_dir)
        logger.info(f"Network files saved to {output_dir}")
    except Exception as e:
        logger.error(
            f"An exception was raised while saving the network to {output_dir}"
        )
        raise e

    return


def find_way_ids_by_ref(
    gmns_network_directory: str,
    osm_filename: str,
    ref: str = "I 880",
    id_whitelist: list[str] = [],
    id_blacklist: list[str] = [],
) -> tuple[set[str], set[str], set[str]]:
    # Find the .osm file
    osm_filepath = find_file_os_walk(
        directory=gmns_network_directory, filename=osm_filename
    )
    if osm_filepath is None:
        logger.error(f"Could not locate {osm_filename} in {gmns_network_directory}")
        raise FileNotFoundError

    # Load the .osm file as an ElementTree
    logger.info(f"Loading {osm_filepath}...")
    tree = ET.parse(osm_filepath)
    root = tree.getroot()

    # Empty set of node IDs attributed with the ref
    node_ids_set = set()

    # Empty list of motorway way IDs attributed with the ref
    motorway_way_ids = []

    # Empty list of motorway_link way IDs
    motorway_link_way_ids = []

    # Search the ET for ways that match the ref
    for way in root.findall("way"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        if ((ref in tags.get("ref", "")) or (way.attrib["id"] in id_whitelist)) and (
            way.attrib["id"] not in id_blacklist
        ):
            # Add the way ID to capture the motorway
            motorway_way_ids.append(int(way.attrib["id"]))
            for nd in way.findall("nd"):
                # Store the nodes to prepare for capturing the motorway_links
                node_ids_set.add(int(nd.attrib["ref"]))

    # Traverse the ET again for any
    for way in root.findall("way"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        if (
            (tags.get("highway", "") == "motorway_link")
            or (way.attrib["id"] in id_whitelist)
        ) and (way.attrib["id"] not in id_blacklist):
            # Get all the nodes for this motorway_link
            nd_refs = [int(nd.attrib["ref"]) for nd in way.findall("nd")]
            # Check if any nodes connect to the ways that were previously identified
            if any(nd in node_ids_set for nd in nd_refs):
                motorway_link_way_ids.append(int(way.attrib["id"]))

    # Convert to sets and merge
    motorway_way_ids = set(motorway_way_ids)
    motorway_link_way_ids = set(motorway_link_way_ids)
    all_way_ids = motorway_way_ids | motorway_link_way_ids

    # Logging output:
    logger.info(
        f"Identified {len(all_way_ids)} total ways associated with {ref}.\n{len(motorway_way_ids)} motorways.\n{len(motorway_link_way_ids)} motorway_links."
    )

    return (all_way_ids, motorway_way_ids, motorway_link_way_ids)


def classify_link_type(
    row: pd.Series, mainline_way_ids: set[str], mainline_node_ids: set[int]
):
    osm_id = row["osm_way_id"]
    from_node = row["from_node_id"]
    to_node = row["to_node_id"]

    if osm_id in mainline_way_ids:
        return "Mainline"
    elif from_node in mainline_node_ids:
        return "Off-ramp"
    elif to_node in mainline_node_ids:
        return "On-ramp"
    else:
        return ""  # fallback for ambiguous cases


def assign_link_type(
    links_df: pd.DataFrame, mainline_way_ids: set[str], overwrite: dict[int, str]
) -> pd.DataFrame:
    # Filter the DF for links with an osm_way_id in the mainline way IDs
    mainline_links = links_df.loc[
        links_df["osm_way_id"].isin(mainline_way_ids), :
    ].reset_index(drop=True)

    # Construct a set of unique mainline node IDs
    mainline_node_ids = set(
        pd.unique(
            mainline_links[["from_node_id", "to_node_id"]].values.ravel().astype(int)
        )
    )

    # Apply the classification method
    links_df["link_type"] = links_df.apply(
        classify_link_type, axis=1, args=(mainline_way_ids, mainline_node_ids)
    )

    # Apply any overwrites
    for link_id, link_type in overwrite.items():
        links_df.loc[links_df["link_id"] == link_id, "link_type"] = link_type

    return links_df


def find_boundary_nodes(graph: nx.DiGraph, links_df: pd.DataFrame) -> tuple[int, int]:
    # Lists of possible node IDs of boundary nodes
    starting_options = [
        int(n)
        for n in graph.nodes
        if (
            (graph.in_degree(n) == 0)
            and (links_df.set_index("from_node_id").loc[n, "link_type"] == "Mainline")
        )
    ]
    ending_options = [
        int(n)
        for n in graph.nodes
        if (
            (graph.out_degree(n) == 0)
            and (links_df.set_index("to_node_id").loc[n, "link_type"] == "Mainline")
        )
    ]

    # Make sure there's only one option
    assert len(starting_options) == 1
    assert len(ending_options) == 1

    # Mark the boundary nodes
    return starting_options[0], ending_options[0]


def assign_direction(links_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    # Add a new column to record the direction
    links_df["direction"] = ""

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges using the from_node_id and to_node_id columns
    for _, row in links_df.iterrows():
        G.add_edge(row["from_node_id"], row["to_node_id"], **row.to_dict())

    # Find weakly-connected components
    components = list(nx.weakly_connected_components(G))
    print(f"Found {len(components)} weakly-connected components")

    # Extract a subgraph for each components
    subgraphs = [G.subgraph(component) for component in components]

    for graph in subgraphs:
        # Get the links that show up in this graph
        link_ids = [link_id for _, _, link_id in graph.edges(data="link_id")]
        links_in_graph = links_df.loc[(links_df["link_id"].isin(link_ids)), :].copy()

        # Extract the starting node and ending node for this graph
        start_node_id, end_node_id = find_boundary_nodes(graph, links_in_graph)

        # Find the relative orientation of the two nodes
        start_node_lat = nodes_df.set_index("node_id").loc[start_node_id, "y_coord"]
        end_node_lat = nodes_df.set_index("node_id").loc[end_node_id, "y_coord"]

        # Southbound
        if start_node_lat > end_node_lat:  # type: ignore
            links_df.loc[(links_df["link_id"].isin(link_ids)), "direction"] = "S"
        # Northbound
        elif start_node_lat < end_node_lat:  # type: ignore
            links_df.loc[(links_df["link_id"].isin(link_ids)), "direction"] = "N"
        else:
            logger.error(f"Comparison of boundary node latitudes raised an error")
            raise RuntimeError

    return links_df


def filter_network_by_way_id(
    gmns_network_directory: str, way_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Find the filepath to the links and nodes
    link_filepath = find_file_os_walk(
        directory=gmns_network_directory, filename="link.csv"
    )
    if link_filepath is None:
        logger.error(f"Could not locate link.csv in {gmns_network_directory}")
        raise FileNotFoundError
    node_filepath = find_file_os_walk(
        directory=gmns_network_directory, filename="node.csv"
    )
    if node_filepath is None:
        logger.error(f"Could not locate node.csv in {gmns_network_directory}")
        raise FileNotFoundError

    # Load the CSVs into DFs
    links_df = pd.read_csv(link_filepath)
    nodes_df = pd.read_csv(node_filepath)

    # Filter links that match the way IDs
    links_associated_with_way_IDs = links_df.loc[
        links_df["osm_way_id"].isin(way_ids), :
    ].reset_index(drop=True)

    # Extract associated node IDs
    associated_node_ids = pd.unique(
        links_associated_with_way_IDs[["from_node_id", "to_node_id"]].values.ravel()
    )
    nodes_associated_with_way_IDs = nodes_df.loc[
        nodes_df["node_id"].isin(associated_node_ids), :
    ].reset_index(drop=True)

    return links_associated_with_way_IDs, nodes_associated_with_way_IDs


def generate_net(
    relation_id: int,
    output_dir: str = os.path.join(".", "gmns_files"),
    osm_filename: str = "map.osm",
    link_types: list[str] = ["motorway"],
    mode_types: list[str] = ["auto"],
    ref: str = "I 880",
    id_whitelist: list[str] = [],
    id_blacklist: list[str] = [],
    link_type_assignment_overwrite: dict[int, str] = {},
) -> tuple[pd.DataFrame, pd.DataFrame]:
    osm_filepath = download_osm_file_by_relation_id(
        relation_id=relation_id, output_dir=output_dir, osm_filename=osm_filename
    )

    # Download the GMNS files
    download_gmns_network_from_osm_file(
        osm_filepath=osm_filepath,
        link_types=link_types,
        mode_types=mode_types,
        output_dir=output_dir,
    )

    # Get a list of way IDs to use for filtering and modifying the network
    way_ids, mainline_way_ids, ramp_way_ids = find_way_ids_by_ref(
        gmns_network_directory=output_dir,
        osm_filename=osm_filename,
        ref=ref,
        id_whitelist=id_whitelist,
        id_blacklist=id_blacklist,
    )

    # Filter the network based on the way IDs
    links_df, nodes_df = filter_network_by_way_id(
        gmns_network_directory=output_dir, way_ids=way_ids
    )

    # Assign custom attributes to the links
    links_df = assign_link_type(
        links_df=links_df,
        mainline_way_ids=mainline_way_ids,
        overwrite=link_type_assignment_overwrite,
    )
    links_df = assign_direction(links_df=links_df, nodes_df=nodes_df)

    return links_df, nodes_df
