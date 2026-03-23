"""
Script for building a fully-defined, CTM-ready, graph-based representation of a freeway section
"""

# %% Import modules
import sys
import logging
from transportation_models.utils.constants import *
from transportation_models.utils.representation import Freeway

# Set up logging to stdout
root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

# %%
# Construct graphical representations for each direction of the freeway
i880_nb = Freeway(
    links_filepath=GMNS_LINKS_FILEPATH,
    nodes_filepath=GMNS_NODES_FILEPATH,
    detector_metadata_filepath=CALTRANS_VDS_METADATA_FILEPATH,
    detector_timeseries_data_directory=CALTRANS_VDS_TIMESERIES_DIRECTORY_PATH,
    postmile_filepath=POSTMILE_FILEPATH,
    bounding_filepath=BOUNDING_FILEPATH,
    direction="N",
    vds_crs="EPSG:32610",
    postmile_crs="EPSG:32610",
    boundary_crs="EPSG:32610",
    common_crs="EPSG:32610",
    postmile_start=POSTMILE_START,
    postmile_end=POSTMILE_END,
    additional_offramp_node_ids=ADDITIONAL_OFFRAMP_NODE_IDS,
    additional_onramp_node_ids=ADDITIONAL_ONRAMP_NODE_IDS,
    mainline_node_ids_to_exclude=MAINLINE_NODE_IDS_TO_EXCLUDE,
    link_to_station_override=NB_LINK_TO_STATION_OVERRIDE,
)
i880_sb = Freeway(
    links_filepath=GMNS_LINKS_FILEPATH,
    nodes_filepath=GMNS_NODES_FILEPATH,
    detector_metadata_filepath=CALTRANS_VDS_METADATA_FILEPATH,
    detector_timeseries_data_directory=CALTRANS_VDS_TIMESERIES_DIRECTORY_PATH,
    postmile_filepath=POSTMILE_FILEPATH,
    bounding_filepath=BOUNDING_FILEPATH,
    direction="S",
    vds_crs="EPSG:32610",
    postmile_crs="EPSG:32610",
    boundary_crs="EPSG:32610",
    common_crs="EPSG:32610",
    postmile_start=POSTMILE_START,
    postmile_end=POSTMILE_END,
    additional_offramp_node_ids=ADDITIONAL_OFFRAMP_NODE_IDS,
    additional_onramp_node_ids=ADDITIONAL_ONRAMP_NODE_IDS,
    mainline_node_ids_to_exclude=MAINLINE_NODE_IDS_TO_EXCLUDE,
    link_to_station_override=SB_LINK_TO_STATION_OVERRIDE,
)

# %%
# One option is to use the built-in method for visualizing the roadway
i880_nb.visualize_roadway_graph()

# Another option is to save the roadway data to CSVs for visualizing externally (e.g., with QGIS)
i880_nb.links.to_csv(
    "/Volumes/easystore/work/boulder/GMNS/WIP/links_with_detectors.csv", index=False
)
i880_nb.nodes.to_csv("/Volumes/easystore/work/boulder/GMNS/WIP/nodes.csv", index=False)
i880_nb.detectors.to_csv(
    "/Volumes/easystore/work/boulder/Caltrans/PeMS/metadata/WIP/detectors_reprojected.csv",
    index=False,
)
# %%
i880_sb.visualize_roadway_graph(ramp_offset=0.2)
i880_sb.links.to_csv(
    "/Volumes/easystore/work/boulder/GMNS/WIP/links_with_detectors.csv", index=False
)
i880_sb.nodes.to_csv("/Volumes/easystore/work/boulder/GMNS/WIP/nodes.csv", index=False)
i880_sb.detectors.to_csv(
    "/Volumes/easystore/work/boulder/Caltrans/PeMS/metadata/WIP/detectors_reprojected.csv",
    index=False,
)

# %%
