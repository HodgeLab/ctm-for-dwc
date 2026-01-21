"""
Script for building a fully-defined, CTM-ready, graph-based representation of a freeway section
"""

# %% Import modules
import sys
import logging
from pathlib import Path
from transportation_models.utils.representation import Freeway

# Set up logging to stdout
root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

# Define constants
GMNS_LINKS_FILEPATH = Path("/Volumes/easystore/work/boulder/GMNS/links_modified.csv")
GMNS_NODES_FILEPATH = Path("/Volumes/easystore/work/boulder/GMNS/nodes_modified.csv")
CALTRANS_VDS_METADATA_FILEPATH = Path(
    "/Volumes/easystore/work/boulder/Caltrans/PeMS/metadata/station_metadata_CALIBRATED.csv"
)
CALTRANS_VDS_TIMESERIES_DIRECTORY_PATH = Path(
    "/Volumes/easystore/work/boulder/Caltrans/PeMS/timeseries_data"
)
POSTMILE_FILEPATH = Path(
    "/Volumes/easystore/work/boulder/Caltrans/State_Highway_Network_Postmiles_Tenth/State_Highway_Network_Postmiles_Tenth.shp"
)
BOUNDING_FILEPATH = Path(
    "/Volumes/easystore/work/boulder/SMART-DS/v1.0/GIS/SFO/P31U/DistribTransf_N.shp"
)
POSTMILE_START = 6.7
POSTMILE_END = 13.7

ADDITIONAL_OFFRAMP_NODE_IDS = [2234]  # OSM Node ID: 258131276
ADDITIONAL_ONRAMP_NODE_IDS = [2498]  # OSM Node ID: 568983599
MAINLINE_NODE_IDS_TO_EXCLUDE = [
    2235,  # OSM Node ID: 258132827
    2251,  # OSM Node ID: 370488795
]
NB_LINK_TO_STATION_OVERRIDE = {
    1973: 420001,  # Warren Ave Off
    3203: 402802,  # N of Warren Ave (mainline detector with 5th lane as on-ramp from Warren Ave)
    2421: 402985,  # Alvarado-Niles Blvd On
    3281: 402957,  # Stevenson Blvd on loop
    200: 402962,  # Mowry Ave off
    3191: 402963,  # Mowry Ave on loop
    1381: 402989,  # Whipple Rd off -> On-ramp from Whipple Rd
    3185: 402943,  # Auto Mall Pkwy Diagonal On-ramp
    153: 402951,  # Auto Mall Pkwy Off-ramp
    199: 402947,  # Fremont Blvd Off-ramp
    2471: 425000,  # Mission Blvd On-ramp
    3203: 418449,  # Warren Ave On-ramp
    1632: 420002,  # Mission Blvd Off-ramp
    "collapsed_1249_2234": 407219,  # S of Mission Blvd
    "collapsed_2234_2250": 402789,  # oppo Mission Blvd rm-s-fly
    "collapsed_2248_2249": 408756,  # Mission Blvd rm-n-diag
    "collapsed_467_290": 400417,  # oppo at the truck scales (GMNS counts 5 lanes, but 1 is HOV and unmeasured)
    "collapsed_97_2436": 400536,  # Alvarado-Niles Rd rm-n-diag (5th lane is the on-ramp from Alvarado-Niles Road)
}

SB_LINK_TO_STATION_OVERRIDE = {
    1934: 402983,  # Off-ramp to Alvarado-Niles Rd
    230: 402984,  # Loop on-ramp from Alvarado-Niles Rd
    1826: 402982,  # Diagonal on-ramp from Alvarado-Niles Rd
    53: 402978,  # Off-ramp to Alvarado Blvd
    2347: 402977,  # Diagonal On-ramp from Alvarado Blvd
    2355: 402974,  # Off-ramp to Decoto Rd
    3174: 402942,  # Loop on-ramp from Mowry Ave
    3142: 402961,  # Diagonal on-ramp from Mowry Ave
    1965: 402945,  # Off-ramp to Fremont Blvd
    1974: 425003,  # Off-ramp to Mission Blvd
    "collapsed_2805_2496": 430136,  # 0.4 mi NO Mission Blvd (moving prior to off-ramp to Mission)
    "collapsed_2498_587": 407221,  # oppo S of Mission Blvd (5th and 6th lanes are on-ramps from Mission)
    "collapsed_2496_2497": 408757,  # Warren Ave rm-s-loop (alternative is 402803 which has a phantom 5th lane)
}
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
