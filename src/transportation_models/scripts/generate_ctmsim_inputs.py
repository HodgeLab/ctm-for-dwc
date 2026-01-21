"""
Script to generate all the necessary inputs for running ctmsim, the Matlab implementation of the
Cell Transmission Model.
NOTE: Currently a WIP
"""

# %% Import modules
import sys
import logging
import pandas as pd
from pathlib import Path
from utils.representation import *

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
# POSTMILE_START = 6.950
# POSTMILE_END = 8.926
POSTMILE_START = 0.0
POSTMILE_END = 100000.0

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

START_DATETIME = "2023-06-04"
END_DATETIME = "2023-06-10"

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

i880_nb.visualize_roadway_graph()

# %%
# Leave out the ramp links at the boundary, since we only care about mainline links on either side
indices_to_drop = i880_nb.links.loc[i880_nb.links["link_id"].isin([3196, 200])].index
links_to_use = i880_nb.links.drop(indices_to_drop).loc[
    :,
    [
        "from_node_id",
        "to_node_id",
        "link_id",
        "link_type",
        "lanes",
        "name",
        "length_mi",
        "Station ID",
    ],
]

# Load the link flows
i880_nb_link_flows = build_link_timeseries_df(
    links_df=links_to_use,
    timeseries_dir=CALTRANS_VDS_TIMESERIES_DIRECTORY_PATH,
    start_datetime=START_DATETIME,
    end_datetime=END_DATETIME,
)

# Compute split ratios at each node
i880_nb_splits = compute_all_split_ratios(i880_nb.graph, merged_df=i880_nb_link_flows)

# %%
# Only one off-ramp in this case
df = i880_nb_splits[540]
df["Total"] = df.apply(lambda row: round((row.sum()), 2), axis=1)
ax = df.rename(
    columns={
        "beta": "$\\beta$",
        "1-beta": "$1-\\beta$",
    }
).plot(title=f"Split Ratio for Thornton Ave Off-ramp (NB)")

# %%
