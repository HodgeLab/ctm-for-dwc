"""
Script for extracting the GMNS links and nodes from OSM data
"""

# %%
# Imports
import os
from utils.representation import *

# Define constants
ALAMEDA_COUNTY_RELATION_ID = 396499
OUTPUT_DIR = "/Volumes/easystore/work/boulder/GMNS"
OSM_FILENAME = "alameda_county.osm"
ID_WHITELIST = ["156651860", "1356869915"]
ID_BLACKLIST = ["861098866"]
LINK_TYPE_ASSIGNMENTS = {1632: "Off-ramp", 3438: "On-ramp"}

# %% Download the OSM file
if Path(os.path.join(OUTPUT_DIR, OSM_FILENAME)).is_file():
    print(f"OSM file already exists")
else:
    osm_filepath = download_osm_file_by_relation_id(
        relation_id=ALAMEDA_COUNTY_RELATION_ID,
        output_dir=OUTPUT_DIR,
        osm_filename=OSM_FILENAME,
    )

# %% Generate the links and nodes
links_filename = os.path.join(OUTPUT_DIR, "links_modified.csv")
nodes_filename = os.path.join(OUTPUT_DIR, "nodes_modified.csv")

if Path(links_filename).is_file() and Path(nodes_filename).is_file():
    print(f"Link and node CSVs already exist")
    links_df = pd.read_csv(links_filename)
    nodes_df = pd.read_csv(nodes_filename)
else:
    links_df, nodes_df = generate_net(
        relation_id=ALAMEDA_COUNTY_RELATION_ID,
        output_dir=OUTPUT_DIR,
        osm_filename=OSM_FILENAME,
        id_whitelist=ID_WHITELIST,
        id_blacklist=ID_BLACKLIST,
        link_type_assignment_overwrite=LINK_TYPE_ASSIGNMENTS,
    )
    links_df.to_csv(os.path.join(OUTPUT_DIR, "links_modified.csv"), index=False)
    nodes_df.to_csv(os.path.join(OUTPUT_DIR, "nodes_modified.csv"), index=False)

# %%
