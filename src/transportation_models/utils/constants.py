from pathlib import Path

# Define constants
ROOT_PATH = Path("/Volumes/easystore/work/boulder/Caltrans/PeMS")
GMNS_LINKS_FILEPATH = Path("/Volumes/easystore/work/boulder/GMNS/links_modified.csv")
GMNS_NODES_FILEPATH = Path("/Volumes/easystore/work/boulder/GMNS/nodes_modified.csv")
CALTRANS_VDS_METADATA_FILEPATH = Path(
    "/Volumes/easystore/work/boulder/Caltrans/PeMS/metadata/station_metadata_calibrated.csv"
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
