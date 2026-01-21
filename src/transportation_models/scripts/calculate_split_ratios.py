"""
Script for calculating the split ratios for freeway off-ramps
"""

# %% Import modules
import sys
import logging
from pathlib import Path
from transportation_models.utils.validation import *
from transportation_models.utils.representation import *

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
POSTMILE_START = 6.950
POSTMILE_END = 8.926

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

# Node ID -> Name Mapping
node_id_to_exit = {
    540: "Thornton Ave (NB)",
    290: "Auto Mall Pkwy (NB)",
    2234: "Mission Blvd (NB)",
    2250: "Warren Ave (NB)",
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
    # postmile_start=POSTMILE_START,
    # postmile_end=POSTMILE_END,
    additional_offramp_node_ids=ADDITIONAL_OFFRAMP_NODE_IDS,
    additional_onramp_node_ids=ADDITIONAL_ONRAMP_NODE_IDS,
    mainline_node_ids_to_exclude=MAINLINE_NODE_IDS_TO_EXCLUDE,
    link_to_station_override=NB_LINK_TO_STATION_OVERRIDE,
)

# %%
i880_nb.visualize_roadway_graph()
i880_nb.links.to_csv(
    "/Volumes/easystore/work/boulder/GMNS/WIP/links_with_detectors.csv", index=False
)
i880_nb.nodes.to_csv("/Volumes/easystore/work/boulder/GMNS/WIP/nodes.csv", index=False)
i880_nb.detectors.to_csv(
    "/Volumes/easystore/work/boulder/Caltrans/PeMS/metadata/WIP/detectors_reprojected.csv",
    index=False,
)
# %%
# Leave out the ramp links at the boundary, since we only care about mainline links on either side
# indices_to_drop = i880_nb.links.loc[i880_nb.links["link_id"].isin([3196, 200])].index
links_to_use = i880_nb.links.loc[
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
    # start_datetime=START_DATETIME,
    # end_datetime=END_DATETIME,
)

# %%
# Compute split ratios at each node
i880_nb_splits = compute_all_split_ratios(i880_nb.graph, merged_df=i880_nb_link_flows)

# %%
for node_id, df in i880_nb_splits.items():
    # # Create a plot showing the non-conservation of flow
    # view_flow_conservation_timeseries(
    #     splits_df=df,
    #     node_id=node_id,
    #     node_id_mapper=node_id_to_exit,
    #     save_path=f"./timeseries_{node_id}.png",
    # )

    # # Create CDF plots
    # view_split_ratio_cdfs(
    #     splits_df=df,
    #     node_id=node_id,
    #     numbins=1000,
    #     col_to_plot="seconds",
    #     node_id_mapper=node_id_to_exit,
    #     save_path=f"./cdf_time_{node_id}.png",
    # )
    # view_split_ratio_cdfs(
    #     splits_df=df,
    #     node_id=node_id,
    #     numbins=50000,
    #     col_to_plot="Percent Error",
    #     node_id_mapper=node_id_to_exit,
    #     save_path=f"./cdf_error_{node_id}.png",
    # )

    # # Create box plot
    # view_split_ratio_box_plot(
    #     splits_df=df,
    #     node_id=node_id,
    #     node_id_mapper=node_id_to_exit,
    #     save_path=f"./box_{node_id}.png",
    # )

    # # Create autocorrelation plot
    # view_split_ratio_autocorrelation_plot(
    #     splits_df=df,
    #     node_id=node_id,
    #     node_id_mapper=node_id_to_exit,
    #     lag_cadence="daily",
    #     save_path=f"./acorr_{node_id}.png",
    # )

    # # Calculate the Pearson Correlation Coefficient for the split ratio and the error
    # _, _ = get_pearson_correlation(
    #     splits_df=df, node_id=node_id, node_id_mapper=node_id_to_exit
    # )

    # Create error plot
    view_split_ratio_errors(
        splits_df=df,
        node_id=node_id,
        node_id_mapper=node_id_to_exit,
        save_path=f"./error_{node_id}.png",
    )

# %%
node_id = 2234
# df = i880_nb_splits[node_id]
df = pd.read_csv("practice_data/node_2234_splits_df.csv", index_col="timestamp")

# Create a plot showing the non-conservation of flow
view_flow_conservation_timeseries(
    splits_df=df,
    node_id=node_id,
    node_id_mapper=node_id_to_exit,
    save_path=None,
)

# Create CDF plots
view_split_ratio_cdfs(
    splits_df=df,
    node_id=node_id,
    numbins=1000,
    col_to_plot="seconds",
    node_id_mapper=node_id_to_exit,
    save_path=None,
)
view_split_ratio_cdfs(
    splits_df=df,
    node_id=node_id,
    numbins=50000,
    col_to_plot="Percent Error",
    node_id_mapper=node_id_to_exit,
    save_path=None,
)
# Create box plot
view_split_ratio_box_plot(
    splits_df=df,
    node_id=node_id,
    node_id_mapper=node_id_to_exit,
    save_path=None,
)
view_split_ratio_autocorrelation_plot(
    splits_df=df,
    node_id=node_id,
    node_id_mapper=node_id_to_exit,
    lag_cadence="weekly",
    save_path=None,
)

# Create error plot
view_split_ratio_errors(
    splits_df=df, node_id=node_id, node_id_mapper=node_id_to_exit, save_path=None
)

# %% # Resample the merged flow data
# List to fill with resampled dataframes
resampled_dataframes = []

for link_id in i880_nb_link_flows["link_id"].unique():
    # Prepare the DF for this one link
    df = (
        i880_nb_link_flows.loc[i880_nb_link_flows["link_id"] == link_id, :]
        .copy()
        .drop(columns="avg_speed_[mph]")
    )

    # Resample
    logging.info(f"Resampling timeseries for Link {link_id}")
    df_15min = resample_timeseries(df=df)

    # Check for empty dataframes
    dropped_df = df_15min.dropna(axis=0, how="any", subset=["total_flow_[veh/15-min]"])

    if dropped_df.empty:
        logging.info(f"Resampled data for Link {link_id} is empty – skipping...")
    else:
        resampled_dataframes.append(df_15min)

logging.info(f"Concatenating {len(resampled_dataframes)} dataframes")
i880_nb_link_flows_15min = pd.concat(resampled_dataframes, ignore_index=False)

# Compute split ratios with 15-minute data
splits_15min = compute_all_split_ratios(
    i880_nb.graph,
    merged_df=i880_nb_link_flows_15min,
    flow_col_name="total_flow_[veh/15-min]",
)

for k, v in splits_15min.items():
    filename = f"./practice_data/node_{k}_15min_splits_df.csv"
    v.to_csv(filename, index=True)

# %%
node_id = 2234
df = pd.read_csv(
    f"practice_data/node_{node_id}_15min_splits_df.csv", index_col="timestamp"
)

# Create a plot showing the non-conservation of flow
view_flow_conservation_timeseries(
    splits_df=df,
    node_id=node_id,
    node_id_mapper=node_id_to_exit,
    save_path=None,
)

# Create CDF plots
view_split_ratio_cdfs(
    splits_df=df,
    node_id=node_id,
    numbins=1000,
    col_to_plot="seconds",
    node_id_mapper=node_id_to_exit,
    save_path=None,
)
view_split_ratio_cdfs(
    splits_df=df,
    node_id=node_id,
    numbins=50000,
    col_to_plot="Percent Error",
    node_id_mapper=node_id_to_exit,
    save_path=None,
)
# Create box plot
view_split_ratio_box_plot(
    splits_df=df,
    node_id=node_id,
    node_id_mapper=node_id_to_exit,
    save_path=None,
)
view_split_ratio_autocorrelation_plot(
    splits_df=df,
    node_id=node_id,
    node_id_mapper=node_id_to_exit,
    lag_cadence="daily",
    save_path=None,
)
view_split_ratio_errors(
    splits_df=df,
    node_id=node_id,
    node_id_mapper=node_id_to_exit,
    save_path=None,
)

# %%
station_distances = (
    i880_nb.detectors.loc[:, ["Station ID", "Abs PM"]]
    .set_index("Station ID")
    .to_dict()["Abs PM"]
)
wide_df = (
    i880_nb_link_flows.reset_index()
    .loc[:, ["timestamp", "pct_observed", "Station ID"]]
    .pivot(index="Station ID", columns="timestamp", values="pct_observed")
)
wide_df = wide_df.dropna(how="all")
sorted_stations = sorted(wide_df.index, key=lambda sid: station_distances[sid])
pivot_df = wide_df.loc[sorted_stations]
pivot_df.columns = pd.to_datetime(pivot_df.columns)


# %%
# scale up all fonts ~2x
plt.rcParams.update(
    {
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 24,
    }
)


plt.figure(figsize=(18, 9))  # also scale up figure size
ax = sns.heatmap(
    pivot_df,
    cmap="viridis",
    cbar_kws={"label": "Percent Observed"},
    linewidths=0,  # no bars
    square=False,  # allow rectangular cells
)

# all datetime values on the x-axis
dates = pivot_df.columns

# pick every 3 months
three_month_starts = pd.date_range(dates.min(), dates.max(), freq="3MS")

# find nearest column positions for those month starts
tick_positions = dates.get_indexer(three_month_starts, method="nearest")

# set ticks + labels
ax.set_xticks(tick_positions)
ax.set_xticklabels(
    [d.strftime("%b %Y") for d in three_month_starts], rotation=45, ha="right"
)


plt.xlabel("Datetime")
plt.ylabel("Vehicle Detector Station ID")
plt.title("Caltrans PeMS Vehicle Detector Station Health\n(I880-N Segment)")
plt.tight_layout()
plt.show()

# %%
