"""
Script for one-off plotting of PEMS VDS uptime
"""

# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


root_dir = "/Volumes/easystore/work/boulder/Caltrans/PeMS"

detectors = [
    407219,
    402789,
    420001,
    420002,
    408755,
    418449,
    402802,
    425000,
    408756,
    402947,
    400189,
    402948,
    402949,
    400309,
    400417,
    400249,
    402952,
    402951,
    401639,
    402943,
    402956,
    400662,
    402957,
    400141,
    402960,
    402962,
    400761,
    402963,
    402965,
    400490,
    401888,
    402967,
    400137,
    402968,
    402964,
    400716,
    401545,
    402972,
    401011,
    402973,
    402976,
    400674,
    400539,
    402980,
    402981,
    400534,
    401062,
    401529,
    401613,
    402985,
    400536,
    402986,
    402989,
    400488,
]  # All NB detectors
# detectors = [
#     407219,
#     402789,
#     408755,
#     402802,
#     408756,
#     400189,
#     400309,
#     400417,
#     400249,
#     401639,
#     400662,
#     400141,
#     400761,
#     400490,
#     401888,
#     400137,
#     400716,
#     401545,
#     401011,
#     400674,
#     400539,
#     400534,
#     401062,
#     401529,
#     401613,
#     400536,
#     400488,
# ]  # Mainline NB detectors
dfs = []

# Load dataframes
for detector in detectors:
    filepath = os.path.join(root_dir, "timeseries_data", f"{detector}.csv")
    df = pd.read_csv(filepath, usecols=["timestamp", "station", "pct_observed"])
    dfs.append(df)

metadata = pd.read_csv(os.path.join(root_dir, "metadata", "station_metadata.csv"))

# combine all dfs into one long dataframe
long_df = pd.concat(dfs)

# Reformatting
long_df.rename(columns={"station": "Station ID"}, inplace=True)
long_df["timestamp"] = pd.to_datetime(long_df["timestamp"])

long_df

# %%
# pivot so we get stations on rows and datetime on columns
pivot_df = long_df.pivot(index="Station ID", columns="timestamp", values="pct_observed")

# Temporarily reset the index to put Station ID as a column
pivot_df.reset_index()

# Merge with metadata and sort by postmile
pivot_df = (
    pivot_df.merge(
        right=metadata.loc[:, ["Station ID", "Abs PM"]], how="left", on="Station ID"
    )
    .sort_values(by="Abs PM")
    .drop(columns=["Abs PM"])
    .set_index("Station ID")
)

# all datetime values on the x-axis
dates = pivot_df.columns

# pick every 6 months
three_month_starts = pd.date_range(dates.min(), dates.max(), freq="3MS")

# find nearest column positions for those dates
tick_positions = dates.get_indexer(three_month_starts, method="nearest")

# %%
# make the heatmap
plt.rcParams.update(
    {
        "font.size": 24,  # Default text size (axes labels, titles, etc.)
        "axes.titlesize": 42,  # Title size
        "axes.labelsize": 30,  # X/Y label size
        "xtick.labelsize": 24,  # X tick labels
        "ytick.labelsize": 24,  # Y tick labels
        "legend.fontsize": 24,  # Legend
    }
)

plt.figure(figsize=(48, 20))
ax = sns.heatmap(
    pivot_df, cmap="viridis", cbar_kws={"label": "Percent Observed"}, linewidths=0
)

# set ticks + labels
ax.set_xticks(tick_positions)
ax.set_xticklabels(
    [d.strftime("%b %Y") for d in three_month_starts], rotation=45, ha="right"
)

plt.xlabel("Time")
plt.ylabel("Vehicle Detector Station ID")
plt.title("Caltrans PeMS Vehicle Detector Station Health\n(I880-N Segment)")
plt.tight_layout()
plt.savefig("I880N_Uptime.png", dpi=600)
plt.show()

# %%
