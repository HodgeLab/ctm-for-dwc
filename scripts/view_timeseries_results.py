"""
Script for one-off plotting of scikit-learn regressor prediction performance
"""

# %%
#####     Setup     #####
# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_percentage_error

# Variable definitions
results_dir = "~/Downloads/models"
root_dir = "./example_data"

kn_results = pd.read_csv(
    os.path.join(results_dir, "KNeighborsRegressor_basic_results.csv")
)
rf_results = pd.read_csv(
    os.path.join(results_dir, "RandomForestRegressor_basic_results.csv")
)
timestamps = pd.read_csv(
    os.path.join(root_dir, "formatted_data", "data_wide.csv"), usecols=["timestamp"]
)
timestamps["timestamp"] = pd.to_datetime(timestamps["timestamp"])
lookup = pd.read_csv(os.path.join(root_dir, "station_metadata_CALIBRATED.csv"))
results = pd.merge(
    left=kn_results,
    right=rf_results,
    left_on=kn_results.index,
    right_on=rf_results.index,
    suffixes=["_KN", "_RF"],
).drop("key_0", axis=1)
results["timestamp"] = timestamps.loc[
    timestamps["timestamp"].apply(lambda ts: ts.year >= 2024), :
].reset_index(drop=True)

results


# %%
# Convert back to long-format
long = results.melt(id_vars="timestamp")

# Extract column information from the variable column
long["Station ID"] = long["variable"].apply(
    lambda key: int(key.split("normalized_flow_[vphpl]_")[-1].split("_")[0])
)


def feature_classifier(row):
    var = row["variable"]
    if "hat_RF" in var:
        return "y_hat_RF"
    elif "hat_KN" in var:
        return "y_hat_KN"
    else:
        return "y"


long["feature"] = long.apply(
    lambda row: feature_classifier(row),
    axis=1,
)

# Remove the unused column
long.drop("variable", axis=1, inplace=True)

# Pivot so each feature becomes its own column again
long = long.pivot_table(
    index=["timestamp", "Station ID"], columns="feature", values="value"
).reset_index()

# Replace imputed values of -1 with NaNs
long.loc[long["y"] == -1, "y"] = np.nan

# Convert the timestamp to a datetime object
long["timestamp"] = pd.to_datetime(long["timestamp"])

long

# %%
# Get a manageable subset
date_range = pd.date_range(
    start="2024-03-09 00:00:00", end="2024-03-12 23:55:00", freq="5min"
)
station_to_plot = 407219
df = long.loc[
    ((long["timestamp"].isin(date_range)) & (long["Station ID"] == station_to_plot)),
    :,
].copy()

# Save a copy without NaNs or zeros for error calcs
df_whole = df.copy()
df_whole.loc[df_whole["y"] == 0.0, "y"] = np.nan
df_whole.dropna(inplace=True)

# Calculate the MAE
mape_kn = mean_absolute_percentage_error(df_whole["y"], df_whole["y_hat_KN"]) * 100
mape_rf = mean_absolute_percentage_error(df_whole["y"], df_whole["y_hat_RF"]) * 100

# Add columns for error
df["ape_KN"] = abs(df_whole["y"] - df["y_hat_KN"]) / df_whole["y"] * 100
df["ape_RF"] = abs(df_whole["y"] - df["y_hat_RF"]) / df_whole["y"] * 100

df
# %%

# Make a plot
plt.rcParams.update(
    {
        "font.size": 24,  # Default text size (axes labels, titles, etc.)
        "axes.titlesize": 24,  # Title size
        "axes.labelsize": 18,  # X/Y label size
        "xtick.labelsize": 14,  # X tick labels
        "ytick.labelsize": 14,  # Y tick labels
        "legend.fontsize": 14,  # Legend
    }
)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(11, 11), sharex=True)

ax1.plot(
    df["timestamp"],
    df["y"],
    label=f"ground truth",
    color="black",
    linestyle="-",
    alpha=0.7,
)
ax1.plot(
    df["timestamp"],
    df["y_hat_KN"],
    label=f"KNN",
    color="tab:blue",
    linestyle="-",
)
ax1.plot(
    df["timestamp"],
    df["y_hat_RF"],
    label=f"RF",
    color="tab:orange",
    linestyle="-",
)

ax1.set_ylabel(r"Volume $[\frac{vehicle}{hour \cdot lane}]$")
ax1.set_title(f"Traffic Volume Predictions\nPeMS Station {station_to_plot}")
ax1.grid()

ax2.plot(df["timestamp"], df["ape_KN"], label="KNN", color="tab:blue", linestyle="-")
ax2.plot(
    df["timestamp"],
    df["ape_RF"],
    label="RF",
    color="tab:orange",
    linestyle="-",
)
ax2.axhline(mape_kn, label="MAPE (KN)", color="tab:blue", linestyle="--")
ax2.axhline(mape_rf, label="MAPE (RF)", color="tab:orange", linestyle="--")


# Set formatter for datetime x-axis
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y %H:%M"))

ax2.set_ylabel("Absolute Percent Error [%]")
ax2.set_xlabel("Timestamp")
ax2.grid()

# Collect handles and labels from both subplots
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Merge them
handles = handles1 + handles2
labels = labels1 + labels2

# To avoid duplicates, use dict (preserves first occurrence)
by_label = dict(zip(labels, handles))

# Adjust spacing between subplots
fig.subplots_adjust(hspace=0.25)  # increase vertical spacing

# Place the legend in the gap between ax1 and ax2
fig.legend(
    by_label.values(),
    by_label.keys(),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.53),  # tune this for exact placement
    ncol=3,
    frameon=True,
    fontsize=14,
)

# Rotate x-tick labels
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

# plt.savefig("./test.png", dpi=300)
plt.show()


# %%
def find_error_summary(df: pd.DataFrame, results_dict) -> dict:
    # Replace zero values with NaNs
    df.loc[df["y"] == 0.0, "y"] = np.nan

    # Drop missing values for error calcs
    df.dropna(subset="y", inplace=True)

    # Extract the Station ID
    station_id = df["Station ID"].iloc[0]

    # Calculate MAPE
    mape_KN = mean_absolute_percentage_error(df["y"], df["y_hat_KN"]) * 100
    mape_RF = mean_absolute_percentage_error(df["y"], df["y_hat_RF"]) * 100

    # Calculate APE
    df["ape_KN"] = abs(df["y"] - df["y_hat_KN"]) / df["y"] * 100
    df["ape_RF"] = abs(df["y"] - df["y_hat_RF"]) / df["y"] * 100

    # Find the peak error and associated timestamp for each
    peak_ape_ts_KN, peak_ape_KN = df.loc[df["ape_KN"].idxmax(), ["timestamp", "ape_KN"]].to_list()  # type: ignore
    peak_ape_ts_RF, peak_ape_RF = df.loc[df["ape_RF"].idxmax(), ["timestamp", "ape_RF"]].to_list()  # type: ignore

    # Append results
    results_dict["Station ID"].append(station_id)
    results_dict["mape_KN"].append(mape_KN)
    results_dict["mape_RF"].append(mape_RF)
    results_dict["peak_ape_ts_KN"].append(peak_ape_ts_KN)
    results_dict["peak_ape_ts_RF"].append(peak_ape_ts_RF)
    results_dict["peak_ape_KN"].append(peak_ape_KN)
    results_dict["peak_ape_RF"].append(peak_ape_RF)

    return results_dict


error_summary = {
    "Station ID": [],
    "mape_KN": [],
    "mape_RF": [],
    "peak_ape_ts_KN": [],
    "peak_ape_ts_RF": [],
    "peak_ape_KN": [],
    "peak_ape_RF": [],
}

stations_to_calc = long["Station ID"].unique()

for station in stations_to_calc:
    df = long.loc[long["Station ID"] == station, :].copy()
    error_summary = find_error_summary(df, error_summary)

error_df = pd.DataFrame.from_dict(data=error_summary)
error_df
# %%

#### Create a plot of the error metrics

# Melt into long format
df_plot = error_df.melt(
    id_vars="Station ID", var_name="MetricCategory", value_name="Value"
)

# Split into Metric + Category
df_plot[["Metric", "Category"]] = df_plot["MetricCategory"].str.rsplit(
    "_", n=1, expand=True
)

# Sort by Abs PM
df_plot = df_plot.merge(lookup, on="Station ID", how="left").sort_values(by="Abs PM")

# Cast stations as strings to improve visibility
df_plot["Station ID"] = df_plot["Station ID"].astype(str)

# Define the metrics
metrics = df_plot["Metric"].unique()
n_metrics = len(metrics)

metric_lookup = {
    "mape": "MAPE [%]",
    "mape": "MAPE [%]",
    "peak_ape": "Peak Absolute Percent Error [%]",
    "peak_ape": "Peak Absolute Percent Error [%]",
    "peak_ape_ts": "Peak APE Timestamp",
    "peak_ape_ts": "Peak APE Timestamp",
}

df_plot
# %%

fig, axes = plt.subplots(nrows=n_metrics, ncols=1, figsize=(12, 12))
axes = axes.flatten()  # Make it a 1d array for easy looping

for ax, metric in zip(axes, metrics):
    subset = df_plot[df_plot["Metric"] == metric]

    # Pivot so that we can group bars
    pivoted = subset.pivot(index="Station ID", columns="Category", values="Value")

    # Calculate mean values for each type
    mean_val_KN = pivoted["KN"].mean()
    mean_val_RF = pivoted["RF"].mean()

    # Plot grouped bars
    x = range(len(pivoted))
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        pivoted["KN"],
        width=width,
        label="KN",
        color="tab:blue",
    )
    ax.bar(
        [i + width / 2 for i in x],
        pivoted["RF"],
        width=width,
        label="RF",
        color="tab:orange",
    )

    ax.axhline(
        mean_val_KN, linestyle="--", color="tab:blue", alpha=0.7, label="KN (mean)"
    )
    ax.axhline(
        mean_val_RF, linestyle="--", color="tab:orange", alpha=0.7, label="RF (mean)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(pivoted.index.astype(str), rotation=45)
    ax.set_xlabel("Station ID")
    ax.set_ylabel(metric_lookup[metric])

    handles, labels = ax.get_legend_handles_labels()

    if "ts" in metric:
        ax.set_ylim([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")])

plt.suptitle(
    "Error Metrics for PeMS Stations in I-880 Corridor\n(Station IDs sorted by Abs PM)"
)
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# %%
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)


def print_error_metrics(df):
    y = df.loc[:, [col for col in df.columns if "hat" not in col]]
    y_hat = df.loc[:, [col for col in df.columns if "hat" in col]]

    rmse = root_mean_squared_error(y, y_hat)
    mae = mean_absolute_error(y, y_hat)
    mape = mean_absolute_percentage_error(y, y_hat)
    r2 = r2_score(y, y_hat)

    print(f"Error Summary:")
    print(f"RMSE = {rmse:.3f}")
    print(f"MAE = {mae:.3f}")
    print(f"MAPE = {mape:.3f}")
    print(f"R2 = {r2:.3f}")


print(f"KNeighbors")
print_error_metrics(df=kn_results)
print(f"Random Forest")
print_error_metrics(df=rf_results)

# %%
