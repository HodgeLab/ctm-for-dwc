"""
Script for one-off plotting of Fundamental Diagrams
"""

# %%
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.validation import validate_timestamp_dtype
from utils.calibration import estimate_free_flow_speed
from utils.calibration import estimate_congestion_wave_speed

detector = 400839
root_path = "/Volumes/easystore/work/boulder/Caltrans/PeMS/"
metadata_df = pd.read_csv(os.path.join(root_path, "metadata", "station_metadata.csv"))
timeseries_df = pd.read_csv(
    os.path.join(root_path, "timeseries_data", f"{detector}.csv")
)
detector_metadata = metadata_df.loc[metadata_df["Station ID"] == detector, :].iloc[0]

imputation_threshold = 100.0
imputation_flag_col_name = "pct_observed"
flow_col_name = "total_flow_[veh/5-min]"
speed_col_name = "avg_speed_[mph]"
num_lanes = None
congestion_speed_threshold = 55.0

# Logging setup
logger = logging.getLogger(__name__)

timeseries_df

# %%
# Empty dictionary to pack with values
params = {
    "capacity": np.nan,
    "free_flow_speed": np.nan,
    "congestion_wave_speed": np.nan,
    "jam_density": np.nan,
    "critical_density": np.nan,
}

# Extract detector ID
detector_id = detector_metadata["Station ID"]

if num_lanes is None:
    # Extract normalization parameters from metadata
    num_lanes = detector_metadata["Lanes"]

    # Use ones as default in case the parameters are bad
    if not num_lanes:
        logger.warning(
            f"No lane information for detector: {detector_id}. Using num_lanes=1"
        )
        num_lanes = 1

# Number of 5-minute periods per hour (used for normalization)
periods_per_hour = 12

# Ensure that the timestamp column exists and contains datetime objects
timeseries_df = validate_timestamp_dtype(timeseries_df)

# Make a copy of the timeseries DF to normalize
timeseries_normalized = timeseries_df.loc[
    :, ["timestamp", imputation_flag_col_name, flow_col_name, speed_col_name]
].copy()

# Set the index to the timestamp to make slicing easier
timeseries_normalized.set_index(keys="timestamp", inplace=True)

# Limit the data to points where the detector is not imputing any measurements
timeseries_normalized = timeseries_normalized.loc[
    timeseries_normalized[imputation_flag_col_name] >= imputation_threshold, :
]

# Normalize the flow and density: [veh/5-min] * [5-min/hr] * [1/lanes] = [veh/hr*lanes]
timeseries_normalized["normalized_flow_[veh/hr-lane]"] = (timeseries_normalized[flow_col_name] * periods_per_hour) / num_lanes  # type: ignore

# Normalize the density: [veh/5-min] * [5-min/h] * [h/mi] * [1/lanes] = [veh/mi*lanes]
timeseries_normalized["density_[veh/mi-lane]"] = (
    timeseries_normalized[flow_col_name] * periods_per_hour
) / (
    timeseries_normalized[speed_col_name] * num_lanes
)  # type: ignore

# Eliminate any NA values
not_na_mask = (~timeseries_normalized["density_[veh/mi-lane]"].isna()) & (
    ~timeseries_normalized["normalized_flow_[veh/hr-lane]"].isna()
)
timeseries_normalized = timeseries_normalized.loc[not_na_mask, :].reset_index(drop=True)

# Extract the capacity and jam density based on the peak observed flow
max_capacity = timeseries_normalized["normalized_flow_[veh/hr-lane]"].max()
params["capacity"] = max_capacity
max_capacity_timestamp = timeseries_normalized["normalized_flow_[veh/hr-lane]"].idxmax()
critical_density = timeseries_normalized.loc[
    max_capacity_timestamp, "density_[veh/mi-lane]"
]
# TODO: Figure out a better method for managing this detector's outlier
if str(detector_id) == "400534":
    logger.info("Overriding max capacity for detector 400534")
    max_capacity = 1755.0
    critical_density = 28.306451612903224
    params["capacity"] = max_capacity

# Add critical density as a parameter
params["critical_density"] = critical_density  # type: ignore

# Linear regression step
free_flow_speed, free_flow_speed_model = estimate_free_flow_speed(
    timeseries_df=timeseries_normalized,
    congestion_speed_threshold=congestion_speed_threshold,
    test_points=[0.0, critical_density],  # type: ignore
    speed_col_name=speed_col_name,
)
params["free_flow_speed"] = free_flow_speed
congestion_slope, jam_density, congestion_wave_speed_model, bin_df = (
    estimate_congestion_wave_speed(
        timeseries_df=timeseries_normalized,
        max_capacity=max_capacity,
        critical_density=critical_density,  # type: ignore
        test_points=[critical_density],  # type: ignore
    )
)
params["congestion_wave_speed"] = -1 * congestion_slope
params["jam_density"] = jam_density

# Log the parameters
station_id = detector_metadata["Station ID"]
logger.info(f"Calibration of Station {station_id} complete: {params}")


params


# %%
# Optionally plot and save the calibration plots
save_path = f"./{detector}_fundamental_diagram.png"


def plot_fundamental_diagram(
    normalized_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    free_flow_model,
    congestion_model,
    params: dict[str, float],
    detector_id: int,
    save_path,
) -> None:
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

    # Generate some prediction points using the models
    free_flow_xvals = np.linspace(
        0,
        params["critical_density"],
    ).reshape(-1, 1)
    free_flow_pred = free_flow_model.predict(free_flow_xvals)

    congestion_xvals = np.linspace(
        bin_df["BinDensity"].min(), bin_df["BinDensity"].max(), 100
    )
    congestion_slope = congestion_model.coef_[0]
    congestion_pred = (
        congestion_slope * (congestion_xvals - params["critical_density"])
        + params["capacity"]
    )

    fig, ax = plt.subplots(figsize=(10.5, 7))

    # Plot the data points
    plt.scatter(
        normalized_df["density_[veh/mi-lane]"],
        normalized_df["normalized_flow_[veh/hr-lane]"],
        label="Raw (data)",
        color="tab:blue",
        marker=".",
        s=5,
        alpha=0.3,
    )
    plt.plot(
        bin_df["BinDensity"],
        bin_df["BinFlow"],
        color="tab:purple",
        marker="o",
        markersize=5,
        linestyle="-",
        label="Congestion Bins",
    )

    # Plot the regression lines
    plt.plot(
        free_flow_xvals,
        free_flow_pred,
        color="tab:orange",
        linewidth=3,
        label="Free flow (fit)",
    )
    plt.plot(
        congestion_xvals,
        congestion_pred,
        color="tab:green",
        linewidth=3,
        label="Congestion (fit)",
    )

    # Plot a horizontal line for capacity
    plt.axhline(y=params["capacity"], color="gray", linestyle="--", label="Capacity")

    # Plot vertical lines for critical and jam densities
    plt.axvline(
        x=params["critical_density"],
        color="gray",
        linestyle="-.",
        label="Critical Density",
    )
    plt.axvline(
        x=params["jam_density"], color="gray", linestyle=":", label="Jam Density"
    )

    # Set y limits
    bottom, top = plt.ylim()
    ax.set_ylim(-25, top)

    # Add plot features
    plt.xlabel(r"Density $[\frac{vehicle}{mile \cdot lane}]$")
    plt.ylabel(r"Volume $[\frac{vehicle}{hour \cdot lane}]$")
    plt.title(f"Fundamental Diagram for Detector: {detector_id}")
    plt.legend(bbox_to_anchor=(0.65, 0.95), loc="upper left")
    plt.tight_layout()

    params_as_text = "\n".join(
        [
            rf"$F_{{\max}}$ = {params['capacity']:.2f} vphpl",
            rf"$\rho_{{jam}}$ = {params['jam_density']:.2f} vpmpl",
            rf"$\rho_{{crit}}$ = {params['critical_density']:.2f} vpmpl",
            rf"$v_f$ = {params['free_flow_speed']:.2f} mph",
            rf"$w$ = {params['congestion_wave_speed']:.2f} mph",
        ]
    )

    plt.annotate(
        text=params_as_text,
        xy=(0.65, 0.25),
        xycoords="figure fraction",
        bbox=dict(boxstyle="square,pad=0.5", fc="lightgray", ec="black", lw=2),
        fontsize=18,
    )

    if save_path is None:
        # Show the plot
        plt.show()
    else:
        try:
            plt.savefig(save_path)
            logger.info(f"Figure for detector {detector_id} saved to: {save_path}")
        except Exception as e:
            logger.info(
                f"An exception was raised while saving the figure for detector {detector_id}: {e}"
            )
    # Close the figure
    plt.close()


plot_fundamental_diagram(
    normalized_df=timeseries_normalized,
    bin_df=bin_df,
    free_flow_model=free_flow_speed_model,
    congestion_model=congestion_wave_speed_model,
    params=params,
    detector_id=detector_id,
    save_path=save_path,
)

# %%
