# 3rd party imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import typing
import logging

# Local imports
from cell_transmission_model.utils.validation import *

# Logger setup
logger = logging.getLogger(__name__)


def plot_fundamental_diagram(
    normalized_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    free_flow_model: LinearRegression,
    congestion_model: LinearRegression,
    params: dict[str, float],
    detector_id: int,
    save_path: typing.Optional[Path] = None,
) -> None:
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

    fig, ax = plt.subplots(figsize=(9, 6))

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
    plt.xlabel("Density (Vehicles Per Mile Per Lane)")
    plt.ylabel("Flow (Vehicles Per Hour Per Lane)")
    plt.title(f"Fundamental Diagram for Detector: {detector_id}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    params_as_text = f"q_max = {params['capacity']:.2f} vphpl\nrho_jam = {params['jam_density']:.2f} vpmpl\nrho_crit = {params['critical_density']:.2f} vpmpl\nv_f = {params['free_flow_speed']:.2f} mph\nw = {(params['congestion_wave_speed']):.2f} mph"
    plt.annotate(
        text=params_as_text,
        xy=(0.77, 0.25),
        xycoords="figure fraction",
        bbox=dict(boxstyle="square,pad=0.5", fc="lightgray", ec="black", lw=2),
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


def plot_parameters_for_freeway(
    calibrated_metadata_filepath: Path,
    save_path: typing.Optional[Path] = None,
) -> None:
    # Load the calibrated metadata
    df = pd.read_csv(calibrated_metadata_filepath)

    # Filter for mainline detectors
    df = df.loc[df["Type"] == "Mainline", :].copy()

    # List of calibrated parameter column names
    calibrated_params = [
        "capacity",
        "free_flow_speed",
        "congestion_wave_speed",
        "jam_density",
        "critical_density",
    ]

    # Prepare labels for plot
    param_labels = {
        "capacity": "Capacity [veh/hr-lane]",
        "free_flow_speed": "Free Flow Speed [mi/hr]",
        "congestion_wave_speed": "Congestion Wave Speed [mi/hr]",
        "jam_density": "Jam Density [veh/mi-lane]",
        "critical_density": "Critical Density [veh/mi-lane]",
    }

    # Drop missing values
    df.dropna(
        axis=0, how="any", subset=calibrated_params, inplace=True, ignore_index=True
    )

    # Split into northbound and southbound stations
    nb_df = df.loc[df["Fwy"].str.contains("-N"), :].copy()
    sb_df = df.loc[df["Fwy"].str.contains("-S"), :].copy()

    # Sort by direction of traffic
    nb_df.sort_values(
        by="Abs PM", ascending=True, inplace=True, ignore_index=True
    )  # NB postmiles increase with traffic
    sb_df.sort_values(
        by="Abs PM", ascending=False, inplace=True, ignore_index=True
    )  # SB postmiles decrease with traffic

    # Build a figure
    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        sharex="col",
        sharey="row",
        figsize=(14, 16),
        constrained_layout=True,
    )
    for row, param in enumerate(calibrated_params):
        for col, source in enumerate([nb_df, sb_df]):
            ax = axes[row, col]
            values = source[param]
            station_ids = source["Station ID"].astype("str")

            # Bar chart
            ax.bar(
                station_ids,
                values,
                color="tab:blue" if col == 0 else "tab:orange",
            )

            # Calculate mean and std
            mean_val = values.mean()
            std_val = values.std()

            # Horizontal lines
            ax.axhline(
                mean_val, color="red", linestyle="-", linewidth=1.5, label="Mean"
            )
            ax.axhline(
                mean_val + std_val,
                color="gray",
                linestyle="--",
                linewidth=1,
                label="±1 Std Dev",
            )
            ax.axhline(mean_val - std_val, color="gray", linestyle="--", linewidth=1)

            # Shaded region for ±1 std
            ax.fill_between(
                [-0.5, len(station_ids) - 0.5],  # full width of the bar chart
                mean_val - std_val,
                mean_val + std_val,
                color="gray",
                alpha=0.2,
            )

            # Y label goes on left column
            if col == 0:
                ax.set_ylabel(param_labels[param])

            # X label goes on the bottom row
            if row == 3:
                ax.set_xlabel("Station ID (traffic flows left to right)")
                ax.set_xticks(station_ids)
                ax.set_xticklabels(station_ids, rotation=45, ha="right")

    # Add legend and title
    axes[0, 1].legend(loc="upper right", bbox_to_anchor=(1.05, 1))
    plt.suptitle("Calibrated Parameters for I-880 Stations")

    # Save or show the plot
    if save_path is None:
        # Show the plot
        plt.show()
    else:
        try:
            plt.savefig(save_path)
            logger.info(f"Calibration summary figure saved to: {save_path}")
        except Exception as e:
            logger.info(
                f"An exception was raised while saving the calibration summary figure: {e}"
            )
    # Close the figure
    plt.close()
