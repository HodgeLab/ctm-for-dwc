# 3rd party imports
import typing
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Local imports
from validation import *

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


class PeMSPlotter(object):
    def __init__(self, df: pd.DataFrame) -> None:
        # Assign attributes
        self.df = self._validate_timestamp_dtype(df)
        self.stations_in_df = self._extract_unique_stations_in_df(df)
        self.num_stations_in_df = self._validate_number_of_stations_in_df(
            self.stations_in_df
        )

        # Pad the DF with entries for the missing dates
        self.df = self._pad_timeseries()

    @staticmethod
    def _validate_timestamp_dtype(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that the 'timestamp' column exists in the dataframe and that it holds datetime objects

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame that should have a 'timestamp' column

        Returns
        -------
        pd.DataFrame
            The DataFrame with a 'timestamp' column with datetime objects

        Raises
        ------
        e
            Any exception that is raised while trying to convert the timestamp column to datetime
        """
        # Check whether the timestamp is already a datetime
        if df.dtypes["timestamp"] == "datetime64[ns]":
            return df
        # Otherwise try and cast it as such before returning
        else:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except Exception as e:
                print(f"Could not convert timestamp column to dtype pd.Timestamp:\n{e}")
                raise e

        return df

    @staticmethod
    def _extract_unique_stations_in_df(df: pd.DataFrame) -> list[int]:
        """
        Get a list of unique station IDs that are present in the DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame which holds an unknown set of station data

        Returns
        -------
        list[int]
            A list of station IDs that are present in the DataFrame

        Raises
        ------
        e
            Any exception that is raised while looking for stations is passed along
        """
        try:
            stations = [int(station) for station in df["station"].unique()]
        except Exception as e:
            print(f"Could not extract station information from DataFrame:\n{e}")
            raise e
        return stations

    @staticmethod
    def _validate_number_of_stations_in_df(stations_in_df: list[int]) -> int:
        """
        Assert that the number of stations in the DataFrame is a strictly positive number

        Parameters
        ----------
        stations_in_df : list[int]
            The output of _extract_stations_in_df()

        Returns
        -------
        int
            The number of unique station IDs in stations_in_df

        Raises
        ------
        AssertionError
            If the number of unique station IDs is not strictly positive
        """
        num_stations = len(stations_in_df)
        if num_stations > 0:
            return num_stations
        else:
            raise AssertionError(f"Expected a strictly positive number of stations")

    @staticmethod
    def _pad_timeseries_single_detector(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pad the timeseries data so that the time index covers the entire year.
        Missing sensor data is filled with NaN values, and missing metadata is forward-filled.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame with timeseries data. Expected to have the following attributes:
             - A single detector
             - A column called 'timestamp' with pd.Timestamp objects
             - The following metadata columns:
                "station",
                "district",
                "freeway",
                "direction of travel",
                "lane_type",
                "station_length"

        Returns
        -------
        pd.DataFrame
            The same DataFrame that was passed, but with the 'timestamp' column now containing all 5-minute
            intervals in the range of years covered in the original DataFrame. Missing sensor data is filled
            with NaN values, and missing metadata is forward-filled.


        Raises
        ------
        Exception
            If the 'station' column contains more than one detector ID.
        """
        # Extract the detector
        try:
            detector = df["station"].unique().item()
        except Exception as e:
            print(f"Could not extract detector ID from dataframe: {e}")
            raise e

        # Extract the year
        start_year = df["timestamp"].iloc[0].year
        end_year = df["timestamp"].iloc[-1].year

        # Create a full datetime index from Jan 1 to Dec 31 with 5-minute intervals
        full_index = pd.date_range(
            start=f"{start_year}-01-01 00:00:00",
            end=f"{end_year}-12-31 23:55:00",
            freq="5min",
        )

        # Set the existing timestamp column as the index
        df = df.set_index("timestamp")

        # Reindex to get the full DF
        df_full = df.reindex(index=full_index)

        # Find missing timestamps and their indices
        missing_mask = df_full.isnull().all(axis=1)

        # Get the timestamps that were missing
        missing_timestamps = df_full.index[missing_mask]

        print(
            f"Detector {detector} was missing entries for the following times:\n{missing_timestamps}"
        )

        # List of columns to forward-fill
        cols_to_fill = [
            "station",
            "district",
            "freeway",
            "direction_of_travel",
            "lane_type",
            "station_length",
        ]

        # Forward fill those columns
        df_full[cols_to_fill] = df_full[cols_to_fill].ffill()

        # Get the original data type of the columns that were forward-filled
        original_dtypes = df[cols_to_fill].dtypes.copy()

        # Try to set the data type back to the original
        for col, dtype in original_dtypes.items():
            try:
                df_full[col] = df_full[col].astype(dtype)
            except Exception as e:
                print(f"Warning: Could not cast column {col} to {dtype}: {e}")

        # Reset the index to put the timestamp back as column
        df_full = df_full.reset_index().rename(columns={"index": "timestamp"})

        return df_full

    def _pad_timeseries(self) -> pd.DataFrame:
        """
        A wrapper for _pad_timeseries_single_detector that can manage DataFrames containing multiple
        unique detectors.

        Returns
        -------
        pd.DataFrame
            A DataFrame (which may be contain multiple unique detectors) with timeseries data for each
            detector padded such that the timeseries spans every interval in the year.
        """
        if self.num_stations_in_df > 1:
            # Split the dataframe into dataframes for each detector
            dataframes_by_detector = {
                detector_id: self.df.loc[(self.df["station"] == detector_id), :].copy()
                for detector_id in self.stations_in_df
            }
            # Call the single detector function on each slice
            for detector_id, detector_df in dataframes_by_detector.items():
                dataframes_by_detector[detector_id] = (
                    self._pad_timeseries_single_detector(detector_df)
                )
            # Recombine the dataframes
            padded_df = pd.concat(
                [detector_df for detector_df in dataframes_by_detector.values()],
                ignore_index=True,
            )
        else:
            # Call the single detector function directly
            padded_df = self._pad_timeseries_single_detector(self.df)

        return padded_df

    def plot_uptime(
        self, title: typing.Optional[str] = None, save_path: typing.Optional[str] = None
    ) -> None:
        """
        Generate a plot of the uptime for the detector(s) represented in the dataframe.
        'Uptime is

        Parameters
        ----------
        title : typing.Optional[str], optional
            Option to override the default title of the plot with a custom string, by default None
        save_path : typing.Optional[str], optional
            Option to save the plot to the path specified, by default None
        """
        # Pick out the timestamp, station, and pct_observed columns
        df = self.df.loc[:, ["timestamp", "station", "pct_observed", "samples"]].copy()

        # Construct uptime plot for a single station
        if self.num_stations_in_df == 1:
            # Get the detector ID
            detector = self.stations_in_df[0]

            # Plot the uptime
            ax = df.loc[:, ["timestamp", "pct_observed"]].plot(
                x="timestamp",
                grid=True,
                # legend=False,
                title=f"VDS: {detector} Uptime",
                xlabel="Datetime",
                ylabel="% Observed",
            )

            # Plot the samples
            ax2 = ax.twinx()
            df.loc[:, ["timestamp", "samples"]].plot(
                x="timestamp",
                grid=True,
                ax=ax2,
                # legend=False,
                # xlabel="Datetime",
                ylabel="Samples",
                color="tab:orange",
            )

            # Optionally override the title
            if title is not None:
                ax.set_title(title)

            # Rotate the x tick labels
            ax.tick_params(axis="x", labelrotation=-25, labelbottom=True)
            plt.show()

            avg_uptime = self.df["pct_observed"].fillna(0).mean()
            print(
                f"Detector {detector} measured data for {avg_uptime:.2f}% of the year"
            )
        # Construct the uptime plot for multiple stations
        elif self.num_stations_in_df > 1:
            # Determine whether the legend should be included
            if self.num_stations_in_df > 10:
                include_legend = False
            else:
                include_legend = True

            # Pivot the DF to get the timeseries data for each station
            station_wide = df.pivot_table(
                index="timestamp", columns="station", values="pct_observed"
            )

            # Fill the NaN values in 'pct_observed' with zeros to calculate the mean
            df["pct_observed"] = df["pct_observed"].fillna(0)

            # Set timestamp column as index to prepare for groupby
            df = df.set_index("timestamp")

            # Group by the timestamp and take the mean across stations
            mean_pct = df.groupby(df.index)["pct_observed"].mean()

            # Plot the mean uptime of all stations
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
            mean_pct.plot(
                x="timestamp",
                color="dimgrey",
                alpha=1.0,
                ax=axes[0],
                grid=True,
                legend=False,
                title="Station-wide Average",
                ylabel="% Observed",
            )

            # Add a subplot with the uptime for all stations
            station_wide.plot(
                ax=axes[1],
                alpha=0.75,
                grid=True,
                legend=include_legend,
                ylabel="% Observed",
                xlabel="Timestamp",
                title="Uptime by Station",
            )

            # Rotate the x tick labels
            axes[1].tick_params(axis="x", labelrotation=-25)

            # Add a super title to the subplots
            plt.subplots_adjust(top=0.92, hspace=0.3)
            if title is None:
                plt.suptitle(
                    f"Uptime for Stations:\n{[station for station in self.stations_in_df]}",
                )
            else:
                plt.suptitle(title)

            # Move the legend outside the plot
            if include_legend:
                axes[1].legend(
                    title="Station", bbox_to_anchor=(1.05, 1), loc="upper left"
                )
            plt.tight_layout()

            # Print the overall average:
            avg_uptime = mean_pct.fillna(0).mean()
            print(f"Average uptime across all detectors is: {avg_uptime:.2f}")

            # Group by the station and take the mean across times to get the average uptime for each station
            avg_uptimes = df.groupby(by="station").mean().reset_index()

            # Print the average uptime for each station
            for _, row in avg_uptimes.iterrows():
                print(
                    f"Detector {int(row['station'])} measured data for {row['pct_observed']:.2f}% of the year"
                )

        if save_path is None:
            plt.show()
        else:
            try:
                plt.savefig(save_path, dpi=300)
            except Exception as e:
                print(f"Attempt to save plot to {save_path} failed: {e}")

    def plot_flow(
        self, title: typing.Optional[str] = None, save_path: typing.Optional[str] = None
    ) -> None:
        """
        Generate a plot of 5-minute traffic flow for the detector(s) represented in the dataframe.

        Parameters
        ----------
        title : typing.Optional[str], optional
            Option to override the default title of the plot with a custom string, by default None
        save_path : typing.Optional[str], optional
            Option to save the plot to the path specified, by default None
        """
        # Define columns to copy
        cols_to_copy = ["timestamp", "station", "total_flow_[veh/5-min]"]
        cols_to_copy += [col for col in self.df.columns if "flow_lane_" in col]

        # Pick out the columns
        df = self.df.loc[:, cols_to_copy].copy().dropna(axis=1, how="all")

        # Prepare subplots
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

        # Construct flow plot for a single station
        if self.num_stations_in_df == 1:
            # Get the detector ID
            detector = self.stations_in_df[0]

            # Define the supertitle
            suptitle = f"Flow for VDS:\n{detector}"

            # Plot the aggregate flow across lanes
            df.loc[:, ["timestamp", "total_flow_[veh/5-min]"]].plot(
                x="timestamp",
                ax=axes[0],
                grid=True,
                legend=False,
                title=f"Aggregate Flow",
                ylabel="Flow (veh/5-min)",
                color="dimgray",
            )

            # Get the lane columns
            lane_cols = [col for col in df.columns if "flow_lane_" in col]

            # Add traces for each lane to the second subplot
            df.loc[:, (["timestamp"] + lane_cols)].plot(
                x="timestamp",
                ax=axes[1],
                grid=True,
                legend=True,
                title="Flow by Lane",
                xlabel="Datetime",
                ylabel="Flow (veh/5-min)",
                alpha=0.6,
            )

            # Move the legend outside the plot
            axes[1].legend(
                [col.split("_")[-1] for col in lane_cols],
                title="Lane",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
            plt.tight_layout()

            # Print the average flow
            avg_flow = self.df["total_flow_[veh/5-min]"].fillna(0).mean()
            print(
                f"Average flow for Detector {detector} is {avg_flow:.2f} vehicles/5-minutes"
            )

            # Add a super title to the subplots
            plt.subplots_adjust(top=0.85, hspace=0.3)
            if title is None:
                plt.suptitle(suptitle)
            else:
                plt.suptitle(title)

            # Rotate the x tick labels
            axes[1].tick_params(axis="x", labelrotation=-25, labelbottom=True)
            plt.show()

            if save_path is not None:
                try:
                    plt.savefig(save_path, dpi=300)
                except Exception as e:
                    print(f"Attempt to save plot to {save_path} failed: {e}")
        else:
            print("Plotting flow for more than one station is not currently supported")
