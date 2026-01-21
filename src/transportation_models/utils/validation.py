"""
Tools for validating data structure and values
"""

# 3rd party imports
import typing
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.stats import cumfreq, pearsonr

# Logger setup
logger = logging.getLogger(__name__)


def validate_timestamp_dtype(df: pd.DataFrame) -> pd.DataFrame:
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
            logger.error(
                f"Could not convert timestamp column to dtype pd.Timestamp:\n{e}"
            )
            raise e

    return df


def slice_lanes(df: pd.DataFrame) -> pd.DataFrame:
    # Drop empty columns – what's left is the actual lanes
    df.dropna(how="all", axis=1, inplace=True, ignore_index=True)

    # Deduce the number of lanes in this DF from the remaining lane flow columns
    num_lanes = len([col for col in df.columns if "flow_lane_" in col])

    # Build a list of column names to extract
    cols_to_extract = [
        "timestamp",
        "pct_observed",
        "total_flow_[veh/5-min]",
        "avg_speed_[mph]",
    ]
    for i in range(1, num_lanes + 1):
        lane_cols = [f"flow_lane_{i}", f"avg_speed_lane_{i}", f"observed_lane_{i}"]
        cols_to_extract += lane_cols

    # Make a copy of the DF with the subset of columns to extract
    sliced_df = df.loc[:, cols_to_extract].copy()

    # Multiply the lane imputation flag by 100 so that positive flags count as 100% observed
    # (Useful for calibration)
    observed_cols = [col for col in sliced_df.columns if "observed_lane_" in col]
    sliced_df.loc[:, observed_cols] *= 100

    return sliced_df


def validate_calibration(calibrated_metadata_filepath: Path) -> dict[str, float]:
    # Constants
    CAPACITY_LOWER_LIMIT = 100.0
    CAPACITY_UPPER_LIMIT = 10000.0
    FREE_FLOW_SPEED_LOWER_LIMIT = 40.0
    FREE_FLOW_SPEED_UPPER_LIMIT = 80.0
    CONGESTION_WAVE_SPEED_LOWER_LIMIT = 0.0
    CONGESTION_WAVE_SPEED_UPPER_LIMIT = 40.0
    JAM_DENSITY_LOWER_LIMIT = 0.0
    JAM_DENSITY_UPPER_LIMIT = 1000.0

    calibrated_metadata_df = pd.read_csv(calibrated_metadata_filepath)

    # Filter for mainline detectors
    calibrated_metadata_df = calibrated_metadata_df.loc[
        calibrated_metadata_df["Type"] == "Mainline", :
    ].copy()

    # List of calibrated parameter column names
    calibrated_params = [
        "capacity",
        "free_flow_speed",
        "congestion_wave_speed",
        "jam_density",
    ]

    # Drop missing values
    calibrated_metadata_df.dropna(
        axis=0, how="any", subset=calibrated_params, inplace=True, ignore_index=True
    )

    # Add a column for each parameter to track validation
    for param_column in calibrated_params:
        calibrated_metadata_df[f"{param_column}_validated"] = False

    # Validate the limits for each parameter
    calibrated_metadata_df = calibrated_metadata_df.apply(
        validate_limits,  # type: ignore
        axis=1,
        args=("capacity", CAPACITY_LOWER_LIMIT, CAPACITY_UPPER_LIMIT),
    )
    calibrated_metadata_df = calibrated_metadata_df.apply(
        validate_limits,  # type: ignore
        axis=1,
        args=(
            "free_flow_speed",
            FREE_FLOW_SPEED_LOWER_LIMIT,
            FREE_FLOW_SPEED_UPPER_LIMIT,
        ),
    )
    calibrated_metadata_df = calibrated_metadata_df.apply(
        validate_limits,  # type: ignore
        axis=1,
        args=(
            "congestion_wave_speed",
            CONGESTION_WAVE_SPEED_LOWER_LIMIT,
            CONGESTION_WAVE_SPEED_UPPER_LIMIT,
        ),
    )
    calibrated_metadata_df = calibrated_metadata_df.apply(
        validate_limits,  # type: ignore
        axis=1,
        args=(
            "jam_density",
            JAM_DENSITY_LOWER_LIMIT,
            JAM_DENSITY_UPPER_LIMIT,
        ),
    )

    # Calculate statistics on the calibrated parameters
    stats = calculate_calibration_statistics(
        calibrated_metadata_df, col_names=calibrated_params
    )

    # Check for large standard deviations in any of the parameters
    thresholds = {
        "std_capacity": 0.9 * stats["mean_capacity"],
        "std_free_flow_speed": 0.9 * stats["mean_free_flow_speed"],
        "std_congestion_wave_speed": 0.9 * stats["mean_congestion_wave_speed"],
        "std_jam_density": 0.9 * stats["mean_jam_density"],
    }
    validate_std(stats, thresholds=thresholds)

    # Check the results of the validation
    for param_column in calibrated_params:
        pct_invalid = quantify_calibration_validity(
            calibrated_metadata_df, f"{param_column}_validated"
        )
        stats[f"{param_column}_pct_invalid"] = pct_invalid

    return stats


def validate_std(
    stats: dict[str, float],
    thresholds: dict[str, float] = {
        "std_capacity": 0.0,
        "std_free_flow_speed": 0.0,
        "std_congestion_wave_speed": 0.0,
        "std_jam_density": 0.0,
    },
) -> None:
    for k, v in stats.items():
        if "std" in k:
            threshold = thresholds.get(k, 0.0)
            if v > threshold:
                logger.warning(
                    f"Standard deviation for {k} exceeds the allowable threshold of {threshold}\n{k}:{v}"
                )
    return


def validate_limits(
    row: pd.Series, parameter: str, lower_limit: float, upper_limit: float
) -> pd.Series:
    # Get the parameter and station ID
    param = row[parameter]
    station_id = row["Station ID"]

    # Check the limits
    if lower_limit <= param <= upper_limit:
        row[f"{parameter}_validated"] = True
    else:
        row[f"{parameter}_validated"] = False
        logger.warning(
            f"Calibration of parameter {parameter} for Station ID {station_id} falls outside the acceptable limits\nCalibrated value: {param:.2f} is not within [{lower_limit},{upper_limit}]"
        )

    return row


def quantify_calibration_validity(df: pd.DataFrame, validated_col_name: str) -> float:
    # Filter the DF for stations with invalid flags set
    invalid_stations = df.loc[~df[validated_col_name], "Station ID"].copy()

    if invalid_stations.empty:
        return 0.0
    else:
        # Find the share of stations with invalid flags set
        num_invalid_stations = len(invalid_stations)  # type: ignore
        num_total_stations = len(df)
        pct_stations_with_invalid_params = (
            num_invalid_stations / num_total_stations
        ) * 100

        logger.warning(
            f"Calibration has resulted in invalid {validated_col_name} calibration for {pct_stations_with_invalid_params:.2f}% of detectors"
        )
        return pct_stations_with_invalid_params


def validate_speed(df: pd.DataFrame, speed_col) -> None:
    # Filter the DF for stations with a negative speed
    negative_speed_stations = df.loc[df[speed_col] < 0.0, "Station ID"].copy()  # type: ignore

    if negative_speed_stations.empty:  # type: ignore
        # logger.info(f"Calibration of {speed_col} looks reasonable")
        return
    else:
        # Find the share of stations with negative speeds
        num_negative_speed_stations = len(negative_speed_stations)  # type: ignore
        num_total_stations = len(df)
        pct_stations_with_negative_speed = (
            num_negative_speed_stations / num_total_stations
        ) * 100

        logger.warning(
            f"Calibration has resulted in negative {speed_col} for {pct_stations_with_negative_speed:.2f}% of detectors:\n{negative_speed_stations.to_list()}"  # type: ignore
        )
        return


def calculate_calibration_statistics(
    df: pd.DataFrame, col_names: list
) -> dict[str, float]:
    stats = {}
    for col_name in col_names:
        stats[f"mean_{col_name}"] = df[col_name].mean()
        stats[f"std_{col_name}"] = df[col_name].std()
    return stats


def view_flow_conservation_timeseries(
    splits_df: pd.DataFrame,
    node_id: int,
    node_id_mapper: dict[int, str] = {},
    save_path: typing.Optional[str] = None,
):
    # Make a local copy
    df = splits_df.copy()

    # Add a 'Total' column
    df["Total"] = df.loc[:, ["beta", "1-beta"]].apply(
        lambda row: round((row.sum()), 2), axis=1
    )

    # Create a plot
    ax = df.rename(
        columns={
            "beta": "$\\beta$",
            "1-beta": "$1-\\beta$",
        }
    ).plot(title=f"Split Ratio\n{node_id_mapper.get(node_id, f'Node {node_id}')}")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def calculate_cdf(
    data: typing.Iterable, numbins: int = 10
) -> tuple[typing.Iterable, typing.Iterable]:
    r = cumfreq(data, numbins=numbins)
    x = r.lowerlimit + np.linspace(0, r.binsize * r.cumcount.size, r.cumcount.size)
    c = r.cumcount / r.cumcount[-1]

    return x, c


def get_seconds_since_start_of_day(timestamp_obj: pd.Timestamp) -> float:
    """
    Convert timestamp to total seconds since the start of that day.
    Used for generating CDF.

    Parameters
    ----------
    timestamp_obj : pd.Timestamp
        A timestamp

    Returns
    -------
    float
        The number of seconds since the start of the day
    """
    start_of_day = timestamp_obj.normalize()
    time_difference = timestamp_obj - start_of_day
    seconds_since_start_of_day = time_difference.total_seconds()
    return seconds_since_start_of_day


# Formatter function for plotting
def seconds_to_time(x, pos):
    """
    A formatter function for plotting. Used to convert a number of seconds since midnight to a
    string "HH:MM"

    Parameters
    ----------
    x : float
        Number of seconds since the start of the day
    pos : unknown
        unknown

    Returns
    -------
    str
        Human-readable representation of the number of seconds since midnight
    """
    # Convert seconds since midnight to HH:MM
    seconds = int(x) % (24 * 3600)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


# Formatter function for plotting
def minutes_to_time(x, pos):
    """
    A formatter function for plotting. Used to convert a number of minutes since midnight to a
    string "HH:MM"

    Parameters
    ----------
    x : float
        Number of minutes since the start of the day
    pos : unknown
        unknown

    Returns
    -------
    str
        Human-readable representation of the number of minutes since midnight
    """
    # Convert minutes since midnight to HH:MM
    hours = int(x // 60)
    minutes = int(x % 60)
    return f"{hours:02d}:{minutes:02d}"


def generate_percent_error_df(splits_df: pd.DataFrame, time_res: str) -> pd.DataFrame:
    # Make a local copy
    df = splits_df.copy()

    # Validate the time_res parameter
    acceptable_time_res_vals = ["hour", "seconds"]
    time_res = time_res.lower()
    if time_res not in acceptable_time_res_vals:
        raise ValueError(
            f"The value of 'time_res' must be one of {acceptable_time_res_vals}, but {time_res} was passed"
        )

    # Rename columns
    df.rename(columns={"beta": "beta (calc)", "1-beta": "1-beta (calc)"}, inplace=True)

    # Add expected value columns
    df["beta (exp)"] = df["1-beta (calc)"].apply(lambda val: 1 - val)
    df["1-beta (exp)"] = df["beta (calc)"].apply(lambda val: 1 - val)

    # Add percentage error columns
    # 1) "mainline pct error" assumes that the ramp detector is always correct. Thus, if the
    # value of (1-beta) that was calculated from the mainline detector exceeds the value expected
    # by the ramp detector, this means that the mainline detector is counting too many vehicles.
    # Positive "mainline pct error" means too many vehicles were counted by the mainline detector,
    # negative "mainline pct error" means too few vehicles were counted by the mainline detector.
    # 2) "ramp pct error" assumes that the mainline detector is always correct.
    # Positive "ramp pct error" means that too many vehicles were counted by the ramp detector,
    # negative "ramp pct error" means that too few vehicles were counted by the mainline detector.
    df["mainline pct error"] = (
        (df["1-beta (calc)"] - df["1-beta (exp)"]) / df["1-beta (exp)"]
    ) * 100
    df["ramp pct error"] = (
        (df["beta (calc)"] - df["beta (exp)"]) / df["beta (exp)"]
    ) * 100

    # Get the timestamp as a column and make sure its a pd.Timestamp
    if "timestamp" not in df.columns:
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract the hour from the timestamp index
    if time_res == "hour":
        df[time_res] = df["timestamp"].dt.hour
    elif time_res == "seconds":
        df[time_res] = df["timestamp"].apply(get_seconds_since_start_of_day)
    else:
        return pd.DataFrame()

    # Keep only the columns we care about
    df_plot = df[[time_res, "timestamp", "mainline pct error", "ramp pct error"]]

    # Melt to long format: hour | type | value
    df_long = df_plot.melt(
        id_vars=[time_res, "timestamp"],
        var_name="Error Type",
        value_name="Percent Error",
    )

    return df_long


def view_split_ratio_cdfs(
    splits_df: pd.DataFrame,
    node_id: int,
    numbins: int = 10,
    col_to_plot: str = "seconds",
    node_id_mapper: dict[int, str] = {},
    save_path: typing.Optional[str] = None,
):
    # Print an update
    logger.info(
        f"Creating CDF plot for Node {node_id_mapper.get(node_id, f'Node {node_id}')}"
    )

    # Mask for good data
    not_missing_mask = ~((splits_df["beta"].isna()) | (splits_df["1-beta"].isna()))

    # Extract the good data
    df = splits_df.loc[not_missing_mask, :].copy()

    # Print some results on the masking
    logger.info(f"{len(df)} valid values out of {len(splits_df)}")

    # Convert to percent error
    pct_error_df = generate_percent_error_df(splits_df=df, time_res="seconds")

    # Validate the column passed
    assert col_to_plot in pct_error_df.columns

    # Slice for different conditions
    ramp_as_baseline = (
        pct_error_df.loc[pct_error_df["Error Type"] == "mainline pct error", :]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    mainline_overestimates = (
        ramp_as_baseline.loc[ramp_as_baseline["Percent Error"] > 0, col_to_plot]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    mainline_underestimates = -1 * (
        ramp_as_baseline.loc[ramp_as_baseline["Percent Error"] < 0, col_to_plot]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    mainline_as_baseline = (
        pct_error_df.loc[pct_error_df["Error Type"] == "ramp pct error", :]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    ramp_overestimates = (
        mainline_as_baseline.loc[mainline_as_baseline["Percent Error"] > 0, col_to_plot]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    ramp_underestimates = -1 * (
        mainline_as_baseline.loc[mainline_as_baseline["Percent Error"] < 0, col_to_plot]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # Print some results on the slicing
    mainline_overestimate_pct = (
        len(mainline_overestimates) / len(ramp_as_baseline)
    ) * 100
    mainline_underestimate_pct = (
        len(mainline_underestimates) / len(ramp_as_baseline)
    ) * 100
    ramp_overestimate_pct = (len(ramp_overestimates) / len(mainline_as_baseline)) * 100
    ramp_underestimate_pct = (
        len(ramp_underestimates) / len(mainline_as_baseline)
    ) * 100
    logger.info(
        f"If the ramp counts are reliable, mainline counts are overstated in {mainline_overestimate_pct:.2f}% of the data"
    )
    logger.info(
        f"If the ramp counts are reliable, mainline counts are understated in {mainline_underestimate_pct:.2f}% of the data"
    )
    logger.info(
        f"If the mainline counts are reliable, ramp counts are overstated in {ramp_overestimate_pct:.2f}% of the data"
    )
    logger.info(
        f"If the mainline counts are reliable, ramp counts are understated in {ramp_underestimate_pct:.2f}% of the data"
    )

    # Computing CDF
    mainline_over_x, mainline_over_c = calculate_cdf(
        mainline_overestimates,
        numbins=numbins,
    )
    mainline_under_x, mainline_under_c = calculate_cdf(
        mainline_underestimates, numbins=numbins
    )
    ramp_over_x, ramp_over_c = calculate_cdf(ramp_overestimates, numbins=numbins)
    ramp_under_x, ramp_under_c = calculate_cdf(ramp_underestimates, numbins=numbins)

    # Determine how to share the x-axis
    if col_to_plot == "seconds":
        sharex = "col"
    else:
        sharex = True

    # Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2, ncols=2, sharex=sharex, figsize=(15, 15)
    )
    ax1.plot(mainline_over_x, mainline_over_c, marker=".", linestyle="none")
    ax1.set_title("Mainline Overestimate (Ramp as Ground Truth)")
    ax1.set_ylabel("Cumulative Probability")
    ax1.grid(True)
    ax2.plot(mainline_under_x, mainline_under_c, marker=".", linestyle="none")
    ax2.set_title("Mainline Underestimate (Ramp as Ground Truth)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.grid(True)
    ax3.plot(ramp_over_x, ramp_over_c, marker=".", linestyle="none")
    ax3.set_title("Ramp Overestimate (Mainline as Ground Truth)")
    ax3.set_ylabel("Cumulative Probability")
    ax3.grid(True)
    if col_to_plot == "seconds":
        ax3.xaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_time))
        ax3.xaxis.set_major_locator(ticker.MultipleLocator(2 * 3600))  # every 2 hours
        ax3.tick_params(axis="x", labelrotation=45)
        ax3.set_xlabel("Time of Day")
    else:
        ax3.set_xlim(0, 100)
        ax3.set_xlabel("Percent Error")
    ax4.plot(ramp_under_x, ramp_under_c, marker=".", linestyle="none")
    ax4.set_title("Ramp Underestimate (Mainline as Ground Truth)")
    ax4.set_ylabel("Cumulative Probability")
    ax4.grid(True)
    if col_to_plot == "seconds":
        ax4.xaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_time))
        ax4.xaxis.set_major_locator(ticker.MultipleLocator(2 * 3600))  # every 2 hours
        ax4.tick_params(axis="x", labelrotation=45)
        ax4.set_xlabel("Time of Day")
    else:
        ax4.set_xlim(0, 100)
        ax4.set_xlabel("Percent Error")
    plt.suptitle(
        f"Cumulative Distribution Function of Split Ratio\n{node_id_mapper.get(node_id, f'Node {node_id}')}"
    )
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def view_split_ratio_box_plot(
    splits_df: pd.DataFrame,
    node_id: int,
    node_id_mapper: dict[int, str] = {},
    save_path: typing.Optional[str] = None,
):
    # Print an update
    logger.info(
        f"Creating box plot for Node {node_id_mapper.get(node_id, f'Node {node_id}')}"
    )

    pct_error_df = generate_percent_error_df(splits_df=splits_df, time_res="hour")

    # Compute Q1 and Q3 for each hour/error type
    quartiles = (
        pct_error_df.groupby(["Error Type", "hour"])["Percent Error"]
        .quantile([0.25, 0.75])  # type: ignore
        .unstack(level=-1)  # columns are now 0.25 and 0.75
        .reset_index()
        .rename(columns={0.25: "Q1", 0.75: "Q3"})
    )
    # Find overall lowest Q1 and highest Q3
    lowest_q1 = quartiles["Q1"].min()
    highest_q3 = quartiles["Q3"].max()

    # Set the boundaries for the zoomed version of the plot to make the error around zero more visible
    # lower_plot_bound = max((lowest_q1 * 1.05), -50)
    lower_plot_bound = lowest_q1
    # upper_plot_bound = min((highest_q3 * 1.05), 50)
    upper_plot_bound = highest_q3

    # Create hour labels
    hour_labels = [f"{h:02d}:00" for h in range(24)]

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 1. Full range
    sns.boxplot(
        x="hour",
        y="Percent Error",
        hue="Error Type",
        data=pct_error_df,
        ax=axes[0],
    )
    axes[0].set_title("")
    axes[0].set_ylabel("Percent Error")
    axes[0].set_xlabel("")
    axes[0].set_xticks(range(24))
    axes[0].set_xticklabels(hour_labels, rotation=45)

    # 2. Zoomed-in range
    sns.boxplot(
        x="hour",
        y="Percent Error",
        hue="Error Type",
        data=pct_error_df,
        ax=axes[1],
    )
    axes[1].axhline(y=0, linestyle="--", color="red")
    axes[1].set_ylim(lower_plot_bound, upper_plot_bound)
    axes[1].set_title("(Zoomed)")
    axes[1].set_ylabel("Percent Error")
    axes[1].set_xlabel("Time of Day")
    axes[1].set_xticks(range(24))
    axes[1].set_xticklabels(hour_labels, rotation=45)

    plt.suptitle(
        f"Hourly Distribution of Flow Conservation Error\n{node_id_mapper.get(node_id, f'Node {node_id}')}"
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def view_split_ratio_autocorrelation_plot(
    splits_df: pd.DataFrame,
    node_id: int,
    node_id_mapper: dict[int, str] = {},
    lag_cadence: str = "daily",
    save_path: typing.Optional[str] = None,
):
    # validate lag_cadence parameter
    if lag_cadence == "daily":
        lag_factor = 2  # 2 days
        tick_frequency = 120  # 2 hours
    elif lag_cadence == "weekly":
        lag_factor = 14  # 2 weeks
        tick_frequency = 60 * 24  # 1 day
    else:
        raise ValueError(
            f"The value of parameter 'lag_cadence' must be one of ['daily', 'weekly'], but {lag_cadence} was passed."
        )

    # Define the number of lags that we want
    num_lags = 12 * 24 * lag_factor  # 5-minute intervals

    # Make a local copy to avoid editing original data
    df = splits_df.copy()

    # Convert index for beta and beta' to DatetimeIndex
    df.index = pd.to_datetime(df.index)

    # Extract beta and beta'
    beta = df.loc[:, "beta"]
    beta_prime = df.loc[:, "1-beta"]

    # Get the error metrics
    pct_error_df = generate_percent_error_df(splits_df=df, time_res="seconds")

    # Replace infs with NaNs
    pct_error_df.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=True)

    # Set the index to the timestamp as a DatetimeIndex
    if "timestamp" in pct_error_df.columns:
        pct_error_df.set_index("timestamp", inplace=True)
    pct_error_df.index = pd.to_datetime(pct_error_df.index)

    # Create a complete 5-minute interval index from first to last timestamp
    full_index = pd.date_range(
        start=pct_error_df.index.min(), end=pct_error_df.index.max(), freq="5min"
    )

    # Extract the mainline and ramp errors as separate timeseries
    mainline_error = pct_error_df.loc[
        pct_error_df["Error Type"] == "mainline pct error", "Percent Error"
    ]
    ramp_error = pct_error_df.loc[
        pct_error_df["Error Type"] == "ramp pct error", "Percent Error"
    ]

    # Reindex each dataframe to include all 5-minute steps
    mainline_error = mainline_error.reindex(full_index).to_numpy()
    ramp_error = ramp_error.reindex(full_index).to_numpy()
    beta = beta.reindex(full_index).to_numpy()
    beta_prime = beta_prime.reindex(full_index).to_numpy()

    # Calculate the autocorrelation of the ramp and mainline percent errors
    mainline_acorr = sm.tsa.acf(
        x=mainline_error, nlags=num_lags, missing="conservative"
    )
    ramp_acorr = sm.tsa.acf(x=ramp_error, nlags=num_lags, missing="conservative")
    beta_acorr = sm.tsa.acf(x=beta, nlags=num_lags, missing="conservative")
    beta_prime_acorr = sm.tsa.acf(x=beta_prime, nlags=num_lags, missing="conservative")

    # Generate some x-values for plotting
    x = np.linspace(0, num_lags * 5, num_lags + 1)

    # Plot the results
    fig, ax = plt.subplots(sharex=True, figsize=(12, 8))
    ax.plot(x, beta_acorr, label="$\\beta$")
    ax.plot(x, beta_prime_acorr, label="$1-\\beta$")
    ax.plot(x, mainline_acorr, label="Mainline Error")
    ax.plot(x, ramp_acorr, label="Ramp Error")

    # Formatting
    plt.minorticks_on()
    ax.set_ylabel("Correlation")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="both")
    ax.set_xlabel("Lag (time)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_frequency))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(minutes_to_time))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="x", which="major")
    plt.legend()
    plt.title(
        f"Autocorrelation of Errors in Split Ratio Calculation\n{node_id_mapper.get(node_id, f'Node {node_id}')}"
    )

    # Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def get_pearson_correlation(
    splits_df: pd.DataFrame,
    node_id: int,
    node_id_mapper: dict[int, str] = {},
) -> tuple:
    # Make a local copy of the data to avoid overwriting
    df = splits_df.copy()

    # Eliminate any NaN values from the main DF
    df.dropna(axis=0, how="any", inplace=True)

    # Calculate errors
    pct_error_df = generate_percent_error_df(splits_df=df, time_res="hour")

    # Eliminating NaN values from the Error DF requires a pivot to make sure that the arrays end up the same size
    pct_error_df = pct_error_df.drop("hour", axis=1).pivot(
        index="timestamp", columns="Error Type", values="Percent Error"
    )
    # Eliminate Inf values before dropping NaNs
    pct_error_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    pct_error_df.dropna(axis=0, how="any", inplace=True)

    # Extract data as arrays
    beta = df.loc[:, "beta"].to_numpy()
    beta_prime = df.loc[:, "1-beta"].to_numpy()
    mainline_errors = pct_error_df.loc[:, "mainline pct error"].to_numpy()
    ramp_errors = pct_error_df.loc[:, "ramp pct error"].to_numpy()

    res_data = pearsonr(x=beta, y=beta_prime)
    res_error = pearsonr(x=mainline_errors, y=ramp_errors)

    logging.info(
        f"The Pearson correlation coefficient for beta and beta' at {node_id_mapper.get(node_id, f'Node {node_id}')} is: {res_data.statistic:.3f}"  # type: ignore
    )
    logging.info(
        f"The Pearson correlation coefficient for mainline and ramp errors at {node_id_mapper.get(node_id, f'Node {node_id}')} is: {res_error.statistic:.3f}"  # type: ignore
    )

    return res_data, res_error


def safe_sum(series: pd.Series):
    """Return NaN if any NaN present, else sum."""
    return series.sum() if not series.isna().any() else np.nan


def weighted_avg(pct: pd.Series, weights: pd.Series):
    """Return weighted average, NaN if any NaN present in either series."""
    if pct.isna().any() or weights.isna().any():
        return np.nan
    return (pct * weights).sum() / weights.sum() if weights.sum() != 0 else np.nan


def resample_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    # Make sure that the index is a datetime
    df.index = pd.to_datetime(df.index)

    # Reindex to fill in any missing values
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5min")
    df = df.reindex(full_index)

    # Resample into 15-minute timeseries, treating each column specifically
    df_15min = df.resample("15min").apply(
        lambda g: pd.Series(
            {
                "total_flow_[veh/15-min]": safe_sum(g["total_flow_[veh/5-min]"]),
                "pct_observed": weighted_avg(
                    g["pct_observed"], g["total_flow_[veh/5-min]"]
                ),
                "from_node_id": g["from_node_id"].iloc[0],
                "to_node_id": g["to_node_id"].iloc[0],
                "link_id": g["link_id"].iloc[0],
                "link_type": g["link_type"].iloc[0],
                "lanes": g["lanes"].iloc[0],
                "name": g["name"].iloc[0],
                "length_mi": g["length_mi"].iloc[0],
                "Station ID": g["Station ID"].iloc[0],
            }
        )
    )

    return df_15min


def view_split_ratio_errors(
    splits_df: pd.DataFrame,
    node_id: int,
    node_id_mapper: dict[int, str] = {},
    save_path: typing.Optional[str] = None,
) -> None:
    # Make a local copy
    df = splits_df.copy()

    # Rename columns
    df.rename(columns={"beta": "beta (calc)", "1-beta": "1-beta (calc)"}, inplace=True)

    # Drop useless columns
    df.dropna(axis=0, how="all", inplace=True)

    # Add expected value columns
    df["beta (exp)"] = df["1-beta (calc)"].apply(lambda val: 1 - val)
    df["1-beta (exp)"] = df["beta (calc)"].apply(lambda val: 1 - val)

    # Prepare "ideal" data
    x = np.linspace(0, 1, 100)

    # Plot data
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
    df.plot(x="beta (exp)", y="beta (calc)", ax=ax1, kind="scatter", label="data")
    # df.plot(x="1-beta (exp)", y="1-beta (calc)", ax=ax2, kind="scatter", label="data")

    # Add lines for "ideal" case
    ax1.plot(x, x, linestyle="--", color="red", label="ideal")
    # ax2.plot(x, x, linestyle="--", color="red", label="ideal")

    # Limit windows
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    # ax2.set_xlim([0, 1])
    # ax2.set_ylim([0, 1])

    # Add legend, labels & title
    ax1.legend()
    # ax2.legend()
    ax1.set_xlabel("$\\beta$ (Expected)")
    ax1.set_ylabel("$\\beta$ (Calculated)")
    # ax2.set_xlabel("$\\beta'$ (Expected)")
    # ax2.set_ylabel("$\\beta'$ (Calculated)")
    plt.title(f"Error in Split Ratio\n{node_id_mapper.get(node_id, f'Node {node_id}')}")

    # Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
