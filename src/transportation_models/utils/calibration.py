"""
Tools for calibrating the fundamental diagram of a cell.
"""

# 3rd party imports
import os
import typing
import logging
import statistics
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Local imports
from transportation_models.utils.validation import *
from transportation_models.utils.plotting import *

# Logging setup
logger = logging.getLogger(__name__)


def calibrate_detector(
    detector_metadata: pd.Series,
    timeseries_df: pd.DataFrame,
    imputation_threshold: float = 100.0,
) -> dict[str, float]:
    # Extract detector ID
    detector_id = detector_metadata["Station ID"]

    # Extract normalization parameters from metadata
    num_lanes = detector_metadata["Lanes"]

    # Use ones as default in case the parameters are bad
    if not num_lanes:
        logger.warning(
            f"No lane information for detector: {detector_id}. Using num_lanes=1"
        )
        num_lanes = 1

    # Empty dictionaries to pack with values
    params = {
        "capacity": np.nan,
        "free_flow_speed": np.nan,
        "congestion_wave_speed": np.nan,
        "jam_density": np.nan,
    }
    lane_params = {}

    # Modify the DF to prepare for per-lane analysis
    timeseries_df = slice_lanes(timeseries_df)

    for i in range(1, num_lanes + 1):
        imputation_flag_col_name = f"observed_lane_{i}"
        flow_col_name = f"flow_lane_{i}"
        speed_col_name = f"avg_speed_lane_{i}"

        # Get the parameters of the lane
        try:
            lane_params[i] = extract_fundamental_diagram_params_from_timeseries(
                detector_metadata=detector_metadata,
                timeseries_df=timeseries_df,
                imputation_threshold=imputation_threshold,
                imputation_flag_col_name=imputation_flag_col_name,
                flow_col_name=flow_col_name,
                speed_col_name=speed_col_name,
                num_lanes=1,
            )
        except ValueError:
            lane_params[i] = {
                "capacity": 0.0,
                "jam_density": 0.0,
                "free_flow_speed": 0.0,
                "congestion_wave_speed": 0.0,
            }
            logger.warning(
                f"Could not extract parameters for detector {detector_id}, lane {i}"
            )

        # Check the speed
        if lane_params[i]["free_flow_speed"] < 0.0:
            logger.warning(
                f"Calibration of free flow speed for detector {detector_id}, lane {i} is likely incorrect.\nCalibrated value={lane_params[i]['free_flow_speed']}\nOverriding to 0.0"
            )
            lane_params[i]["free_flow_speed"] = 0.0
        if lane_params[i]["congestion_wave_speed"] < 0.0:
            logger.warning(
                f"Calibration of congestion wave speed for detector {detector_id}, lane {i} is likely incorrect.\nCalibrated value={lane_params[i]['congestion_wave_speed']}\nOverriding to 0.0"
            )
            lane_params[i]["congestion_wave_speed"] = 0.0

    # Calculate the complete calibration from the individually calibrated lanes
    params["capacity"] = sum(
        [lane_param["capacity"] for lane_param in lane_params.values()]
    )
    params["jam_density"] = sum(
        [lane_param["jam_density"] for lane_param in lane_params.values()]
    )
    params["free_flow_speed"] = statistics.harmonic_mean(
        [lane_param["free_flow_speed"] for lane_param in lane_params.values()]
    )
    params["congestion_wave_speed"] = statistics.harmonic_mean(
        [lane_param["congestion_wave_speed"] for lane_param in lane_params.values()]
    )

    # Log the parameters
    station_id = detector_metadata["Station ID"]
    logger.info(f"Calibration of Station {station_id} complete: {params}")

    return params


def extract_max_capacity(
    detector_metadata: pd.Series,
    timeseries_df: pd.DataFrame,
    imputation_threshold: float = 100.0,
    imputation_flag_col_name: str = "pct_observed",
    flow_col_name: str = "total_flow_[veh/5-min]",
) -> float:
    # Extract detector ID
    detector_id = detector_metadata["Station ID"]

    # Ensure that the timestamp column exists and contains datetime objects
    timeseries_df = validate_timestamp_dtype(timeseries_df)

    # Aggregate flow over the hour
    hourly_flow_df = (
        timeseries_df.loc[:, ["timestamp", flow_col_name]]
        .set_index("timestamp")
        .resample("h")
        .sum()
        .reset_index()
    )

    # Rename flow column
    hourly_flow_df.rename(columns={flow_col_name: "flow_[veh/hour]"})

    # Average imputation over the hour
    hourly_imputation_df = (
        timeseries_df.loc[:, ["timestamp", imputation_flag_col_name]]
        .set_index("timestamp")
        .resample("h")
        .mean()
        .reset_index()
    )

    # Recombine
    hourly_df = hourly_flow_df.merge(
        right=hourly_imputation_df, how="left", on="timestamp"
    )

    # Limit the data to points where the detector is not imputing any measurements
    hourly_df = hourly_df.loc[
        hourly_df[imputation_flag_col_name] >= imputation_threshold, :
    ]

    # Handle missing data
    if hourly_df.empty:
        logger.warning(
            f"Peak flow extraction for detector {detector_id} failed:\nAfter eliminating imputed measurements, the timeseries data is empty."
        )
        # Return 0 by default
        return 0.0

    # Extract the peak flow (a.k.a., max capacity)
    max_capacity = hourly_df["flow_[veh/hour]"].max()

    return max_capacity


def extract_fundamental_diagram_params_from_timeseries(
    detector_metadata: pd.Series,
    timeseries_df: pd.DataFrame,
    imputation_threshold: float = 100.0,
    make_plots: bool = False,
    save_plots: bool = False,
    imputation_flag_col_name: str = "pct_observed",
    flow_col_name: str = "total_flow_[veh/5-min]",
    speed_col_name: str = "avg_speed_[mph]",
    num_lanes: typing.Optional[int] = None,
    congestion_speed_threshold: float = 55.0,
) -> dict[str, float]:
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

    # Handle missing data
    if timeseries_normalized.empty:
        logger.warning(
            f"Calibration failed for detector {detector_id}:\nAfter eliminating imputed measurements, the timeseries data is empty."
        )
        # Return the empty dictionary
        return params

    # Normalize the flow and density: [veh/5-min] * [5-min/hr] * [1/lanes] = [veh/hr*lanes]
    timeseries_normalized["normalized_flow_[veh/hr-lane]"] = (
        timeseries_normalized[flow_col_name] * periods_per_hour
    ) / num_lanes

    # Normalize the density: [veh/5-min] * [5-min/h] * [h/mi] * [1/lanes] = [veh/mi*lanes]
    timeseries_normalized["density_[veh/mi-lane]"] = (
        timeseries_normalized[flow_col_name] * periods_per_hour
    ) / (timeseries_normalized[speed_col_name] * num_lanes)

    # Eliminate any NA values
    not_na_mask = (~timeseries_normalized["density_[veh/mi-lane]"].isna()) & (
        ~timeseries_normalized["normalized_flow_[veh/hr-lane]"].isna()
    )
    timeseries_normalized = timeseries_normalized.loc[not_na_mask, :].reset_index(
        drop=True
    )

    # Extract the capacity and jam density based on the peak observed flow
    max_capacity = timeseries_normalized["normalized_flow_[veh/hr-lane]"].max()
    params["capacity"] = max_capacity
    max_capacity_timestamp = timeseries_normalized[
        "normalized_flow_[veh/hr-lane]"
    ].idxmax()
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

    # Optionally plot and save the calibration plots
    if make_plots:
        if save_plots:
            # Make a directory
            save_dir = os.path.join(os.getcwd(), "calibration_plots")
            Path.mkdir(Path(save_dir), exist_ok=True)

            # Construct a path for the plot
            save_path = Path(os.path.join(save_dir), f"{detector_id}.png")
        else:
            save_path = None
        plot_fundamental_diagram(
            normalized_df=timeseries_normalized,
            bin_df=bin_df,
            free_flow_model=free_flow_speed_model,
            congestion_model=congestion_wave_speed_model,
            params=params,
            detector_id=detector_id,
            save_path=save_path,
        )

    return params


def estimate_congestion_wave_speed(
    timeseries_df: pd.DataFrame,
    max_capacity: float,
    critical_density: float,
    test_points: list[float],
    bin_size=10,
) -> tuple[float, float, LinearRegression, pd.DataFrame]:
    # 1. Filter points with density > critical density (congested regime)
    congested = timeseries_df.loc[
        timeseries_df["density_[veh/mi-lane]"] > critical_density
    ].copy()
    congested = congested.sort_values(by="density_[veh/mi-lane]").reset_index(drop=True)

    # 2. Partition into non-overlapping bins of 10
    num_bins = len(congested) // bin_size
    bin_densities = []
    bin_flows = []

    for i in range(num_bins):
        bin_data = congested.iloc[i * bin_size : (i + 1) * bin_size]
        flows = bin_data["normalized_flow_[veh/hr-lane]"].values
        densities = bin_data["density_[veh/mi-lane]"].values

        Q1 = np.percentile(flows, 25)  # type: ignore
        Q3 = np.percentile(flows, 75)  # type: ignore
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR

        # Filter out high-end outliers
        valid_flows = flows[flows <= outlier_threshold]

        if len(valid_flows) == 0:
            continue  # skip if all flows are outliers

        BinFlow = np.max(valid_flows)
        BinDensity = np.mean(densities)  # type: ignore

        bin_densities.append(BinDensity)
        bin_flows.append(BinFlow)

    # Convert to DataFrame
    bin_df = pd.DataFrame({"BinDensity": bin_densities, "BinFlow": bin_flows})

    # 3. Fit constrained regression: line must pass through (critical_density, capacity_flow)
    # Model: BinFlow = w * (BinDensity - critical_density) + capacity_flow
    # => y = w * x + intercept where x = BinDensity - critical_density, y = BinFlow

    X = (bin_df["BinDensity"] - critical_density).values.reshape(-1, 1)  # type: ignore
    y = bin_df["BinFlow"].values - max_capacity  # type: ignore

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)  # Shifted regression
    wave_speed = model.coef_[0]

    # 4. Compute x-intercept (jam density)
    jam_density = critical_density - (max_capacity / wave_speed)

    # Print the results
    logger.info(f"Model fit score: {model.score(X,y)}")
    for test_point in test_points:
        test_point = np.array(test_point).reshape(-1, 1)  # type: ignore
        logger.info(
            f"Model predicts flow = {model.predict(test_point)} for density = {test_point}"  # type: ignore
        )

    return wave_speed, jam_density, model, bin_df


def estimate_free_flow_speed(
    timeseries_df: pd.DataFrame,
    congestion_speed_threshold: float,
    test_points: list[float],
    speed_col_name: str = "avg_speed_[mph]",
) -> tuple[float, LinearRegression]:

    # 1. Filter points with speed > congestion speed threshold
    free_flow_points = timeseries_df.loc[
        timeseries_df[speed_col_name] >= congestion_speed_threshold
    ].copy()

    # 2. Fit regression
    X = free_flow_points["density_[veh/mi-lane]"].copy().to_numpy().reshape(-1, 1)
    y = free_flow_points["normalized_flow_[veh/hr-lane]"].copy().to_numpy()

    # Initialize the model
    model = LinearRegression(fit_intercept=False)

    # Get the fit
    model.fit(X, y)

    # Get the free flow speed
    free_flow_speed = model.coef_[0]

    # Print the results
    logger.info(f"Model fit score: {model.score(X,y)}")
    for test_point in test_points:
        test_point = np.array(test_point).reshape(-1, 1)  # type: ignore
        logger.info(
            f"Model predicts flow = {model.predict(test_point)} for density = {test_point}"  # type: ignore
        )

    # Return the model
    return free_flow_speed, model


def get_metadata_series_by_id(
    detector_metadata_df: pd.DataFrame, detector_id: int
) -> typing.Optional[pd.Series]:
    # Look up the metadata for this detector
    detector_metadata = detector_metadata_df.loc[
        detector_metadata_df["Station ID"] == detector_id, :
    ]
    if detector_metadata.empty:
        logger.warning(f"No metadata found for detector: {detector_id}")
        detector_metadata_series = None
    else:
        if len(detector_metadata) > 1:
            logger.warning(
                f"More than one metadata entry found for detector: {detector_id}\nUsing the first entry only"
            )
        detector_metadata_series = detector_metadata.iloc[0]
    return detector_metadata_series


def main(
    detector_timeseries_data_directory: str,
    detector_metadata_filepath: str,
    calibrated_metadata_filepath: Path,
    imputation_threshold: float = 80.0,
    save_calibration_data: bool = False,
    make_calibration_plots: bool = False,
    save_calibration_plots: bool = False,
) -> None:
    # Load the metadata
    try:
        detector_metadata_df = pd.read_csv(detector_metadata_filepath)
    except Exception as e:
        logger.error(
            f"Calibration failed: Could not read detector metadata from {detector_metadata_filepath}"
        )
        raise e

    # Get a list of detectors with timeseries data
    detectors_with_timeseries = [
        int(filename.split(".csv")[0])
        for filename in os.listdir(detector_timeseries_data_directory)
        if filename.endswith(".csv") and not filename.startswith("V")
    ]

    # Try to calibrate each detector with timeseries data
    for detector_id in detectors_with_timeseries:
        logger.info(f"Attempting to calibrate detector {detector_id}")

        # Load the timeseries
        timeseries_filepath = os.path.join(
            detector_timeseries_data_directory, f"{detector_id}.csv"
        )
        try:
            timeseries_df = pd.read_csv(timeseries_filepath)
        except Exception as e:
            logger.error(
                f"Calibration failed for detector {detector_id}: Timeseries data could not be loaded due to an exception.\n{e}"
            )
            continue

        detector_metadata_series = get_metadata_series_by_id(
            detector_metadata_df, detector_id
        )
        if detector_metadata_series is None:
            logger.error(
                f"Calibration failed for detector {detector_id}: No metadata found!"
            )
            continue

        # Get the detector type
        detector_type = detector_metadata_series["Type"]

        if detector_type == "Mainline":
            # Get the fundamental diagram parameters
            detector_params = extract_fundamental_diagram_params_from_timeseries(
                detector_metadata=detector_metadata_series,
                timeseries_df=timeseries_df,
                imputation_threshold=100.0,
                make_plots=make_calibration_plots,
                save_plots=save_calibration_plots,
            )
        elif detector_type == "On Ramp":
            # Get the maximum capacity of the on-ramp
            max_capacity = extract_max_capacity(
                detector_metadata=detector_metadata_series,
                timeseries_df=timeseries_df,
                imputation_threshold=80.0,
            )
            # Load the max capacity into the same kind of dictionary as mainline detector parameters
            detector_params = {"capacity": max_capacity}
        else:
            logger.info(
                f"Detector {detector_id} is not a mainline or on-ramp detector. Skipping calibration."
            )
            continue

        # Optionally save the parameters
        if save_calibration_data:
            # Load the calibrated file
            if calibrated_metadata_filepath.is_file() and str(
                calibrated_metadata_filepath
            ).endswith(".csv"):
                try:
                    calibrated_metadata_df = pd.read_csv(calibrated_metadata_filepath)
                    logger.info(
                        f"Loaded calibrated metadata from {calibrated_metadata_filepath}"
                    )
                except Exception as e:
                    logger.error(
                        f"Calibrated metadata file found, but could not be loaded: {e}"
                    )
                    continue
            else:
                # Use the existing metadata as default
                calibrated_metadata_df = detector_metadata_df.copy()
                logger.info(
                    f"No file found with existing calibrated metadata. Creating a new file at: {calibrated_metadata_filepath}"
                )

            # Make sure that calibrated metadata has expected columns
            for calibration_col_name in detector_params.keys():
                if calibration_col_name not in calibrated_metadata_df.columns:
                    logger.info(
                        f"No {calibration_col_name} data found in calibrated metadata, assigning NaNs"
                    )
                    calibrated_metadata_df[calibration_col_name] = np.nan

            # Save the calibration parameters to the calibrated metadata DF
            for col, val in detector_params.items():
                # Check for existing data
                existing_val = calibrated_metadata_df.loc[
                    calibrated_metadata_df["Station ID"] == detector_id, col
                ]
                if existing_val.any():
                    logger.info(
                        f"Overwriting existing value of {col}.\nExisting value: {existing_val}"
                    )

                # Write new data
                calibrated_metadata_df.loc[
                    calibrated_metadata_df["Station ID"] == detector_id, col
                ] = val

            # Save the calibrated metadata DF
            try:
                calibrated_metadata_df.to_csv(calibrated_metadata_filepath, index=False)
            except Exception as e:
                logger.error(
                    f"Failed to save calibrated metadata to {calibrated_metadata_filepath} while working on detector {detector_id}:\n{e}"
                )


if __name__ == "__main__":
    # Logging setup
    Path.mkdir(Path(os.path.join(os.getcwd(), "logs")), exist_ok=True)
    logging.basicConfig(
        filename="logs/calibration.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Running calibration...")

    # Specify location of files on machine
    detector_metadata_filepath = (
        "/Volumes/easystore/work/boulder/Caltrans/PeMS/metadata/station_metadata.csv"
    )
    detector_timeseries_data_directory = (
        "/Volumes/easystore/work/boulder/Caltrans/PeMS/timeseries_data"
    )
    calibrated_metadata_filepath = Path(
        detector_metadata_filepath.split(".csv")[0] + "_CALIBRATED.csv"
    )

    # Run calibration
    main(
        detector_timeseries_data_directory=detector_timeseries_data_directory,
        detector_metadata_filepath=detector_metadata_filepath,
        calibrated_metadata_filepath=calibrated_metadata_filepath,
        imputation_threshold=80.0,
        save_calibration_data=True,
        make_calibration_plots=True,
        save_calibration_plots=True,
    )

    # Validate calibration
    stats = validate_calibration(
        calibrated_metadata_filepath=calibrated_metadata_filepath
    )
    logger.info("Calibration stats:")
    for k, v in stats.items():
        logger.info(f"{k}:{v}")

    # Logging shutdown
    logger.info("Calibration complete.")
    logging.shutdown()
