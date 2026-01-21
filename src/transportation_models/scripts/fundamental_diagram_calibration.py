"""
Script for one-off calibration of Fundamental Diagram parameters
"""

# %%
# Imports
import os
import pandas as pd
from pathlib import Path
from transportation_models.utils.calibration import *

# Data
detector_metadata_filepath = Path(
    "/Volumes/easystore/work/boulder/Caltrans/PeMS/metadata/station_metadata_CALIBRATED.csv"
)
detector_metadata_df = pd.read_csv(detector_metadata_filepath)
detector_timeseries_data_directory = (
    "/Volumes/easystore/work/boulder/Caltrans/PeMS/timeseries_data"
)

detectors_to_calibrate = [400141, 400249]

detector_filepaths = {
    detector: Path(os.path.join(detector_timeseries_data_directory, f"{detector}.csv"))
    for detector in detectors_to_calibrate
}
detector_dataframes = {
    detector: pd.read_csv(detector_filepaths[detector])
    for detector in detectors_to_calibrate
}

# %% Setup
# Start with a single detector
detector = detectors_to_calibrate[1]
detector_metadata = detector_metadata_df.loc[
    detector_metadata_df["Station ID"] == detector, :
].iloc[0]
timeseries_df = detector_dataframes[detector]

# %%
detector_type = detector_metadata["Type"]
detector_id = detector_metadata["Station ID"]
if detector_type == "Mainline":
    params = extract_fundamental_diagram_params_from_timeseries(
        detector_metadata=detector_metadata,
        timeseries_df=timeseries_df,
        imputation_threshold=80.0,
        make_plots=True,
    )
else:
    print(f"Detector {detector_id} is not a mainline detector, so calibration fails")
# %%
