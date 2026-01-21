"""
Script for plotting some timeseries data from the Caltrans PeMS database
"""

# %%
# Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transportation_models.utils.plotting import PeMSPlotter

# Data
detectors_to_plot = [
    400761,  # Mainline upstream
    402963,  # On-ramp (link 1)
    400490,  # Mainline (link 1)
    402965,  # On-ramp (link 2)
    401888,  # Mainline (link 2)
    402967,  # ﻿Off-ramp (link 2)
    400137,  # Mainline downstream
]
detector_metadata = pd.read_csv(
    "/Volumes/easystore/work/boulder/Caltrans/PeMS/metadata/station_metadata_CALIBRATED.csv"
)
detector_counts_directory = (
    "/Volumes/easystore/work/boulder/Caltrans/PeMS/timeseries_data"
)
possible_filepaths = [
    Path(os.path.join(detector_counts_directory, f"{detector}.csv"))
    for detector in detectors_to_plot
]
detector_filepaths = {
    detector: Path(os.path.join(detector_counts_directory, f"{detector}.csv"))
    for detector in detectors_to_plot
}

# Make sure that every filepath exists
detector_filepaths = {
    detector: filepath
    for detector, filepath in detector_filepaths.items()
    if filepath.is_file()
}

detector_dataframes = {
    detector: pd.read_csv(filepath) for detector, filepath in detector_filepaths.items()
}

# %% Test single detector:
df = detector_dataframes[detectors_to_plot[2]]

single_plotter = PeMSPlotter(df)
single_plotter.plot_uptime()
# single_plotter.plot_flow()

# %% Test multiple detectors:
big_df = pd.concat(
    [detector_df for detector_df in detector_dataframes.values()], ignore_index=True
)

multi_plotter = PeMSPlotter(big_df)
multi_plotter.plot_uptime(title="", save_path="i880_detector_uptime_2024.png")

# %%
multi_plotter.plot_uptime()
