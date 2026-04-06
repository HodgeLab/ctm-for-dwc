# 3rd party imports
from pathlib import Path

# Logging setup
import logging
import transportation_models.utils.logs as logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logs.make_logger(
    include_stdout=True,
    log_prefix="fundamental_diagram_calibration",
    log_level=logging.DEBUG    
)

# Local imports
from transportation_models.utils.data_processing import PeMSDataProcessor

if __name__ == "__main__":
    # Create a DataProcessor instance
    DataProcessor = PeMSDataProcessor(
        root_directory="/Volumes/easystore/work/boulder/Caltrans/PeMS",
        save_transforms=False
    )

    # Calibrate all detectors
    DataProcessor.calibrate_fundamental_diagrams(
        make_plots=True,
        saved_plot_dir=Path("/Volumes/easystore/work/boulder/Caltrans/PeMS/metadata/calibration_plots"),
        save_params=True,
        output_metadata_path="/Volumes/easystore/work/boulder/Caltrans/PeMS/metadata/station_metadata_calibrated.csv"
    )