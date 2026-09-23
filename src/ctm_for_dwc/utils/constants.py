"""
Project-wide constants.

Filesystem paths to the local PeMS data store. These point at an external
drive; expect to repoint them for the current machine.
"""

from pathlib import Path

# Root of the PeMS data store (raw downloads, metadata, calibrated outputs).
ROOT_PATH = Path("/Volumes/easystore/work/boulder/Caltrans/PeMS")

# Per-station 5-minute timeseries CSVs, one file per VDS.
CALTRANS_VDS_TIMESERIES_DIRECTORY_PATH = Path(
    "/Volumes/easystore/work/boulder/Caltrans/PeMS/timeseries_data"
)
