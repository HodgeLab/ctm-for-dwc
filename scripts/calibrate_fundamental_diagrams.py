"""Step 4: calibrate triangular-FD parameters from PeMS timeseries.

Wraps :meth:`PeMSDataProcessor.calibrate_fundamental_diagrams` in a CLI
matching the style of the other CTM-module demo scripts
(:mod:`scripts.build_ctm_from_osm`,
:mod:`scripts.download_pems_timeseries`,
``download_caltrans_postmiles``). The script:

1. Loads PeMS station metadata + the per-detector timeseries directory
   produced by Step 3.
2. Runs the calibration loop, fitting the free-flow and congestion-wave
   regimes for each requested detector.
3. Writes the calibrated parameters
   (``capacity``, ``free_flow_speed``, ``congestion_wave_speed``,
   ``jam_density``, ``critical_density``) back into the metadata as
   ``station_metadata_calibrated.csv``.
4. Optionally renders one fundamental-diagram PNG per detector.

The output CSV is keyed on ``Station ID`` and is what Step 1's
``cells.csv`` (via the ``vds_id`` column attached in Step 2) joins
against to finish the freeway schema before passing to
:func:`utils.ctm.io.freeway_from_dataframe`.

Run from the repo root::

    # Default paths (assumes Step 3 has populated data/pems/timeseries/).
    python scripts/calibrate_fundamental_diagrams.py

    # Calibrate just a couple of stations with FD plots:
    python scripts/calibrate_fundamental_diagrams.py \\
        --detectors 400839,400840 --make-plots

    # Override paths (e.g. when timeseries data lives on an external drive):
    python scripts/calibrate_fundamental_diagrams.py \\
        --root-directory /Volumes/data/pems \\
        --metadata /Volumes/data/pems/metadata/station_metadata.csv \\
        --out-dir scripts/output/fd_calibration
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from transportation_models.utils.data_processing import PeMSDataProcessor


def _enable_stdout_logging() -> None:
    """Stream the data_processing logger's records to the terminal.

    ``data_processing`` configures the root logger with a file handler only
    (``include_stdout=False``), so its INFO/WARNING records never reach the
    console. Attach a stdout StreamHandler so calibration progress and the
    "0 rows survived the imputation threshold" warning are visible here.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(handler)

    # The root logger runs at DEBUG, so matplotlib's noisy font-manager/backend
    # DEBUG records would otherwise flood the terminal. Quiet them to WARNING.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "data" / "pems"
DEFAULT_OUT_DIR = REPO / "data" / "pems" / "calibrated"


def _parse_detectors(value: str | None) -> list[str] | None:
    """``"400839,400840"`` -> ``["400839", "400840"]``; ``None`` -> ``None``."""
    if value is None:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


def main() -> None:
    _enable_stdout_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root-directory", type=Path, default=DEFAULT_ROOT,
        help=("PeMS root directory containing 'metadata/' and "
              "'timeseries_data/'. Defaults to <repo>/data/pems/"),
    )
    parser.add_argument(
        "--metadata", type=Path, default=None,
        help=("explicit path to the station metadata CSV; defaults to "
              "<root-directory>/metadata/station_metadata.csv"),
    )
    parser.add_argument(
        "--detectors", default=None,
        help=("comma-separated VDS station ids to calibrate; default = all "
              "mainline detectors present in both metadata and the "
              "timeseries directory"),
    )
    parser.add_argument(
        "--ramp-detectors", default=None,
        help=("comma-separated ramp VDS station ids to estimate capacity "
              "for. Ramp detectors only report total flow (no speed), so "
              "we can't fit a triangular FD; instead we take a high "
              "quantile of the cleaned 5-minute flow as the ramp's "
              "capacity. The output goes to a separate "
              "ramp_metadata_calibrated.csv alongside the mainline "
              "calibration CSV. Omit to skip the ramp pass."),
    )
    parser.add_argument(
        "--ramp-capacity-quantile", type=float, default=0.99,
        help=("quantile of cleaned 5-minute flow to report as ramp "
              "capacity (default 0.99 -- robust to a single noisy spike "
              "while preserving the genuine observed peak)"),
    )
    parser.add_argument(
        "--imputation-threshold", type=float, default=100.0,
        help=("minimum pct_observed to keep a row during calibration "
              "(default 100.0 -- drops any row with any missing lane data)"),
    )
    parser.add_argument(
        "--iqr-multiplier", type=float, default=1.0,
        help=("IQR-fence multiplier for the row-level outlier filter "
              "(default 1.0, matching PeMSDataProcessor)"),
    )
    parser.add_argument(
        "--bin-iqr-multiplier", type=float, default=1.0,
        help=("IQR-fence multiplier for the per-bin congestion-branch filter "
              "(default 1.0, matching PeMSDataProcessor)"),
    )
    parser.add_argument(
        "--bin-size", type=int, default=10,
        help=("number of points per congestion density bin (default 10, "
              "matching PeMSDataProcessor)"),
    )
    parser.add_argument(
        "--triangle-rtol", type=float, default=0.05,
        help=("relative tolerance for the triangle-consistency validation "
              "gate (default 0.05, matching PeMSDataProcessor)"),
    )
    parser.add_argument(
        "--make-plots", action="store_true",
        help="write one FD scatter+fit PNG per detector under <out-dir>/plots/",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=(f"directory for the calibrated metadata CSV (and plots, if "
              f"requested). Default: {DEFAULT_OUT_DIR}"),
    )
    args = parser.parse_args()

    root_dir = args.root_directory.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.metadata
    if metadata_path is None:
        metadata_path = root_dir / "metadata" / "station_metadata.csv"
    metadata_path = metadata_path.expanduser().resolve()
    if not metadata_path.exists():
        raise SystemExit(
            f"station metadata CSV not found at {metadata_path}. "
            "Run scripts/download_pems_timeseries.py (or "
            "download_pems_station_metadata) first, or pass --metadata."
        )

    detectors = _parse_detectors(args.detectors)
    print(f"fd calibration: root={root_dir}", file=sys.stderr)
    print(f"  metadata = {metadata_path}", file=sys.stderr)
    print(f"  out-dir  = {out_dir}", file=sys.stderr)
    print(f"  detectors= {detectors if detectors else 'all mainline'}",
          file=sys.stderr)

    proc = PeMSDataProcessor(
        root_directory=str(root_dir),
        metadata_filepath=str(metadata_path),
        imputation_threshold=args.imputation_threshold,
        save_directory=str(out_dir),
    )

    plots_dir = (out_dir / "plots") if args.make_plots else None
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    proc.calibrate_fundamental_diagrams(
        detectors=detectors,
        iqr_multiplier=args.iqr_multiplier,
        bin_iqr_multiplier=args.bin_iqr_multiplier,
        bin_size=args.bin_size,
        triangle_rtol=args.triangle_rtol,
        make_plots=args.make_plots,
        saved_plot_dir=plots_dir,
        save_params=True,
    )

    calibrated_csv = out_dir / "station_metadata_calibrated.csv"
    print(f"fd calibration: done. calibrated metadata -> {calibrated_csv}",
          file=sys.stderr)
    if plots_dir is not None:
        print(f"                FD plots -> {plots_dir}", file=sys.stderr)

    ramp_detectors = _parse_detectors(args.ramp_detectors)
    if ramp_detectors:
        print(f"ramp calibration: {len(ramp_detectors)} VDSs", file=sys.stderr)
        proc.calibrate_ramp_capacities(
            detectors=ramp_detectors,
            quantile=args.ramp_capacity_quantile,
            save_params=True,
        )
        ramp_csv = out_dir / "ramp_metadata_calibrated.csv"
        print(f"ramp calibration: done. ramp metadata -> {ramp_csv}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
