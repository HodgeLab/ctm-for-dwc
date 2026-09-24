"""Step 3b: extract per-detector CSVs from downloaded PeMS 5-minute .gz files.

Wraps :class:`ctm_for_dwc.utils.pems.PeMSExtractor`. Reads every ``.gz`` at
the top level of ``--gz-dir`` (default: ``<repo_root>/data/pems/timeseries/``,
where ``scripts/download_pems_timeseries.py`` saves them) and writes one CSV
per ``vds_id`` under ``--gz-dir/csv_files/``. Existing detector CSVs are
appended to and then deduplicated, so re-runs after downloading more files
are incremental. No network access or credentials are needed.

Run from the repo root::

    # Two stations, every downloaded file.
    python scripts/extract_pems_timeseries.py --detectors 400839,400840

    # Restrict to a date range (rows outside it are dropped).
    python scripts/extract_pems_timeseries.py --detectors 400839 \\
        --start-date 2022-01-15 --end-date 2022-01-22
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ctm_for_dwc.utils.pems import PeMSExtractor

REPO = Path(__file__).resolve().parents[1]
DEFAULT_GZ_DIR = REPO / "data" / "pems" / "timeseries"


def _parse_detectors(value: str | None) -> list[int] | None:
    """``"400839,400840"`` -> ``[400839, 400840]``; ``None`` -> ``None``."""
    if value is None:
        return None
    return [int(s) for s in value.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gz-dir", type=Path, default=DEFAULT_GZ_DIR,
        help=f"directory of downloaded .gz files; CSVs go to its csv_files/ "
             f"(default: {DEFAULT_GZ_DIR})",
    )
    parser.add_argument(
        "--detectors", default=None,
        help="comma-separated VDS station ids to extract (default: every "
             "station present in the .gz files)",
    )
    parser.add_argument("--start-date", default=None,
                        help="YYYY-MM-DD lower bound on rows kept")
    parser.add_argument("--end-date", default=None,
                        help="YYYY-MM-DD upper bound on rows kept")
    args = parser.parse_args(argv)

    gz_dir = args.gz_dir.expanduser().resolve()
    detectors = _parse_detectors(args.detectors)
    print(f"pems timeseries: extracting from {gz_dir}"
          + (f" detectors={detectors}" if detectors else " (all detectors)"),
          file=sys.stderr)
    extractor = PeMSExtractor(
        str(gz_dir),
        detectors=detectors,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    extractor.process_gz_files()

    print(f"pems timeseries: done. per-detector CSVs under {gz_dir / 'csv_files'}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
