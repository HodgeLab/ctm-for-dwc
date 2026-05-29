"""Step 3: download per-detector PeMS 5-minute timeseries data.

Wraps the existing :class:`PeMSDownloader` + :class:`PeMSExtractor` (in
``utils/data_downloading.py``) in a CLI matching the style of the other
CTM-module downloaders (:mod:`scripts.build_ctm_from_osm`,
``download_caltrans_postmiles``, ``download_pems_station_metadata``). The
script:

1. Queries the PeMS clearinghouse for ``station_5min`` files matching the
   district / year / month filter you specify.
2. Downloads each matching ``.gz`` to ``--out-dir`` (default:
   ``<repo_root>/data/pems/timeseries/``, gitignored).
3. Extracts the requested detectors into per-station CSVs under
   ``--out-dir/csv_files/`` (one CSV per ``vds_id``, appended-to on
   subsequent runs so the suite is incremental-friendly).

Credentials come from ``PEMS_USERNAME`` / ``PEMS_PASSWORD`` (auto-loaded
from ``.env`` via ``python-dotenv``) unless passed via ``--username`` /
``--password``.

Manual download
---------------
If you'd rather grab the ``.gz`` files by hand, browse to
``https://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min``, sign
in, pick the district / year / month, save the result under
``--out-dir``, then re-invoke this script with ``--extract-only`` to run
just the extraction step on what's already on disk.

Run from the repo root::

    # Download + extract two stations for January 2022, District 4.
    python scripts/download_pems_timeseries.py \\
        --district 4 --year 2022 --month January \\
        --detectors 400839,400840

    # Restrict to a specific date range (clipped during extraction).
    python scripts/download_pems_timeseries.py \\
        --district 4 --year 2022 --month January \\
        --detectors 400839 \\
        --start-date 2022-01-15 --end-date 2022-01-22

    # Skip the download; just re-extract whatever .gz files are already on disk.
    python scripts/download_pems_timeseries.py \\
        --extract-only --detectors 400839 \\
        --out-dir data/pems/timeseries
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from transportation_models.utils.data_downloading import (
    PeMSDownloader,
    PeMSExtractor,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO / "data" / "pems" / "timeseries"


# ---- Credential + env loading --------------------------------------------


def _resolve_credentials(username: Optional[str], password: Optional[str]):
    """Pull PeMS credentials from arg-passed values or environment.

    Loads ``.env`` from the repo root on demand (same pattern as
    ``download_caltrans_postmiles`` / ``download_pems_station_metadata``).
    """
    username = username or os.environ.get("PEMS_USERNAME", "").strip() or None
    password = password or os.environ.get("PEMS_PASSWORD", "").strip() or None
    if not username or not password:
        try:
            from dotenv import load_dotenv
            load_dotenv(REPO / ".env")
        except ImportError:
            pass
        username = username or os.environ.get("PEMS_USERNAME", "").strip() or None
        password = password or os.environ.get("PEMS_PASSWORD", "").strip() or None
    if not username or not password:
        raise SystemExit(
            "PeMS credentials not found. Set PEMS_USERNAME / PEMS_PASSWORD in "
            "the environment (a .env file works) or pass --username / "
            "--password explicitly."
        )
    return username, password


# ---- CLI ------------------------------------------------------------------


def _parse_detectors(value: str | None) -> list[int] | None:
    """``"400839,400840"`` -> ``[400839, 400840]``; ``None`` -> ``None``."""
    if value is None:
        return None
    return [int(s) for s in value.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--district", type=int, default=None,
        help="Caltrans district (1-12). Required unless --extract-only is given.",
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="single calendar year to query (use --year-start / --year-end "
             "for a range)",
    )
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end", type=int, default=None)
    parser.add_argument(
        "--month", default=None,
        help='comma-separated month names ("January,February,...") or empty '
             "for all months in each year",
    )
    parser.add_argument(
        "--detectors", default=None,
        help="comma-separated VDS station ids to extract (default: every "
             "station present in the downloaded .gz files)",
    )
    parser.add_argument("--start-date", default=None,
                        help="YYYY-MM-DD lower bound on rows kept by the extractor")
    parser.add_argument("--end-date", default=None,
                        help="YYYY-MM-DD upper bound on rows kept by the extractor")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"download + extraction directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="skip the network query and just run the extractor over "
             "whatever .gz files are already in --out-dir",
    )
    parser.add_argument("--username", default=None, help="PeMS account username")
    parser.add_argument("--password", default=None, help="PeMS account password")
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Download (unless --extract-only) ----------------------------
    if not args.extract_only:
        if args.district is None:
            parser.error("--district is required unless --extract-only is set")
        year_start, year_end = _resolve_year_range(args)
        months = _parse_months(args.month)

        username, password = _resolve_credentials(args.username, args.password)
        print(f"pems timeseries: querying district {args.district}, "
              f"years {year_start}-{year_end}, "
              f"months={months or 'all'}", file=sys.stderr)
        downloader = PeMSDownloader(username=username, password=password)
        downloader.download_files(
            start_year=year_start, end_year=year_end,
            districts=[str(args.district)],
            file_types=["station_5min"],
            months=months,
            save_path=str(out_dir),
        )

    # ---- 2. Extract -----------------------------------------------------
    detectors = _parse_detectors(args.detectors)
    print(f"pems timeseries: extracting from {out_dir}"
          + (f" detectors={detectors}" if detectors else " (all detectors)"),
          file=sys.stderr)
    extractor = PeMSExtractor(
        str(out_dir),
        detectors=detectors,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    extractor.process_gz_files()

    csv_dir = out_dir / "csv_files"
    print(f"pems timeseries: done. per-detector CSVs under {csv_dir}",
          file=sys.stderr)


def _resolve_year_range(args: argparse.Namespace) -> tuple[int, int]:
    """Map ``--year`` / ``--year-start`` / ``--year-end`` to an inclusive range."""
    if args.year is not None and (args.year_start or args.year_end):
        raise SystemExit("pass either --year or --year-start/--year-end, not both")
    if args.year is not None:
        return args.year, args.year
    if args.year_start is None or args.year_end is None:
        raise SystemExit(
            "must supply --year, or both --year-start and --year-end"
        )
    return args.year_start, args.year_end


def _parse_months(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    months = [m.strip() for m in raw.split(",") if m.strip()]
    return months or None


if __name__ == "__main__":
    main()
