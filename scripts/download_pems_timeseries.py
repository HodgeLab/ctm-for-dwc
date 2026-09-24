"""Step 3a: download PeMS 5-minute station timeseries (.gz) from the clearinghouse.

Wraps :class:`ctm_for_dwc.pems.PeMSDownloader` in a CLI matching the
style of the other CTM-module downloaders (:mod:`scripts.build_ctm_from_osm`,
``download_caltrans_postmiles``, ``download_pems_station_metadata``). The
script queries the PeMS clearinghouse for ``station_5min`` files matching the
district / year / month filter you specify and downloads each matching
``.gz`` to ``--out-dir`` (default: ``<repo_root>/data/pems/timeseries/``,
gitignored). Files already recorded in ``--out-dir/saved_files.csv`` are
skipped, so re-runs are incremental.

Turn the downloaded files into per-detector CSVs with
``scripts/extract_pems_timeseries.py`` (Step 3b).

Credentials come from ``PEMS_USERNAME`` / ``PEMS_PASSWORD`` (auto-loaded
from ``.env`` via ``python-dotenv``) unless passed via ``--username`` /
``--password``.

Manual download
---------------
If you'd rather grab the ``.gz`` files by hand, browse to
``https://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min``, sign
in, pick the district / year / month, save the result under
``data/pems/timeseries/``, then run ``scripts/extract_pems_timeseries.py``.

Run from the repo root::

    # January 2022, District 4.
    python scripts/download_pems_timeseries.py \\
        --district 4 --year 2022 --month January

    # All months of 2022-2023.
    python scripts/download_pems_timeseries.py \\
        --district 4 --year-start 2022 --year-end 2023
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ctm_for_dwc.pems import PeMSDownloader, resolve_credentials

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO / "data" / "pems" / "timeseries"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--district", type=int, required=True, help="Caltrans district (1-12).",
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
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"download directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--username", default=None, help="PeMS account username")
    parser.add_argument("--password", default=None, help="PeMS account password")
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    year_start, year_end = _resolve_year_range(args)
    months = _parse_months(args.month)

    try:
        username, password = resolve_credentials(args.username, args.password)
    except RuntimeError as e:
        raise SystemExit(str(e))
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
    print(f"pems timeseries: done. .gz files under {out_dir}; extract with "
          "scripts/extract_pems_timeseries.py", file=sys.stderr)


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
