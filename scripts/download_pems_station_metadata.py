"""Download PeMS station metadata and write an I-880-scoped CSV.

Standalone producer for the ``station_meta`` artifact the CTM pipeline
consumes via a path flag (``build_ctm_from_osm.py --pems-metadata``,
``calibrate_fundamental_diagrams.py --metadata``). Mirrors the CLI style of
the sibling downloader :mod:`scripts.download_pems_timeseries`.

The script always re-downloads the *latest* available PeMS ``station_meta``
text file for the district (no local-file fallback), filters the rows to a
single freeway (default I-880, both directions), keeps every PeMS column
(including ``Length``), and writes the result as a CSV.

Credentials come from ``PEMS_USERNAME`` / ``PEMS_PASSWORD`` (auto-loaded from
``.env`` via ``python-dotenv``) unless passed via ``--username`` /
``--password``.

Run from the repo root::

    python scripts/download_pems_station_metadata.py
    python scripts/download_pems_station_metadata.py --district 4 --freeway 880 \\
        --out data/pems/station_metadata.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ctm_for_dwc.utils.ctm.vds import download_pems_station_metadata

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "data" / "pems" / "station_metadata.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--district", type=int, default=4,
        help="Caltrans district to download (default: 4 = Bay Area / I-880)",
    )
    parser.add_argument(
        "--freeway", type=int, default=880,
        help="freeway number to keep (default: 880); both directions are kept",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"output CSV path (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--username", default=None, help="PeMS account username")
    parser.add_argument("--password", default=None, help="PeMS account password")
    args = parser.parse_args()

    # Always re-fetch the latest snapshot (year=None) -- no local fallback.
    txt_path = download_pems_station_metadata(
        district=args.district,
        year=None,
        overwrite=True,
        progress=True,
        username=args.username,
        password=args.password,
    )

    df = pd.read_csv(txt_path, sep=None, engine="python")
    scoped = df[df["Fwy"] == args.freeway]
    if scoped.empty:
        raise SystemExit(
            f"no stations with Fwy=={args.freeway} in {txt_path.name} "
            f"(available freeways: {sorted(df['Fwy'].unique())[:15]}...)"
        )

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scoped.to_csv(out_path, index=False)
    print(
        f"pems metadata: wrote {len(scoped)} stations "
        f"(Fwy=={args.freeway}, dirs={sorted(scoped['Dir'].unique())}) to "
        f"{out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
