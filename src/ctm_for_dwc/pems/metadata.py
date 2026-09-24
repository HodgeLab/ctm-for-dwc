"""Download PeMS ``station_meta`` files from the clearinghouse.

:func:`download_pems_station_metadata` fetches one district's metadata
snapshot; :func:`ctm_for_dwc.ctm.vds.load_pems_station_metadata` reads
and filters it for the CTM pipeline (Step 2).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

from .credentials import resolve_credentials

# Default download target lives in the repo so subsequent runs find it without
# re-querying the clearinghouse. The directory is gitignored.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PEMS_DIR = _REPO_ROOT / "data" / "pems"
PEMS_META_FILE_TYPE = "meta"


def download_pems_station_metadata(
    district: int,
    *,
    year: Optional[int] = None,
    out_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    progress: bool = True,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Path:
    """Fetch one PeMS ``station_meta`` text file for a district + year.

    Uses :class:`ctm_for_dwc.pems.downloader.PeMSDownloader`
    against the live clearinghouse (``pems.dot.ca.gov``). Credentials come
    from the ``PEMS_USERNAME`` / ``PEMS_PASSWORD`` environment variables
    (loaded from ``.env`` via ``python-dotenv`` if present) unless
    explicitly passed in. The default cache directory is
    ``<repo_root>/data/pems/`` (gitignored); pass ``out_path`` to write
    elsewhere.

    Parameters
    ----------
    district : int
        Caltrans district (1-12; 7 = LA / I-210; 4 = Bay Area / I-880).
    year : int, optional
        Calendar year to query. When omitted, picks the most recent year
        for which a metadata file exists (PeMS only publishes one snapshot
        per district per year at most, so this is usually unambiguous).
    out_path : str or Path, optional
        Explicit destination filename. When omitted, the file lands at
        ``DEFAULT_PEMS_DIR / d{district}_meta_latest.txt`` (or, when
        ``year`` is given, ``d{district}_meta_{year}.txt``).
    overwrite : bool, default False
        Skip the download when the target already exists (False) or
        re-fetch (True).
    progress : bool, default True
        Print one-line status to stderr.
    username, password : str, optional
        PeMS credentials. Falls back to env vars.

    Returns
    -------
    Path
        Absolute path to the downloaded metadata file (a tab-delimited
        ``.txt`` despite the extension; load via
        :func:`load_pems_station_metadata` which sniffs the separator).

    Manual download
    ---------------
    Don't want to use this wrapper? Browse to
    ``https://pems.dot.ca.gov/?dnode=Clearinghouse&type=meta``, sign in,
    pick the district + year, and drop the resulting ``d{N}_text_meta_*.txt``
    file under ``data/pems/`` (or pass its path to
    :func:`load_pems_station_metadata` directly).
    """
    # Local import so this module stays importable when the downloader's
    # heavy/optional deps (mechanize, bs4) aren't available.
    from ctm_for_dwc.pems.downloader import PeMSDownloader

    username, password = resolve_credentials(username, password)

    if out_path is None:
        suffix = "latest" if year is None else str(year)
        out_path = DEFAULT_PEMS_DIR / f"d{int(district):02d}_meta_{suffix}.txt"
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        if progress:
            print(f"pems metadata: cache hit at {out_path} "
                  f"({out_path.stat().st_size / 1024:.1f} KB) -- pass "
                  f"overwrite=True to refresh", file=sys.stderr)
        return out_path

    if progress:
        print(f"pems metadata: querying clearinghouse for district {district}"
              f"{f', year {year}' if year else ''}...", file=sys.stderr)
    dl = PeMSDownloader(username=username, password=password, debug=False)

    # Search across a wide window when year is None so we always find *some*
    # snapshot; PeMS typically keeps one per district per recent year.
    search_window = (year, year) if year else (2010, 2030)
    files = dl.get_files(
        start_year=search_window[0], end_year=search_window[1],
        districts=[str(int(district))], file_types=[PEMS_META_FILE_TYPE],
    )
    if not files:
        raise RuntimeError(
            f"No PeMS metadata files found for district {district}"
            f"{f' year {year}' if year else ''}"
        )
    # Pick the lexicographically-latest filename (PeMS embeds a date in the
    # filename, e.g. d07_text_meta_2023_12_22.txt, so this works).
    chosen = sorted(files, key=lambda f: f["file_name"])[-1]
    if progress:
        print(f"  found {chosen['file_name']} "
              f"({chosen['megabites']:.2f} MB)", file=sys.stderr)

    tmp = Path(tempfile.mkstemp(prefix=".pems_dl_", dir=out_path.parent,
                                suffix=".txt")[1])
    try:
        # Use the underlying mechanize browser to stream the file directly.
        from ctm_for_dwc.pems.settings import BASE_URL
        dl.browser.retrieve(
            f"{BASE_URL}{chosen['download_url']}", str(tmp),
        )
        shutil.move(str(tmp), str(out_path))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if progress:
        size_kb = out_path.stat().st_size / 1024
        print(f"  done: {out_path} ({size_kb:.1f} KB)", file=sys.stderr)
    return out_path
