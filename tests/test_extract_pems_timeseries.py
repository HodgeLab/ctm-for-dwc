"""Smoke test for ``scripts/extract_pems_timeseries.py`` (Step 3b CLI)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from extract_pems_timeseries import main  # noqa: E402
from test_pems_extractor import _ts_range, _write_pems_gz  # noqa: E402


def test_extracts_requested_detectors_within_date_window(tmp_path):
    for day in ("2022-01-15", "2022-01-16", "2022-01-17"):
        _write_pems_gz(tmp_path / f"d4_{day}.gz",
                       timestamps=_ts_range(f"{day} 08:00:00", 4), station=400839)
    _write_pems_gz(tmp_path / "other.gz",
                   timestamps=_ts_range("2022-01-15 08:00:00", 4), station=400840)

    main(["--gz-dir", str(tmp_path), "--detectors", "400839",
          "--start-date", "2022-01-16", "--end-date", "2022-01-17"])

    out_dir = tmp_path / "csv_files"
    assert sorted(p.name for p in out_dir.iterdir()) == ["400839.csv"]
    ts = pd.to_datetime(pd.read_csv(out_dir / "400839.csv")["timestamp"])
    assert len(ts) == 4
    assert (ts.dt.date == pd.Timestamp("2022-01-16").date()).all()
