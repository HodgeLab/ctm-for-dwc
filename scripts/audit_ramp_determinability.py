#!/usr/bin/env python3
"""Audit ramp-flow determinability over PeMS-native stretches.

Consumes ``stretches.csv`` (from ``build_pems_stretches.py`` -- unit stretches
between consecutive mainline detectors, with their on/off ramps, keyed purely by
PeMS ``Abs_PM``) and scans the **entire multi-year timeseries record** for small
windows in which each stretch's ramp flows are determinable. Rather than judging
one fixed window (which makes the corpus artificially small when ramp detectors
are intermittently down), it counts every sliding window of ``--window-size``
consecutive 5-min samples in which the stretch is determinable -- the way Kan
extracts training samples from 3 consecutive points.

For each sliding window a stretch is *determinable* when:

* both bounding mainline detectors are genuinely observed across the window, and
* at most one ramp flow is unobserved across the window (conservation
  ``q_up + sum(r) = q_down + sum(s)`` recovers a single unknown).

"Observed" = non-NaN ``total_flow`` with ``pct_observed`` above the floor for
*every* sample in the window (strict raw). A ramp with no PeMS VDS (a manual
*virtual* string id, or a not-downloaded station) is never observed, so it is a
permanent unknown -- exactly the flow conservation must solve for.

A stretch enters the training corpus when it is type-(c), free of unresolved FF
connectors, and determinable in at least ``--min-windows`` windows (both ramp
flows determinable there -- directly measured, or one measured + one recovered).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from transportation_models.utils.ramp_flow_estimation.observations import sliding_all

_FLOW_COL = "total_flow_[veh/5-min]"
_PCT_COL = "pct_observed"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STRETCHES = _REPO_ROOT / "data/pems/stretches"
_DEFAULT_TIMESERIES = Path(
    "/Volumes/easystore/work/boulder/Caltrans/PeMS/timeseries_data"
)


# --------------------------------------------------------------------------- #
# Pure core.
# --------------------------------------------------------------------------- #
def station_id(value):
    """Normalize one detector id: ``int`` for a PeMS station, ``str`` for a
    manually-added *virtual* detector (a physical ramp/mainline point with no
    PeMS VDS), ``None`` for missing."""
    if value is None:
        return None
    if isinstance(value, str):
        tok = value.strip()
        if not tok or tok.lower() == "nan":
            return None
        return int(tok) if tok.lstrip("+-").isdigit() else tok
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return int(value)


def parse_ids(value) -> list:
    """Parse a ';'-joined id string into station ids (ints for PeMS stations,
    strings for virtual detectors)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    return [sid for tok in s.split(";") if (sid := station_id(tok)) is not None]


def count_determinable_windows(row: dict, win, n_windows_total: int, min_windows: int,
                               fully_determined: bool = False) -> dict:
    """Count sliding windows in which the stretch is determinable.

    ``win(station_id) -> np.ndarray[bool]`` gives each station's window-granular
    observation mask (all-False for virtual / not-downloaded / missing ids).

    By default a window is determinable with <= 1 unmeasured ramp (conservation
    recovers one). With ``fully_determined=True`` every ramp must be measured
    (0 unmeasured) -- the stricter corpus of directly-measured on/off pairs.
    """
    up, down = station_id(row.get("up_ml_id")), station_id(row.get("down_ml_id"))
    on_ids, off_ids = parse_ids(row.get("on_ids")), parse_ids(row.get("off_ids"))
    review_ids = parse_ids(row.get("review_ids"))

    mainline = win(up) & win(down)
    unobserved = np.zeros(n_windows_total, dtype=np.int32)
    for r in on_ids + off_ids:
        unobserved += ~win(r)
    unobserved += len(review_ids)          # unresolved FF: unknown side -> unknown flow
    max_unobserved = 0 if fully_determined else 1
    determinable = mainline & (unobserved <= max_unobserved)
    n_windows = int(determinable.sum())

    config_type = row.get("config_type")
    n_on, n_off = int(row.get("n_on", 0)), int(row.get("n_off", 0))
    has_review = len(review_ids) > 0
    training_corpus = (
        config_type == "c" and not has_review and n_windows >= min_windows
    )
    clean_pair = training_corpus and n_on == 1 and n_off == 1
    return {
        "n_windows": n_windows,
        "has_review": has_review,
        "training_corpus": training_corpus,
        "clean_pair": clean_pair,
    }


# --------------------------------------------------------------------------- #
# IO: load full-record observation masks.
# --------------------------------------------------------------------------- #
def load_observed_series(path: Path, pct_floor: float) -> pd.Series | None:
    """Boolean Series (indexed by timestamp) of genuinely-observed samples for
    one station's full timeseries; None if the file is unreadable."""
    try:
        df = pd.read_csv(
            path, usecols=["timestamp", _FLOW_COL, _PCT_COL], parse_dates=["timestamp"]
        )
    except (ValueError, KeyError, FileNotFoundError):
        return None
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return df[_FLOW_COL].notna() & (df[_PCT_COL].fillna(-1.0) > pct_floor)


def build_observation_windows(
    station_ids, tsdir: Path, pct_floor: float, window: int,
    record_start: pd.Timestamp | None, record_end: pd.Timestamp | None,
):
    """Load each station once, align to a common 5-min grid, and reduce to
    window-granular observation masks. Returns ``(win, n_windows_total, grid,
    missing)`` where ``win(id) -> bool array`` and ``missing`` are ids without a
    readable file."""
    series: dict[int, pd.Series] = {}
    missing: list[int] = []
    for sid in station_ids:
        s = load_observed_series(tsdir / f"{sid}.csv", pct_floor)
        if s is None or s.empty:
            missing.append(sid)
        else:
            series[sid] = s

    if record_start is not None and record_end is not None:
        grid = pd.date_range(record_start, record_end, freq="5min", inclusive="left")
    elif series:
        lo = min(s.index.min() for s in series.values()).floor("5min")
        hi = max(s.index.max() for s in series.values()).ceil("5min")
        grid = pd.date_range(lo, hi, freq="5min")
    else:
        raise ValueError("no readable station files and no --record-start/--record-end")

    n_windows_total = max(len(grid) - window + 1, 0)
    obs_windows: dict[int, np.ndarray] = {
        sid: sliding_all(s.reindex(grid, fill_value=False).to_numpy(), window)
        for sid, s in series.items()
    }
    false_windows = np.zeros(n_windows_total, dtype=bool)

    def win(sid) -> np.ndarray:
        arr = obs_windows.get(sid)
        return arr if arr is not None else false_windows

    return win, n_windows_total, grid, missing


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def _referenced_station_ids(frames: list[pd.DataFrame]) -> list[int]:
    ids: set[int] = set()
    for df in frames:
        for col in ("up_ml_id", "down_ml_id"):
            for v in df[col].map(station_id).dropna():
                if isinstance(v, int):
                    ids.add(v)
        for col in ("on_ids", "off_ids", "review_ids"):
            if col in df.columns:
                for cell in df[col]:
                    ids.update(i for i in parse_ids(cell) if isinstance(i, int))
    return sorted(ids)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stretches", type=Path, default=_DEFAULT_STRETCHES,
                    help="A stretches.csv or a directory of them.")
    ap.add_argument("--timeseries-dir", type=Path, default=_DEFAULT_TIMESERIES)
    ap.add_argument("--pct-observed-floor", type=float, default=0.0,
                    help="A sample counts as observed iff pct_observed > floor (default 0).")
    ap.add_argument("--window-size", type=int, default=3,
                    help="Consecutive 5-min samples per window (Kan uses 3).")
    ap.add_argument("--fully-determined", action="store_true",
                    help="Corpus requires every ramp measured (no conservation-recovered "
                         "ramp); default admits <=1 unmeasured ramp.")
    ap.add_argument("--min-windows", type=int, default=1,
                    help="Determinable windows a stretch needs to enter the corpus.")
    ap.add_argument("--record-start", type=pd.Timestamp, default=None,
                    help="Clip the scan to [start, end); default = full data span.")
    ap.add_argument("--record-end", type=pd.Timestamp, default=None)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "scripts/output/ramp_audit.csv")
    args = ap.parse_args(argv)

    if not args.timeseries_dir.exists():
        warnings.warn(f"timeseries-dir not found: {args.timeseries_dir}", RuntimeWarning)

    csvs = (sorted(args.stretches.glob("*.csv")) if args.stretches.is_dir()
            else [args.stretches])
    if not csvs:
        ap.error(f"no stretches CSV under {args.stretches}")
    frames = {csv.stem: pd.read_csv(csv) for csv in csvs}

    win, n_windows_total, grid, missing = build_observation_windows(
        _referenced_station_ids(list(frames.values())),
        args.timeseries_dir, args.pct_observed_floor, args.window_size,
        args.record_start, args.record_end,
    )
    print(f"record: {grid[0]} .. {grid[-1]}  ({len(grid)} samples, "
          f"{n_windows_total} windows of {args.window_size})")
    if missing:
        print(f"stations without a readable file: {len(missing)}")

    rows = []
    for source, df in frames.items():
        for row in df.to_dict("records"):
            rec = count_determinable_windows(row, win, n_windows_total, args.min_windows,
                                             fully_determined=args.fully_determined)
            rec.update(source=source, stretch_id=row.get("stretch_id"),
                       pm_up=row.get("pm_up"), pm_down=row.get("pm_down"),
                       config_type=row.get("config_type"),
                       n_on=row.get("n_on"), n_off=row.get("n_off"),
                       up_ml_id=row.get("up_ml_id"), down_ml_id=row.get("down_ml_id"),
                       on_ids=row.get("on_ids"), off_ids=row.get("off_ids"))
            rows.append(rec)

    out = pd.DataFrame(rows)[[
        "source", "stretch_id", "pm_up", "pm_down", "config_type", "n_on", "n_off",
        "has_review", "up_ml_id", "down_ml_id", "on_ids", "off_ids",
        "n_windows", "training_corpus", "clean_pair",
    ]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} stretch rows -> {args.out}")
    _print_summary(out)
    return 0


def _print_summary(df: pd.DataFrame) -> None:
    corpus = df[df.training_corpus]
    print("\n=== training corpus (type-(c) with determinable r,s) ===")
    print(f"  corpus stretches       : {len(corpus)}")
    print(f"    clean 1-on/1-off     : {int(corpus.clean_pair.sum())}")
    print(f"    multi-ramp weaves    : {int((~corpus.clean_pair).sum())}")
    print(f"  total training windows : {int(corpus.n_windows.sum())}")
    if len(corpus):
        print("  by source:", corpus.groupby("source").size().to_dict())


if __name__ == "__main__":
    sys.exit(main())
