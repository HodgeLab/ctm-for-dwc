#!/usr/bin/env python3
"""Build PeMS-postmile-native ramp stretches for a freeway direction.

A *stretch* (a.k.a. Kan/Zhang "unit stretch") is the span between two
consecutive mainline (``ML``) detectors along a ``(freeway, direction)``,
together with the on-ramp (``OR``), off-ramp (``FR``), and freeway connector
(``FF``) detectors whose postmiles fall between them. Everything comes from PeMS
station metadata -- ``Abs_PM`` for ordering and ``Type`` for role -- so stretch
identification is independent of the CTM cell discretization (which snaps ramps
to cell edges and filters detectors through cell assignment).

The output ``stretches.csv`` is a **reviewable checkpoint**, like ``cells.csv``:
inspect and fix the flagged rows (FF connectors whose on/off flow the Name can't
resolve, open end stretches, multi-ramp interchanges) before the ramp-flow
determinability audit consumes it.

Role assignment
---------------
* ``OR`` -> on-flow, ``FR`` -> off-flow (authoritative from Type).
* ``FF`` -> parse ``Name``: "off"/"to X" => off, "on"/"from X" => on, else
  ``review`` (left out of the on/off counts, flagged for manual classification).
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_METADATA = _REPO_ROOT / "data/pems/station_metadata.csv"
_DEFAULT_OUT_DIR = _REPO_ROOT / "data/pems/stretches"
_RAMP_TYPES = ("OR", "FR", "FF")
_ON_RE = re.compile(r"\bon\b", re.I)
_OFF_RE = re.compile(r"\boff\b", re.I)
_TO_RE = re.compile(r"\bto\b", re.I)
_FROM_RE = re.compile(r"\bfrom\b", re.I)


def classify_flow(station_type: str, name: object) -> str:
    """Return 'on', 'off', or 'review' for a ramp/connector station."""
    if station_type == "OR":
        return "on"
    if station_type == "FR":
        return "off"
    # FF (or anything unexpected): lean on the Name.
    n = str(name or "")
    if _OFF_RE.search(n) or _TO_RE.search(n):
        return "off"
    if _ON_RE.search(n) or _FROM_RE.search(n):
        return "on"
    return "review"


def build_stretches(meta: pd.DataFrame, fwy: int, direction: str) -> pd.DataFrame:
    """One row per stretch between consecutive ML detectors on ``(fwy, direction)``.

    ``meta`` is PeMS station metadata (``ID, Fwy, Dir, Abs_PM, Type, Name``).
    Ramps upstream of the first / downstream of the last ML detector become
    open end stretches (one boundary ``<NA>``), flagged in ``notes``.
    """
    d = meta[(meta["Fwy"] == fwy) & (meta["Dir"] == direction)]
    d = d[d["Type"].isin(("ML",) + _RAMP_TYPES)].dropna(subset=["Abs_PM"])
    d = d.sort_values("Abs_PM").reset_index(drop=True)

    ml = d[d["Type"] == "ML"].reset_index(drop=True)
    ml_pms = ml["Abs_PM"].to_numpy()
    # Traffic runs toward increasing Abs_PM for N, decreasing for S; so the
    # upstream boundary is the lower-PM ML for N and the higher-PM ML for S.
    downstream_is_higher_pm = direction == "N"

    # Bucket ramps by the ML gap they fall in (index of the lower-PM boundary).
    buckets: dict[int | None, list[dict]] = {}
    for row in d[d["Type"].isin(_RAMP_TYPES)].itertuples(index=False):
        j = int(ml_pms.searchsorted(row.Abs_PM, side="left"))  # first ML pm >= ramp pm
        if j == 0:
            key: int | None = -1            # upstream of first ML (open)
        elif j >= len(ml):
            key = len(ml) - 1               # downstream of last ML (open)
        else:
            key = j - 1                     # interior gap [ml[j-1], ml[j]]
        buckets.setdefault(key, []).append(
            {"id": int(row.ID), "flow": classify_flow(row.Type, row.Name),
             "type": row.Type, "name": row.Name, "pm": float(row.Abs_PM)}
        )

    rows: list[dict] = []
    for key in sorted(buckets, key=lambda k: (k is None, k)):
        ramps = buckets[key]
        lo = ml.iloc[key] if 0 <= key < len(ml) else None
        hi = ml.iloc[key + 1] if 0 <= key + 1 < len(ml) else None

        if key == -1:                       # open upstream end
            lo, hi = None, (ml.iloc[0] if len(ml) else None)
        elif key == len(ml) - 1 and hi is None:  # open downstream end
            lo, hi = (ml.iloc[-1] if len(ml) else None), None

        low_ml, high_ml = lo, hi            # low_ml.Abs_PM < high_ml.Abs_PM
        if downstream_is_higher_pm:
            up, down = low_ml, high_ml
        else:
            up, down = high_ml, low_ml

        on_ids = [r["id"] for r in ramps if r["flow"] == "on"]
        off_ids = [r["id"] for r in ramps if r["flow"] == "off"]
        review_ids = [r["id"] for r in ramps if r["flow"] == "review"]
        n_on, n_off = len(on_ids), len(off_ids)
        config_type = "c" if (n_on and n_off) else "a" if n_on else "b" if n_off else "mainline"

        notes = []
        if up is None or down is None:
            notes.append("open_end")
        if review_ids:
            notes.append(f"ff_review({';'.join(map(str, review_ids))})")
        if n_on > 1 or n_off > 1:
            notes.append("multi_ramp")

        rows.append({
            "fwy": fwy, "dir": direction,
            "pm_up": up["Abs_PM"] if up is not None else pd.NA,
            "pm_down": down["Abs_PM"] if down is not None else pd.NA,
            "up_ml_id": int(up["ID"]) if up is not None else pd.NA,
            "down_ml_id": int(down["ID"]) if down is not None else pd.NA,
            "on_ids": ";".join(map(str, on_ids)),
            "off_ids": ";".join(map(str, off_ids)),
            "review_ids": ";".join(map(str, review_ids)),
            "n_on": n_on, "n_off": n_off, "config_type": config_type,
            "notes": ";".join(notes),
        })

    out = pd.DataFrame(rows)
    # Order geographically by the lower boundary postmile (direction-agnostic;
    # up/down columns still carry the travel orientation for conservation).
    lo_pm = out[["pm_up", "pm_down"]].apply(pd.to_numeric, errors="coerce").min(axis=1)
    out = out.iloc[lo_pm.argsort(kind="stable").to_numpy()].reset_index(drop=True)
    out.insert(0, "stretch_id", range(len(out)))
    return out


def _warn_review(df: pd.DataFrame, fwy: int, direction: str) -> None:
    flagged = df[df["notes"].str.len() > 0]
    for r in flagged.itertuples():
        warnings.warn(f"{fwy}{direction} stretch {r.stretch_id}: {r.notes}", RuntimeWarning)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metadata", type=Path, default=_DEFAULT_METADATA)
    ap.add_argument("--fwy", type=int, required=True, help="Freeway number, e.g. 880.")
    ap.add_argument("--dir", dest="direction", choices=["N", "S", "E", "W"], default=None,
                    help="Direction; omit to build every direction present.")
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)

    meta = pd.read_csv(args.metadata)
    directions = [args.direction] if args.direction else sorted(
        meta.loc[meta["Fwy"] == args.fwy, "Dir"].dropna().unique()
    )
    if not directions:
        ap.error(f"no stations for Fwy {args.fwy} in {args.metadata}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for direction in directions:
        df = build_stretches(meta, args.fwy, direction)
        out = args.out_dir / f"{args.fwy}_{direction}.csv"
        df.to_csv(out, index=False)
        _warn_review(df, args.fwy, direction)
        counts = df["config_type"].value_counts().to_dict()
        print(f"{args.fwy}{direction}: {len(df)} stretches -> {out}  types={counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
