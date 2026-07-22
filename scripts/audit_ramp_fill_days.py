#!/usr/bin/env python3
"""Audit ramp-fill demand per day across a date range for a fixed cell set.

Reproduces the single-day ramp-fill provenance (Q1-Q3) for every day in a
range and ranks days by the share of ramp cells that required any fill -- a
tool for finding the cleanest days of a corridor to promote for further
analysis.

For each ramp cell (from ``cells.csv``) and each ramp detector, the fill
method per 5-min sample follows ``build_ctm_ramp_scenario.py``:

* a detector with data *somewhere* in its timeseries is gap-filled per day --
  ``persistence`` for a gap of <=2 samples with a prior value, else
  ``historical_average``;
* a detector with no data at all (virtual ``v`` ids, or a dead real id) is
  synthetic every day -- ``conservation`` on a type-(a)/(b) stretch, else
  ``estimator``.

A cell "requires fill" on a day if any of its ramp samples that day is not
directly measured. Cells whose ramp has no detector ever therefore require
fill on *every* day (a constant floor); ``n_cells_fill_datagap`` excludes them
so the day-to-day signal is visible, while ``share_cells_fill`` is over all
ramp cells.

Each detector's timeseries is read once and sliced per day, so the whole
range costs one pass over the CSVs (not one pass per day).

Run from the repo root::

    python scripts/audit_ramp_fill_days.py \\
        --cells case_studies/I880N_10mi/cells.csv \\
        --stretches data/pems/stretches/880_N.csv \\
        --start-date 2022-01-01 --end-date 2025-01-01 \\
        --out case_studies/I880N_10mi/ramp_fill_by_day.csv --top 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from transportation_models.utils import constants
from transportation_models.utils.ctm.assembly import parse_ramp_vds_ids
from build_ctm_ramp_scenario import _has_any_data, _vds_to_stretch_map

_FLOW = "total_flow_[veh/5-min]"
_TS = "timestamp"
_SLOTS = 288  # 5-min samples per day

# label codes
MEASURED, PERSIST, HISTAVG, CONSERV, ESTIM, UNMAPPED = range(6)
_FILL_NAMES = {PERSIST: "persistence", HISTAVG: "historical_average",
               CONSERV: "conservation", ESTIM: "estimator", UNMAPPED: "unmapped"}


def _classify_full(raw: np.ndarray) -> np.ndarray:
    """Per-sample label codes for a real detector over the full grid.

    ``measured`` where present; a missing run of <=2 samples with a prior
    observed value is ``persistence``; every other missing sample (longer or
    leading run) is ``historical_average``.
    """
    miss = np.isnan(raw)
    code = np.where(miss, HISTAVG, MEASURED).astype(np.int8)
    if not miss.any():
        return code
    edges = np.diff(miss.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if miss[0]:
        starts = np.r_[0, starts]
    if miss[-1]:
        ends = np.r_[ends, len(raw)]
    for s, e in zip(starts, ends):
        if (e - s) <= 2 and s > 0:      # short gap with a prior value
            code[s:e] = PERSIST
    return code


def _absent_code(vid, stretch_map) -> int:
    stretch = stretch_map.get(vid)
    if stretch is None:
        return UNMAPPED
    return CONSERV if str(stretch["config_type"]) in ("a", "b") else ESTIM


def _detector_codes(vid, ts_dir, grid) -> tuple[str, np.ndarray]:
    """(kind, per-sample code array over grid) for one ramp detector."""
    df = pd.read_csv(ts_dir / f"{vid}.csv", usecols=[_TS, _FLOW], parse_dates=[_TS])
    raw = (df.drop_duplicates(_TS).set_index(_TS).reindex(grid)[_FLOW]
           .to_numpy(dtype=float))
    return "measured", _classify_full(raw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--stretches", type=Path, required=True)
    ap.add_argument("--timeseries-dir", type=Path,
                    default=constants.CALTRANS_VDS_TIMESERIES_DIRECTORY_PATH)
    ap.add_argument("--start-date", default="2022-01-01")
    ap.add_argument("--end-date", default="2025-01-01",
                    help="Exclusive; the range must be whole days.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=30,
                    help="Print this many lowest-fill-share days.")
    args = ap.parse_args(argv)

    grid = pd.date_range(args.start_date, args.end_date, freq="5min", inclusive="left")
    if len(grid) % _SLOTS:
        ap.error("date range must be a whole number of days")
    n_days = len(grid) // _SLOTS
    days = pd.date_range(args.start_date, periods=n_days, freq="D")

    cells = pd.read_csv(args.cells)
    stretches = pd.read_csv(args.stretches)
    smap = {"on": _vds_to_stretch_map(stretches, "on_ids"),
            "off": _vds_to_stretch_map(stretches, "off_ids")}
    ts_dir = args.timeseries_dir

    # Build per-cell detector code matrices (n_days, 288), reading each CSV once.
    ramp_cells: list[int] = []
    cell_dets: dict[int, list[np.ndarray]] = {}
    structural: set[int] = set()   # cells with an always-absent ramp side
    for cell_idx, row in cells.iterrows():
        dets = []
        for side in ("on", "off"):
            if not bool(row[f"{side}_ramp"]):
                continue
            for vid in parse_ramp_vds_ids(row.get(f"{side}_ramp_vds_id")):
                real = isinstance(vid, int) and _has_any_data(vid, ts_dir)
                if real:
                    _, code = _detector_codes(vid, ts_dir, grid)
                else:
                    code = np.full(len(grid), _absent_code(vid, smap[side]), np.int8)
                    structural.add(cell_idx)
                dets.append(code.reshape(n_days, _SLOTS))
        if dets:
            ramp_cells.append(cell_idx)
            cell_dets[cell_idx] = dets
    print(f"{len(ramp_cells)} ramp cells; {len(structural)} always-absent "
          f"(structural) cell(s): {sorted(structural)}", file=sys.stderr)

    # Per-day aggregation.
    n_cells = len(ramp_cells)
    fill_any = np.zeros((n_days, n_cells), bool)
    cell_share = np.zeros((n_days, n_cells))
    method_samples = {c: np.zeros(n_days, int) for c in _FILL_NAMES}
    for ci, cell_idx in enumerate(ramp_cells):
        filled = np.zeros(n_days, int)
        total = 0
        for day in cell_dets[cell_idx]:
            filled += (day != MEASURED).sum(axis=1)
            fill_any[:, ci] |= (day != MEASURED).any(axis=1)
            total += _SLOTS
            for code in _FILL_NAMES:
                method_samples[code] += (day == code).sum(axis=1)
        cell_share[:, ci] = filled / total

    n_cells_fill = fill_any.sum(axis=1)
    struct_mask = np.array([c in structural for c in ramp_cells])
    out = pd.DataFrame({
        "date": days.strftime("%Y-%m-%d"),
        "weekday": days.day_name().str[:3],
        "n_cells_fill": n_cells_fill,
        "share_cells_fill": np.round(n_cells_fill / n_cells, 4),
        "n_cells_fill_datagap": fill_any[:, ~struct_mask].sum(axis=1),
        "mean_cell_fill_share": np.round(cell_share.mean(axis=1), 4),
        **{name: method_samples[code] for code, name in _FILL_NAMES.items()},
    })
    out = out.sort_values(["share_cells_fill", "mean_cell_fill_share", "date"]) \
             .reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    floor = int(struct_mask.sum())
    print(f"\n{n_days} days scanned. Structural floor = {floor}/{n_cells} cells "
          f"always require fill ({floor / n_cells:.0%}).", file=sys.stderr)
    print(f"\nTop {args.top} cleanest days (lowest share of cells requiring fill):")
    cols = ["date", "weekday", "n_cells_fill", "share_cells_fill",
            "n_cells_fill_datagap", "mean_cell_fill_share"]
    print(out.head(args.top)[cols].to_string(index=False))
    print(f"\nfull ranking -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
