"""Impute unobserved cells of a diffsim bundle's observed density/flow grids.

The differentiable ramp optimizer (`optimize_ramp_flows_diffsim.py`) scores the
mainline density+flow only at cells with a direct VDS, leaving unobserved cells
free to drift to unrealistic values. This step fills the full
``(n_cells, n_5min)`` grid from the PeMS measurements before scoring
(docs/ctm_module.md, Step 7, "Refinement with diffsim"):

* **Type 2** (a direct-VDS cell missing at some 5-min steps) is filled from the
  detector's own ``(day-of-week, time-of-day)`` history;
* **Type 1** (a cell with no direct VDS -- a full missing column) is filled by a
  spatial-temporal KNN over the surrounding observed and Type-2-filled cells.

Both fills live in ``ctm.impute``. Output is written per neighborhood
config to ``<bundle>/imputed/k{K}_d{decay}/observed_{density,flow}.csv`` with
columns ``cell, m_5min, value, source`` (``source`` in
``observed``/``hist``/``knn``). The imputed *values* depend on ``--k`` and
``--knn-decay`` only; how much they count in the loss is the optimizer's
``--imputed-weight``, so a weight sweep needs no re-augmentation.

    python scripts/augment_observed_grid.py \
        --bundle case_studies/I880N_5mi/output/diffsim_inputs \
        --cells  case_studies/I880N_5mi/cells.csv \
        --timeseries-dir data/pems/csv_files \
        --k 2 --knn-decay 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ctm_for_dwc.ctm.impute import (
    historical_slot_fill,
    knn_impute_grid,
)

_DIRECT_SOURCES = ("direct", "direct_tiebreak")
_QUANTITIES = ("density", "flow")


def _dense_grid(path: Path, value_col: str, n_cells: int, n_5min: int):
    """Sparse ``(cell, m_5min, value)`` CSV -> dense grid + present mask."""
    grid = np.full((n_cells, n_5min), np.nan)
    present = np.zeros((n_cells, n_5min), dtype=bool)
    df = pd.read_csv(path)
    for cell, m, val in zip(df["cell"], df["m_5min"], df[value_col]):
        grid[int(cell), int(m)] = float(val)
        present[int(cell), int(m)] = True
    return grid, present


def _vds_series(timeseries_dir: Path, vds_id: int, cache: dict):
    """Full ``(timestamps, flow_veh_h, density_veh_mi, observed)`` for a VDS.

    Same flow/density definition as the bundle builder's ``_load_observed_window``
    (``flow = total_flow * 12``, ``density = flow / speed``); ``observed`` is
    where density is finite.
    """
    if vds_id not in cache:
        cache[vds_id] = pd.read_csv(
            timeseries_dir / f"{vds_id}.csv", parse_dates=["timestamp"],
        ).sort_values("timestamp").reset_index(drop=True)
    df = cache[vds_id]
    flow = df["total_flow_[veh/5-min]"].to_numpy(dtype=float) * 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = df["avg_speed_[mph]"].to_numpy(dtype=float)
        density = np.where(speed > 0, flow / speed, np.nan)
    observed = np.isfinite(density) & np.isfinite(flow)
    return df["timestamp"], flow, density, observed


def _write_contours(out_dir, cells_df, raw, filled, n_5min, k, knn_decay):
    """Raw-vs-augmented space-time contours per quantity, for visual inspection.

    Reuses ``ctm.plots.plot_flow_density_contour`` (matplotlib imported
    lazily so ``--no-plot`` runs stay light). Raw grids show unobserved cells in
    black; raw and augmented share a color scale per quantity so the fill is
    directly comparable.
    """
    from ctm_for_dwc.ctm.plots import plot_flow_density_contour

    n = next(iter(filled.values())).shape[0]
    if {"pm_start", "pm_end"} <= set(cells_df.columns):
        pm_edges = np.concatenate([cells_df["pm_start"].to_numpy(dtype=float),
                                   [float(cells_df["pm_end"].iloc[-1])]])
        x_label = "Distance from upstream end [mi]"
    else:
        pm_edges = np.arange(n + 1, dtype=float)
        x_label = "Cell index"
    horizon_h = n_5min / 12.0
    styles = {"density": ("plasma", "Density [veh/mi]"),
              "flow": ("viridis", "Flow [veh/h]")}
    for q, (cmap, cbar) in styles.items():
        finite = filled[q][np.isfinite(filled[q])]
        vmin = float(finite.min()) if finite.size else None
        vmax = float(finite.max()) if finite.size else None
        for label, grid in (("raw", raw[q]), ("augmented", filled[q])):
            plot_flow_density_contour(
                grid.T,
                cbar_label=cbar, cmap=cmap, horizon_h=horizon_h,
                pm_edges=pm_edges, x_label=x_label,
                y_label="Window time [h]",
                vmin=vmin, vmax=vmax, missing_color="black",
                out_path=out_dir / f"{label}_{q}_contour.png",
            )


def augment_bundle(
    bundle: Path,
    cells_path: Path,
    timeseries_dir: Path,
    *,
    k: int,
    knn_decay: float,
    hist_start: pd.Timestamp | None = None,
    hist_end: pd.Timestamp | None = None,
    out_dir: Path | None = None,
    plot: bool = False,
) -> dict:
    """Fill Type-2 then Type-1 cells; write the augmented grids. Return counts.

    ``plot`` writes raw-vs-augmented contours next to the CSVs (needs
    ``pm_start``/``pm_end`` in ``cells.csv`` for a distance axis, else falls back
    to cell index).
    """
    bundle = Path(bundle)
    meta = json.loads((bundle / "meta.json").read_text())
    n, n_5min = int(meta["n_cells"]), int(meta["n_5min"])
    start = pd.Timestamp(meta["start"])
    window_ts = pd.date_range(start, periods=n_5min, freq="5min")

    cells_df = pd.read_csv(cells_path)
    if len(cells_df) != n:
        raise SystemExit(
            f"cells.csv has {len(cells_df)} rows but meta.json has n_cells={n}."
        )
    is_direct = cells_df["vds_source"].isin(_DIRECT_SOURCES).to_numpy()
    type1_cells = np.flatnonzero(~is_direct)

    grids = {
        "density": _dense_grid(bundle / "observed_density.csv", "rho_obs", n, n_5min),
        "flow": _dense_grid(bundle / "observed_flow.csv", "flow_obs", n, n_5min),
    }
    values = {q: g for q, (g, _) in grids.items()}
    present = {q: p for q, (_, p) in grids.items()}
    raw = {q: values[q].copy() for q in _QUANTITIES}  # observed-only snapshot for plots
    # source codes: 0 unfilled, 1 observed, 2 hist, 3 knn
    source = {q: np.where(present[q], 1, 0).astype(np.int8) for q in _QUANTITIES}

    # --- Type 2: fill a direct cell's in-window gaps from its own history ---
    cache: dict = {}
    for cell in np.flatnonzero(is_direct):
        gap = ~present["density"][cell] | ~present["flow"][cell]
        if not gap.any():
            continue
        vds_id = int(cells_df["vds_id"].iloc[cell])
        ts, flow, density, observed = _vds_series(timeseries_dir, vds_id, cache)
        if hist_start is not None or hist_end is not None:
            keep = np.ones(len(ts), dtype=bool)
            if hist_start is not None:
                keep &= ts.to_numpy() >= np.datetime64(hist_start)
            if hist_end is not None:
                keep &= ts.to_numpy() < np.datetime64(hist_end)
            ts, flow, density, observed = ts[keep], flow[keep], density[keep], observed[keep]
        for q, series in (("density", density), ("flow", flow)):
            filled, _ = historical_slot_fill(series, observed, ts)
            filled_win = pd.Series(filled, index=ts).reindex(window_ts).to_numpy()
            need = ~present[q][cell] & np.isfinite(filled_win)
            values[q][cell, need] = filled_win[need]
            present[q][cell, need] = True
            source[q][cell, need] = 2

    # --- Type 1: KNN over observed + Type-2-filled neighbors ---
    for q in _QUANTITIES:
        target = np.zeros((n, n_5min), dtype=bool)
        target[type1_cells] = ~present[q][type1_cells]
        filled, imputed = knn_impute_grid(
            values[q], usable=present[q], target=target, k=k, decay=knn_decay,
        )
        values[q] = filled
        present[q] |= imputed
        source[q][imputed] = 3

    if out_dir is None:
        out_dir = bundle / "imputed" / f"k{k}_d{knn_decay}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = {1: "observed", 2: "hist", 3: "knn"}
    counts = {}
    for q in _QUANTITIES:
        rows = []
        for cell in range(n):
            for m in range(n_5min):
                code = source[q][cell, m]
                if code:
                    rows.append({"cell": cell, "m_5min": m,
                                 "value": values[q][cell, m], "source": labels[code]})
        pd.DataFrame(rows, columns=["cell", "m_5min", "value", "source"]).to_csv(
            out_dir / f"observed_{q}.csv", index=False,
        )
        counts[q] = {name: int((source[q] == code).sum())
                     for code, name in labels.items()}
        counts[q]["unfilled"] = int(n * n_5min - present[q].sum())

    config = {"k": k, "knn_decay": knn_decay,
              "hist_start": str(hist_start) if hist_start is not None else None,
              "hist_end": str(hist_end) if hist_end is not None else None,
              "n_type1_cells": int(len(type1_cells)), "counts": counts}
    (out_dir / "impute_meta.json").write_text(json.dumps(config, indent=2))
    if plot:
        _write_contours(out_dir, cells_df, raw, values, n_5min, k, knn_decay)
    config["out_dir"] = str(out_dir)
    return config


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bundle", required=True, type=Path,
                   help="Diffsim bundle dir (has meta.json + observed_*.csv).")
    p.add_argument("--cells", required=True, type=Path,
                   help="cells.csv (vds_id / vds_source), one row per cell.")
    p.add_argument("--timeseries-dir", required=True, type=Path,
                   help="Dir of per-VDS timeseries CSVs (<vds_id>.csv).")
    p.add_argument("--k", type=int, required=True, choices=(1, 2, 3),
                   help="KNN Manhattan radius for Type-1 fill.")
    p.add_argument("--knn-decay", type=float, default=1.0,
                   help="Exponential decay rate lambda for KNN weights.")
    p.add_argument("--hist-start", type=pd.Timestamp, default=None,
                   help="Restrict the Type-2 history to >= this time (default: full file).")
    p.add_argument("--hist-end", type=pd.Timestamp, default=None,
                   help="Restrict the Type-2 history to < this time (default: full file).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <bundle>/imputed/k{K}_d{decay}.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the raw-vs-augmented contour plots.")
    args = p.parse_args()

    cfg = augment_bundle(
        args.bundle, args.cells, args.timeseries_dir,
        k=args.k, knn_decay=args.knn_decay,
        hist_start=args.hist_start, hist_end=args.hist_end, out_dir=args.out_dir,
        plot=not args.no_plot,
    )

    def line(q):
        c = cfg["counts"][q]
        return (f"  {q:8s}: {c['observed']} observed, {c['hist']} hist-filled, "
                f"{c['knn']} knn-filled, {c['unfilled']} unfilled")

    print(
        f"\nAugmented grids written to {cfg['out_dir']}\n"
        f"  k = {cfg['k']}, knn_decay = {cfg['knn_decay']}, "
        f"{cfg['n_type1_cells']} Type-1 cell(s)\n"
        f"{line('density')}\n{line('flow')}"
        + ("" if args.no_plot else
           "\n  contours     : {raw,augmented}_{density,flow}_contour.png")
    )


if __name__ == "__main__":
    main()
