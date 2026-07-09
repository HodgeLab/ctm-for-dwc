"""Step 9 (ramp): gap-fill accuracy validation for ramp VDS timeseries.

The Step-7 gap fillers (:mod:`utils.ctm.ramp_flow`) reconstruct on-ramp
demand and off-ramp flow from PeMS ramp detectors that suffer frequent
outages. Those reconstructed values feed the CTM scenario directly
(`demand_from_ramp_vds` / `beta_from_off_ramp_vds`), so their accuracy
bounds how well the simulation can match reality -- yet the real gaps
they fill have no ground truth to score against.

This module measures fill accuracy by **synthetic hold-out**: take the
*known* (non-NaN) stretches of a ramp VDS series, blank fixed-length
blocks of them, fill with each strategy, and compare the fill to the
held-out truth. Errors are reported **stratified by gap length** -- the
evidence needed to decide which filler wins at which outage duration
(and to settle the deferred outage-duration-aware composition).

Masking scheme: a **fixed gap-length sweep**. For each gap length in
``gap_lengths_min`` (default 5/15/30/60/120 min), non-overlapping blocks
of that length are laid down on a regular schedule over fully-observed
stretches. Each block is masked **in isolation** (one synthetic outage
at a time on an otherwise-intact copy of the series), so the
historical-average fillers compute their per-(weekday, time-of-day) bin
statistics from the series *minus only that one gap* -- exactly the
masked-stats regime that production filling sees when a real gap occurs.

Two entry points, both returning a :class:`RampFillAccuracy` with a
per-VDS table and a corridor-pooled table:

* :func:`onramp_fill_accuracy` -- scores the on-ramp flow series
  (the ``d_i`` driver) directly.
* :func:`offramp_split_accuracy` -- masks the off-ramp flow ``s_i``,
  fills, then reconstructs the split ratio
  ``beta_i = s_i / (f_i + s_i)`` against the unmasked downstream
  mainline ``f_i`` (mirroring :func:`beta_from_off_ramp_vds`). Reports
  error in **both** flow units (``s``) and split units (``beta``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from .assembly import parse_ramp_vds_ids
from .ramp_flow import (
    historical_average_fill,
    persistence_fill,
)

# A filler bound to a flow column: (df, flow_col) -> filled df.
FlowFiller = Callable[[pd.DataFrame, str], pd.DataFrame]

DEFAULT_GAP_LENGTHS_MIN: tuple[int, ...] = (5, 15, 30, 60, 120)
DEFAULT_GAP_SPACING_MIN: float = 120.0
_FLOW_COL = "total_flow_[veh/5-min]"
_TIME_COL = "timestamp"
_CADENCE_MIN = 5.0


def _persistence(df: pd.DataFrame, flow_col: str) -> pd.DataFrame:
    return persistence_fill(df, flow_col=flow_col)


def _historical_average(df: pd.DataFrame, flow_col: str) -> pd.DataFrame:
    return historical_average_fill(df, flow_col=flow_col)


DEFAULT_STRATEGIES: dict[str, FlowFiller] = {
    "persistence": _persistence,
    "historical_average": _historical_average,
}


@dataclass
class RampFillAccuracy:
    """Gap-fill accuracy, per VDS and pooled across the corridor.

    Both frames carry one row per ``(gap_length_min, strategy)`` group
    (the per-VDS frame additionally per VDS / off-ramp cell). Columns
    use a ``<prefix>_`` namespace per scored quantity -- ``flow_`` for
    the ramp flow series, and (off-ramp only) ``beta_`` for the
    reconstructed split ratio:

    * ``<prefix>_n`` -- scored samples (non-NaN truth *and* fill)
    * ``<prefix>_n_unfilled`` -- masked samples the filler left NaN
      (e.g. persistence with no prior value, or a historical bin with no
      other history)
    * ``<prefix>_rmse`` / ``<prefix>_mape`` / ``<prefix>_bias`` -- error
      across the scored samples (MAPE in %, skipping zero-truth samples;
      bias is mean signed ``fill - truth``)
    """

    per_vds: pd.DataFrame
    pooled: pd.DataFrame


# ---- core sweep ----------------------------------------------------------


def _collect_fill_residuals(
    series_df: pd.DataFrame,
    *,
    flow_col: str,
    time_col: str = _TIME_COL,
    gap_lengths_min: Sequence[int] = DEFAULT_GAP_LENGTHS_MIN,
    strategies: Mapping[str, FlowFiller] = DEFAULT_STRATEGIES,
    gap_spacing_min: float = DEFAULT_GAP_SPACING_MIN,
    cadence_min: float = _CADENCE_MIN,
) -> pd.DataFrame:
    """Mask fixed-length blocks, fill, and return per-sample residuals.

    Returns a long frame with columns ``gap_length_min``, ``strategy``,
    ``gap_id``, ``pos`` (row index into ``series_df``), ``truth``, and
    ``pred``. Only blocks that are fully observed (no NaN truth) are
    masked, so every row has a ground-truth ``truth`` value; ``pred`` is
    NaN when the filler couldn't fill that masked sample.
    """
    df = series_df.reset_index(drop=True)
    truth = df[flow_col].to_numpy(dtype=float)
    n = len(df)
    spacing = max(int(round(gap_spacing_min / cadence_min)), 1)

    chunks: list[pd.DataFrame] = []
    for gap_min in gap_lengths_min:
        gap_len = max(int(round(gap_min / cadence_min)), 1)
        stride = gap_len + spacing
        # Start one spacing in so the first masked block always has prior
        # context (persistence needs a known value before the gap).
        for gap_id, start in enumerate(range(spacing, n - gap_len + 1, stride)):
            sl = slice(start, start + gap_len)
            if np.isnan(truth[sl]).any():
                continue  # need full ground truth to score this block
            masked = df.copy()
            masked.loc[start : start + gap_len - 1, flow_col] = np.nan
            for name, filler in strategies.items():
                with warnings.catch_warnings():
                    # Binned fillers warn on bins with no other history;
                    # expected here and surfaced via <prefix>_n_unfilled.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    filled = filler(masked, flow_col)
                pred = filled[flow_col].to_numpy(dtype=float)[sl]
                chunks.append(
                    pd.DataFrame(
                        {
                            "gap_length_min": gap_min,
                            "strategy": name,
                            "gap_id": gap_id,
                            "pos": np.arange(start, start + gap_len),
                            "truth": truth[sl],
                            "pred": pred,
                        }
                    )
                )

    if not chunks:
        return pd.DataFrame(
            columns=["gap_length_min", "strategy", "gap_id", "pos", "truth", "pred"]
        )
    return pd.concat(chunks, ignore_index=True)


def _metrics(truth: np.ndarray, pred: np.ndarray, prefix: str) -> dict:
    """RMSE / MAPE / bias for one residual group, under a column prefix."""
    have_truth = ~np.isnan(truth)
    valid = have_truth & ~np.isnan(pred)
    n_unfilled = int((have_truth & np.isnan(pred)).sum())
    t = truth[valid]
    p = pred[valid]
    err = p - t
    rmse = float(np.sqrt(np.mean(err**2))) if err.size else float("nan")
    bias = float(np.mean(err)) if err.size else float("nan")
    nz = t != 0
    mape = float(np.mean(np.abs(err[nz] / t[nz])) * 100.0) if nz.any() else float("nan")
    return {
        f"{prefix}_n": int(valid.sum()),
        f"{prefix}_n_unfilled": n_unfilled,
        f"{prefix}_rmse": rmse,
        f"{prefix}_mape": mape,
        f"{prefix}_bias": bias,
    }


def _summarize(
    resid: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_specs: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    """Reduce per-sample residuals to one metrics row per group.

    ``value_specs`` is a list of ``(prefix, truth_col, pred_col)``; each
    contributes its own ``<prefix>_*`` metric columns to every row.
    """
    group_cols = list(group_cols)
    if resid.empty:
        cols = group_cols + [
            f"{prefix}_{stat}"
            for prefix, _, _ in value_specs
            for stat in ("n", "n_unfilled", "rmse", "mape", "bias")
        ]
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    for keys, g in resid.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for prefix, truth_col, pred_col in value_specs:
            row.update(
                _metrics(
                    g[truth_col].to_numpy(dtype=float),
                    g[pred_col].to_numpy(dtype=float),
                    prefix,
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ---- series loading ------------------------------------------------------


def _load_series(
    timeseries_dir: Path, vds_id: int, *, out_col: str, flow_col: str = _FLOW_COL
) -> pd.DataFrame:
    """Load a per-VDS CSV down to ``timestamp`` + a renamed flow column."""
    df = pd.read_csv(timeseries_dir / f"{vds_id}.csv", parse_dates=[_TIME_COL])
    return (
        df[[_TIME_COL, flow_col]]
        .rename(columns={flow_col: out_col})
        .sort_values(_TIME_COL)
        .reset_index(drop=True)
    )


def _load_summed_series(
    timeseries_dir: Path,
    vds_ids: Sequence[int],
    *,
    out_col: str,
    flow_col: str = _FLOW_COL,
) -> pd.DataFrame:
    """Sum one or more ramp VDS flow series onto a shared 5-min grid.

    Mirrors the Step-5 multi-ramp-per-cell convention (sum per-ramp
    flows into the cell's total). A timestamp where every contributing
    VDS is NaN stays NaN (a genuine gap); partial coverage sums the
    present detectors (``min_count=1``).
    """
    if len(vds_ids) == 1:
        return _load_series(timeseries_dir, vds_ids[0], out_col=out_col, flow_col=flow_col)

    merged: pd.DataFrame | None = None
    for vid in vds_ids:
        part = _load_series(timeseries_dir, vid, out_col=f"_f{vid}", flow_col=flow_col)
        merged = part if merged is None else merged.merge(part, on=_TIME_COL, how="outer")
    flow_cols = [c for c in merged.columns if c != _TIME_COL]
    merged[out_col] = merged[flow_cols].sum(axis=1, min_count=1)
    return merged[[_TIME_COL, out_col]].sort_values(_TIME_COL).reset_index(drop=True)


# ---- public entry points -------------------------------------------------


def onramp_fill_accuracy(
    cells_df: pd.DataFrame,
    timeseries_dir: Union[Path, str],
    *,
    gap_lengths_min: Sequence[int] = DEFAULT_GAP_LENGTHS_MIN,
    strategies: Mapping[str, FlowFiller] = DEFAULT_STRATEGIES,
    gap_spacing_min: float = DEFAULT_GAP_SPACING_MIN,
) -> RampFillAccuracy:
    """Gap-fill accuracy for every on-ramp VDS in ``cells_df``.

    For each cell with ``on_ramp=True`` and a non-NaN ``on_ramp_vds_id``,
    the on-ramp flow series (summed across the cell's ramp VDS, per the
    Step-5 multi-ramp convention) is run through the synthetic hold-out
    sweep. A given VDS set is scored once even if it spans multiple
    cells.

    Returns a :class:`RampFillAccuracy`; ``per_vds`` is keyed by
    ``vds_id`` (comma-joined for multi-VDS cells), ``pooled`` pools the
    residuals of every on-ramp VDS per ``(gap_length_min, strategy)``.
    """
    timeseries_dir = Path(timeseries_dir)
    resid_frames: list[pd.DataFrame] = []
    seen: set[tuple[int, ...]] = set()

    for cell_idx, row in enumerate(cells_df.itertuples(index=False)):
        if not bool(getattr(row, "on_ramp", False)):
            continue
        ids = parse_ramp_vds_ids(getattr(row, "on_ramp_vds_id", None))
        if not ids:
            continue
        key = tuple(sorted(ids))
        if key in seen:
            continue
        seen.add(key)

        series = _load_summed_series(timeseries_dir, ids, out_col=_FLOW_COL)
        resid = _collect_fill_residuals(
            series,
            flow_col=_FLOW_COL,
            gap_lengths_min=gap_lengths_min,
            strategies=strategies,
            gap_spacing_min=gap_spacing_min,
        )
        resid.insert(0, "vds_id", ",".join(str(i) for i in ids))
        resid.insert(1, "cell", cell_idx)
        resid_frames.append(resid)

    return _assemble(
        resid_frames,
        per_vds_cols=["vds_id", "cell"],
        value_specs=[("flow", "truth", "pred")],
    )


def offramp_split_accuracy(
    cells_df: pd.DataFrame,
    timeseries_dir: Union[Path, str],
    *,
    gap_lengths_min: Sequence[int] = DEFAULT_GAP_LENGTHS_MIN,
    strategies: Mapping[str, FlowFiller] = DEFAULT_STRATEGIES,
    gap_spacing_min: float = DEFAULT_GAP_SPACING_MIN,
) -> RampFillAccuracy:
    """Gap-fill accuracy for off-ramp flow and the reconstructed split.

    For each cell with ``off_ramp=True`` and a non-NaN
    ``off_ramp_vds_id`` (and a downstream mainline VDS to pair with), the
    off-ramp flow ``s_i`` is masked + filled by the sweep, then the split
    ``beta_i = s_i / (f_i + s_i)`` is reconstructed against the unmasked
    downstream mainline flow ``f_i`` (cell ``i+1``'s ``vds_id``), exactly
    as :func:`beta_from_off_ramp_vds` forms it at sim time.

    The off-ramp and mainline series are inner-joined on timestamp, so
    the split is scored only where both detectors report. Off-ramp cells
    at the corridor's downstream end (no ``i+1``), or whose downstream
    neighbor has no ``vds_id``, are skipped with a ``UserWarning``.

    Returns a :class:`RampFillAccuracy` with both ``flow_`` (off-ramp
    flow ``s``) and ``beta_`` (split ratio) metric columns.
    """
    timeseries_dir = Path(timeseries_dir)
    last_cell_idx = len(cells_df) - 1
    resid_frames: list[pd.DataFrame] = []

    for cell_idx, row in enumerate(cells_df.itertuples(index=False)):
        if not bool(getattr(row, "off_ramp", False)):
            continue
        off_ids = parse_ramp_vds_ids(getattr(row, "off_ramp_vds_id", None))
        if not off_ids:
            continue
        if cell_idx == last_cell_idx:
            warnings.warn(
                f"Cell {cell_idx} is the corridor's last cell with "
                "off_ramp=True; no downstream mainline VDS to reconstruct "
                "beta = s/(f+s). Skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        downstream_raw = cells_df.iloc[cell_idx + 1]["vds_id"]
        if pd.isna(downstream_raw):
            warnings.warn(
                f"Cell {cell_idx + 1} (downstream of off-ramp cell "
                f"{cell_idx}) has no vds_id; skipping split reconstruction.",
                UserWarning,
                stacklevel=2,
            )
            continue
        downstream_vds = int(downstream_raw)

        s = _load_summed_series(timeseries_dir, off_ids, out_col="s")
        f = _load_series(timeseries_dir, downstream_vds, out_col="f")
        merged = s.merge(f, on=_TIME_COL, how="inner")

        resid = _collect_fill_residuals(
            merged,
            flow_col="s",
            gap_lengths_min=gap_lengths_min,
            strategies=strategies,
            gap_spacing_min=gap_spacing_min,
        )
        resid = resid.rename(columns={"truth": "s_truth", "pred": "s_pred"})

        f_vals = merged["f"].to_numpy(dtype=float)[resid["pos"].to_numpy()]
        denom_t = f_vals + resid["s_truth"].to_numpy(dtype=float)
        denom_p = f_vals + resid["s_pred"].to_numpy(dtype=float)
        f_ok = np.isfinite(f_vals)
        with np.errstate(divide="ignore", invalid="ignore"):
            resid["beta_truth"] = np.where(
                f_ok & (denom_t > 0), resid["s_truth"] / denom_t, np.nan
            )
            resid["beta_pred"] = np.where(
                f_ok & (denom_p > 0), resid["s_pred"] / denom_p, np.nan
            )

        resid.insert(0, "off_ramp_vds_id", ",".join(str(i) for i in off_ids))
        resid.insert(1, "downstream_vds_id", downstream_vds)
        resid.insert(2, "cell", cell_idx)
        resid_frames.append(resid)

    return _assemble(
        resid_frames,
        per_vds_cols=["off_ramp_vds_id", "downstream_vds_id", "cell"],
        value_specs=[("flow", "s_truth", "s_pred"), ("beta", "beta_truth", "beta_pred")],
    )


def _assemble(
    resid_frames: Sequence[pd.DataFrame],
    *,
    per_vds_cols: Sequence[str],
    value_specs: Sequence[tuple[str, str, str]],
) -> RampFillAccuracy:
    """Build the per-VDS + pooled summaries from tagged residual frames."""
    group = ["gap_length_min", "strategy"]
    if not resid_frames:
        empty = _summarize(
            pd.DataFrame(), group_cols=group, value_specs=value_specs
        )
        per_vds = _summarize(
            pd.DataFrame(),
            group_cols=list(per_vds_cols) + group,
            value_specs=value_specs,
        )
        return RampFillAccuracy(per_vds=per_vds, pooled=empty)

    resid = pd.concat(resid_frames, ignore_index=True)
    per_vds = _summarize(
        resid, group_cols=list(per_vds_cols) + group, value_specs=value_specs
    )
    pooled = _summarize(resid, group_cols=group, value_specs=value_specs)
    return RampFillAccuracy(per_vds=per_vds, pooled=pooled)
