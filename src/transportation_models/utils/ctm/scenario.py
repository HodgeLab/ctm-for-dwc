"""Step 8: data-shaping adapters that turn PeMS VDS timeseries into the
wide-format inputs :func:`utils.ctm.io.scenario_from_dataframes` expects.

Two pieces live here today:

* :func:`inflow_from_vds`        -- upstream boundary inflow ``f_0(k)`` from
                                    a single per-VDS 5-minute CSV, sliced
                                    to a user-supplied time window and
                                    resampled to sim cadence.
* :func:`initial_state_from_vds` -- initial density ``\\rho_i(0)`` per cell
                                    from the cell's mainline VDS at a
                                    single timestamp.

The on-ramp demand (``demand``) and off-ramp split-ratio (``beta``)
adapters are intentionally deferred to Step 7 ("Ramp Flow Estimation"); see
``docs/ctm_module.md`` Step 7 for the punch list and target signatures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

_FIVE_MIN_H = 5.0 / 60.0   # 5 minutes in hours
_ONE_HOUR_H = 1.0
_FIVE_MIN_S = 300.0
_ONE_HOUR_S = 3600.0
_GRID_TOL = 1e-9           # float tolerance for "evenly divides" checks


# ---- Upstream boundary inflow --------------------------------------------


def inflow_from_vds(
    vds_csv_path: Union[Path, str],
    dt: float,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    """Upstream boundary inflow ``f_0(k)`` from a Step-3 per-VDS CSV.

    Reads the 5-minute timeseries written by ``PeMSExtractor`` for a single
    VDS (typically the corridor's upstream-most mainline detector), slices
    to the half-open window ``[start, end)``, and resamples to sim
    cadence. The output is in veh/h and is ready to pass as ``inflow`` to
    :func:`utils.ctm.io.scenario_from_dataframes`.

    Resampling strategy (no intermediate ``× 12`` round-trip)
    ---------------------------------------------------------
    The PeMS native sampling cadence is 5 minutes; the column
    ``total_flow_[veh/5-min]`` carries the count over each 5-minute
    interval. We resample to sim cadence directly:

    * **``dt`` ≤ 5 min** (the common case): constant-hold each 5-minute
      sample across the ``300 / dt_s`` sim steps it covers. The single
      unit change ``× 12 -> veh/h`` happens once at output time.
    * **``dt`` > 5 min**: aggregate 5-minute counts to **hourly** by
      summing the 12 samples in each hour. The hourly sum is *already*
      veh/h (counts per hour), so no further factor is applied -- we
      then constant-hold from hourly to sim cadence.

    Parameters
    ----------
    vds_csv_path : Path or str
        Per-VDS timeseries CSV (the file ``PeMSExtractor`` writes).
        Must carry a ``timestamp`` column (parseable as datetime64) and
        a ``total_flow_[veh/5-min]`` column.
    dt : float
        Sim time step in hours. Pick this from Step 6's CFL advisory.
        Must evenly divide 5 minutes when ``dt <= 5 min``, or 1 hour
        when ``dt > 5 min``; otherwise the resampling has no integer
        constant-hold length.
    start, end : pandas.Timestamp
        Calibration window. ``start`` is inclusive, ``end`` is exclusive.
        Both must align to the relevant resampling grid (5 minutes when
        ``dt <= 5 min``, 1 hour when ``dt > 5 min``) so a whole number
        of sim steps fits the window.

    Returns
    -------
    numpy.ndarray, shape ``(T,)``
        Upstream boundary flow in veh/h. ``T = int((end - start) / dt)``.

    Raises
    ------
    ValueError
        * ``dt`` is non-positive, exceeds 1 hour, or doesn't evenly
          divide the resampling grid.
        * ``start`` / ``end`` are misaligned to the resampling grid.
        * The window's row count doesn't match its duration (the CSV is
          short or non-contiguous in the window).
        * Any ``total_flow_[veh/5-min]`` sample in the window is NaN.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end <= start:
        raise ValueError(f"end ({end}) must be strictly after start ({start})")
    if dt <= 0 or dt > _ONE_HOUR_H + _GRID_TOL:
        raise ValueError(
            f"dt must be in (0, 1 hour]; got {dt} h "
            f"({dt * 3600.0:.3f} s)."
        )

    df = pd.read_csv(vds_csv_path, parse_dates=["timestamp"])
    mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
    df_window = (
        df.loc[mask].sort_values("timestamp").reset_index(drop=True)
    )

    window_s = (end - start).total_seconds()
    if dt <= _FIVE_MIN_H + _GRID_TOL:
        # Sub-5-min branch: constant-hold from 5-min samples to sim dt.
        _require_grid_alignment(start, end, _FIVE_MIN_S, label="5-minute")
        steps_per_sample = _exact_divisor(_FIVE_MIN_H, dt, label="dt", grid_label="5 min")
        expected_n = int(round(window_s / _FIVE_MIN_S))
        if len(df_window) != expected_n:
            raise ValueError(
                f"VDS timeseries has {len(df_window)} 5-min sample(s) in "
                f"window [{start}, {end}); expected {expected_n} on a "
                "contiguous 5-min grid. CSV likely doesn't cover the "
                "requested window."
            )
        counts = df_window["total_flow_[veh/5-min]"].to_numpy(dtype=float)
        _require_no_nan(counts, label="total_flow_[veh/5-min]")
        flow_veh_h = counts * 12.0
        return np.repeat(flow_veh_h, steps_per_sample)

    # Super-5-min branch: aggregate to hourly veh/h, then constant-hold.
    _require_grid_alignment(start, end, _ONE_HOUR_S, label="1-hour")
    steps_per_hour = _exact_divisor(
        _ONE_HOUR_H, dt, label="dt > 5 min", grid_label="1 hour",
    )
    expected_n_5min = int(round(window_s / _FIVE_MIN_S))
    if len(df_window) != expected_n_5min:
        raise ValueError(
            f"VDS timeseries has {len(df_window)} 5-min sample(s) in "
            f"window [{start}, {end}); expected {expected_n_5min} on a "
            "contiguous 5-min grid. CSV likely doesn't cover the "
            "requested window."
        )
    counts = df_window["total_flow_[veh/5-min]"].to_numpy(dtype=float)
    _require_no_nan(counts, label="total_flow_[veh/5-min]")
    # Sum 5-min counts within each hour. 12 contiguous samples per hour.
    n_hours = int(round(window_s / _ONE_HOUR_S))
    hourly_veh_h = counts.reshape(n_hours, 12).sum(axis=1)
    return np.repeat(hourly_veh_h, steps_per_hour)


# ---- Initial state -------------------------------------------------------


def initial_state_from_vds(
    cells_df: pd.DataFrame,
    timeseries_dir: Union[Path, str],
    *,
    at_time: pd.Timestamp,
) -> np.ndarray:
    """Initial density ``\\rho_i(0)`` per cell from each cell's mainline VDS.

    For each row in ``cells_df``, locates the sample at ``at_time`` in the
    mainline VDS's timeseries CSV (``timeseries_dir / <vds_id>.csv``) and
    computes the per-cell density. The result feeds the ``rho0`` argument
    of :func:`utils.ctm.io.scenario_from_dataframes`.

    Lane-count contract
    -------------------
    The OSM-derived ``lanes`` column (the cell's lane count, after any
    Step-5 manual edits) **must equal** the Step-2-attached ``vds_lanes``
    column (the VDS's PeMS-reported lane count) for every row. The
    function raises on any mismatch and points the user to fix it before
    sampling -- there's no defensible default scaling when the two
    disagree, and the lane-mismatch warning was already surfaced in Step 1
    + Step 2's console summary.

    When the two columns agree, per-cell density reduces to the VDS's
    total-roadway density:

    .. math::
        \\rho_i = \\frac{\\text{total\\_flow}_{\\text{veh/h}}}
                       {\\text{avg\\_speed}_{\\text{mi/h}}}
               = \\frac{\\text{total\\_flow}_{\\text{veh/5-min}} \\times 12}
                       {\\text{avg\\_speed}_{\\text{mi/h}}}

    Neither lane-count column appears in the formula -- they're read only
    for the equality precondition. The numerator and denominator both
    come straight from the VDS row at ``at_time``.

    Parameters
    ----------
    cells_df : pandas.DataFrame
        Step 1+2 ``cells.csv`` (must carry ``vds_id``, ``lanes``, and
        ``vds_lanes``). Cells whose ``vds_source`` is ``"missing"``
        (``vds_id`` is NaN) are rejected -- run Step 2's upstream-inherit
        fallback or fix ``cells.csv`` first.
    timeseries_dir : Path or str
        Directory containing per-VDS timeseries CSVs (e.g.
        ``data/pems/csv_files/``), one ``<vds_id>.csv`` per detector.
        Files are cached across cells so a VDS shared by multiple cells
        (via Step 2's Dervisoglu fallback) is read only once.
    at_time : pandas.Timestamp
        Sim start time. Must align to the PeMS 5-minute grid and be
        present in every cell's VDS timeseries.

    Returns
    -------
    numpy.ndarray, shape ``(n_cells,)``
        Per-cell initial density in veh/mi, in ``cells_df`` row order.

    Raises
    ------
    KeyError
        ``cells_df`` is missing ``vds_id``, ``lanes``, or ``vds_lanes``.
    ValueError
        * Any cell has a NaN ``vds_id``.
        * Any cell has ``lanes != vds_lanes`` (see lane-count contract).
        * Any VDS has no row at ``at_time``.
        * Any sampled flow or speed at ``at_time`` is NaN.
    """
    required = {"vds_id", "lanes", "vds_lanes"}
    missing = required - set(cells_df.columns)
    if missing:
        raise KeyError(
            f"cells_df is missing column(s): {sorted(missing)}. "
            "`vds_id` and `vds_lanes` are added by Step 2's "
            "`assign_vds_to_cells`; rerun that step before sampling "
            "initial state."
        )
    if cells_df["vds_id"].isna().any():
        unresolved = cells_df.index[cells_df["vds_id"].isna()].to_list()
        raise ValueError(
            f"Cell(s) at row(s) {unresolved} have no `vds_id` "
            "(Step 2's `vds_source='missing'`). Initial state cannot be "
            "sampled for them; run Step 2's upstream-inherit fallback or "
            "edit cells.csv."
        )
    mismatch_mask = cells_df["lanes"].astype(int) != cells_df["vds_lanes"].astype(int)
    if mismatch_mask.any():
        rows = cells_df.index[mismatch_mask].to_list()
        raise ValueError(
            f"Cell(s) at row(s) {rows} have `lanes != vds_lanes` "
            "(OSM-derived lane count disagrees with PeMS-reported VDS "
            "Lanes). Resolve in cells.csv (Step 5 manual edit) before "
            "sampling initial state; PeMS Lanes is typically the "
            "authoritative source."
        )

    at_time = pd.Timestamp(at_time)
    timeseries_dir = Path(timeseries_dir)
    cache: dict[int, pd.DataFrame] = {}
    rho = np.empty(len(cells_df), dtype=float)
    for i, vds_id_raw in enumerate(cells_df["vds_id"].to_list()):
        vds_id = int(vds_id_raw)
        if vds_id not in cache:
            cache[vds_id] = pd.read_csv(
                timeseries_dir / f"{vds_id}.csv",
                parse_dates=["timestamp"],
            )
        df = cache[vds_id]
        sample = df.loc[df["timestamp"] == at_time]
        if sample.empty:
            raise ValueError(
                f"VDS {vds_id}: no sample at {at_time}. Make sure "
                "`at_time` aligns to the PeMS 5-min grid and falls "
                "inside the downloaded window."
            )
        flow_5min = float(sample["total_flow_[veh/5-min]"].iloc[0])
        speed = float(sample["avg_speed_[mph]"].iloc[0])
        if np.isnan(flow_5min) or np.isnan(speed):
            raise ValueError(
                f"VDS {vds_id}: NaN flow or speed at {at_time}. Pick a "
                "different `at_time` or fill the gap in the source CSV."
            )
        rho[i] = (flow_5min * 12.0) / speed

    return rho


# ---- Helpers --------------------------------------------------------------


def _require_grid_alignment(
    start: pd.Timestamp, end: pd.Timestamp, grid_s: float, *, label: str,
) -> None:
    for name, ts in (("start", start), ("end", end)):
        # Use Unix epoch as the grid origin; this matches how PeMS timestamps
        # are reported (5-min and hourly tick on UTC-aligned boundaries).
        offset_s = (ts - pd.Timestamp(0)).total_seconds()
        if abs(offset_s % grid_s) > _GRID_TOL:
            raise ValueError(
                f"{name}={ts} must align to a {label} grid; "
                f"its offset from the epoch is not a multiple of "
                f"{int(grid_s)} s."
            )


def _exact_divisor(
    grid_h: float, dt_h: float, *, label: str, grid_label: str,
) -> int:
    ratio = grid_h / dt_h
    rounded = int(round(ratio))
    if rounded < 1 or abs(ratio - rounded) > 1e-6:
        raise ValueError(
            f"{label}={dt_h} h must evenly divide {grid_label} "
            f"(={grid_h} h); got ratio {ratio}."
        )
    return rounded


def _require_no_nan(values: np.ndarray, *, label: str) -> None:
    n_nan = int(np.isnan(values).sum())
    if n_nan > 0:
        raise ValueError(
            f"VDS timeseries has {n_nan} NaN `{label}` sample(s) in the "
            "requested window. Pick a different window or fill the gaps "
            "in the source CSV."
        )
