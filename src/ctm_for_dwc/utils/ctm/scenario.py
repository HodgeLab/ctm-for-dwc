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

import warnings
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from .assembly import parse_ramp_vds_ids, warn_presence_without_vds

FillStrategy = Callable[[pd.DataFrame], pd.DataFrame]

_FIVE_MIN_H = 5.0 / 60.0   # 5 minutes in hours
_ONE_HOUR_H = 1.0
_FIVE_MIN_S = 300.0
_ONE_HOUR_S = 3600.0
_GRID_TOL = 1e-9           # float tolerance for "evenly divides" checks


# ---- Upstream boundary inflow --------------------------------------------


def _validate_window(
    dt: float, start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Coerce + validate the (dt, start, end) triple shared by every adapter."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end <= start:
        raise ValueError(f"end ({end}) must be strictly after start ({start})")
    if dt <= 0 or dt > _ONE_HOUR_H + _GRID_TOL:
        raise ValueError(
            f"dt must be in (0, 1 hour]; got {dt} h ({dt * 3600.0:.3f} s)."
        )
    return start, end


def _sim_step_count(
    dt: float, start: pd.Timestamp, end: pd.Timestamp,
) -> int:
    """Number of sim steps in the half-open window ``[start, end)``."""
    return int(round((end - start).total_seconds() / (dt * 3600.0)))


def _resample_5min_to_sim_cadence(
    df: pd.DataFrame, *,
    dt: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    flow_col: str = "total_flow_[veh/5-min]",
) -> np.ndarray:
    """Slice a 5-min DataFrame to ``[start, end)`` and resample to sim dt.

    The shared core of :func:`inflow_from_vds`,
    :func:`demand_from_ramp_vds`, and :func:`beta_from_off_ramp_vds`.
    Inputs may have been gap-filled by a
    :mod:`utils.ctm.ramp_fill` strategy before being passed in; the
    resampler doesn't care where the values came from, only that the
    window contains no NaN.

    Resampling rule (no intermediate ``× 12`` round-trip):

    * **``dt`` ≤ 5 min**: constant-hold each 5-min sample across the
      ``300 / dt_s`` sim steps it covers; one ``× 12 -> veh/h`` happens
      at output time.
    * **``dt`` > 5 min**: sum 5-min counts within each hour (= veh/h
      directly), then constant-hold from hourly to sim cadence.
    """
    mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
    df_window = (
        df.loc[mask].sort_values("timestamp").reset_index(drop=True)
    )
    window_s = (end - start).total_seconds()

    if dt <= _FIVE_MIN_H + _GRID_TOL:
        _require_grid_alignment(start, end, _FIVE_MIN_S, label="5-minute")
        steps_per_sample = _exact_divisor(
            _FIVE_MIN_H, dt, label="dt", grid_label="5 min",
        )
        expected_n = int(round(window_s / _FIVE_MIN_S))
        if len(df_window) != expected_n:
            raise ValueError(
                f"VDS timeseries has {len(df_window)} 5-min sample(s) in "
                f"window [{start}, {end}); expected {expected_n} on a "
                "contiguous 5-min grid. CSV likely doesn't cover the "
                "requested window."
            )
        counts = df_window[flow_col].to_numpy(dtype=float)
        _require_no_nan(counts, label=flow_col)
        flow_veh_h = counts * 12.0
        return np.repeat(flow_veh_h, steps_per_sample)

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
    counts = df_window[flow_col].to_numpy(dtype=float)
    _require_no_nan(counts, label=flow_col)
    n_hours = int(round(window_s / _ONE_HOUR_S))
    hourly_veh_h = counts.reshape(n_hours, 12).sum(axis=1)
    return np.repeat(hourly_veh_h, steps_per_hour)


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
    cadence via :func:`_resample_5min_to_sim_cadence`. Output is in
    veh/h and ready to pass as ``inflow`` to
    :func:`utils.ctm.io.scenario_from_dataframes`.

    Parameters
    ----------
    vds_csv_path : Path or str
        Per-VDS timeseries CSV (the file ``PeMSExtractor`` writes).
        Must carry a ``timestamp`` column (parseable as datetime64) and
        a ``total_flow_[veh/5-min]`` column.
    dt : float
        Sim time step in hours. Pick this from Step 6's CFL advisory.
        Must evenly divide 5 minutes when ``dt <= 5 min``, or 1 hour
        when ``dt > 5 min``.
    start, end : pandas.Timestamp
        Calibration window. ``start`` inclusive, ``end`` exclusive.
        Both must align to the relevant resampling grid.

    Returns
    -------
    numpy.ndarray, shape ``(T,)``
        Upstream boundary flow in veh/h. ``T = int((end - start) / dt)``.

    Raises
    ------
    ValueError
        See :func:`_resample_5min_to_sim_cadence`. Notable cases:
        misaligned bounds, short / non-contiguous CSV, any NaN sample
        in the window.
    """
    start, end = _validate_window(dt, start, end)
    df = pd.read_csv(vds_csv_path, parse_dates=["timestamp"])
    return _resample_5min_to_sim_cadence(
        df, dt=dt, start=start, end=end,
    )


# ---- On-ramp demand and off-ramp split ratio (Step 7) --------------------


def _load_ramp_vds(
    vds_id: int,
    timeseries_dir: Path,
    fill_strategy: Optional[FillStrategy],
) -> pd.DataFrame:
    """Load a per-VDS CSV and optionally apply ``fill_strategy`` to the full file.

    The filler sees every sample in the CSV (not the sim window) so the
    historical-average versions can use the widest possible history.
    """
    df = pd.read_csv(
        timeseries_dir / f"{vds_id}.csv", parse_dates=["timestamp"],
    )
    if fill_strategy is not None:
        df = fill_strategy(df)
    return df


def _sum_ramp_vds_flows(
    vds_ids: list[int],
    *,
    timeseries_dir: Path,
    fill_strategy: Optional[FillStrategy],
    dt: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    """Load + resample each VDS's flow, return the per-step sum.

    A Step-5 manual edit can give a single cell more than one ramp VDS;
    the cell's total ramp flow is the sum of the per-VDS flows (each
    detector measures its own ramp; the CTM treats :math:`d_i` /
    :math:`s_i` as the cell-wide total).
    """
    total = None
    for vds_id in vds_ids:
        flow = _resample_5min_to_sim_cadence(
            _load_ramp_vds(vds_id, timeseries_dir, fill_strategy),
            dt=dt, start=start, end=end,
        )
        total = flow if total is None else total + flow
    return total


def demand_from_ramp_vds(
    cells_df: pd.DataFrame,
    timeseries_dir: Union[Path, str],
    dt: float,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fill_strategy: Optional[FillStrategy] = None,
) -> pd.DataFrame:
    """Wide on-ramp demand ``d_i(k)`` [veh/h] for the scenario's ``demand`` arg.

    For each row of ``cells_df`` with ``on_ramp=True`` *and* a non-NaN
    ``on_ramp_vds_id``, load the ramp VDS's 5-min timeseries, apply
    ``fill_strategy`` if supplied, slice to ``[start, end)``, and
    resample to sim cadence via :func:`_resample_5min_to_sim_cadence`.
    The result is keyed by the cell's integer row index, which is
    what :func:`utils.ctm.io.scenario_from_dataframes` expects.

    Cells without an on-ramp, **or** with ``on_ramp=True`` but no matched
    ramp VDS (``on_ramp_vds_id`` is NaN), produce no column. The
    scenario validator treats absent columns as zero, which is the
    intended fallback when no ramp detector data is available.

    ``on_ramp_vds_id`` may be a single id, ``NaN``, or a list-valued
    Step-5 manual edit (parsed via
    :func:`utils.ctm.assembly.parse_ramp_vds_ids`). When the column
    carries multiple ids for a cell, the per-VDS flows are loaded,
    filled, resampled, and **summed** into one demand column.

    Parameters
    ----------
    cells_df : pandas.DataFrame
        Step 1+2 ``cells.csv`` (must carry ``on_ramp`` and
        ``on_ramp_vds_id``). Row order defines cell index.
    timeseries_dir : Path or str
        Directory of per-VDS CSVs (Step 3 output, one ``<vds_id>.csv``
        per detector).
    dt : float
        Sim time step in hours. Same constraints as in
        :func:`inflow_from_vds`.
    start, end : pandas.Timestamp
        Sim window; ``end`` exclusive. Same alignment rules as
        :func:`inflow_from_vds`.
    fill_strategy : callable, optional
        Gap-filler applied to each ramp VDS's full timeseries before
        windowing. See :mod:`utils.ctm.ramp_fill` for ready-made
        Level 1 / 2 / 2b strategies. With ``None`` (the default), any
        NaN sample in the sim window raises -- matching
        :func:`inflow_from_vds`'s strict default.

    Returns
    -------
    pandas.DataFrame
        Wide ``T x N`` (``T = (end - start) / dt``, ``N`` = number of
        on-ramp cells with a matched VDS); columns are integer cell
        indices.
    """
    start, end = _validate_window(dt, start, end)
    timeseries_dir = Path(timeseries_dir)
    warn_presence_without_vds(
        cells_df, presence_col="on_ramp", vds_col="on_ramp_vds_id",
    )

    columns: dict[int, np.ndarray] = {}
    for cell_idx, row in enumerate(cells_df.itertuples(index=False)):
        if not bool(row.on_ramp):
            continue
        ids = parse_ramp_vds_ids(getattr(row, "on_ramp_vds_id", None))
        virtual = [i for i in ids if isinstance(i, str)]
        if virtual:
            warnings.warn(
                f"Cell {cell_idx}: skipping virtual on-ramp VDS id(s) "
                f"{virtual} -- no PeMS timeseries to fill, so they "
                "contribute 0 demand. Use scripts/build_ctm_ramp_scenario.py "
                "to estimate flows for virtual ramps.",
                UserWarning, stacklevel=2,
            )
            ids = [i for i in ids if isinstance(i, int)]
        if not ids:
            continue
        columns[cell_idx] = _sum_ramp_vds_flows(
            ids,
            timeseries_dir=timeseries_dir,
            fill_strategy=fill_strategy,
            dt=dt, start=start, end=end,
        )

    if not columns:
        return pd.DataFrame(index=range(_sim_step_count(dt, start, end)))
    return pd.DataFrame(columns)


def _build_off_vds_to_stretch(stretches_df: pd.DataFrame) -> dict[int | str, list[int]]:
    """Map each off-ramp VDS id to the list of row-index positions in ``stretches_df``.

    Integer ids map as ints; virtual ids (e.g. ``'v400001'``, ramps absent
    from PeMS) map as strings so they still resolve a downstream mainline VDS.
    """
    mapping: dict[int | str, list[int]] = {}
    for pos, (idx, row) in enumerate(stretches_df.iterrows()):
        for vid in str(row.get("off_ids", "")).split(";"):
            vid = vid.strip()
            if vid and vid.lower() != "nan":
                mapping.setdefault(int(vid) if vid.isdigit() else vid, []).append(pos)
    return mapping


def _furthest_downstream_ml_id(
    off_vds_ids: list[int | str],
    stretches_df: pd.DataFrame,
    vds_to_pos: dict[int | str, list[int]],
) -> int | None:
    """Return the ``down_ml_id`` of the furthest-downstream stretch that
    contains any of ``off_vds_ids``, or ``None`` if none found."""
    positions: set[int] = set()
    for vid in off_vds_ids:
        positions.update(vds_to_pos.get(vid, []))
    if not positions:
        return None
    candidates = stretches_df.iloc[sorted(positions)].dropna(subset=["down_ml_id"])
    if candidates.empty:
        return None
    direction = str(stretches_df["dir"].iloc[0])
    if direction in ("N", "E"):
        chosen = candidates.loc[candidates["pm_down"].idxmax()]
    else:
        chosen = candidates.loc[candidates["pm_down"].idxmin()]
    return int(chosen["down_ml_id"])


def beta_from_off_ramp_vds(
    cells_df: pd.DataFrame,
    timeseries_dir: Union[Path, str],
    dt: float,
    stretches_df: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fill_strategy: Optional[FillStrategy] = None,
) -> pd.DataFrame:
    """Wide off-ramp split ratio ``beta_i(k)`` [unitless] for ``Scenario.beta``.

    For each row with ``off_ramp=True`` and a non-NaN ``off_ramp_vds_id``,
    compute :math:`\\beta_i(k) = s_i(k) / (f_i(k) + s_i(k))` where:

    * :math:`s_i(k)` is the off-ramp VDS's flow (from
      ``off_ramp_vds_id``). When the cell carries a list of off-ramp
      VDS ids (Step-5 manual edit), the per-VDS flows are summed.
    * :math:`f_i(k)` is the flow from the ``down_ml_id`` of the
      furthest-downstream stretch (by ``pm_down``) in ``stretches_df``
      that contains the cell's off-ramp VDS ids. This is the mainline
      flow past all off-ramp exits in the cell.

    Both flows are gap-filled (via ``fill_strategy`` applied to each
    underlying CSV), then sliced + resampled by
    :func:`_resample_5min_to_sim_cadence`.

    **Clipping to [0, 1).** Numerical noise (or a mis-matched
    mainline / off-ramp VDS pair) can produce raw ratios outside
    ``[0, 1)``. We clip and emit aggregate stats (count of clipped
    samples, mean pre-clip value on each side) via a
    ``UserWarning`` so the user can spot pathological VDS pairings.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Step-1+2 ``cells.csv``. Must carry ``off_ramp`` and
        ``off_ramp_vds_id``.
    timeseries_dir : path-like
        Directory of per-VDS 5-min CSVs (Step 3 output).
    dt : float
        Sim timestep in hours.
    stretches_df : pd.DataFrame
        Output of ``build_pems_stretches.py``. Must carry ``off_ids``
        (semicolon-separated VDS ids), ``down_ml_id``, ``pm_down``,
        and ``dir``.
    start, end, fill_strategy
        Same conventions as :func:`demand_from_ramp_vds`.

    Returns
    -------
    pandas.DataFrame
        Wide ``T x N`` with integer cell-index columns.
    """
    start, end = _validate_window(dt, start, end)
    timeseries_dir = Path(timeseries_dir)
    warn_presence_without_vds(
        cells_df, presence_col="off_ramp", vds_col="off_ramp_vds_id",
    )
    vds_to_pos = _build_off_vds_to_stretch(stretches_df)

    raw_columns: dict[int, np.ndarray] = {}
    for cell_idx, row in enumerate(cells_df.itertuples(index=False)):
        if not bool(row.off_ramp):
            continue
        off_ids = parse_ramp_vds_ids(getattr(row, "off_ramp_vds_id", None))
        if not off_ids:
            continue
        measured_ids = [i for i in off_ids if isinstance(i, int)]
        if len(measured_ids) < len(off_ids):
            virtual = [i for i in off_ids if isinstance(i, str)]
            warnings.warn(
                f"Cell {cell_idx}: virtual off-ramp VDS id(s) {virtual} have "
                "no PeMS timeseries; they are excluded from s_i, so beta is "
                "underestimated (0 if no measured off-ramp remains). Use "
                "scripts/build_ctm_ramp_scenario.py to estimate flows for "
                "virtual ramps.",
                UserWarning, stacklevel=2,
            )
        if not measured_ids:
            continue

        # The downstream-ML lookup uses the full id list (virtual included):
        # f_i must be the mainline flow past *all* the cell's exits.
        ml_id = _furthest_downstream_ml_id(off_ids, stretches_df, vds_to_pos)
        if ml_id is None:
            warnings.warn(
                f"Cell {cell_idx}: no downstream mainline VDS found in "
                "stretches_df for off-ramp VDS id(s) "
                f"{off_ids}. Setting beta = 0 (off-ramp modeled as no-op).",
                UserWarning, stacklevel=2,
            )
            raw_columns[cell_idx] = np.zeros(_sim_step_count(dt, start, end))
            continue

        s_i = _sum_ramp_vds_flows(
            measured_ids,
            timeseries_dir=timeseries_dir,
            fill_strategy=fill_strategy,
            dt=dt, start=start, end=end,
        )
        f_i = _resample_5min_to_sim_cadence(
            _load_ramp_vds(ml_id, timeseries_dir, fill_strategy),
            dt=dt, start=start, end=end,
        )
        denom = f_i + s_i
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_columns[cell_idx] = np.where(denom > 0, s_i / denom, 0.0)

    if not raw_columns:
        return pd.DataFrame(index=range(_sim_step_count(dt, start, end)))

    # Clip to [0, 1) and report aggregate clipping stats.
    _report_beta_clipping(raw_columns)
    clipped = {
        idx: np.clip(arr, 0.0, np.nextafter(1.0, 0.0))
        for idx, arr in raw_columns.items()
    }
    return pd.DataFrame(clipped)


def _report_beta_clipping(columns: dict[int, np.ndarray]) -> None:
    """Warn with aggregate stats when any column needs clipping to ``[0, 1)``."""
    n_below = 0
    n_above = 0
    below_diffs = []
    above_diffs = []
    upper = np.nextafter(1.0, 0.0)
    for arr in columns.values():
        below = arr < 0.0
        above = arr >= 1.0
        n_below += int(below.sum())
        n_above += int(above.sum())
        below_diffs.extend((0.0 - arr[below]).tolist())
        above_diffs.extend((arr[above] - upper).tolist())
    if n_below == 0 and n_above == 0:
        return
    total = sum(arr.size for arr in columns.values())
    msg_parts = [
        f"{n_below + n_above} of {total} off-ramp split-ratio samples "
        f"needed clipping to [0, 1):",
    ]
    if n_below > 0:
        mean_below = float(np.mean(below_diffs))
        msg_parts.append(
            f"  - {n_below} value(s) < 0 (mean magnitude {mean_below:.4f}, "
            "clipped to 0)"
        )
    if n_above > 0:
        mean_above = float(np.mean(above_diffs))
        msg_parts.append(
            f"  - {n_above} value(s) >= 1 (mean magnitude {mean_above:.4f} "
            "above 1, clipped just below 1)"
        )
    msg_parts.append(
        "Consider double-checking the off-ramp / downstream-mainline VDS "
        "pairings in cells.csv (off_ramp_vds_id vs the next cell's vds_id)."
    )
    warnings.warn("\n".join(msg_parts), UserWarning, stacklevel=3)


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
        * Any cell has ``lanes != vds_lanes`` (see lane-count contract) AND vds_source == 'direct'
        * Any VDS has no row at ``at_time``.
        * Any sampled flow or speed at ``at_time`` is NaN.
    """

    required = {"vds_id", "lanes", "vds_lanes", "vds_source"}
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
    mismatch_mask &= cells_df["vds_source"] == "direct"
    if mismatch_mask.any():
        rows = cells_df.index[mismatch_mask].to_list()
        raise ValueError(
            f"Cell(s) at row(s) {rows} have `lanes != vds_lanes` "
            "(OSM-derived lane count disagrees with PeMS-reported VDS "
            "Lanes) with `vds_source='direct'`. Resolve in cells.csv "
            "(Step 5 manual edit) before sampling initial state; PeMS "
            "Lanes is typically the authoritative source."
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
