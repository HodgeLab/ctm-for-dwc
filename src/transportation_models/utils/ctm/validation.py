"""Compare CTM simulation results against ground-truth observations.

Two ground-truth sources are supported today:

* **Historical PeMS** (:func:`compare_against_historical`) -- the field
  data the model is calibrated against. Compares sim density and
  mainline flow to the cell's assigned mainline VDS over the sim
  window. RMSE/MAPE per cell, scored only at cells whose VDS
  assignment reads an observed station directly (``vds_source`` of
  ``direct`` / ``direct_tiebreak``); cells with an inherited
  (``nearest_upstream``) or ``missing`` assignment are reported as
  NaN rows.

* **CTMSIM reference output** (:func:`compare_against_ctmsim`) -- the
  Kurzhanskiy 2007 / Aurora reference CSVs bundled under
  ``utils/ctm/ctmsim_results/<day>/``. Compares sim density, mainline
  flow, and the four aggregate metrics (VHT, VMT, delay, productivity
  loss) to CTMSIM v1.1's CSV exports for the same .mat config.
  Useful for regression-style validation of engine behavior against
  the canonical reference implementation.

:func:`compare_against_historical` and :func:`compare_against_ctmsim`
return RMSE and MAPE; MAPE skips zero-observed samples (undefined
ratio).

A third helper, :func:`compare_corridor_aggregates`, rolls the
historical comparison up to two corridor scalars -- total VMT and VHT,
sim vs observed -- computed once per *unique* mainline VDS (at its
``direct`` / ``direct_tiebreak`` cell, with the station's
``station_metadata`` length) rather than per cell. It answers "does the
model reproduce aggregate corridor productivity?" as opposed to the
per-cell point fits.

A fourth, :func:`compute_flow_geh`, reports the **GEH** flow statistic
(``sqrt(2*(M-C)^2/(M+C))`` on hourly volumes) at the same
``direct`` / ``direct_tiebreak`` cells -- per-cell median + share with
GEH < 5, plus the per-cell-per-hour matrix used for the heatmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .results import SimulationResult


_FIVE_MIN_S = 300.0
_FIVE_MIN_H = 5.0 / 60.0
_GRID_TOL = 1e-9

# VDS-assignment sources that read an observed station directly (one VDS
# per cell), as opposed to interpolated/propagated assignments. Every
# historical comparison in this module scores on these only.
_DIRECT_SOURCES = ("direct", "direct_tiebreak")


def compare_against_historical(
    result: SimulationResult,
    cells_df: pd.DataFrame,
    timeseries_dir: Union[Path, str],
    *,
    start: pd.Timestamp,
) -> pd.DataFrame:
    """Per-cell sim-vs-observed RMSE and MAPE for density and mainline flow.

    Only cells whose ``vds_source`` is ``direct`` / ``direct_tiebreak``
    are scored -- their VDS physically sits inside the cell. Cells with a
    propagated (``nearest_upstream``) or ``missing`` assignment would be
    judged against a detector somewhere else on the corridor, so they are
    reported as NaN rows instead.

    Parameters
    ----------
    result : SimulationResult
        Output of :func:`utils.ctm.engine.simulate`. The freeway's
        ``dt`` must evenly divide 5 minutes, and the sim horizon
        ``n_steps * dt`` must be a positive whole multiple of 5 minutes.
    cells_df : pandas.DataFrame
        Step 1+2 ``cells.csv`` (must carry ``vds_id``, ``vds_source``,
        ``lanes``, and ``vds_lanes``). Row count and order must match
        ``result.freeway.cells``. Cells whose ``vds_id`` is NaN or whose
        ``vds_source`` is not direct/direct_tiebreak are skipped.
    timeseries_dir : Path or str
        Directory of per-VDS timeseries CSVs (one ``<vds_id>.csv``
        each), in the format ``PeMSExtractor`` writes.
    start : pandas.Timestamp
        Wallclock time of the sim's first step (``step k=0`` of
        ``result``). Used to align sim columns with PeMS rows. Must
        align to PeMS's 5-min grid.

    Returns
    -------
    pandas.DataFrame
        One row per cell, in cell-index order:

        ============================= ============================================
        column                        meaning
        ============================= ============================================
        ``cell``                      cell index in ``result.freeway.cells``
        ``vds_id``                    mainline VDS used for observed data, or
                                      ``<NA>`` if the cell had no direct
                                      VDS assignment (skipped in stats)
        ``n_density_samples``         non-NaN observed density samples
        ``density_rmse``              RMSE [veh/mi] of (sim, observed) density
        ``density_mape``              MAPE [%] of (sim, observed) density
        ``n_flow_samples``            non-NaN observed flow samples
        ``flow_rmse``                 RMSE [veh/h] of (sim, observed) flow
        ``flow_mape``                 MAPE [%] of (sim, observed) flow
        ============================= ============================================

        Cells with a NaN ``vds_id`` have all sample counts at 0 and all
        stats NaN. Cells whose VDS has no non-NaN samples in the window
        also get NaN stats.

    Raises
    ------
    ValueError
        * ``cells_df`` row count doesn't match ``result.freeway.n_cells``.
        * Sim ``dt`` doesn't evenly divide 5 minutes, or the horizon
          isn't a whole multiple of 5 minutes.
        * ``start`` is misaligned to the 5-min grid.
        * Any cell has ``lanes != vds_lanes`` (per the
          :func:`utils.ctm.scenario.initial_state_from_vds` contract --
          observed density isn't directly comparable to sim density
          when the OSM-reported and PeMS-reported lane counts disagree).
    KeyError
        ``cells_df`` is missing ``vds_id``, ``vds_source``, ``lanes``, or
        ``vds_lanes``.
    """
    n_cells = result.freeway.n_cells
    if len(cells_df) != n_cells:
        raise ValueError(
            f"cells_df has {len(cells_df)} rows but result.freeway has "
            f"{n_cells} cells; one row per cell is required."
        )
    required = {"vds_id", "vds_source", "lanes", "vds_lanes"}
    missing = required - set(cells_df.columns)
    if missing:
        raise KeyError(
            f"cells_df is missing column(s): {sorted(missing)}. "
            "vds_id / vds_source / vds_lanes come from Step 2 "
            "(assign_vds_to_cells)."
        )

    # Resolve sim cadence vs 5-min cadence and the wallclock window.
    steps_per_5min, n_5min, start, end = _resolve_window(result, start)

    # Lane-count contract (same as initial_state_from_vds): observed
    # density derived from a VDS with N lanes is only directly
    # comparable to a per-cell density on a cell with the same lane
    # count. Otherwise the user must reconcile the discrepancy in
    # cells.csv (typically by trusting the PeMS-reported value).
    # Checked on the scored (direct) cells only.
    cells_with_vds = cells_df[
        cells_df["vds_source"].isin(_DIRECT_SOURCES)
    ].dropna(subset=["vds_id"])
    mismatch_mask = (
        cells_with_vds["lanes"].astype(int)
        != cells_with_vds["vds_lanes"].astype(int)
    )
    if mismatch_mask.any():
        rows = cells_with_vds.index[mismatch_mask].to_list()
        raise ValueError(
            f"Cell(s) at row(s) {rows} have lanes != vds_lanes. "
            "Resolve in cells.csv (Step 5 manual edit) before "
            "validating against historical data; PeMS Lanes is the "
            "authoritative source."
        )

    # Sim density at 5-min boundaries (instantaneous state at the start
    # of each interval; we drop the trailing boundary so the array has
    # one row per interval, lined up with the PeMS row at that interval).
    sim_density_5min = result.density[:, : n_5min * steps_per_5min : steps_per_5min]
    # Sim flow averaged across each 5-min window.
    sim_flow_per_step = result.mainline_flow[:, : n_5min * steps_per_5min]
    sim_flow_5min = sim_flow_per_step.reshape(
        n_cells, n_5min, steps_per_5min,
    ).mean(axis=2)

    timeseries_dir = Path(timeseries_dir)
    rows: list[dict] = []
    vds_cache: dict[int, pd.DataFrame] = {}
    for cell_idx, cell_row in enumerate(cells_df.itertuples(index=False)):
        vds_id_raw = cell_row.vds_id
        if pd.isna(vds_id_raw) or cell_row.vds_source not in _DIRECT_SOURCES:
            rows.append(_nan_row(cell_idx, vds_id=pd.NA))
            continue
        vds_id = int(vds_id_raw)
        observed_flow, observed_density = _load_observed_window(
            timeseries_dir, vds_id, start=start, end=end,
            n_5min=n_5min, cache=vds_cache,
        )

        n_flow, flow_rmse, flow_mape = _rmse_mape(
            sim_flow_5min[cell_idx], observed_flow,
        )
        n_density, density_rmse, density_mape = _rmse_mape(
            sim_density_5min[cell_idx], observed_density,
        )
        rows.append({
            "cell": cell_idx,
            "vds_id": vds_id,
            "n_density_samples": n_density,
            "density_rmse": density_rmse,
            "density_mape": density_mape,
            "n_flow_samples": n_flow,
            "flow_rmse": flow_rmse,
            "flow_mape": flow_mape,
        })

    return pd.DataFrame(
        rows, columns=[
            "cell", "vds_id",
            "n_density_samples", "density_rmse", "density_mape",
            "n_flow_samples", "flow_rmse", "flow_mape",
        ],
    )


def restrict_to_direct_tiebreak(cells_df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``cells_df`` keeping only the unique direct/tiebreak VDS.

    Returns a copy in which ``vds_id`` is retained only at the *first*
    cell whose ``vds_source`` is ``direct`` / ``direct_tiebreak`` for each
    unique VDS; every other cell's ``vds_id`` is blanked to ``<NA>``.

    :func:`compare_against_historical` now applies the direct-only
    restriction itself, so this helper is no longer needed there; it
    remains for consumers that need the one-cell-per-unique-VDS view of
    ``cells.csv`` directly (e.g. ``scripts/build_ctm_qp_inputs.py``).

    Raises
    ------
    KeyError
        ``cells_df`` is missing ``vds_id`` or ``vds_source``.
    """
    missing = {"vds_id", "vds_source"} - set(cells_df.columns)
    if missing:
        raise KeyError(
            f"cells_df is missing column(s): {sorted(missing)}. "
            "vds_id / vds_source come from Step 2 (assign_vds_to_cells)."
        )
    out = cells_df.copy()
    keep = np.zeros(len(out), dtype=bool)
    seen: set[int] = set()
    for pos, (vds_id, source) in enumerate(zip(out["vds_id"], out["vds_source"])):
        if source in _DIRECT_SOURCES and pd.notna(vds_id) and int(vds_id) not in seen:
            seen.add(int(vds_id))
            keep[pos] = True
    out.loc[~keep, "vds_id"] = pd.NA
    return out


# ---- Corridor aggregates --------------------------------------------------


@dataclass
class CorridorAggregates:
    """Corridor-total VMT and VHT, sim vs observed, over the sim window.

    Computed once per *unique* mainline VDS (at its ``direct`` /
    ``direct_tiebreak`` cell), summing over the cells' direct VDS and the
    window. ``n_vds`` is the number of unique VDS that contributed.

    Attributes
    ----------
    sim_vmt, obs_vmt : float
        Vehicle-miles traveled [veh*mi].
    sim_vht, obs_vht : float
        Vehicle-hours traveled [veh*h].
    n_vds : int
        Unique direct/tiebreak VDS summed.
    """

    sim_vmt: float
    obs_vmt: float
    sim_vht: float
    obs_vht: float
    n_vds: int

    @property
    def vmt_pct_diff(self) -> float:
        """100 * (sim - obs) / obs for VMT; NaN if ``obs_vmt`` is 0."""
        return _pct_diff(self.sim_vmt, self.obs_vmt)

    @property
    def vht_pct_diff(self) -> float:
        """100 * (sim - obs) / obs for VHT; NaN if ``obs_vht`` is 0."""
        return _pct_diff(self.sim_vht, self.obs_vht)


def compare_corridor_aggregates(
    result: SimulationResult,
    cells_df: pd.DataFrame,
    timeseries_dir: Union[Path, str],
    station_metadata: pd.DataFrame,
    *,
    start: pd.Timestamp,
) -> CorridorAggregates:
    """Corridor-total VMT and VHT, sim vs observed PeMS, over the sim window.

    Unlike the per-cell corridor-long metrics in :mod:`utils.ctm.metrics`
    (which sum every cell with its own cell length and on-ramp queues),
    this restricts to one term per *unique* mainline VDS -- the cell whose
    ``vds_source`` is ``direct`` or ``direct_tiebreak`` -- and uses the
    station's ``Length`` from ``station_metadata`` on both sides, so the
    comparison reflects flow/density accuracy rather than geometry.

    Both sides share the same definitions:

    * ``VMT = sum rho * v * L * dt`` -- the sim integrates ``rho * v`` at
      native ``dt`` (the eq. 4.12 VMT definition); the observed side uses
      ``flow * L * dt_5min`` (identical, since observed ``rho * v == flow``
      because observed density is ``flow / speed``).
    * ``VHT = sum rho * L * dt`` -- sim density integrated at native
      ``dt``; observed ``rho = flow / speed``.

    For each VDS, a 5-min window with a NaN observed value is dropped from
    *both* sides of that metric (paired support), mirroring how the
    per-cell RMSE only scores sim against non-NaN observed samples. VMT
    uses flow validity; VHT uses density validity.

    Parameters
    ----------
    result : SimulationResult
        Output of :func:`utils.ctm.engine.simulate`. Same ``dt`` /
        horizon constraints as :func:`compare_against_historical`.
    cells_df : pandas.DataFrame
        Step 1+2 ``cells.csv``; must carry ``vds_id`` and ``vds_source``.
        Row count and order must match ``result.freeway.cells``.
    timeseries_dir : Path or str
        Directory of per-VDS timeseries CSVs (``<vds_id>.csv``).
    station_metadata : pandas.DataFrame
        PeMS ``station_metadata.csv``; must carry ``ID`` (VDS id) and
        ``Length`` [mi]. Supplies each VDS's segment length.
    start : pandas.Timestamp
        Wallclock time of the sim's first step; must align to the 5-min
        grid.

    Returns
    -------
    CorridorAggregates

    Raises
    ------
    ValueError
        Row-count mismatch, bad cadence/horizon, misaligned ``start``, or
        a contributing VDS missing from ``station_metadata`` (hard error
        -- no silent skip).
    KeyError
        ``cells_df`` missing ``vds_id`` / ``vds_source``, or
        ``station_metadata`` missing ``ID`` / ``Length``.
    """
    n_cells = result.freeway.n_cells
    if len(cells_df) != n_cells:
        raise ValueError(
            f"cells_df has {len(cells_df)} rows but result.freeway has "
            f"{n_cells} cells; one row per cell is required."
        )
    missing = {"vds_id", "vds_source"} - set(cells_df.columns)
    if missing:
        raise KeyError(
            f"cells_df is missing column(s): {sorted(missing)}. "
            "vds_id / vds_source come from Step 2 (assign_vds_to_cells)."
        )
    meta_missing = {"ID", "Length"} - set(station_metadata.columns)
    if meta_missing:
        raise KeyError(
            f"station_metadata is missing column(s): {sorted(meta_missing)}."
        )

    steps_per_5min, n_5min, start, end = _resolve_window(result, start)
    timeseries_dir = Path(timeseries_dir)
    length_by_vds = station_metadata.set_index("ID")["Length"]

    # Post-step density aligned with speed (k = 0..T-1), same as metrics.py.
    horizon = n_5min * steps_per_5min
    rho = result.density[:, 1: horizon + 1]   # (n_cells, T)  rho_i(k+1)
    speed = result.speed[:, :horizon]         # (n_cells, T)  V_i(k+1)
    dt = result.freeway.dt

    sim_vmt = obs_vmt = sim_vht = obs_vht = 0.0
    seen: set[int] = set()
    cache: dict[int, pd.DataFrame] = {}
    for cell_idx, cell_row in enumerate(cells_df.itertuples(index=False)):
        if cell_row.vds_source not in _DIRECT_SOURCES:
            continue
        vds_id = int(cell_row.vds_id)
        if vds_id in seen:
            continue
        seen.add(vds_id)

        if vds_id not in length_by_vds.index:
            raise ValueError(
                f"VDS {vds_id} (cell {cell_idx}, vds_source="
                f"{cell_row.vds_source!r}) has no row in station_metadata; "
                "cannot resolve its segment Length."
            )
        length = float(length_by_vds.loc[vds_id])

        # Sim side: integrate rho*v and rho at native dt, bucketed into the
        # n_5min windows so NaN-observed windows can be dropped pairwise.
        rho_c = rho[cell_idx].reshape(n_5min, steps_per_5min)
        flux_c = (rho[cell_idx] * speed[cell_idx]).reshape(n_5min, steps_per_5min)
        sim_vmt_w = flux_c.sum(axis=1) * dt * length    # (n_5min,) veh*mi
        sim_vht_w = rho_c.sum(axis=1) * dt * length     # (n_5min,) veh*h

        observed_flow, observed_density = _load_observed_window(
            timeseries_dir, vds_id, start=start, end=end,
            n_5min=n_5min, cache=cache,
        )
        obs_vmt_w = observed_flow * _FIVE_MIN_H * length
        obs_vht_w = observed_density * _FIVE_MIN_H * length

        flow_valid = ~np.isnan(observed_flow)
        density_valid = ~np.isnan(observed_density)
        sim_vmt += sim_vmt_w[flow_valid].sum()
        obs_vmt += obs_vmt_w[flow_valid].sum()
        sim_vht += sim_vht_w[density_valid].sum()
        obs_vht += obs_vht_w[density_valid].sum()

    return CorridorAggregates(
        sim_vmt=float(sim_vmt), obs_vmt=float(obs_vmt),
        sim_vht=float(sim_vht), obs_vht=float(obs_vht),
        n_vds=len(seen),
    )


# ---- GEH statistic --------------------------------------------------------


_GEH_THRESHOLD = 5.0
_FIVE_MIN_PER_HOUR = 12


@dataclass
class GEHResult:
    """Per-cell flow GEH plus the hourly matrix for the heatmap.

    GEH = ``sqrt(2*(M-C)^2 / (M+C))`` on *hourly volumes* (M = sim, C =
    observed), computed once per unique ``direct`` / ``direct_tiebreak``
    VDS, at each clock-hour bin in the window.

    Attributes
    ----------
    per_cell : pandas.DataFrame
        One row per direct/tiebreak cell (in cell-index order): ``cell``,
        ``vds_id``, ``n_geh_hours`` (hours with a non-NaN observed
        volume), ``flow_geh_median``, ``flow_geh_pct_under5`` (share of
        those hours with GEH < 5).
    hourly_geh : numpy.ndarray
        ``(n_direct_cells, n_hours)`` GEH per cell per hour, NaN where the
        observed hourly volume was incomplete. Rows align with
        ``per_cell``; columns with ``hour_starts``.
    hour_starts : pandas.DatetimeIndex
        Start wallclock time of each hourly bin (heatmap x-axis).
    corridor_pct_under5 : float
        Share [%] of all non-NaN ``(cell, hour)`` GEH samples with
        GEH < 5 (NaN if there were none).
    n_geh_samples : int
        Number of non-NaN ``(cell, hour)`` GEH samples pooled into
        ``corridor_pct_under5``.
    """

    per_cell: pd.DataFrame
    hourly_geh: np.ndarray
    hour_starts: pd.DatetimeIndex
    corridor_pct_under5: float
    n_geh_samples: int


def compute_flow_geh(
    result: SimulationResult,
    cells_df: pd.DataFrame,
    timeseries_dir: Union[Path, str],
    *,
    start: pd.Timestamp,
) -> GEHResult:
    """Per-cell flow GEH on hourly volumes, at direct/tiebreak cells.

    GEH is a volume-comparison statistic; it is applied to **flow only**
    (not density) and to **hourly volumes** -- the twelve 5-min samples in
    each clock-hour bin are summed into an hourly volume on both sides
    before computing ``GEH = sqrt(2*(M-C)^2/(M+C))``. This matches the
    magnitude the conventional ``GEH < 5`` acceptance criterion is
    defined for.

    As with :func:`compare_corridor_aggregates`, only the cell whose
    ``vds_source`` is ``direct`` or ``direct_tiebreak`` contributes, and a
    VDS spanning several cells is scored once.

    An hour with any NaN observed 5-min sample yields a NaN hourly volume
    (incomplete count) and is dropped from that cell's GEH.

    Parameters
    ----------
    result, cells_df, timeseries_dir, start
        As in :func:`compare_corridor_aggregates`, except ``cells_df``
        needs only ``vds_id`` and ``vds_source``.

    Returns
    -------
    GEHResult

    Raises
    ------
    ValueError
        Row-count mismatch, bad cadence, misaligned ``start``, or a window
        that is not a whole number of hours (GEH needs whole-hour bins).
    KeyError
        ``cells_df`` missing ``vds_id`` / ``vds_source``.
    """
    n_cells = result.freeway.n_cells
    if len(cells_df) != n_cells:
        raise ValueError(
            f"cells_df has {len(cells_df)} rows but result.freeway has "
            f"{n_cells} cells; one row per cell is required."
        )
    missing = {"vds_id", "vds_source"} - set(cells_df.columns)
    if missing:
        raise KeyError(
            f"cells_df is missing column(s): {sorted(missing)}. "
            "vds_id / vds_source come from Step 2 (assign_vds_to_cells)."
        )

    steps_per_5min, n_5min, start, end = _resolve_window(result, start)
    if n_5min % _FIVE_MIN_PER_HOUR != 0:
        raise ValueError(
            f"GEH needs a whole-hour window: the sim covers {n_5min} 5-min "
            "sample(s), which is not a multiple of 12 (60 min)."
        )
    n_hours = n_5min // _FIVE_MIN_PER_HOUR
    hour_starts = pd.DatetimeIndex(
        [start + pd.Timedelta(hours=h) for h in range(n_hours)]
    )
    timeseries_dir = Path(timeseries_dir)

    # Sim mainline flow averaged per 5-min window (veh/h), then per hour.
    sim_flow_5min = result.mainline_flow[
        :, : n_5min * steps_per_5min
    ].reshape(n_cells, n_5min, steps_per_5min).mean(axis=2)

    rows: list[dict] = []
    hourly: list[np.ndarray] = []
    seen: set[int] = set()
    cache: dict[int, pd.DataFrame] = {}
    for cell_idx, cell_row in enumerate(cells_df.itertuples(index=False)):
        if cell_row.vds_source not in _DIRECT_SOURCES:
            continue
        vds_id = int(cell_row.vds_id)
        if vds_id in seen:
            continue
        seen.add(vds_id)

        observed_flow, _ = _load_observed_window(
            timeseries_dir, vds_id, start=start, end=end,
            n_5min=n_5min, cache=cache,
        )
        # Hourly volume [veh] = mean of the twelve 5-min veh/h rates in the
        # bin. NaN propagates -> an incomplete hour is excluded.
        sim_hourly = sim_flow_5min[cell_idx].reshape(n_hours, _FIVE_MIN_PER_HOUR).mean(axis=1)
        obs_hourly = observed_flow.reshape(n_hours, _FIVE_MIN_PER_HOUR).mean(axis=1)

        geh = _geh(sim_hourly, obs_hourly)
        hourly.append(geh)
        valid = ~np.isnan(geh)
        n_valid = int(valid.sum())
        rows.append({
            "cell": cell_idx,
            "vds_id": vds_id,
            "n_geh_hours": n_valid,
            "flow_geh_median": (
                float(np.median(geh[valid])) if n_valid else float("nan")
            ),
            "flow_geh_pct_under5": (
                100.0 * float(np.mean(geh[valid] < _GEH_THRESHOLD))
                if n_valid else float("nan")
            ),
        })

    per_cell = pd.DataFrame(
        rows, columns=[
            "cell", "vds_id", "n_geh_hours",
            "flow_geh_median", "flow_geh_pct_under5",
        ],
    )
    hourly_geh = (
        np.vstack(hourly) if hourly else np.empty((0, n_hours), dtype=float)
    )
    pooled = hourly_geh[~np.isnan(hourly_geh)]
    n_geh_samples = int(pooled.size)
    corridor_pct_under5 = (
        100.0 * float(np.mean(pooled < _GEH_THRESHOLD))
        if n_geh_samples else float("nan")
    )
    return GEHResult(
        per_cell=per_cell,
        hourly_geh=hourly_geh,
        hour_starts=hour_starts,
        corridor_pct_under5=corridor_pct_under5,
        n_geh_samples=n_geh_samples,
    )


# ---- QQ error-distribution samples ---------------------------------------


@dataclass
class QQCellSamples:
    """Paired sim/observed 5-min samples for one direct/tiebreak VDS cell.

    Each pair is on paired support: a 5-min window with a NaN observed
    value is dropped from *both* sides of that quantity (flow validity for
    flow, density validity for density), mirroring the per-cell RMSE
    convention. Sim values are never NaN.

    Attributes
    ----------
    cell : int
        Cell index in ``result.freeway.cells``.
    vds_id : int
        The cell's direct/tiebreak mainline VDS.
    flow_sim, flow_obs : numpy.ndarray
        Paired mainline flow [veh/h]; length = non-NaN observed flow
        windows.
    density_sim, density_obs : numpy.ndarray
        Paired density [veh/mi]; length = non-NaN observed density windows.
    """

    cell: int
    vds_id: int
    flow_sim: np.ndarray
    flow_obs: np.ndarray
    density_sim: np.ndarray
    density_obs: np.ndarray


@dataclass
class QQResult:
    """Per-cell paired samples for QQ error-distribution analysis.

    Holds one :class:`QQCellSamples` per unique direct/tiebreak VDS, in
    cell-index order. :meth:`pooled` concatenates a quantity's paired
    arrays across all cells for the corridor-pooled plots.
    """

    per_cell: list[QQCellSamples]

    def pooled(self, quantity: str) -> tuple[np.ndarray, np.ndarray]:
        """Corridor-pooled ``(sim, obs)`` for ``"flow"`` or ``"density"``."""
        if quantity not in ("flow", "density"):
            raise ValueError(
                f"quantity must be 'flow' or 'density'; got {quantity!r}."
            )
        sim_parts = [getattr(c, f"{quantity}_sim") for c in self.per_cell]
        obs_parts = [getattr(c, f"{quantity}_obs") for c in self.per_cell]
        if not sim_parts:
            empty = np.empty(0, dtype=float)
            return empty, empty
        return np.concatenate(sim_parts), np.concatenate(obs_parts)


def compute_qq_samples(
    result: SimulationResult,
    cells_df: pd.DataFrame,
    timeseries_dir: Union[Path, str],
    *,
    start: pd.Timestamp,
) -> QQResult:
    """Collect paired sim/observed 5-min samples for QQ analysis.

    Like :func:`compute_flow_geh`, this restricts to the cell whose
    ``vds_source`` is ``direct`` or ``direct_tiebreak`` and scores a VDS
    spanning several cells once. For each such cell it pairs the sim 5-min
    flow/density against the observed PeMS window, dropping windows whose
    observed value is NaN (flow validity for flow, density validity for
    density). The paired arrays feed the Normal residual QQ (``sim - obs``)
    and two-sample sim-vs-observed QQ plots.

    Sim quantities use the same definitions as
    :func:`compare_against_historical`: flow is the per-5-min window mean
    of ``result.mainline_flow``; density is the instantaneous state at each
    5-min boundary.

    Parameters
    ----------
    result, cells_df, timeseries_dir, start
        As in :func:`compute_flow_geh`; ``cells_df`` needs ``vds_id`` and
        ``vds_source``.

    Returns
    -------
    QQResult

    Raises
    ------
    ValueError
        Row-count mismatch, bad cadence/horizon, or misaligned ``start``.
    KeyError
        ``cells_df`` missing ``vds_id`` / ``vds_source``.
    """
    n_cells = result.freeway.n_cells
    if len(cells_df) != n_cells:
        raise ValueError(
            f"cells_df has {len(cells_df)} rows but result.freeway has "
            f"{n_cells} cells; one row per cell is required."
        )
    missing = {"vds_id", "vds_source"} - set(cells_df.columns)
    if missing:
        raise KeyError(
            f"cells_df is missing column(s): {sorted(missing)}. "
            "vds_id / vds_source come from Step 2 (assign_vds_to_cells)."
        )

    steps_per_5min, n_5min, start, end = _resolve_window(result, start)
    timeseries_dir = Path(timeseries_dir)

    sim_density_5min = result.density[:, : n_5min * steps_per_5min : steps_per_5min]
    sim_flow_5min = result.mainline_flow[
        :, : n_5min * steps_per_5min
    ].reshape(n_cells, n_5min, steps_per_5min).mean(axis=2)

    per_cell: list[QQCellSamples] = []
    seen: set[int] = set()
    cache: dict[int, pd.DataFrame] = {}
    for cell_idx, cell_row in enumerate(cells_df.itertuples(index=False)):
        if cell_row.vds_source not in _DIRECT_SOURCES:
            continue
        vds_id = int(cell_row.vds_id)
        if vds_id in seen:
            continue
        seen.add(vds_id)

        observed_flow, observed_density = _load_observed_window(
            timeseries_dir, vds_id, start=start, end=end,
            n_5min=n_5min, cache=cache,
        )
        flow_valid = ~np.isnan(observed_flow)
        density_valid = ~np.isnan(observed_density)
        per_cell.append(QQCellSamples(
            cell=cell_idx,
            vds_id=vds_id,
            flow_sim=sim_flow_5min[cell_idx][flow_valid],
            flow_obs=observed_flow[flow_valid],
            density_sim=sim_density_5min[cell_idx][density_valid],
            density_obs=observed_density[density_valid],
        ))

    return QQResult(per_cell=per_cell)


@dataclass
class ObservedGrid:
    """Measured-PeMS 5-min space-time grids for direct/tiebreak VDS cells.

    ``flow`` and ``density`` are ``(n_cells, n_5min)`` with one row per
    unique direct/tiebreak VDS (in cell-index order) and one column per
    5-min window over the sim's wallclock horizon. Invalid samples are
    NaN. ``cell_labels`` annotate the cells (with the VDS absolute
    postmile); ``times`` are the window-start timestamps.
    """

    cell_labels: list[str]
    times: pd.DatetimeIndex
    flow: np.ndarray
    density: np.ndarray


def compute_observed_grid(
    result: SimulationResult,
    cells_df: pd.DataFrame,
    timeseries_dir: Union[Path, str],
    station_metadata: pd.DataFrame,
    *,
    start: pd.Timestamp,
) -> ObservedGrid:
    """Collect measured-PeMS flow/density 5-min grids over the sim window.

    Like :func:`compute_qq_samples`, restricts to the cell whose
    ``vds_source`` is ``direct`` or ``direct_tiebreak`` and reads each VDS
    once. For each such cell it loads the observed 5-min flow [veh/h] and
    density [veh/mi] over the sim's wallclock ``[start, end)`` window,
    stacking them into ``(n_cells, n_5min)`` grids for space-time heatmaps.

    Parameters
    ----------
    result, cells_df, timeseries_dir, start
        As in :func:`compute_qq_samples`.
    station_metadata : pandas.DataFrame
        PeMS ``station_metadata.csv``; must carry ``ID`` (VDS id) and
        ``Abs_PM`` (Caltrans absolute postmile). Supplies each VDS's
        absolute postmile for the cell labels.

    Returns
    -------
    ObservedGrid
    """
    n_cells = result.freeway.n_cells
    if len(cells_df) != n_cells:
        raise ValueError(
            f"cells_df has {len(cells_df)} rows but result.freeway has "
            f"{n_cells} cells; one row per cell is required."
        )
    missing = {"vds_id", "vds_source"} - set(cells_df.columns)
    if missing:
        raise KeyError(
            f"cells_df is missing column(s): {sorted(missing)}. "
            "vds_id / vds_source come from Step 2 (assign_vds_to_cells)."
        )
    meta_missing = {"ID", "Abs_PM"} - set(station_metadata.columns)
    if meta_missing:
        raise KeyError(
            f"station_metadata is missing column(s): {sorted(meta_missing)}."
        )
    abs_pm_by_vds = station_metadata.set_index("ID")["Abs_PM"]

    _, n_5min, start, end = _resolve_window(result, start)
    timeseries_dir = Path(timeseries_dir)
    times = pd.date_range(start, periods=n_5min, freq="5min")

    cell_labels: list[str] = []
    flow_rows: list[np.ndarray] = []
    density_rows: list[np.ndarray] = []
    seen: set[int] = set()
    cache: dict[int, pd.DataFrame] = {}
    for cell_idx, cell_row in enumerate(cells_df.itertuples(index=False)):
        if cell_row.vds_source not in _DIRECT_SOURCES:
            continue
        vds_id = int(cell_row.vds_id)
        if vds_id in seen:
            continue
        seen.add(vds_id)

        observed_flow, observed_density = _load_observed_window(
            timeseries_dir, vds_id, start=start, end=end,
            n_5min=n_5min, cache=cache,
        )
        abs_pm = abs_pm_by_vds.get(vds_id, float("nan"))
        label = f"cell {cell_idx} (VDS {vds_id}"
        label += f", PM {float(abs_pm):.2f})" if pd.notna(abs_pm) else ")"
        cell_labels.append(label)
        flow_rows.append(observed_flow)
        density_rows.append(observed_density)

    shape = (len(flow_rows), n_5min)
    flow = np.vstack(flow_rows) if flow_rows else np.empty(shape, dtype=float)
    density = (
        np.vstack(density_rows) if density_rows else np.empty(shape, dtype=float)
    )
    return ObservedGrid(
        cell_labels=cell_labels, times=times, flow=flow, density=density,
    )


def corridor_rmse_mape(qq: QQResult) -> dict[str, tuple[int, float, float]]:
    """Corridor-pooled ``(n_samples, RMSE, MAPE)`` for flow and density.

    Pools the paired sim/observed 5-min samples across every direct/
    tiebreak cell (:meth:`QQResult.pooled`) and applies the same
    definitions as the per-cell :func:`compare_against_historical`: RMSE
    over all (non-NaN-observed) pooled samples, MAPE additionally
    excluding zero-observed samples. This is the true corridor error --
    one RMSE/MAPE over the pooled residuals, not an average of per-cell
    RMSEs.

    Returns a dict keyed by ``"flow"`` / ``"density"``; each value is
    ``(n_samples, rmse, mape)`` with RMSE in the quantity's native unit
    (flow veh/h, density veh/mi) and MAPE in %. Empty pools give
    ``(0, nan, nan)``.
    """
    return {q: _rmse_mape(*qq.pooled(q)) for q in ("flow", "density")}


# ---- Helpers --------------------------------------------------------------


def _geh(modeled: np.ndarray, counted: np.ndarray) -> np.ndarray:
    """Elementwise GEH = sqrt(2*(M-C)^2/(M+C)); 0 where M+C==0, NaN where NaN."""
    m = np.asarray(modeled, dtype=float)
    c = np.asarray(counted, dtype=float)
    denom = m + c
    with np.errstate(divide="ignore", invalid="ignore"):
        geh = np.sqrt(2.0 * (m - c) ** 2 / denom)
    return np.where(denom == 0.0, 0.0, geh)


def _pct_diff(sim: float, obs: float) -> float:
    return float("nan") if obs == 0.0 else 100.0 * (sim - obs) / obs


def _pct_diff(sim: float, obs: float) -> float:
    return float("nan") if obs == 0.0 else 100.0 * (sim - obs) / obs


def _resolve_window(
    result: SimulationResult, start: pd.Timestamp,
) -> tuple[int, int, pd.Timestamp, pd.Timestamp]:
    """Resolve sim cadence against the PeMS 5-min grid and the wallclock window.

    Returns ``(steps_per_5min, n_5min, start, end)``. Raises
    :class:`ValueError` if ``dt`` doesn't evenly divide 5 minutes, the
    horizon isn't a positive whole multiple of 5 minutes, or ``start`` is
    off the 5-min grid.
    """
    dt = result.freeway.dt
    steps_per_5min = _exact_steps_per(_FIVE_MIN_H, dt, label="dt", grid="5 min")
    horizon_steps = result.n_steps
    n_5min = horizon_steps // steps_per_5min
    if n_5min < 1 or n_5min * steps_per_5min != horizon_steps:
        raise ValueError(
            f"Sim horizon {horizon_steps} step(s) of dt={dt} h "
            "must be a positive whole multiple of 5 minutes; got "
            f"{horizon_steps * dt:.5f} h."
        )

    start = pd.Timestamp(start)
    if abs((start - pd.Timestamp(0)).total_seconds() % _FIVE_MIN_S) > _GRID_TOL:
        raise ValueError(f"start={start} must align to PeMS's 5-min grid.")
    end = start + pd.Timedelta(seconds=n_5min * _FIVE_MIN_S)
    return steps_per_5min, n_5min, start, end


def _load_observed_window(
    timeseries_dir: Path, vds_id: int, *,
    start: pd.Timestamp, end: pd.Timestamp, n_5min: int,
    cache: dict[int, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    """Load a VDS's ``[start, end)`` window as ``(flow_veh_h, density_veh_mi)``.

    Flow is ``total_flow_[veh/5-min] * 12``; density is ``flow / avg_speed``.
    Reads are memoized in ``cache``. Raises :class:`ValueError` unless the
    window holds exactly ``n_5min`` contiguous 5-min samples.
    """
    if vds_id not in cache:
        cache[vds_id] = pd.read_csv(
            timeseries_dir / f"{vds_id}.csv", parse_dates=["timestamp"],
        )
    df = cache[vds_id]
    window = df.loc[
        (df["timestamp"] >= start) & (df["timestamp"] < end)
    ].sort_values("timestamp").reset_index(drop=True)
    if len(window) != n_5min:
        raise ValueError(
            f"VDS {vds_id}: {len(window)} 5-min sample(s) in window "
            f"[{start}, {end}); expected {n_5min} on a contiguous "
            "5-min grid."
        )
    observed_flow = window["total_flow_[veh/5-min]"].to_numpy(dtype=float) * 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        observed_density = observed_flow / window[
            "avg_speed_[mph]"
        ].to_numpy(dtype=float)
    return observed_flow, observed_density


def _exact_steps_per(grid_h: float, dt_h: float, *, label: str, grid: str) -> int:
    ratio = grid_h / dt_h
    rounded = int(round(ratio))
    if rounded < 1 or abs(ratio - rounded) > 1e-6:
        raise ValueError(
            f"{label}={dt_h} h must evenly divide {grid} (={grid_h} h); "
            f"got ratio {ratio}."
        )
    return rounded


def _rmse_mape(
    predicted: np.ndarray, observed: np.ndarray,
) -> tuple[int, float, float]:
    """RMSE and MAPE [%] between predicted and observed across non-NaN obs.

    MAPE additionally skips samples where ``observed`` is zero
    (undefined ratio). Returns (n_samples_used_for_rmse, rmse, mape).
    With zero non-NaN observed samples, ``rmse`` and ``mape`` are NaN
    and the count is 0.
    """
    obs_valid = ~np.isnan(observed)
    if not obs_valid.any():
        return 0, float("nan"), float("nan")
    diff = predicted[obs_valid] - observed[obs_valid]
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    obs_for_mape = observed[obs_valid]
    pred_for_mape = predicted[obs_valid]
    nonzero = obs_for_mape != 0.0
    if nonzero.any():
        mape = float(
            np.mean(
                np.abs(pred_for_mape[nonzero] - obs_for_mape[nonzero])
                / np.abs(obs_for_mape[nonzero])
            )
            * 100.0
        )
    else:
        mape = float("nan")
    return int(obs_valid.sum()), rmse, mape


def _nan_row(cell_idx: int, *, vds_id) -> dict:
    return {
        "cell": cell_idx,
        "vds_id": vds_id,
        "n_density_samples": 0,
        "density_rmse": float("nan"),
        "density_mape": float("nan"),
        "n_flow_samples": 0,
        "flow_rmse": float("nan"),
        "flow_mape": float("nan"),
    }


# ---- CTMSIM ground-truth comparison --------------------------------------


_CTMSIM_AGG_COLUMNS = ("vht", "vmt", "delay", "productivity_loss")


@dataclass
class CTMSIMValidation:
    """Two-part sim-vs-CTMSIM comparison report.

    Attributes
    ----------
    per_cell : pandas.DataFrame
        One row per cell with density + mainline-flow RMSE and MAPE.
        Columns mirror :func:`compare_against_historical`'s output for
        consistency: ``cell``, ``n_density_samples``, ``density_rmse``,
        ``density_mape``, ``n_flow_samples``, ``flow_rmse``,
        ``flow_mape``. Density in veh/mi, flow in veh/h, MAPE in %.
    per_metric : pandas.DataFrame
        One row per aggregate metric (vht, vmt, delay,
        productivity_loss). Columns: ``metric``, ``n_samples``,
        ``rmse``, ``mape``. RMSE in the metric's native unit
        (veh*h, veh*mi, veh*h, mi*h respectively); MAPE in %.
    """

    per_cell: pd.DataFrame
    per_metric: pd.DataFrame


def compare_against_ctmsim(
    result: SimulationResult,
    ctmsim_results_dir: Union[Path, str],
    *,
    plot_samples: int = 288,
    sim_steps_per_plot_sample: int = 30,
) -> CTMSIMValidation:
    """Compare our CTM trajectory + aggregate metrics to CTMSIM reference output.

    CTMSIM exports its trajectories at a fixed plotting cadence
    (``plotTS = 5 min`` by default, i.e. 30 sim steps for the canonical
    ``TS = 10 s`` over a 24-hour horizon -> 288 plotting samples).
    Our sim runs at the native ``dt`` cadence; we sample it at the same
    plotting boundaries before computing per-cell and per-metric
    RMSE/MAPE.

    The CSV layout under ``ctmsim_results/<day>/`` (per CTMSIM User
    Guide §8, transposed to wide format):

    * ``density.csv``           -- ``(K+1, N+1)`` state at plotting
                                    boundaries; the last column is a
                                    to-be-ignored placeholder.
    * ``mainline_flow.csv``     -- ``(K+1, N+1)`` flow; col 0 is the
                                    upstream boundary inflow, cols
                                    ``1..N`` are the flow entering each
                                    cell (= our ``f_0..f_{N-1}``); the
                                    trailing row is a junk sample.
    * ``aggregate_metrics.csv`` -- ``(K+1, 4)`` per-period totals; row
                                    0 is the initial-state snapshot
                                    (skipped); rows ``1..K`` are
                                    ``[vht, vmt, delay, ploss]``.

    Parameters
    ----------
    result : SimulationResult
        Output of :func:`utils.ctm.engine.simulate`. Must have horizon
        ``plot_samples * sim_steps_per_plot_sample``.
    ctmsim_results_dir : Path or str
        Directory containing the three reference CSVs above (typically
        ``utils/ctm/ctmsim_results/<day>/``).
    plot_samples : int, default 288
        Number of plotting samples (``K``) in the CTMSIM CSVs --
        24 hours / 5 min per sample = 288 for the I-210W reference.
    sim_steps_per_plot_sample : int, default 30
        Sim steps per plotting period (``plotTS / TS``). 30 for the
        canonical 10-s sim / 5-min plot cadence.

    Returns
    -------
    CTMSIMValidation
        See class docstring.

    Raises
    ------
    ValueError
        Sim horizon doesn't match ``plot_samples * sim_steps_per_plot_sample``,
        or the reference CSVs have unexpected shapes.
    FileNotFoundError
        Required reference CSVs are missing from ``ctmsim_results_dir``.
    """
    P = int(sim_steps_per_plot_sample)
    K = int(plot_samples)
    if result.n_steps != K * P:
        raise ValueError(
            f"result.n_steps={result.n_steps} doesn't match "
            f"plot_samples * sim_steps_per_plot_sample = {K} * {P} = {K * P}; "
            "pass matching --plot-samples / --sim-steps-per-plot-sample "
            "or re-run the sim at the canonical CTMSIM cadence."
        )

    ctmsim_results_dir = Path(ctmsim_results_dir)
    n_cells = result.freeway.n_cells

    # ---- Density: state at plotting boundaries (N, K+1).
    ours_density = result.density[:, ::P]
    ctm_density = _load_ctmsim_csv(
        ctmsim_results_dir / "density.csv",
        expected_shape=(K + 1, n_cells + 1),
        label="density",
    )[:, :-1].T

    # ---- Mainline flow: (N, K) at the start of each plotting period.
    ours_flow = result.mainline_flow[:, ::P]
    ctm_flow_full = _load_ctmsim_csv(
        ctmsim_results_dir / "mainline_flow.csv",
        expected_shape=(K + 1, n_cells + 1),
        label="mainline_flow",
    )
    # Drop the trailing junk row, drop col 0 (upstream boundary inflow,
    # not one of our per-cell mainline flows), transpose to (N, K).
    ctm_flow = ctm_flow_full[:-1, 1:].T

    # ---- Per-cell stats.
    per_cell_rows = []
    for i in range(n_cells):
        n_d, dens_rmse, dens_mape = _rmse_mape(ours_density[i], ctm_density[i])
        n_f, flow_rmse, flow_mape = _rmse_mape(ours_flow[i], ctm_flow[i])
        per_cell_rows.append({
            "cell": i,
            "n_density_samples": n_d,
            "density_rmse": dens_rmse,
            "density_mape": dens_mape,
            "n_flow_samples": n_f,
            "flow_rmse": flow_rmse,
            "flow_mape": flow_mape,
        })
    per_cell = pd.DataFrame(
        per_cell_rows, columns=[
            "cell",
            "n_density_samples", "density_rmse", "density_mape",
            "n_flow_samples", "flow_rmse", "flow_mape",
        ],
    )

    # ---- Aggregate-metrics comparison.
    m = compute_metrics(result)
    ctm_agg = _load_ctmsim_csv(
        ctmsim_results_dir / "aggregate_metrics.csv",
        expected_shape=(K + 1, len(_CTMSIM_AGG_COLUMNS)),
        label="aggregate_metrics",
    )[1: K + 1, :]   # drop the initial-state row

    ours_agg_per_period = {
        "vht": _aggregate_to_periods(m.vht, P, K),
        "vmt": _aggregate_to_periods(m.vmt, P, K),
        "delay": _aggregate_to_periods(m.delay, P, K),
        "productivity_loss": _aggregate_to_periods(m.productivity_loss, P, K),
    }
    per_metric_rows = []
    for j, name in enumerate(_CTMSIM_AGG_COLUMNS):
        n, rmse, mape = _rmse_mape(ours_agg_per_period[name], ctm_agg[:, j])
        per_metric_rows.append({
            "metric": name, "n_samples": n,
            "rmse": rmse, "mape": mape,
        })
    per_metric = pd.DataFrame(
        per_metric_rows, columns=["metric", "n_samples", "rmse", "mape"],
    )

    return CTMSIMValidation(per_cell=per_cell, per_metric=per_metric)


def _load_ctmsim_csv(
    path: Path, *, expected_shape: tuple[int, int], label: str,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"CTMSIM reference {label}.csv not found at {path}."
        )
    arr = np.loadtxt(path, delimiter=",")
    if arr.shape != expected_shape:
        raise ValueError(
            f"CTMSIM reference {label}.csv has shape {arr.shape}, "
            f"expected {expected_shape}."
        )
    return arr


def _aggregate_to_periods(per_step: np.ndarray, P: int, K: int) -> np.ndarray:
    """Sum a (T,) per-sim-step series into per-plotting-period totals (K,)."""
    return per_step[: K * P].reshape(K, P).sum(axis=1)
