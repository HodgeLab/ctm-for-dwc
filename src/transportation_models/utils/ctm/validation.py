"""Compare CTM simulation results against historical PeMS observations.

Step 1 of the validation pipeline: for each cell, compare the sim's
predicted mainline flow and density to the historical 5-min PeMS
samples at the cell's assigned mainline VDS (``vds_id`` in
``cells.csv`` -- not ``fd_vds_id``; we want observed traffic data,
not the calibration source).

Both quantities are evaluated at PeMS's native 5-min cadence: sim
density is sampled at the start of each 5-min interval (instantaneous
state); sim mainline flow is averaged across the interval (matching
PeMS's interval-rate semantics). RMSE and MAPE are computed across
all non-NaN observed samples per cell; MAPE additionally skips
samples where the observed value is zero (undefined ratio).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .results import SimulationResult


_FIVE_MIN_S = 300.0
_FIVE_MIN_H = 5.0 / 60.0
_GRID_TOL = 1e-9


def compare_against_historical(
    result: SimulationResult,
    cells_df: pd.DataFrame,
    timeseries_dir: Union[Path, str],
    *,
    start: pd.Timestamp,
) -> pd.DataFrame:
    """Per-cell sim-vs-observed RMSE and MAPE for density and mainline flow.

    Parameters
    ----------
    result : SimulationResult
        Output of :func:`utils.ctm.engine.simulate`. The freeway's
        ``dt`` must evenly divide 5 minutes, and the sim horizon
        ``n_steps * dt`` must be a positive whole multiple of 5 minutes.
    cells_df : pandas.DataFrame
        Step 1+2 ``cells.csv`` (must carry ``vds_id``, ``lanes``, and
        ``vds_lanes``). Row count and order must match
        ``result.freeway.cells``. Cells whose ``vds_id`` is NaN are
        skipped.
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
                                      ``<NA>`` if the cell had no assigned VDS
                                      (skipped in stats)
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
        ``cells_df`` is missing ``vds_id``, ``lanes``, or ``vds_lanes``.
    """
    n_cells = result.freeway.n_cells
    if len(cells_df) != n_cells:
        raise ValueError(
            f"cells_df has {len(cells_df)} rows but result.freeway has "
            f"{n_cells} cells; one row per cell is required."
        )
    required = {"vds_id", "lanes", "vds_lanes"}
    missing = required - set(cells_df.columns)
    if missing:
        raise KeyError(
            f"cells_df is missing column(s): {sorted(missing)}. "
            "vds_id / vds_lanes come from Step 2 (assign_vds_to_cells)."
        )

    # Resolve sim cadence vs 5-min cadence.
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
        raise ValueError(
            f"start={start} must align to PeMS's 5-min grid."
        )
    end = start + pd.Timedelta(seconds=n_5min * _FIVE_MIN_S)

    # Lane-count contract (same as initial_state_from_vds): observed
    # density derived from a VDS with N lanes is only directly
    # comparable to a per-cell density on a cell with the same lane
    # count. Otherwise the user must reconcile the discrepancy in
    # cells.csv (typically by trusting the PeMS-reported value).
    cells_with_vds = cells_df.dropna(subset=["vds_id"])
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
        if pd.isna(vds_id_raw):
            rows.append(_nan_row(cell_idx, vds_id=pd.NA))
            continue
        vds_id = int(vds_id_raw)
        if vds_id not in vds_cache:
            vds_cache[vds_id] = pd.read_csv(
                timeseries_dir / f"{vds_id}.csv", parse_dates=["timestamp"],
            )
        df = vds_cache[vds_id]
        window = df.loc[
            (df["timestamp"] >= start) & (df["timestamp"] < end)
        ].sort_values("timestamp").reset_index(drop=True)
        if len(window) != n_5min:
            raise ValueError(
                f"VDS {vds_id}: {len(window)} 5-min sample(s) in window "
                f"[{start}, {end}); expected {n_5min} on a contiguous "
                "5-min grid."
            )
        observed_flow = window["total_flow_[veh/5-min]"].to_numpy(
            dtype=float,
        ) * 12.0
        with np.errstate(divide="ignore", invalid="ignore"):
            observed_density = observed_flow / window[
                "avg_speed_[mph]"
            ].to_numpy(dtype=float)

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


# ---- Helpers --------------------------------------------------------------


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
