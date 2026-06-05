"""Compare CTM simulation results against ground-truth observations.

Two ground-truth sources are supported today:

* **Historical PeMS** (:func:`compare_against_historical`) -- the field
  data the model is calibrated against. Compares sim density and
  mainline flow to the cell's assigned mainline VDS over the sim
  window. RMSE/MAPE per cell.

* **CTMSIM reference output** (:func:`compare_against_ctmsim`) -- the
  Kurzhanskiy 2007 / Aurora reference CSVs bundled under
  ``utils/ctm/ctmsim_results/<day>/``. Compares sim density, mainline
  flow, and the four aggregate metrics (VHT, VMT, delay, productivity
  loss) to CTMSIM v1.1's CSV exports for the same .mat config.
  Useful for regression-style validation of engine behavior against
  the canonical reference implementation.

Both functions return RMSE and MAPE; MAPE skips zero-observed samples
(undefined ratio).
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
