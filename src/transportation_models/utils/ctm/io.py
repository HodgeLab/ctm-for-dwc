"""DataFrame / CSV adapters between tabular external data and the CTM engine.

Two seams are exposed:

* ``freeway_*`` -- the static, *physical* configuration: cell geometry, FD
  parameters, ramp presence/capacity, and on-ramp blending. One row per cell,
  in DataFrame order upstream-to-downstream.
* ``scenario_*`` -- the time-varying *experimental* inputs to run on that
  freeway: on-ramp demand, off-ramp split ratios, and upstream boundary
  inflow, plus initial state. Wide format, time index by step, cell columns.

Freeway schema
==============

required columns
    length              mi      cell length l_i
    q_max               veh/h   capacity q_max,i
    v_f                 mi/h    free-flow speed v_f,i
    w                   mi/h    congestion-wave speed w_i
    rho_jam             veh/mi  jam density rho_jam,i

optional columns (fall back to :class:`Cell` defaults if missing or NaN)
    rho_crit            veh/mi  critical density; defaults to q_max / v_f
    on_ramp             bool    whether the cell has an on-ramp;  default False
    off_ramp            bool    whether the cell has an off-ramp; default False
    on_ramp_capacity    veh/h   R_i; defaults to inf (must stay inf if on_ramp=False)
    off_ramp_capacity   veh/h   S_i; defaults to inf (must stay inf if off_ramp=False)
    gamma               -       on-ramp blending factor; defaults to 0
    xi                  -       on-ramp allocation factor; defaults to 1

Scenario schema
===============

``scenario_from_dataframes`` takes the freeway, the boundary ``inflow`` (1-D,
one value per step -- a Series or single-column DataFrame), and optional
wide DataFrames for ``demand`` and ``beta`` (rows = steps, columns = cell
indices). Cells absent from a wide DataFrame default to zero. Initial
density ``rho0`` and queue ``q0`` can be passed as arrays / scalars.

The upstream pipeline that builds the per-cell freeway table -- VDS-level
FD calibration, the VDS-to-cell mapping, ramp placement -- is a separate
workstream (steps 2-5 in ``docs/ctm_module.md``); this module assumes it
has already been done.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .model import Cell, Freeway, Scenario

REQUIRED_COLUMNS: tuple[str, ...] = ("length", "q_max", "v_f", "w", "rho_jam")
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "rho_crit",
    "on_ramp",
    "off_ramp",
    "on_ramp_capacity",
    "off_ramp_capacity",
    "gamma",
    "xi",
)
_BOOL_COLUMNS: frozenset[str] = frozenset({"on_ramp", "off_ramp"})
_ALL_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


# ---- Freeway adapter ------------------------------------------------------


def _to_bool(value) -> bool:
    """Coerce a DataFrame cell to ``bool``, handling string CSV values."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "t")
    return bool(value)


def freeway_from_dataframe(df: pd.DataFrame, dt: float) -> Freeway:
    """Build a :class:`Freeway` from a tabular cell config.

    Rows are taken in DataFrame order as the upstream-to-downstream cell
    sequence; the time step ``dt`` is supplied separately. Required and
    optional columns are documented in the module docstring. Cell-level and
    freeway-level validation runs before return (FD consistency, CFL rule,
    ramp/capacity consistency); construction fails loudly on a malformed
    table.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Expected at minimum: {list(REQUIRED_COLUMNS)}."
        )

    cells: list[Cell] = []
    for _, row in df.iterrows():
        kwargs: dict = {}
        for col in _ALL_COLUMNS:
            if col not in df.columns or pd.isna(row[col]):
                continue
            kwargs[col] = _to_bool(row[col]) if col in _BOOL_COLUMNS else float(row[col])
        cells.append(Cell(**kwargs))
    return Freeway(dt=float(dt), cells=cells).validate()


def freeway_to_dataframe(freeway: Freeway) -> pd.DataFrame:
    """Dump a :class:`Freeway` to a one-row-per-cell DataFrame using this module's schema."""
    rows = [{col: getattr(c, col) for col in _ALL_COLUMNS} for c in freeway.cells]
    return pd.DataFrame(rows, columns=list(_ALL_COLUMNS))


def freeway_from_csv(path: Union[str, Path], dt: float) -> Freeway:
    """Convenience: read a CSV with the freeway schema and build a :class:`Freeway`."""
    df = pd.read_csv(path)
    return freeway_from_dataframe(df, dt=dt)


# ---- Scenario adapter -----------------------------------------------------


def _inflow_to_array(inflow) -> np.ndarray:
    if isinstance(inflow, pd.DataFrame):
        if inflow.shape[1] != 1:
            raise ValueError(
                f"inflow DataFrame must have exactly 1 column; got {inflow.shape[1]}"
            )
        return inflow.iloc[:, 0].to_numpy(dtype=float)
    if isinstance(inflow, pd.Series):
        return inflow.to_numpy(dtype=float)
    return np.asarray(inflow, dtype=float).reshape(-1)


def _wide_frame_to_cell_array(
    df: pd.DataFrame | None, name: str, n_cells: int, n_steps: int
) -> np.ndarray:
    """Convert a wide ``(steps x cells)`` DataFrame into a ``(n_cells, n_steps)`` array.

    Cells absent from ``df.columns`` default to zero. Column labels must be
    cell indices in ``[0, n_cells)``.
    """
    if df is None:
        return np.zeros((n_cells, n_steps))
    if df.shape[0] != n_steps:
        raise ValueError(
            f"{name} has {df.shape[0]} rows but the scenario has {n_steps} steps"
        )
    arr = np.zeros((n_cells, n_steps))
    for col in df.columns:
        try:
            idx = int(col)
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} column labels must be integer cell indices; got {col!r}"
            )
        if not 0 <= idx < n_cells:
            raise ValueError(
                f"{name} column {idx} is out of range [0, {n_cells})"
            )
        arr[idx, :] = df[col].to_numpy(dtype=float)
    return arr


def scenario_from_dataframes(
    freeway: Freeway,
    *,
    inflow,
    demand: pd.DataFrame | None = None,
    beta: pd.DataFrame | None = None,
    rho0=0.0,
    q0=0.0,
) -> Scenario:
    """Build a :class:`Scenario` from wide-format time-varying DataFrames.

    Parameters
    ----------
    freeway : Freeway
        Used to size the scenario (``n_cells``) and to validate ramp consistency.
    inflow : Series / DataFrame / array-like, shape (T,)
        Upstream boundary demand ``f_0(k)`` -- required; its length sets ``T``.
    demand, beta : pd.DataFrame, optional
        Wide ``(T x n_cells)`` frames; cells absent from the columns default to
        zero. Column labels must be integer cell indices in ``[0, n_cells)``.
    rho0, q0 : array-like or scalar, optional
        Initial density and on-ramp queue, broadcast to ``(n_cells,)``.
    """
    inflow_arr = _inflow_to_array(inflow)
    n = freeway.n_cells
    T = inflow_arr.shape[0]
    demand_arr = _wide_frame_to_cell_array(demand, "demand", n, T)
    beta_arr = _wide_frame_to_cell_array(beta, "beta", n, T)
    return Scenario.build(
        freeway,
        steps=T,
        rho0=rho0,
        q0=q0,
        inflow=inflow_arr,
        demand=demand_arr,
        beta=beta_arr,
    )


def scenario_to_dataframes(
    scenario: Scenario,
) -> dict[str, Union[pd.DataFrame, pd.Series]]:
    """Dump a :class:`Scenario`'s time-varying inputs to wide-format frames.

    Returns ``demand``, ``beta`` (DataFrames, step index, cell columns),
    ``inflow`` (Series, step index), and ``rho0``, ``q0`` (Series, cell index).
    """
    n, T = scenario.demand.shape
    steps = pd.Index(np.arange(T), name="step")
    cells = pd.Index(np.arange(n), name="cell")
    return {
        "demand": pd.DataFrame(scenario.demand.T, index=steps, columns=cells),
        "beta": pd.DataFrame(scenario.beta.T, index=steps, columns=cells),
        "inflow": pd.Series(scenario.inflow, index=steps, name="inflow"),
        "rho0": pd.Series(scenario.rho0, index=cells, name="rho0"),
        "q0": pd.Series(scenario.q0, index=cells, name="q0"),
    }
