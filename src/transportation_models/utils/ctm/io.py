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
    gamma               -       on-ramp blending factor; defaults to 1 (CTMSIM
                                canonical, matches our cells.cells_from_corridor
                                convention of on-ramps at the upstream edge;
                                set 0 to recover Gomes & Horowitz)
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
import scipy.io as sio

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


# ---- CTMSIM .mat adapter --------------------------------------------------
#
# Reads the MATLAB configuration file schema documented in Kurzhanskiy diss.
# Appendix A (p. 150-154). The on-ramp blending factor gamma and allocation
# factor xi default to 1 in CTMSIM (vs. 0 / 1 in our Cell), so the adapter
# pulls them through explicitly from each cell struct.


def _ctmsim_has_name(field) -> bool:
    """True if a CTMSIM ``ORname`` / ``FRname`` field carries a ramp.

    Empty char arrays come through as ``ndarray`` with size 0; empty strings
    come through as ``""``. Either means "no ramp."
    """
    if isinstance(field, str):
        return bool(field)
    return hasattr(field, "size") and field.size > 0


def _cell_from_ctmsim_struct(struct) -> Cell:
    """Translate one CTMSIM cell struct into a :class:`Cell`.

    ``v_f`` and ``w`` are derived from the triangular FD identity
    ``q_max = v_f * rho_crit = w * (rho_jam - rho_crit)`` (CTMSIM stores the
    triangle by capacity and densities, not by slopes).
    """
    length = abs(float(struct.PMend) - float(struct.PMstart))
    rho_crit = float(struct.FDrhocrit)
    rho_jam = float(struct.FDrhojam)
    q_max = float(struct.FDfmax)
    v_f = q_max / rho_crit
    w = q_max / (rho_jam - rho_crit)
    on_ramp = _ctmsim_has_name(struct.ORname)
    off_ramp = _ctmsim_has_name(struct.FRname)
    kwargs: dict = dict(
        length=length,
        q_max=q_max,
        v_f=v_f,
        w=w,
        rho_jam=rho_jam,
        rho_crit=rho_crit,
        on_ramp=on_ramp,
        off_ramp=off_ramp,
    )
    if on_ramp:
        kwargs["on_ramp_capacity"] = float(struct.ORfmax)
        kwargs["gamma"] = float(struct.ORgamma)
        kwargs["xi"] = float(struct.ORxi)
    if off_ramp:
        kwargs["off_ramp_capacity"] = float(struct.FRfmax)
    return Cell(**kwargs)


def _load_ctmsim_mat(path: Union[str, Path]) -> dict:
    """Load a CTMSIM ``.mat`` file with squeeze + struct-as-attribute access."""
    return sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)


def freeway_from_ctmsim_mat(path: Union[str, Path]) -> Freeway:
    """Build a :class:`Freeway` from a CTMSIM MATLAB configuration file.

    Reads ``celldata`` (cell geometry, FD, ramp presence and capacity, on-ramp
    blending) and ``TS`` (simulation step in hours) from a CTMSIM ``.mat``;
    schema: Kurzhanskiy diss. Appendix A (p. 150-154). Free-flow speed and
    congestion-wave speed are derived from the triangular FD identities;
    standard Cell + Freeway validation runs before return.
    """
    mat = _load_ctmsim_mat(path)
    cells = [_cell_from_ctmsim_struct(c) for c in mat["celldata"]]
    return Freeway(dt=float(mat["TS"]), cells=cells).validate()


def ctmsim_initial_densities(path: Union[str, Path]) -> np.ndarray:
    """Extract the initial density vector ``rho0`` from a CTMSIM ``.mat``."""
    mat = _load_ctmsim_mat(path)
    return np.asarray(mat["initialDensities"], dtype=float).reshape(-1)


def ctmsim_demand_at_sim_steps(path: Union[str, Path]) -> np.ndarray:
    """Expand CTMSIM's ``demandProfile`` to per-step on-ramp demand.

    ``demandProfile`` is sampled at the display cadence ``plotTS`` (e.g. 5 min),
    one column per cell with a 1-based offset (column ``k`` corresponds to
    cell ``k-1`` in our 0-indexed convention); the boundary "ML" cell's
    upstream demand lives in this same array. The simulation step ``TS`` is
    typically much finer (e.g. 10 s), and CTMSIM holds each sample constant
    across ``plotTS / TS`` simulation steps (zero-order hold).

    Returns an array of shape ``(n_cells, T)`` where ``T = n_samples *
    plotTS / TS``. Cells without an on-ramp receive a zero column, which is
    what :meth:`Scenario.validate` requires.
    """
    mat = _load_ctmsim_mat(path)
    profile = np.asarray(mat["demandProfile"], dtype=float)
    n_samples = profile.shape[0]
    n_cells = len(mat["celldata"])
    steps_per_sample = int(round(float(mat["plotTS"]) / float(mat["TS"])))

    # Column k in demandProfile corresponds to cell k-1 (CTMSIM is 1-indexed).
    # Columns 0 and the trailing N+1 are unused/padding and dropped here.
    per_cell = profile[:, 1 : n_cells + 1]                       # (n_samples, n_cells)
    expanded = np.repeat(per_cell, steps_per_sample, axis=0)      # (T, n_cells)
    return expanded.T                                             # (n_cells, T)


# ---- Scenario adapter -----------------------------------------------------


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
