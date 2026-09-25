"""Output container for a CTM simulation run.

A :class:`SimulationResult` packs the per-step state and flows produced by
:func:`engine.simulate`. State arrays carry ``T + 1`` columns (the initial
condition at column 0 plus the post-step values at columns ``1..T``), while
flow arrays carry ``T`` columns (one per step). The :meth:`to_dataframes`
helper exposes the same data in wide format with a time index in hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .model import Cell, Freeway, Scenario

DataFrameOrSeries = Union[pd.DataFrame, pd.Series]

# Per-cell Cell fields persisted in the .npz (covers all of Cell's parameters,
# so freeway geometry / FD / ramp config round-trips exactly).
_CELL_FIELDS: tuple[str, ...] = (
    "length", "q_max", "v_f", "w", "rho_jam", "rho_crit",
    "on_ramp", "off_ramp", "on_ramp_capacity", "off_ramp_capacity",
    "gamma", "xi",
)
# The per-step arrays carried on a SimulationResult.
_RESULT_ARRAYS: tuple[str, ...] = (
    "density", "queue", "mainline_flow", "off_ramp", "on_ramp",
    "speed", "boundary_inflow",
)
# The Scenario arrays (the run's inputs), kept so the round-trip is faithful.
_SCENARIO_ARRAYS: tuple[str, ...] = (
    "rho0", "inflow", "demand", "beta", "q0",
)


@dataclass
class SimulationResult:
    """Per-step state, flows, and ancillaries from a CTM simulation run."""

    freeway: Freeway
    scenario: Scenario
    density: np.ndarray          # (n_cells, T+1)  rho_i(k) [veh/mi]
    queue: np.ndarray            # (n_cells, T+1)  q_i(k)   [veh]
    mainline_flow: np.ndarray    # (n_cells, T)    f_i(k)   [veh/h]
    off_ramp: np.ndarray         # (n_cells, T)    s_i(k)   [veh/h]
    on_ramp: np.ndarray          # (n_cells, T)    r_i(k)   [veh/h]
    speed: np.ndarray            # (n_cells, T)    V_i(k)   [mi/h]
    boundary_inflow: np.ndarray  # (T,)            admitted f_0(k) [veh/h]
    start: Optional[pd.Timestamp] = None  # wall-clock anchor of step 0, if known

    @property
    def n_cells(self) -> int:
        return self.freeway.n_cells

    @property
    def n_steps(self) -> int:
        return self.mainline_flow.shape[1]

    @property
    def time_state(self) -> np.ndarray:
        """Time at each state sample [h from start], length ``T + 1``."""
        return np.arange(self.n_steps + 1) * self.freeway.dt

    @property
    def time_flow(self) -> np.ndarray:
        """Time at the start of each step's flow [h from start], length ``T``."""
        return np.arange(self.n_steps) * self.freeway.dt

    def to_dataframes(self) -> dict[str, DataFrameOrSeries]:
        """Return wide-format frames keyed by quantity: time index, cell columns.

        State frames (``density``, ``queue``) have ``T + 1`` rows indexed at the
        step boundaries; flow frames (``mainline_flow``, ``off_ramp``,
        ``on_ramp``, ``speed``) have ``T`` rows indexed at the *start* of each
        step. ``boundary_inflow`` is a scalar time series, returned as a
        :class:`pandas.Series`.
        """
        cells = pd.Index(range(self.n_cells), name="cell")
        idx_state = pd.Index(self.time_state, name="time_h")
        idx_flow = pd.Index(self.time_flow, name="time_h")

        def frame(arr: np.ndarray, idx: pd.Index) -> pd.DataFrame:
            # arrays are (n_cells, T_arr); transpose so time runs down rows.
            return pd.DataFrame(arr.T, index=idx, columns=cells)

        return {
            "density": frame(self.density, idx_state),
            "queue": frame(self.queue, idx_state),
            "mainline_flow": frame(self.mainline_flow, idx_flow),
            "off_ramp": frame(self.off_ramp, idx_flow),
            "on_ramp": frame(self.on_ramp, idx_flow),
            "speed": frame(self.speed, idx_flow),
            "boundary_inflow": pd.Series(
                self.boundary_inflow, index=idx_flow, name="boundary_inflow"
            ),
        }

    def to_npz(self, path: Union[str, Path]) -> None:
        """Save this result to a single self-contained ``.npz`` file.

        Bundles the per-step arrays plus enough of the ``Freeway`` (``dt`` and
        every per-cell parameter) and ``Scenario`` to reconstruct an equivalent
        :class:`SimulationResult` via :meth:`from_npz` -- no original config
        objects or CSVs needed. Intended for handing a run off to the DWC
        demand pipeline (``dwc.adapter``).
        """
        data: dict[str, np.ndarray] = {
            f"result__{name}": getattr(self, name) for name in _RESULT_ARRAYS
        }
        data["freeway__dt"] = np.asarray(self.freeway.dt, dtype=float)
        for field in _CELL_FIELDS:
            data[f"cell__{field}"] = np.array(
                [getattr(c, field) for c in self.freeway.cells]
            )
        for name in _SCENARIO_ARRAYS:
            data[f"scenario__{name}"] = getattr(self.scenario, name)
        if self.start is not None:
            data["meta__start"] = np.array(pd.Timestamp(self.start).isoformat())
        np.savez(path, **data)

    @classmethod
    def from_npz(cls, path: Union[str, Path]) -> "SimulationResult":
        """Reconstruct a :class:`SimulationResult` saved by :meth:`to_npz`.

        Rebuilds the ``Freeway`` and ``Scenario`` and revalidates both (so a
        corrupted or hand-edited file fails loudly), then re-packs the per-step
        arrays.
        """
        npz = np.load(path)
        dt = float(npz["freeway__dt"])
        cols = {field: npz[f"cell__{field}"] for field in _CELL_FIELDS}
        cells = [
            Cell(**{
                field: bool(cols[field][i]) if field in ("on_ramp", "off_ramp")
                else float(cols[field][i])
                for field in _CELL_FIELDS
            })
            for i in range(len(cols["length"]))
        ]
        freeway = Freeway(dt=dt, cells=cells).validate()
        scenario = Scenario(
            **{name: npz[f"scenario__{name}"] for name in _SCENARIO_ARRAYS}
        ).validate(freeway)
        start = (
            pd.Timestamp(str(npz["meta__start"])) if "meta__start" in npz
            else None
        )
        return cls(
            freeway=freeway,
            scenario=scenario,
            start=start,
            **{name: npz[f"result__{name}"] for name in _RESULT_ARRAYS},
        )
