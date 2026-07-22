"""Data model for the DWPT demand module: ``PadSpec`` and ``CorridorSpec``.

``PadSpec`` carries the physical Tx/Rx pad parameters from Newbolt 2024a;
``CorridorSpec`` composes a corridor's per-cell layout with a ``PadSpec``
and a position-grid spacing ``dx_grid``. Both validate on construction.

See ``docs/dwpt_demand_module.md`` §"Spatial Dependence" for the
definitions of ``alpha``, ``delta``, ``lambda_gap``, ``beta``,
``beta_prime``, ``gamma``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Float tolerance for "is ``value`` an integer multiple of ``dx_grid``?"
# checks. ~1e-6 is comfortably above float64 round-trip noise for the
# scales we care about (meters, sub-millimeter grids).
_GRID_ALIGN_TOL = 1e-6


@dataclass(frozen=True)
class PadSpec:
    """Tx/Rx pad geometry and power capacities.

    Parameters
    ----------
    alpha : float
        Tx pad length, meters.
    delta : float
        Rx pad length, meters.
    lambda_gap : float
        Gap between consecutive Tx pads along the corridor, meters.
    beta : float
        Rx pad rated power capacity, watts.
    beta_prime : float
        Tx pad rated power capacity, watts.
    """

    alpha: float
    delta: float
    lambda_gap: float
    beta: float
    beta_prime: float

    def __post_init__(self) -> None:
        for name in ("alpha", "delta", "lambda_gap", "beta", "beta_prime"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(
                    f"PadSpec.{name} must be positive, got {value}"
                )

    @property
    def gamma(self) -> float:
        """Power transfer cap, ``min(beta, beta_prime)`` in watts."""
        return min(self.beta, self.beta_prime)


@dataclass(frozen=True)
class CorridorSpec:
    """Corridor cell layout, pad spec, and position-grid spacing.

    Parameters
    ----------
    cell_lengths_m : tuple[float, ...]
        Per-cell lengths in meters, ordered along the corridor.
    pad : PadSpec
        Pad geometry and capacities.
    dx_grid : float
        Spatial discretization for the spatial submodule, meters. All
        pad dimensions and cell lengths must be integer multiples of
        this value (split-position case is deferred per the spec).
    """

    cell_lengths_m: tuple[float, ...]
    pad: PadSpec
    dx_grid: float

    def __post_init__(self) -> None:
        if self.dx_grid <= 0:
            raise ValueError(
                f"dx_grid must be positive, got {self.dx_grid}"
            )
        if not self.cell_lengths_m:
            raise ValueError("cell_lengths_m must be non-empty")
        for i, length in enumerate(self.cell_lengths_m):
            if length <= 0:
                raise ValueError(
                    f"cell_lengths_m[{i}] must be positive, got {length}"
                )
            if not _is_multiple_of(length, self.dx_grid):
                raise ValueError(
                    f"cell_lengths_m[{i}]={length} is not an integer "
                    f"multiple of dx_grid={self.dx_grid}"
                )
        for name in ("alpha", "delta", "lambda_gap"):
            value = getattr(self.pad, name)
            if not _is_multiple_of(value, self.dx_grid):
                raise ValueError(
                    f"PadSpec.{name}={value} is not an integer multiple "
                    f"of dx_grid={self.dx_grid}"
                )

    @property
    def corridor_length_m(self) -> float:
        return sum(self.cell_lengths_m)

    @property
    def n_corridor(self) -> int:
        """Number of position-grid units spanning the corridor."""
        return round(self.corridor_length_m / self.dx_grid)

    @property
    def alpha_grid(self) -> int:
        return round(self.pad.alpha / self.dx_grid)

    @property
    def delta_grid(self) -> int:
        return round(self.pad.delta / self.dx_grid)

    @property
    def lambda_grid(self) -> int:
        return round(self.pad.lambda_gap / self.dx_grid)

    @property
    def m_traversal(self) -> int:
        """Rx traversal positions (corridor + Rx entry/exit extent)."""
        return self.n_corridor + self.delta_grid - 1

    @property
    def positions_per_cell(self) -> tuple[int, ...]:
        return tuple(
            round(length / self.dx_grid) for length in self.cell_lengths_m
        )

    @property
    def position_m(self) -> np.ndarray:
        """Column position labels (meters), center-reference convention.

        ``position_m[j] = (j - off) * dx_grid`` with ``off = (delta_grid-1)//2``;
        off-corridor boundary positions get out-of-``[0, L)`` labels.
        """
        off = (self.delta_grid - 1) // 2
        return (np.arange(self.m_traversal) - off) * self.dx_grid

    def build_P_y(self, *, method: str = "auto") -> np.ndarray:
        """Power-vs-position profile for this corridor (spatial submodule).

        Composes ``build_S_T``, ``build_S_R``, and ``build_P_y`` over the
        corridor's grid properties, returning ``P_y`` of length
        ``m_traversal``. ``method`` is forwarded to ``build_P_y``; the default
        ``"auto"`` selects convolution or FFT by kernel size.
        """
        from .spatial import build_P_y, build_S_R, build_S_T

        S_T = build_S_T(
            self.n_corridor, self.alpha_grid, self.lambda_grid, self.pad.beta_prime
        )
        S_R = build_S_R(self.delta_grid, self.pad.beta)
        return build_P_y(S_T, S_R, self.pad.gamma, method=method)


@dataclass(frozen=True)
class DemandResult:
    """Spatiotemporal DWPT demand: ``E`` (Wh) indexed by (timestep, position).

    Parameters
    ----------
    E : ndarray (T, m)
        Energy in watt-hours; rows are timesteps, columns Rx-pad positions.
    position_m : ndarray (m,)
        Column position labels, meters.
    timesteps : ndarray (T,)
        Row labels (timestep index).
    """

    E: np.ndarray
    position_m: np.ndarray
    timesteps: np.ndarray

    def __post_init__(self) -> None:
        if self.E.ndim != 2:
            raise ValueError(f"E must be 2-D (T, m), got shape {self.E.shape}")
        n_t, n_m = self.E.shape
        if self.timesteps.shape != (n_t,):
            raise ValueError(
                f"timesteps must have length T={n_t}, got {self.timesteps.shape}"
            )
        if self.position_m.shape != (n_m,):
            raise ValueError(
                f"position_m must have length m={n_m}, got {self.position_m.shape}"
            )

    def to_dataframe(self):
        """Long-format ``(timestep, position_m, energy_wh)`` table."""
        import pandas as pd

        n_t, n_m = self.E.shape
        return pd.DataFrame(
            {
                "timestep": np.repeat(self.timesteps, n_m),
                "position_m": np.tile(self.position_m, n_t),
                "energy_wh": self.E.ravel(),
            }
        )


def _is_multiple_of(value: float, step: float) -> bool:
    ratio = value / step
    return abs(ratio - round(ratio)) <= _GRID_ALIGN_TOL
