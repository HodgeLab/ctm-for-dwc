"""CTM data model: ``Cell``, ``Freeway``, and ``Scenario`` configuration objects.

Static configuration for the canonical CTMSIM update documented in
``docs/ctm_module.md`` (Kurzhanskiy 2007, sec. 4.2.1). All quantities are
dimensional:

==========  =========
quantity    unit
==========  =========
length      mi
time        h
speed       mi/h
flow        veh/h
density     veh/mi
queue       veh
==========  =========
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

# Relative tolerance for the triangular-FD consistency check. Step-2 calibrated
# parameters are fit to a triangle and should satisfy it tightly.
FD_RTOL = 1e-3


@dataclass
class Cell:
    """One CTM cell: geometry, triangular fundamental diagram, ramp parameters.

    The triangular FD has three degrees of freedom; its five parameters satisfy
    ``q_max = v_f * rho_crit = w * (rho_jam - rho_crit)`` (docs/ctm_module.md,
    eq. 3.1). ``rho_crit`` defaults to ``q_max / v_f`` when omitted.
    """

    length: float           # l_i       cell length [mi]
    q_max: float            # q_max,i   mainline capacity [veh/h]
    v_f: float              # v_f,i     free-flow speed [mi/h]
    w: float                # w_i       congestion-wave speed [mi/h]
    rho_jam: float          # rho_jam,i jam density [veh/mi]
    rho_crit: float | None = None      # rho_crit,i critical density [veh/mi]
    on_ramp: bool = False              # whether the cell has an on-ramp
    off_ramp: bool = False             # whether the cell has an off-ramp
    on_ramp_capacity: float = np.inf   # R_i [veh/h]
    off_ramp_capacity: float = np.inf  # S_i [veh/h]
    gamma: float = 1.0      # gamma_i   on-ramp blending factor [-]
    xi: float = 1.0         # xi_i      on-ramp allocation factor [-]
    # gamma defaults to 1.0 to match the CTMSIM canonical update + the
    # cell-boundary convention encoded in `cells.cells_from_corridor`
    # (on-ramps live at the *upstream* edge of their cell, so on-ramp arrivals
    # have a full ΔT to traverse the cell at free-flow speed and contribute to
    # this step's sending function). Set gamma=0.0 explicitly to recover the
    # Gomes & Horowitz / four-mode form used in the dissertation's Examples.

    def __post_init__(self) -> None:
        if self.rho_crit is None:
            self.rho_crit = self.q_max / self.v_f
        self._check_structure()

    def _check_structure(self) -> None:
        problems = []
        for name in ("length", "q_max", "v_f", "w", "rho_jam", "rho_crit"):
            if getattr(self, name) <= 0:
                problems.append(f"{name} must be > 0 (got {getattr(self, name)})")
        if not 0 < self.rho_crit < self.rho_jam:
            problems.append(
                f"rho_crit must lie in (0, rho_jam); got rho_crit={self.rho_crit}, "
                f"rho_jam={self.rho_jam}"
            )
        if not 0.0 <= self.gamma <= 1.0:
            problems.append(f"gamma must lie in [0, 1] (got {self.gamma})")
        if not 0.0 < self.xi <= 1.0:
            problems.append(f"xi must lie in (0, 1] (got {self.xi})")
        if self.on_ramp_capacity <= 0:
            problems.append(f"on_ramp_capacity must be > 0 (got {self.on_ramp_capacity})")
        if self.off_ramp_capacity <= 0:
            problems.append(f"off_ramp_capacity must be > 0 (got {self.off_ramp_capacity})")
        # Capacity must be left at the default (inf) when the corresponding ramp
        # is absent; otherwise the config is contradictory.
        if not self.on_ramp and np.isfinite(self.on_ramp_capacity):
            problems.append(
                f"on_ramp_capacity={self.on_ramp_capacity} given but on_ramp=False; "
                "set on_ramp=True or leave on_ramp_capacity at its default (inf)"
            )
        if not self.off_ramp and np.isfinite(self.off_ramp_capacity):
            problems.append(
                f"off_ramp_capacity={self.off_ramp_capacity} given but off_ramp=False; "
                "set off_ramp=True or leave off_ramp_capacity at its default (inf)"
            )
        if problems:
            raise ValueError("Invalid Cell: " + "; ".join(problems))

    def fd_residuals(self) -> tuple[float, float]:
        """Relative errors of the two triangular-FD identities (0.0 == consistent)."""
        free = abs(self.v_f * self.rho_crit - self.q_max) / self.q_max
        cong = abs(self.w * (self.rho_jam - self.rho_crit) - self.q_max) / self.q_max
        return free, cong

    def check_fd_consistency(self, rtol: float = FD_RTOL) -> None:
        free, cong = self.fd_residuals()
        if free > rtol or cong > rtol:
            raise ValueError(
                "Triangular FD inconsistent (q_max must equal v_f*rho_crit and "
                f"w*(rho_jam-rho_crit)): free-branch rel.err={free:.2e}, "
                f"cong-branch rel.err={cong:.2e}, rtol={rtol:.0e}"
            )


class FreewayArrays(NamedTuple):
    """Per-cell parameters compiled to NumPy arrays for the update loop.

    Field order matches the unpacking in :func:`engine.step`.
    """

    length: np.ndarray
    q_max: np.ndarray
    v_f: np.ndarray
    w: np.ndarray
    rho_jam: np.ndarray
    on_ramp_capacity: np.ndarray
    off_ramp_capacity: np.ndarray
    gamma: np.ndarray
    xi: np.ndarray


@dataclass
class Freeway:
    """An ordered chain of cells with a common time step ``dt`` (ΔT)."""

    dt: float
    cells: list[Cell]

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0 (got {self.dt})")
        if len(self.cells) == 0:
            raise ValueError("Freeway must have at least one cell")

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def validate(self, rtol: float = FD_RTOL) -> "Freeway":
        """Check FD consistency and the CFL cell-length rule; raise on any failure.

        The CFL rule ``v_f * dt <= length`` (docs/ctm_module.md, Step 5 / eq. 4.1)
        keeps a free-flow vehicle from crossing more than one cell per step.
        """
        problems = []
        for i, c in enumerate(self.cells):
            try:
                c.check_fd_consistency(rtol)
            except ValueError as exc:
                problems.append(f"cell {i}: {exc}")
            if c.v_f * self.dt > c.length:
                problems.append(
                    f"cell {i}: CFL violated, v_f*dt={c.v_f * self.dt:.4g} > "
                    f"length={c.length:.4g}"
                )
        if problems:
            raise ValueError("Invalid Freeway:\n  " + "\n  ".join(problems))
        return self

    def arrays(self) -> FreewayArrays:
        def col(name: str) -> np.ndarray:
            return np.array([getattr(c, name) for c in self.cells], dtype=float)

        return FreewayArrays(
            length=col("length"),
            q_max=col("q_max"),
            v_f=col("v_f"),
            w=col("w"),
            rho_jam=col("rho_jam"),
            on_ramp_capacity=col("on_ramp_capacity"),
            off_ramp_capacity=col("off_ramp_capacity"),
            gamma=col("gamma"),
            xi=col("xi"),
        )


@dataclass
class Scenario:
    """Initial conditions and time-varying inputs for a simulation run.

    ``demand`` and ``beta`` are shaped ``(n_cells, n_steps)``; ``inflow`` is
    ``(n_steps,)``; ``rho0`` and ``q0`` are ``(n_cells,)``. Build with
    :meth:`build` to broadcast scalar shorthands and validate in one step.
    """

    rho0: np.ndarray      # initial density [veh/mi]
    inflow: np.ndarray    # upstream boundary demand f_0(k) [veh/h]
    demand: np.ndarray    # on-ramp arrival demand d_i(k) [veh/h]
    beta: np.ndarray      # off-ramp split ratio beta_i(k) [-]
    q0: np.ndarray        # initial on-ramp queue [veh]

    @classmethod
    def build(
        cls,
        freeway: Freeway,
        steps: int,
        *,
        rho0=0.0,
        inflow=0.0,
        demand=0.0,
        beta=0.0,
        q0=0.0,
    ) -> "Scenario":
        n = freeway.n_cells

        def vec(value, shape) -> np.ndarray:
            return np.array(np.broadcast_to(np.asarray(value, dtype=float), shape), dtype=float)

        scn = cls(
            rho0=vec(rho0, (n,)),
            inflow=vec(inflow, (steps,)),
            demand=vec(demand, (n, steps)),
            beta=vec(beta, (n, steps)),
            q0=vec(q0, (n,)),
        )
        scn.validate(freeway, steps)
        return scn

    @property
    def n_steps(self) -> int:
        return self.inflow.shape[0]

    def validate(self, freeway: Freeway, steps: int | None = None) -> "Scenario":
        n = freeway.n_cells
        t = self.n_steps if steps is None else steps
        problems = []
        if self.rho0.shape != (n,):
            problems.append(f"rho0 shape {self.rho0.shape} != ({n},)")
        if self.q0.shape != (n,):
            problems.append(f"q0 shape {self.q0.shape} != ({n},)")
        if self.inflow.shape != (t,):
            problems.append(f"inflow shape {self.inflow.shape} != ({t},)")
        if self.demand.shape != (n, t):
            problems.append(f"demand shape {self.demand.shape} != ({n}, {t})")
        if self.beta.shape != (n, t):
            problems.append(f"beta shape {self.beta.shape} != ({n}, {t})")
        if np.any(self.beta < 0) or np.any(self.beta > 1):
            problems.append("beta must lie in [0, 1]")
        if np.any(self.demand < 0):
            problems.append("demand must be >= 0")
        if np.any(self.inflow < 0):
            problems.append("inflow must be >= 0")
        if np.any(self.rho0 < 0):
            problems.append("rho0 must be >= 0")
        if np.any(self.q0 < 0):
            problems.append("q0 must be >= 0")
        rho_jam = np.array([c.rho_jam for c in freeway.cells])
        if self.rho0.shape == (n,) and np.any(self.rho0 > rho_jam):
            problems.append("rho0 must be <= rho_jam")
        # Ramp-presence cross-check: a cell without an on-ramp cannot receive
        # demand; a cell without an off-ramp cannot have a non-zero split.
        if self.demand.shape == (n, t):
            bad_on = [
                i for i, c in enumerate(freeway.cells)
                if not c.on_ramp and np.any(self.demand[i, :] != 0)
            ]
            if bad_on:
                problems.append(
                    f"demand must be 0 for cells without an on-ramp; offending cells: {bad_on}"
                )
        if self.beta.shape == (n, t):
            bad_off = [
                i for i, c in enumerate(freeway.cells)
                if not c.off_ramp and np.any(self.beta[i, :] != 0)
            ]
            if bad_off:
                problems.append(
                    f"beta must be 0 for cells without an off-ramp; offending cells: {bad_off}"
                )
        if problems:
            raise ValueError("Invalid Scenario:\n  " + "\n  ".join(problems))
        return self
