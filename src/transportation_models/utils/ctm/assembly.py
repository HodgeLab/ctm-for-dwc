"""Step 6: assemble a CTM-ready freeway table from cells.csv + calibrated VDS data.

This module joins three independent artifacts into a single per-cell freeway
table that slots directly into :func:`utils.ctm.io.freeway_from_dataframe`:

  * Step 1+2 (`cells.csv`)
      One row per cell with cell geometry, lane count, ramp presence, and
      the mainline / on-ramp / off-ramp ``vds_id`` columns from
      :func:`utils.ctm.vds.assign_vds_to_cells` and
      :func:`utils.ctm.vds.assign_ramp_vds_to_cells`.
  * Step 4 mainline (`station_metadata_calibrated.csv`)
      Per-VDS triangular FD parameters in **per-lane** units (``capacity``
      [veh/hr-lane], ``jam_density`` [veh/mi-lane], ``critical_density``
      [veh/mi-lane], plus ``free_flow_speed`` and ``congestion_wave_speed``
      in mi/h -- intensive, no scaling).
  * Step 4 ramps (`ramp_metadata_calibrated.csv`, optional)
      Per-VDS ramp capacity ``ramp_capacity_[veh/hr]`` (total flow, not
      per-lane).

The assembly does three things:

  1. **Mainline join + per-lane → per-cell scaling.** Joins each cell's
     ``vds_id`` against the mainline calibration to look up FD params,
     then scales ``capacity``, ``jam_density``, and ``critical_density``
     by the cell's lane count.  Free-flow and congestion-wave speeds pass
     through unchanged.
  2. **Ramp capacity lookup.** Each cell with ``on_ramp=True``
     (resp. ``off_ramp=True``) takes its on/off-ramp capacity from the
     ramp calibration via ``on_ramp_vds_id`` / ``off_ramp_vds_id``.
     Cells with no matching ramp VDS (or a missing/NaN calibrated
     capacity) get NaN, which :func:`freeway_from_dataframe` translates
     into :math:`R_i = \\infty` (resp. :math:`S_i = \\infty`) per the
     :class:`utils.ctm.model.Cell` defaults -- a deliberate "no constraint"
     fall-back when no measurement is available.
  3. **CFL advisory.** Computes per-cell :math:`\\Delta T_{\\max,i} =
     l_i / v_{f,i}` (Kurzhanskiy 2007 eq. 4.1) and surfaces the binding
     cell so users can pick a sim ``dt`` confidently. Stop short of
     building a :class:`utils.ctm.model.Freeway` -- the user decides
     ``dt`` from this advisory and passes the table into
     :func:`freeway_from_dataframe` themselves at simulation time.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Output schema for assemble_freeway_table -- matches the required +
# optional columns of utils.ctm.io.freeway_from_dataframe.
FREEWAY_COLUMNS: tuple[str, ...] = (
    "length", "q_max", "v_f", "w", "rho_jam", "rho_crit",
    "on_ramp", "off_ramp",
    "on_ramp_capacity", "off_ramp_capacity",
    "lanes",
)

_RAMP_LIST_SEP_RE = re.compile(r"[,;\s]+")


def parse_ramp_vds_ids(value) -> list[int]:
    """Parse a cells.csv ``*_ramp_vds_id`` cell value into a list of ints.

    Step 1+2 emits ``on_ramp_vds_id`` / ``off_ramp_vds_id`` as scalar
    Int64 (one VDS per cell), but a Step-5 manual edit can replace a
    scalar with a list when multiple ramp detectors apply to the same
    cell (e.g. two adjacent on-ramps the user merged into a single
    cell). After such an edit the column's dtype falls back to object
    and individual values can be:

      * ``NaN`` / ``pd.NA`` / ``None`` -> ``[]`` (no ramp VDS).
      * A scalar int / numpy integer -> ``[int(value)]``.
      * A scalar float (e.g. ``900.0`` from a CSV) -> ``[int(value)]``.
      * A string of one or more integer VDS ids separated by ``,``,
        ``;``, or whitespace, optionally bracketed -- e.g.
        ``"900;901"``, ``"900, 901"``, ``"[900, 901]"``.
      * An already-iterable list / tuple / ``numpy.ndarray`` /
        ``pandas.Series`` of integer-like values.

    Returns ``[]`` when no ids are recoverable; callers treat that the
    same as "no matched ramp VDS for this cell."
    """
    if value is None:
        return []
    if pd.api.types.is_scalar(value):
        if pd.isna(value):
            return []
        if isinstance(value, str):
            cleaned = value.strip().strip("[](){}").strip()
            if not cleaned:
                return []
            return [int(p) for p in _RAMP_LIST_SEP_RE.split(cleaned) if p]
        # Numeric scalar.
        return [int(value)]
    # Iterable (list / tuple / ndarray / Series).
    return [int(x) for x in value if not (
        pd.api.types.is_scalar(x) and pd.isna(x)
    )]


def assemble_freeway_table(
    cells_df: pd.DataFrame,
    calibrated_df: pd.DataFrame,
    *,
    ramp_calibrated_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Join cells + calibration into a per-cell freeway table.

    Parameters
    ----------
    cells_df : pandas.DataFrame
        Output of Step 1+2 (``cells.csv``), one row per cell. Must carry
        ``length``, ``lanes``, ``on_ramp``, ``off_ramp``, and a ``vds_id``
        column. ``on_ramp_vds_id`` / ``off_ramp_vds_id`` are required only
        when a corresponding ramp flag is True for at least one row.

        **Optional FD override column.** If ``cells_df`` carries an
        ``fd_vds_id`` column, each non-NaN entry replaces the ``vds_id``
        of that row **only for FD parameter lookup** in the calibrated
        metadata. ``vds_id`` itself is left alone so the rest of the
        pipeline (initial-state extraction, off-ramp split-ratio
        computation) continues to use the Step-2-assigned, physically
        nearest detector. This lets the Step-5 reviewer redirect a cell
        with a poorly-calibrated detector to a neighbor's FD without
        losing the "what's actually here" audit trail.
    calibrated_df : pandas.DataFrame
        Mainline FD calibration (``station_metadata_calibrated.csv``).
        Keyed on ``Station ID`` with columns ``capacity``,
        ``free_flow_speed``, ``congestion_wave_speed``, ``jam_density``,
        ``critical_density``. ``capacity`` / ``jam_density`` /
        ``critical_density`` are interpreted as **per-lane** quantities.
    ramp_calibrated_df : pandas.DataFrame, optional
        Ramp calibration (``ramp_metadata_calibrated.csv``). Keyed on
        ``Station ID`` with a ``ramp_capacity_[veh/hr]`` column. When
        omitted (or for cells whose ramp VDS isn't in the table), the
        cell's ``on_ramp_capacity`` / ``off_ramp_capacity`` is left NaN
        and :func:`freeway_from_dataframe` will default to
        :math:`R_i = \\infty` / :math:`S_i = \\infty`.

    Returns
    -------
    pandas.DataFrame
        One row per input cell, in input order, with the columns listed
        in :data:`FREEWAY_COLUMNS`. The result is ready to hand to
        :func:`utils.ctm.io.freeway_from_dataframe` once the user picks a
        ``dt`` (see :func:`compute_cfl_advisory`).

    Raises
    ------
    KeyError
        If ``cells_df`` is missing a required column.
    ValueError
        If any cell has neither a ``vds_id`` nor an ``fd_vds_id``
        (``vds_source='missing'`` or NaN) -- those cells need a
        calibrated FD before assembly can proceed (see Step 2/4 docs
        for the upstream-inherit fallback options).
    """
    _required_cells = {"length", "lanes", "on_ramp", "off_ramp", "vds_id"}
    missing = _required_cells - set(cells_df.columns)
    if missing:
        raise KeyError(
            f"cells_df is missing required column(s): {sorted(missing)}"
        )

    # Resolve the effective FD-lookup id per cell: fd_vds_id when present
    # and non-NaN, else vds_id. This is the only place fd_vds_id is read;
    # downstream consumers (initial state, β) keep using vds_id directly.
    # `pd.to_numeric` normalizes both columns to float64 with NaN for
    # missing, so the fillna doesn't trip pandas' object-dtype downcast
    # warnings when one column is mixed-type.
    vds_base = pd.to_numeric(cells_df["vds_id"], errors="coerce")
    if "fd_vds_id" in cells_df.columns:
        fd_override = pd.to_numeric(cells_df["fd_vds_id"], errors="coerce")
        fd_ids_raw = fd_override.fillna(vds_base)
    else:
        fd_ids_raw = vds_base
    if fd_ids_raw.isna().any():
        unresolved = cells_df.index[fd_ids_raw.isna()].to_list()
        col_hint = (
            "vds_id (or fd_vds_id, if you intended to override)"
            if "fd_vds_id" in cells_df.columns else "vds_id"
        )
        raise ValueError(
            f"Cell(s) at row(s) {unresolved} have no {col_hint} and "
            "therefore no calibrated FD. Run Step 2's "
            "`assign_vds_to_cells` with the Dervisoglu upstream-inherit "
            "fallback (or pick a different VDS) before assembly."
        )

    fd_lookup = calibrated_df.set_index("Station ID")
    # Pull per-lane FD params for each cell's effective FD VDS.
    fd_ids = fd_ids_raw.astype(int)
    fd_for_cell = fd_lookup.reindex(fd_ids)

    lanes = cells_df["lanes"].to_numpy(dtype=float)
    out = pd.DataFrame(index=cells_df.index)
    out["length"] = cells_df["length"].to_numpy(dtype=float)
    out["q_max"] = fd_for_cell["capacity"].to_numpy(dtype=float) * lanes
    out["v_f"] = fd_for_cell["free_flow_speed"].to_numpy(dtype=float)
    out["w"] = fd_for_cell["congestion_wave_speed"].to_numpy(dtype=float)
    out["rho_jam"] = fd_for_cell["jam_density"].to_numpy(dtype=float) * lanes
    out["rho_crit"] = fd_for_cell["critical_density"].to_numpy(dtype=float) * lanes
    out["on_ramp"] = cells_df["on_ramp"].to_numpy(dtype=bool)
    out["off_ramp"] = cells_df["off_ramp"].to_numpy(dtype=bool)
    out["lanes"] = lanes.astype(int)

    # Ramp-capacity lookup. NaN ramp capacities pass through and become
    # +inf in freeway_from_dataframe via the Cell default.
    out["on_ramp_capacity"] = _ramp_capacity_column(
        cells_df, ramp_calibrated_df, vds_col="on_ramp_vds_id", presence_col="on_ramp",
    )
    out["off_ramp_capacity"] = _ramp_capacity_column(
        cells_df, ramp_calibrated_df, vds_col="off_ramp_vds_id", presence_col="off_ramp",
    )

    return out[list(FREEWAY_COLUMNS)]


def _ramp_capacity_column(
    cells_df: pd.DataFrame,
    ramp_calibrated_df: Optional[pd.DataFrame],
    *,
    vds_col: str,
    presence_col: str,
) -> pd.Series:
    """Look up calibrated ramp capacities for cells flagged ``presence_col=True``.

    For cells where ``vds_col`` carries a **list** of VDS ids (a valid
    Step-5 manual edit when multiple ramps fall in one cell), the
    per-cell capacity is the **sum** of the per-ramp capacities -- the
    CTM models :math:`R_i` / :math:`S_i` as the total inflow / outflow
    capacity of the cell's ramps. When some ids in a list are present
    in the ramp calibration table and others aren't, the present ones
    are summed and a :class:`UserWarning` flags the partial coverage so
    the user knows the result is a lower bound.

    Returns a Series aligned to ``cells_df.index`` with capacities in
    veh/hr, or NaN where:
      * ``presence_col`` is False (no ramp on this cell -- must be NaN so
        :func:`freeway_from_dataframe` keeps :math:`R_i = \\infty`);
      * ``vds_col`` is absent, NaN, or an empty list (no ramp VDS was
        matched to this cell);
      * ``ramp_calibrated_df`` is omitted or doesn't contain any of the
        VDSs in the list;
      * the calibrated capacity is NaN (no timeseries / empty after
        outlier filtering).
    """
    n = len(cells_df)
    if ramp_calibrated_df is None or vds_col not in cells_df.columns:
        return pd.Series([np.nan] * n, index=cells_df.index, dtype=float)

    ramp_lookup = ramp_calibrated_df.set_index("Station ID")[
        "ramp_capacity_[veh/hr]"
    ]
    presence = cells_df[presence_col].to_numpy(dtype=bool)
    vds_ids_raw = cells_df[vds_col]
    capacities = np.full(n, np.nan, dtype=float)
    partial_misses: list[tuple[int, list[int]]] = []
    for i, (has_ramp, raw) in enumerate(zip(presence, vds_ids_raw)):
        if not has_ramp:
            continue
        ids = parse_ramp_vds_ids(raw)
        if not ids:
            continue
        caps: list[float] = []
        missing: list[int] = []
        for vds_id in ids:
            try:
                cap = float(ramp_lookup.loc[vds_id])
            except KeyError:
                missing.append(vds_id)
                continue
            if np.isnan(cap):
                missing.append(vds_id)
                continue
            caps.append(cap)
        if not caps:
            # Every VDS missing from the calibration -- leave NaN so
            # freeway_from_dataframe defaults to infinite capacity.
            continue
        capacities[i] = float(sum(caps))
        if missing:
            partial_misses.append((i, missing))

    if partial_misses:
        warnings.warn(
            f"{vds_col}: {len(partial_misses)} cell(s) had only partial "
            "ramp-calibration coverage; the present capacities were "
            "summed (treating missing ids as 0). Affected (cell_idx, "
            f"missing_ids) pairs (first 5): {partial_misses[:5]}"
            + (" ..." if len(partial_misses) > 5 else ""),
            UserWarning, stacklevel=3,
        )
    return pd.Series(capacities, index=cells_df.index, dtype=float)


# ---- CFL (Δt) advisory ----------------------------------------------------


@dataclass(frozen=True)
class CFLAdvisory:
    """Per-cell and binding-cell CFL bounds for a freeway table.

    The CTM update requires :math:`v_{f,i}\\,\\Delta T \\le l_i`
    (Kurzhanskiy 2007 eq. 4.1), so the largest stable sim timestep is
    :math:`\\Delta T_{\\max} = \\min_i l_i / v_{f,i}`.

    Attributes
    ----------
    per_cell_dt_max_h : numpy.ndarray
        ``l_i / v_{f,i}`` per cell, in hours.
    binding_cell_idx : int
        Index of the cell whose CFL bound is the minimum.
    binding_dt_max_h : float
        ``min_i l_i / v_{f,i}`` -- the absolute upper bound for the sim
        ``dt``. In hours.
    binding_dt_max_s : float
        Same as :attr:`binding_dt_max_h`, in seconds (the unit users
        usually pick a ``dt`` in).
    binding_cell_length_mi : float
        ``l_i`` of the binding cell.
    binding_cell_v_f_mph : float
        ``v_{f,i}`` of the binding cell.
    suggested_dt_s : float
        Recommended round ``dt`` in seconds (the largest multiple of 5 s
        that is strictly less than :attr:`binding_dt_max_s`), capped at
        :attr:`binding_dt_max_s` itself if the CFL bound is sub-5-s.
    """

    per_cell_dt_max_h: np.ndarray
    binding_cell_idx: int
    binding_dt_max_h: float
    binding_dt_max_s: float
    binding_cell_length_mi: float
    binding_cell_v_f_mph: float
    suggested_dt_s: float


def compute_cfl_advisory(freeway_df: pd.DataFrame) -> CFLAdvisory:
    """Compute per-cell Δt bounds and the binding (most-restrictive) cell.

    Parameters
    ----------
    freeway_df : pandas.DataFrame
        Output of :func:`assemble_freeway_table` (or any DataFrame with
        ``length`` [mi] and ``v_f`` [mi/h] columns).

    Returns
    -------
    CFLAdvisory
        See class docstring; the suggested ``dt`` is rounded down to the
        nearest 5-second multiple strictly below the binding CFL bound
        (or to the bound itself when sub-5-s).
    """
    lengths = freeway_df["length"].to_numpy(dtype=float)
    v_f = freeway_df["v_f"].to_numpy(dtype=float)
    per_cell_h = lengths / v_f
    binding_idx = int(np.argmin(per_cell_h))
    binding_h = float(per_cell_h[binding_idx])
    binding_s = binding_h * 3600.0
    if binding_s >= 5.0:
        # Largest 5-s multiple strictly below the CFL bound. Strict inequality
        # because the engine uses `dt * v_f <= length`, and floating-point
        # equality is risky at the boundary.
        suggested_s = float(np.floor((binding_s - 1e-9) / 5.0) * 5.0)
        if suggested_s <= 0.0:
            suggested_s = binding_s
    else:
        suggested_s = binding_s
    return CFLAdvisory(
        per_cell_dt_max_h=per_cell_h,
        binding_cell_idx=binding_idx,
        binding_dt_max_h=binding_h,
        binding_dt_max_s=binding_s,
        binding_cell_length_mi=float(lengths[binding_idx]),
        binding_cell_v_f_mph=float(v_f[binding_idx]),
        suggested_dt_s=suggested_s,
    )


def format_cfl_advisory(advisory: CFLAdvisory) -> str:
    """Format a :class:`CFLAdvisory` as a multi-line, human-readable summary.

    The shape mirrors :func:`scripts.build_ctm_from_osm.print_summary`'s
    one-block-per-aspect style so the assembly script can drop the result
    straight into stdout.
    """
    per_cell_s = advisory.per_cell_dt_max_h * 3600.0
    lines = [
        "=== CFL (Δt) advisory ===",
        f"  binding cell idx : {advisory.binding_cell_idx}",
        f"  binding cell     : length {advisory.binding_cell_length_mi:.3f} mi "
        f"@ v_f {advisory.binding_cell_v_f_mph:.1f} mi/h",
        f"  max allowable Δt : "
        f"{advisory.binding_dt_max_h:.5f} h "
        f"({advisory.binding_dt_max_s:.1f} s)",
        f"  per-cell Δt range: "
        f"{per_cell_s.min():.1f}-{per_cell_s.max():.1f} s "
        f"(mean {per_cell_s.mean():.1f} s, n={per_cell_s.size})",
        f"  suggested Δt     : {advisory.suggested_dt_s:.1f} s "
        f"(largest 5-s multiple strictly below max)",
        "",
        "  Choose any Δt <= max allowable; the suggested value gives a "
        "safety margin",
        "  and a round, sub-5-second sim cadence. Pass it as --dt to "
        "freeway_from_dataframe.",
    ]
    return "\n".join(lines)
