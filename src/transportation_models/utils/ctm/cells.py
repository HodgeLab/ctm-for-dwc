"""Step 1, layer 2: lay out CTM cells along an extracted :class:`Corridor`.

Cell boundaries are placed at the union of:

* on-ramp gore postmiles,
* off-ramp gore postmiles,
* mainline lane-count changes,
* the corridor endpoints,
* any extra break postmiles the caller passes in (Step 2 uses this to push
  VDS postmiles in so each cell contains at most one detector).

Per the CTMSIM convention (User Guide §5, area 4), a ramp is "always at the
beginning of its cell" for on-ramps and "always at the end of its cell" for
off-ramps. We encode that exactly:

    on_ramp  = True  iff cell.pm_start == an on-ramp gore postmile
    off_ramp = True  iff cell.pm_end   == an off-ramp gore postmile

So an on-ramp at the corridor's upstream end belongs to the first cell, and
an off-ramp at the corridor's downstream end belongs to the last cell.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .osm import Corridor

# All postmile comparisons are quantized to this many decimal places (~6 cm at
# mile scale). Keeps floating-point noise from creating zero-length cells when
# two breakpoints arrive from independent computations.
_PM_ROUND_DECIMALS = 6


def cells_from_corridor(
    corridor: Corridor,
    *,
    extra_break_postmiles: Iterable[float] = (),
) -> pd.DataFrame:
    """Lay out cells along ``corridor`` and return a freeway-schema table.

    Parameters
    ----------
    corridor : Corridor
        Output of :func:`corridor_from_graph` (or constructed by hand for
        tests). Mainline segments must be ordered upstream→downstream.
    extra_break_postmiles : iterable of float, optional
        Additional postmiles at which to force a cell boundary. Values
        outside ``[corridor.pm_start, corridor.pm_end]`` are silently
        ignored. Step 2 (VDS → cell) uses this to guarantee one detector
        per cell.

    Returns
    -------
    pd.DataFrame
        One row per cell, ordered upstream → downstream, with columns:

        ====================  ======  ============================================
        column                dtype   meaning
        ====================  ======  ============================================
        pm_start              float   upstream postmile of the cell [mi]
        pm_end                float   downstream postmile of the cell [mi]
        length                float   cell length [mi]  (= pm_end - pm_start)
        lanes                 int     mainline lane count
        on_ramp               bool    True if cell.pm_start sits on an on-ramp gore
        off_ramp              bool    True if cell.pm_end sits on an off-ramp gore
        on_ramp_name          object  OSM name/ref of that on-ramp (None if absent)
        off_ramp_name         object  OSM name/ref of that off-ramp (None if absent)
        ====================  ======  ============================================

        ``length``, ``on_ramp``, and ``off_ramp`` match the optional/required
        columns expected by
        :func:`transportation_models.utils.ctm.io.freeway_from_dataframe`;
        Step 4 (FD calibration) and Step 2 (VDS-to-cell mapping) fill in the
        rest of the freeway schema before that function is called.
    """
    pm_lo = corridor.pm_start
    pm_hi = corridor.pm_end

    breaks: set[float] = {_q(pm_lo), _q(pm_hi)}
    for ramp in corridor.ramp_junctions:
        breaks.add(_q(ramp.postmile))
    for pm in corridor.lane_change_postmiles():
        breaks.add(_q(pm))
    for pm in extra_break_postmiles:
        if pm_lo - 1e-9 <= pm <= pm_hi + 1e-9:
            breaks.add(_q(pm))

    ordered_breaks = sorted(breaks)

    on_at: dict[float, Optional[str]] = {
        _q(r.postmile): r.name
        for r in corridor.ramp_junctions if r.kind == "on"
    }
    off_at: dict[float, Optional[str]] = {
        _q(r.postmile): r.name
        for r in corridor.ramp_junctions if r.kind == "off"
    }

    has_caltrans = corridor.caltrans_postmiles is not None
    caltrans_lookup = _build_caltrans_lookup(corridor) if has_caltrans else None

    rows: list[dict] = []
    for pm_start, pm_end in zip(ordered_breaks, ordered_breaks[1:]):
        midpoint = 0.5 * (pm_start + pm_end)
        row = {
            "pm_start": pm_start,
            "pm_end": pm_end,
            "length": pm_end - pm_start,
            "lanes": _lanes_at(corridor, midpoint),
            "on_ramp": pm_start in on_at,
            "off_ramp": pm_end in off_at,
            "on_ramp_name": on_at.get(pm_start),
            "off_ramp_name": off_at.get(pm_end),
        }
        if has_caltrans:
            row["caltrans_pm_start"] = caltrans_lookup(pm_start)   # type: ignore[misc]
            row["caltrans_pm_end"] = caltrans_lookup(pm_end)       # type: ignore[misc]
        rows.append(row)

    columns = [
        "pm_start", "pm_end", "length", "lanes",
        "on_ramp", "off_ramp", "on_ramp_name", "off_ramp_name",
    ]
    if has_caltrans:
        columns += ["caltrans_pm_start", "caltrans_pm_end"]
    return pd.DataFrame(rows, columns=columns)


# ---- Helpers --------------------------------------------------------------


def _q(pm: float) -> float:
    """Quantize a postmile for set membership / equality."""
    return round(float(pm), _PM_ROUND_DECIMALS)


def _build_caltrans_lookup(corridor: Corridor):
    """Return a function ``cum_pm -> caltrans_pm`` for any cell-boundary postmile.

    Caltrans PMs are known only at mainline node positions; the cell-boundary
    postmiles passed in are in cumulative-length coordinates and may land
    between nodes (when ``extra_break_postmiles`` is supplied). We
    linear-interpolate inside the mainline segment containing the query.
    """
    assert corridor.caltrans_postmiles is not None
    pm_to_caltrans = corridor.caltrans_postmiles
    # Sort segments by their cumulative pm_start so a linear scan is monotonic.
    segs = list(corridor.mainline_segments)

    def lookup(cum_pm: float) -> float:
        for seg in segs:
            if seg.pm_start - 1e-9 <= cum_pm <= seg.pm_end + 1e-9:
                if seg.length <= 0.0:
                    return float(pm_to_caltrans[seg.u])
                t = (cum_pm - seg.pm_start) / seg.length
                u_pm = pm_to_caltrans[seg.u]
                v_pm = pm_to_caltrans[seg.v]
                return float(u_pm + t * (v_pm - u_pm))
        # Outside the corridor: fall back to the nearest endpoint.
        if cum_pm < segs[0].pm_start:
            return float(pm_to_caltrans[segs[0].u])
        return float(pm_to_caltrans[segs[-1].v])

    return lookup


def _lanes_at(corridor: Corridor, pm: float) -> int:
    """Lane count of the mainline segment containing ``pm`` (midpoint of a cell)."""
    for seg in corridor.mainline_segments:
        if seg.pm_start - 1e-9 <= pm <= seg.pm_end + 1e-9:
            return seg.lanes
    # Fall back to the closest segment's lanes; only reachable for degenerate
    # break placements outside the corridor span, which the caller has already
    # been warned about by Corridor's pm_start/pm_end.
    closest = min(
        corridor.mainline_segments,
        key=lambda s: min(abs(pm - s.pm_start), abs(pm - s.pm_end)),
    )
    return closest.lanes
