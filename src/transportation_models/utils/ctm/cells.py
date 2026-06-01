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

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, substring

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


# ---- GIS output -----------------------------------------------------------


def cells_to_geodataframe(
    cells_df: pd.DataFrame, corridor: Corridor,
) -> gpd.GeoDataFrame:
    """Wrap ``cells_df`` with a per-cell LineString from the corridor's mainline.

    Each row's geometry is the substring of the corridor's mainline polyline
    that runs from ``pm_start`` to ``pm_end`` (in cumulative miles). The
    output GeoDataFrame is in ``corridor.crs`` (EPSG:4326 by default) and is
    suitable for ``GeoDataFrame.to_file(..., driver='GeoJSON')`` so the cell
    layout can be loaded into QGIS / Google Earth / Mapbox alongside the
    underlying basemap.
    """
    geometries = [
        _substring_along_corridor(corridor, row.pm_start, row.pm_end)
        for row in cells_df.itertuples(index=False)
    ]
    return gpd.GeoDataFrame(cells_df.copy(), geometry=geometries, crs=corridor.crs)


def ramp_junctions_to_geodataframe(corridor: Corridor) -> gpd.GeoDataFrame:
    """Return one POINT per ramp junction, suitable for QGIS overlay.

    Carries `kind` ("on" / "off"), `postmile` (cumulative mi), the OSM
    `mainline_node` / `ramp_terminus_node` ids, `lanes`, and `name`.
    """
    rows = [
        {
            "kind": r.kind,
            "postmile": r.postmile,
            "mainline_node": r.mainline_node,
            "ramp_terminus_node": r.ramp_terminus_node,
            "lanes": r.lanes,
            "name": r.name,
        }
        for r in corridor.ramp_junctions
    ]
    geometries = [r.geometry for r in corridor.ramp_junctions]
    return gpd.GeoDataFrame(rows, geometry=geometries, crs=corridor.crs)


# ---- Helpers --------------------------------------------------------------


def _substring_along_corridor(
    corridor: Corridor, pm_start: float, pm_end: float,
):
    """Sub-LineString of the corridor's mainline between two cumulative postmiles.

    Walks each mainline segment, takes the slice that overlaps the requested
    range via :func:`shapely.ops.substring` (normalized by segment length),
    and merges the pieces. Returns a ``LineString`` for single-segment cells
    and a (possibly merged) ``LineString`` / ``MultiLineString`` for cells
    spanning multiple corridor segments. Degenerate ``pm_start == pm_end``
    cells return a ``Point`` at that location.
    """
    pieces: list[LineString] = []
    for seg in corridor.mainline_segments:
        if seg.pm_end <= pm_start + 1e-12:
            continue
        if seg.pm_start >= pm_end - 1e-12:
            break
        slice_start_mi = max(pm_start, seg.pm_start)
        slice_end_mi = min(pm_end, seg.pm_end)
        if seg.length <= 0.0:
            continue
        start_frac = (slice_start_mi - seg.pm_start) / seg.length
        end_frac = (slice_end_mi - seg.pm_start) / seg.length
        # `substring` clamps start/end into [0, 1] internally.
        sub = substring(seg.geometry, start_frac, end_frac, normalized=True)
        if isinstance(sub, Point):
            continue
        pieces.append(sub)

    if not pieces:
        # Zero-length cell -- return a point at the cell's start position.
        return _point_along_corridor(corridor, pm_start)
    if len(pieces) == 1:
        return pieces[0]
    merged = linemerge(pieces)
    return merged


def _point_along_corridor(corridor: Corridor, pm: float) -> Point:
    """Single-point interpolation at the given cumulative postmile."""
    for seg in corridor.mainline_segments:
        if seg.pm_start - 1e-9 <= pm <= seg.pm_end + 1e-9:
            if seg.length <= 0.0:
                return Point(seg.geometry.coords[0])
            t = (pm - seg.pm_start) / seg.length
            return seg.geometry.interpolate(t, normalized=True)
    # Fall back to corridor endpoints if pm is outside the range.
    if pm < corridor.pm_start:
        return Point(corridor.mainline_segments[0].geometry.coords[0])
    return Point(corridor.mainline_segments[-1].geometry.coords[-1])


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
