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

import json
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, substring

from .osm import Corridor, MainlineSegment, RampJunction

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


# ---- Cell-length sanity check --------------------------------------------


# Default cell-length bounds for CTM dynamics, in miles.
#
# * The lower bound keeps the CFL-bounded sampling period from getting
#   uncomfortably small. CTMSIM / Kurzhanskiy 2007 eq. 4.1 requires
#   :math:`v_{f,i} \\Delta T \\le l_i`, so a 0.2-mi cell at v_f = 65 mph caps
#   :math:`\\Delta T` at about 11 s -- already a fairly aggressive
#   sampling rate for batch FD calibration.
# * The upper bound guards against cells that average over too much
#   spatial heterogeneity (multiple lane-add/-drop sections, more than one
#   bottleneck) and so produce unrealistic CTM dynamics.
DEFAULT_MIN_CELL_LENGTH_MI = 0.2
DEFAULT_MAX_CELL_LENGTH_MI = 1.0


def flag_cell_length_warnings(
    cells_df: pd.DataFrame,
    *,
    min_length: float = DEFAULT_MIN_CELL_LENGTH_MI,
    max_length: float = DEFAULT_MAX_CELL_LENGTH_MI,
) -> pd.DataFrame:
    """Annotate ``cells_df`` with a ``length_warning`` column.

    Adds (or overwrites) a single string column flagging cells whose length
    sits outside ``[min_length, max_length]`` miles:

    ================  ============================================================
    value             meaning
    ================  ============================================================
    ``"ok"``          ``min_length <= length <= max_length``
    ``"too_short"``   ``length < min_length`` -- forces ``Δt`` below the CFL
                      bound to be uncomfortably small (Kurzhanskiy 2007
                      eq. 4.1: :math:`v_{f,i} \\Delta T \\le l_i`).
    ``"too_long"``    ``length > max_length`` -- the cell averages over too
                      much spatial heterogeneity for the CTM dynamics to stay
                      realistic.
    ================  ============================================================

    Parameters
    ----------
    cells_df : pandas.DataFrame
        Output of :func:`cells_from_corridor` (must have a ``length`` column).
    min_length, max_length : float
        Inclusive bounds in miles. Defaults to 0.2 / 1.0 -- see module-level
        constants for the rationale.

    Returns
    -------
    pandas.DataFrame
        Copy of ``cells_df`` with the new ``length_warning`` column appended.
        Row count and existing columns are unchanged.
    """
    if min_length > max_length:
        raise ValueError(
            f"min_length ({min_length}) must be <= max_length ({max_length})"
        )
    if "length" not in cells_df.columns:
        raise ValueError("cells_df must have a 'length' column")
    out = cells_df.copy()
    lengths = out["length"].to_numpy()
    flags = np.where(
        lengths < min_length,
        "too_short",
        np.where(lengths > max_length, "too_long", "ok"),
    )
    out["length_warning"] = flags
    return out


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


def mainline_segments_to_geodataframe(corridor: Corridor) -> gpd.GeoDataFrame:
    """Return one LineString per :class:`MainlineSegment`, suitable for round-tripping.

    Carries the upstream/downstream OSM node ids, cumulative postmile
    endpoints, and lane count -- everything :func:`corridor_from_artifacts`
    needs to rebuild a :class:`Corridor` without the live osmnx graph.
    """
    rows = [
        {
            "u": int(seg.u),
            "v": int(seg.v),
            "pm_start": float(seg.pm_start),
            "pm_end": float(seg.pm_end),
            "lanes": int(seg.lanes),
        }
        for seg in corridor.mainline_segments
    ]
    geometries = [seg.geometry for seg in corridor.mainline_segments]
    return gpd.GeoDataFrame(rows, geometry=geometries, crs=corridor.crs)


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


# ---- Corridor artifact round-trip ----------------------------------------


# Filenames written by `corridor_to_artifacts` / read by `corridor_from_artifacts`.
MAINLINE_GEOJSON = "mainline.geojson"
RAMPS_GEOJSON = "ramps.geojson"
CORRIDOR_JSON = "corridor.json"


def corridor_to_artifacts(corridor: Corridor, out_dir: Path | str) -> None:
    """Persist ``corridor`` as ``mainline.geojson`` + ``ramps.geojson`` + ``corridor.json``.

    These three files are sufficient to reconstruct an equivalent
    :class:`Corridor` via :func:`corridor_from_artifacts` -- the regenerator
    script (:mod:`scripts.regenerate_cell_artifacts`) uses them to rebuild the
    cell GeoJSON and the PNG plots after the user hand-edits ``cells.csv``,
    without needing the original osmnx graph.

    The corridor JSON carries scalar metadata (``ref``, ``direction``,
    ``target_bearing``, ``crs``) plus the optional Caltrans-postmile lookup
    (``caltrans_postmiles`` keyed by OSM node id, as a string-keyed dict for
    JSON compatibility).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mainline_gdf = mainline_segments_to_geodataframe(corridor)
    mainline_gdf.to_file(out_dir / MAINLINE_GEOJSON, driver="GeoJSON")

    ramps_gdf = ramp_junctions_to_geodataframe(corridor)
    if not ramps_gdf.empty:
        ramps_gdf.to_file(out_dir / RAMPS_GEOJSON, driver="GeoJSON")

    meta: dict = {
        "ref": corridor.ref,
        "direction": corridor.direction,
        "target_bearing": float(corridor.target_bearing),
        "crs": corridor.crs,
    }
    if corridor.caltrans_postmiles is not None:
        meta["caltrans_postmiles"] = {
            str(int(node_id)): float(pm)
            for node_id, pm in corridor.caltrans_postmiles.items()
        }
    (out_dir / CORRIDOR_JSON).write_text(json.dumps(meta, indent=2))


def corridor_from_artifacts(in_dir: Path | str) -> Corridor:
    """Reconstruct a :class:`Corridor` from the sidecars written by ``corridor_to_artifacts``.

    Required files in ``in_dir``: ``mainline.geojson`` and ``corridor.json``.
    ``ramps.geojson`` is optional -- corridors with no ramps don't have one.

    The reconstructed corridor round-trips with the original for all fields
    used downstream by ``cells_to_geodataframe`` and the plotting helpers:
    geometry, postmiles, lane counts, ramp gore positions, and the optional
    Caltrans postmile lookup.
    """
    in_dir = Path(in_dir)
    meta = json.loads((in_dir / CORRIDOR_JSON).read_text())

    mainline_gdf = gpd.read_file(in_dir / MAINLINE_GEOJSON)
    mainline_gdf = mainline_gdf.sort_values("pm_start").reset_index(drop=True)
    segments = [
        MainlineSegment(
            u=int(row.u),
            v=int(row.v),
            geometry=row.geometry,
            length=float(row.pm_end) - float(row.pm_start),
            lanes=int(row.lanes),
            pm_start=float(row.pm_start),
            pm_end=float(row.pm_end),
        )
        for row in mainline_gdf.itertuples(index=False)
    ]

    ramps: list[RampJunction] = []
    ramps_path = in_dir / RAMPS_GEOJSON
    if ramps_path.exists():
        ramps_gdf = gpd.read_file(ramps_path).sort_values(
            ["postmile", "kind"]
        ).reset_index(drop=True)
        for row in ramps_gdf.itertuples(index=False):
            name = row.name if isinstance(row.name, str) else None
            ramps.append(RampJunction(
                kind=row.kind,
                postmile=float(row.postmile),
                mainline_node=int(row.mainline_node),
                ramp_terminus_node=int(row.ramp_terminus_node),
                geometry=row.geometry,
                lanes=int(row.lanes),
                name=name,
            ))

    caltrans = None
    if "caltrans_postmiles" in meta:
        caltrans = {int(k): float(v) for k, v in meta["caltrans_postmiles"].items()}

    return Corridor(
        ref=meta["ref"],
        direction=meta["direction"],
        target_bearing=float(meta["target_bearing"]),
        mainline_segments=segments,
        ramp_junctions=ramps,
        crs=meta.get("crs", "EPSG:4326"),
        caltrans_postmiles=caltrans,
    )


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
