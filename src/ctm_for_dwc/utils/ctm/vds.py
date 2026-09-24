"""Step 2: map PeMS vehicle detector stations (VDSs) onto CTM cells.

Three thin layers, each independently testable, mirroring the Step-3 layout:

* :func:`load_pems_station_metadata`  -- read a PeMS ``station_meta`` CSV,
  filter to one freeway + direction, normalize column names, and return a
  WGS-84 GeoDataFrame.
* :func:`project_vds_to_corridor`     -- project each VDS POINT onto the
  corridor's mainline LineStrings and report ``(pm_cum, pm_caltrans,
  seg_index, lateral_distance_m)`` per VDS.
* :func:`assign_vds_to_cells`         -- attach a VDS to every cell using the
  CTMSIM convention (VDS inside the cell wins) with the Dervisoglu et al.
  2014 nearest-upstream fallback for cells without their own VDS.

Kurzhanskiy 2007's I-210 case study has one mainline VDS per cell; this
module's output is the data ``cells_from_corridor`` needs to honor that
convention end-to-end. Step 4 (FD calibration) consumes the resulting
``vds_id`` column to look up the matching time-series file per cell.

PeMS station_meta schema
------------------------
The PeMS clearinghouse exports station metadata with these columns (mix of
spaces and underscores depending on the export):

* ``ID``                                   -- VDS station id (int)
* ``Fwy``                                  -- freeway number (int, e.g. 210)
* ``Dir``                                  -- direction (str, e.g. "W")
* ``District``                             -- Caltrans district (int)
* ``County``                               -- 3-letter county code
* ``Abs PM`` / ``Abs_PM``                  -- Caltrans absolute postmile
* ``Latitude`` / ``Lat``                   -- station latitude (WGS-84)
* ``Longitude`` / ``Lng`` / ``Long``       -- station longitude (WGS-84)
* ``Type``                                 -- station type: ``ML`` (mainline),
                                              ``OR``/``FR`` (on/off ramp),
                                              ``HV`` (HOV), ``FF`` (FF/HOV),
                                              ``CH`` (cross-highway), etc.
* ``Lanes``                                -- number of lanes the station covers
* ``Name``                                 -- human-readable location name

:func:`load_pems_station_metadata` accepts either spelling and normalizes
to lowercase / underscored attribute names on the output GeoDataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from .osm import Corridor

# Canonical (lowercase, underscore) column names emitted by
# :func:`load_pems_station_metadata`. Inputs are tolerated in either space
# or underscore form.
_CANONICAL_COLUMNS = (
    "vds_id", "fwy", "direction", "district", "county",
    "abs_pm", "latitude", "longitude", "lanes", "type", "name",
)

# Aliases the PeMS export sometimes uses for the same logical column.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "vds_id":     ("ID", "Station ID", "Station_ID", "station_id"),
    "fwy":        ("Fwy", "Freeway", "freeway"),
    "direction":  ("Dir", "Direction", "direction"),
    "district":   ("District", "district"),
    "county":     ("County", "county"),
    "abs_pm":     ("Abs PM", "Abs_PM", "Abs.PM", "Abs_Pm", "abs_pm"),
    "latitude":   ("Latitude", "Lat", "latitude", "lat"),
    "longitude":  ("Longitude", "Lng", "Long", "longitude", "lng", "lon"),
    "lanes":      ("Lanes", "lanes"),
    "type":       ("Type", "type"),
    "name":       ("Name", "name"),
}

# Default station type filter when ``mainline_only`` is True. PeMS uses ``ML``
# for mainline detectors; everything else (OR, FR, HV, FF, CH, ...) is a
# non-mainline detector.
_MAINLINE_TYPES = frozenset({"ML"})


# ---- Layer 4a: load_pems_station_metadata ---------------------------------


def load_pems_station_metadata(
    source: Union[str, Path, pd.DataFrame],
    *,
    freeway: Union[str, int],
    direction: str,
    mainline_only: bool = True,
) -> gpd.GeoDataFrame:
    """Read a PeMS ``station_meta`` CSV, filter, and emit a WGS-84 GeoDataFrame.

    Parameters
    ----------
    source : str, Path, or pd.DataFrame
        Path to a PeMS station_meta CSV / TSV, or a pre-loaded DataFrame.
        Column names are matched case-insensitively against the aliases in
        the module docstring.
    freeway : str or int
        Freeway number to keep (``210``, ``"210"``, or ``"I 210"`` all work
        -- digits are extracted and matched).
    direction : str
        Direction to keep (``"N"``/``"S"``/``"E"``/``"W"``; case-insensitive).
    mainline_only : bool, default True
        When True (the default and the right thing for Step-4 cell
        assignment), keep only ``Type == "ML"`` stations. Pass False to
        keep on-ramp, off-ramp, HOV, etc. detectors too.

    Returns
    -------
    gpd.GeoDataFrame
        One row per VDS, with normalized lowercase / underscored attribute
        columns (see :attr:`_CANONICAL_COLUMNS`) and POINT geometry in
        EPSG:4326. The index is the VDS station id (``vds_id``).
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        # PeMS exports are TSV-disguised-as-CSV occasionally; let pandas sniff.
        df = pd.read_csv(source, sep=None, engine="python")

    df = _normalize_columns(df)

    fwy_num = _extract_route_digits(freeway)
    df = df.loc[df["fwy"].astype(int) == fwy_num]
    if df.empty:
        raise ValueError(
            f"No stations found for freeway={freeway!r}; "
            f"available: {sorted(df['fwy'].unique())[:10] if 'fwy' in df else '<none>'}"
        )

    dir_norm = direction.strip().upper()
    df = df.loc[df["direction"].astype(str).str.upper() == dir_norm]
    if df.empty:
        raise ValueError(
            f"No stations found for direction={direction!r} on freeway {fwy_num}"
        )

    if mainline_only:
        df = df.loc[df["type"].astype(str).str.upper().isin(_MAINLINE_TYPES)]
        if df.empty:
            raise ValueError(
                f"No mainline (Type=ML) stations on freeway {fwy_num} "
                f"direction {dir_norm}"
            )

    # Coerce numerics now that we've filtered to the rows we want.
    df["vds_id"] = df["vds_id"].astype(int)
    df["abs_pm"] = pd.to_numeric(df["abs_pm"], errors="coerce")
    df["lanes"] = pd.to_numeric(df["lanes"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    bad_geom = df["latitude"].isna() | df["longitude"].isna()
    if bad_geom.any():
        df = df.loc[~bad_geom]

    geometry = [Point(x, y) for x, y in zip(df["longitude"], df["latitude"])]
    out = gpd.GeoDataFrame(
        df[list(_CANONICAL_COLUMNS)].copy(), geometry=geometry, crs="EPSG:4326",
    )
    return out.set_index("vds_id", drop=False).sort_values("abs_pm").reset_index(drop=True)


# ---- Layer 4b: project_vds_to_corridor -----------------------------------


def project_vds_to_corridor(
    vds_gdf: gpd.GeoDataFrame,
    corridor: Corridor,
    *,
    max_lateral_distance_m: float = 100.0,
    keep_off_corridor: bool = False,
) -> pd.DataFrame:
    """For each VDS, project onto the corridor mainline and record postmile + side info.

    Parameters
    ----------
    vds_gdf : gpd.GeoDataFrame
        Output of :func:`load_pems_station_metadata` (or anything with the
        same schema + POINT geometry in EPSG:4326).
    corridor : Corridor
        The :class:`~.osm.Corridor` to project against.
    max_lateral_distance_m : float, default 100.0
        Drop VDSs whose perpendicular distance from the mainline polyline
        exceeds this threshold -- these are VDSs that aren't actually on the
        corridor (e.g. a station on the same freeway but outside the bbox we
        extracted, in which case ``LineString.project`` clamps them to an
        endpoint and the postmile is meaningless). Set
        ``keep_off_corridor=True`` to retain them for diagnostics. 100 m
        comfortably covers divided-highway median offsets + a few m of
        OSM-vs-Caltrans centerline drift.
    keep_off_corridor : bool, default False
        When True, off-corridor VDSs are emitted with their (clamped)
        ``pm_cum`` so callers can inspect why they were dropped; the
        ``within_corridor`` column flags which rows the assigner should
        actually consume.

    Returns
    -------
    pd.DataFrame
        One row per VDS, sorted by ``pm_cum`` (so upstream VDSs come first).
        Columns:

        =====================  ============================================
        column                 meaning
        =====================  ============================================
        vds_id                 VDS station id
        pm_cum                 corridor postmile [mi from upstream end]
        pm_caltrans            Caltrans postmile (None if corridor has no
                               Caltrans annotation)
        seg_index              index of the corridor mainline segment the
                               VDS projected onto
        lateral_distance_m     perpendicular distance from VDS POINT to
                               the mainline polyline, meters
        within_corridor        bool: ``lateral_distance_m <=
                               max_lateral_distance_m``
        abs_pm                 Caltrans Abs PM from PeMS metadata (sanity
                               check against ``pm_caltrans``)
        lanes                  station lane count (carried through)
        name                   station name (carried through)
        =====================  ============================================
    """
    columns = [
        "vds_id", "pm_cum", "pm_caltrans", "seg_index",
        "lateral_distance_m", "within_corridor",
        "abs_pm", "lanes", "name",
    ]
    if vds_gdf.empty:
        return pd.DataFrame(columns=columns)

    caltrans_lookup = _build_caltrans_pm_lookup(corridor)

    rows: list[dict] = []
    for _, row in vds_gdf.iterrows():
        pt = row.geometry
        best = _project_to_corridor(pt, corridor)
        if best is None:
            continue
        seg_index, seg_fraction, lateral_m = best
        within = lateral_m <= max_lateral_distance_m
        if not within and not keep_off_corridor:
            continue
        seg = corridor.mainline_segments[seg_index]
        pm_cum = seg.pm_start + seg_fraction * seg.length
        pm_caltrans = caltrans_lookup(pm_cum) if caltrans_lookup is not None else None
        rows.append({
            "vds_id": int(row.vds_id),
            "pm_cum": pm_cum,
            "pm_caltrans": pm_caltrans,
            "seg_index": seg_index,
            "lateral_distance_m": lateral_m,
            "within_corridor": within,
            "abs_pm": float(row.abs_pm) if pd.notna(row.abs_pm) else np.nan,
            "lanes": int(row.lanes) if pd.notna(row.lanes) else 0,
            "name": row.get("name"),
        })

    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values("pm_cum", ignore_index=True)


# ---- Layer 4c: assign_vds_to_cells ----------------------------------------


_TIEBREAKER = Literal["midpoint", "furthest_downstream", "highest_pm_observed", "lowest_id"]


def assign_vds_to_cells(
    cells_df: pd.DataFrame,
    vds_projected: pd.DataFrame,
    *,
    fallback: Literal["upstream", "none"] = "upstream",
    tiebreaker: _TIEBREAKER = "midpoint",
    min_lane_match: bool = False,
) -> pd.DataFrame:
    """Attach a VDS to every cell, with Dervisoglu et al. 2014 fallback.

    Algorithm:

    1. For each cell, find VDSs whose ``pm_cum`` lies in
       ``(cell.pm_start, cell.pm_end]`` -- closed at end so a VDS exactly on
       a cell boundary belongs to the upstream cell (same half-open
       convention as ramps in :mod:`utils.ctm.cells`).
    2. If the cell has one direct VDS, assign it (``vds_source ==
       "direct"``).
    3. If multiple, apply ``tiebreaker``:

       * ``"midpoint"`` -- VDS closest to the cell's pm midpoint (default).
       * ``"furthest_downstream"`` -- VDS with the highest ``pm_cum``
         (closest to the cell's downstream boundary).
       * ``"highest_pm_observed"`` -- highest ``pct_observed`` (not yet
         hooked up; falls back to midpoint until time-series data lands).
       * ``"lowest_id"`` -- smallest ``vds_id``, fully deterministic.

       ``vds_source == "direct_tiebreak"`` flags the cell so downstream
       consumers can choose to inspect the displaced candidates.
    4. If zero direct VDSs and ``fallback == "upstream"``, inherit the
       ``vds_id`` from the closest cell upstream that *does* have a direct
       (or upstream-inherited) VDS -- this is the downstream-assignment
       scheme from Dervisoglu, Kurzhanskiy, Gomes & Horowitz, "Macroscopic
       Freeway Model Calibration with Partially Observed Data, a Case
       Study," American Control Conference 2014, §III. ``vds_source ==
       "nearest_upstream"``; ``vds_distance_mi`` measures how far upstream
       the inherited VDS is.
    5. If still no VDS available (cell upstream of every detector):
       ``vds_source == "missing"``.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Output of :func:`cells.cells_from_corridor`.
    vds_projected : pd.DataFrame
        Output of :func:`project_vds_to_corridor`.
    fallback : {"upstream", "none"}, default "upstream"
        ``"upstream"`` enables the Dervisoglu et al. 2014 nearest-upstream
        inheritance (downstream-assignment scheme: each detector's FD is
        assigned downstream until another detector is encountered);
        ``"none"`` leaves cells without a direct VDS marked as ``"missing"``.
    tiebreaker : {"midpoint", "furthest_downstream", "highest_pm_observed", "lowest_id"}, default "midpoint"
        How to pick when multiple VDSs land in the same cell.
        ``"furthest_downstream"`` picks the VDS with the highest ``pm_cum``
        (closest to the cell's downstream boundary).
    min_lane_match : bool, default False
        If True, prefer VDSs whose lane count matches the cell's ``lanes``
        column when present; falls back to lane-agnostic if no match.

    Returns
    -------
    pd.DataFrame
        ``cells_df`` copy with these new columns:

        =====================  ===================================================
        vds_id                 PeMS station id assigned to this cell (Int64; <NA>
                               only when ``vds_source == "missing"``)
        vds_pm                 the assigned VDS's ``pm_cum``
        vds_source             "direct" / "direct_tiebreak" / "nearest_upstream" /
                               "missing"
        vds_distance_mi        cumulative-mile distance from cell midpoint to the
                               assigned VDS (negative = upstream); only non-zero
                               for "nearest_upstream"
        vds_n_candidates       how many VDSs landed directly in the cell (>=1
                               means a tiebreaker fired iff this is > 1)
        =====================  ===================================================
    """
    out = cells_df.copy()
    n = len(out)
    assigned_id: list[Optional[int]] = [None] * n
    assigned_pm: list[float] = [float("nan")] * n
    assigned_source: list[str] = ["missing"] * n
    assigned_distance: list[float] = [float("nan")] * n
    assigned_lanes: list[Optional[int]] = [None] * n
    n_candidates: list[int] = [0] * n

    pm_start = out["pm_start"].to_numpy()
    pm_end = out["pm_end"].to_numpy()
    pm_mid = 0.5 * (pm_start + pm_end)
    cell_lanes = out["lanes"].to_numpy() if "lanes" in out.columns else np.zeros(n, dtype=int)

    # Drop off-corridor VDSs if the caller left them in for diagnostics. They
    # have meaningless clamped pm_cum values and would pollute the cell match.
    if "within_corridor" in vds_projected.columns:
        vds_projected = vds_projected.loc[vds_projected["within_corridor"].astype(bool)]

    # Direct pass: bucket VDSs into cells by pm_cum. Interval is half-open at
    # the start and closed at the end -- (pm_start, pm_end] -- so a VDS that
    # lands exactly on a cell boundary belongs to the *upstream* cell. The very
    # first cell is special-cased to include its own pm_start so a VDS at the
    # corridor's upstream end still gets matched.
    if not vds_projected.empty:
        for i in range(n):
            lower_strict = i > 0
            pm = vds_projected["pm_cum"]
            in_cell = vds_projected[
                ((pm > pm_start[i] + 1e-9) if lower_strict
                 else (pm >= pm_start[i] - 1e-9))
                & (pm <= pm_end[i] + 1e-9)
            ]
            n_candidates[i] = len(in_cell)
            if in_cell.empty:
                continue
            if min_lane_match and "lanes" in in_cell.columns and cell_lanes[i] > 0:
                same_lanes = in_cell[in_cell["lanes"] == cell_lanes[i]]
                if not same_lanes.empty:
                    in_cell = same_lanes
            chosen = _apply_tiebreaker(in_cell, pm_mid[i], tiebreaker)
            assigned_id[i] = int(chosen["vds_id"])
            assigned_pm[i] = float(chosen["pm_cum"])
            assigned_source[i] = "direct" if len(in_cell) == 1 else "direct_tiebreak"
            assigned_distance[i] = 0.0
            if pd.notna(chosen.get("lanes")):
                assigned_lanes[i] = int(chosen["lanes"])

    # Upstream fallback: walk forward; cells without a direct VDS inherit
    # whatever the most recent upstream cell holds.
    if fallback == "upstream":
        last_id: Optional[int] = None
        last_pm: float = float("nan")
        last_lanes: Optional[int] = None
        for i in range(n):
            if assigned_id[i] is not None:
                last_id = assigned_id[i]
                last_pm = assigned_pm[i]
                last_lanes = assigned_lanes[i]
                continue
            if last_id is None:
                continue  # still upstream of every VDS -> stays "missing"
            assigned_id[i] = last_id
            assigned_pm[i] = last_pm
            assigned_lanes[i] = last_lanes
            assigned_source[i] = "nearest_upstream"
            # Negative distance because the source VDS is upstream of the
            # cell's midpoint.
            assigned_distance[i] = last_pm - pm_mid[i]

    out["vds_id"] = pd.array(assigned_id, dtype="Int64")
    out["vds_pm"] = assigned_pm
    out["vds_lanes"] = pd.array(assigned_lanes, dtype="Int64")
    out["vds_source"] = assigned_source
    out["vds_distance_mi"] = assigned_distance
    out["vds_n_candidates"] = n_candidates
    return out


def assign_ramp_vds_to_cells(
    cells_df: pd.DataFrame,
    corridor: Corridor,
    ramp_vds_gdf: gpd.GeoDataFrame,
    *,
    max_distance_m: float = 200.0,
) -> pd.DataFrame:
    """Attach the nearest on-ramp / off-ramp PeMS VDS to each cell.

    Splits ``ramp_vds_gdf`` into PeMS ``Type='OR'`` (on-ramp) and
    ``Type='FR'`` (off-ramp) detectors, then for every cell that has
    ``on_ramp=True`` / ``off_ramp=True`` finds the matching corridor
    :class:`RampJunction` and assigns the closest matching-type ramp VDS
    within ``max_distance_m`` meters (great-circle, not along-corridor).
    Cells without a ramp on the relevant side, or whose nearest candidate is
    farther than the threshold, get ``<NA>``.

    Adds two ``Int64`` columns to ``cells_df`` and returns a copy:

    =================  =========================================================
    on_ramp_vds_id     PeMS station id of the on-ramp detector attached to the
                       on-ramp gore at ``pm_start`` (or ``<NA>``)
    off_ramp_vds_id    PeMS station id of the off-ramp detector attached to the
                       off-ramp gore at ``pm_end`` (or ``<NA>``)
    =================  =========================================================
    """
    out = cells_df.copy()
    n = len(out)
    on_ids: list[Optional[int]] = [None] * n
    off_ids: list[Optional[int]] = [None] * n

    if ramp_vds_gdf is None or ramp_vds_gdf.empty:
        out["on_ramp_vds_id"] = pd.array(on_ids, dtype="Int64")
        out["off_ramp_vds_id"] = pd.array(off_ids, dtype="Int64")
        return out

    or_vds = ramp_vds_gdf[ramp_vds_gdf["type"].astype(str).str.upper() == "OR"]
    fr_vds = ramp_vds_gdf[ramp_vds_gdf["type"].astype(str).str.upper() == "FR"]
    on_assignments = _assign_ramp_vds_per_gore(
        [r for r in corridor.ramp_junctions if r.kind == "on"],
        or_vds, max_distance_m,
    )
    off_assignments = _assign_ramp_vds_per_gore(
        [r for r in corridor.ramp_junctions if r.kind == "off"],
        fr_vds, max_distance_m,
    )

    pm_start = out["pm_start"].to_numpy() if "pm_start" in out.columns else None
    pm_end = out["pm_end"].to_numpy() if "pm_end" in out.columns else None
    for i in range(n):
        if pm_start is not None and out.iloc[i].get("on_ramp"):
            on_ids[i] = _pick_ramp_vds_at(on_assignments, pm_start[i])
        if pm_end is not None and out.iloc[i].get("off_ramp"):
            off_ids[i] = _pick_ramp_vds_at(off_assignments, pm_end[i])

    out["on_ramp_vds_id"] = pd.array(on_ids, dtype="Int64")
    out["off_ramp_vds_id"] = pd.array(off_ids, dtype="Int64")
    return out


def _assign_ramp_vds_per_gore(
    gores, vds_gdf, max_distance_m: float,
) -> dict[float, int]:
    """Map ``gore.postmile -> nearest VDS id`` within ``max_distance_m`` meters."""
    assignments: dict[float, int] = {}
    if vds_gdf is None or vds_gdf.empty or not gores:
        return assignments
    vds_lat = vds_gdf["latitude"].to_numpy(dtype=float)
    vds_lon = vds_gdf["longitude"].to_numpy(dtype=float)
    vds_id = vds_gdf["vds_id"].to_numpy(dtype=int)
    for gore in gores:
        gx, gy = gore.geometry.x, gore.geometry.y
        # Great-circle distance from gore to every candidate VDS.
        dists = np.array([
            _haversine_m(gy, gx, lat, lon)
            for lat, lon in zip(vds_lat, vds_lon)
        ])
        if dists.size == 0:
            continue
        j = int(np.argmin(dists))
        if dists[j] <= max_distance_m:
            assignments[round(gore.postmile, 6)] = int(vds_id[j])
    return assignments


def _pick_ramp_vds_at(
    assignments: dict[float, int], postmile: float,
) -> Optional[int]:
    """Look up the ramp VDS id matched at this gore postmile (or None)."""
    key = round(float(postmile), 6)
    return assignments.get(key)


# ---- Internals ------------------------------------------------------------


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename PeMS columns to the canonical lowercase/underscore form."""
    lookup = {col.strip(): col for col in df.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                rename[lookup[alias]] = canonical
                break
    missing = [c for c in _CANONICAL_COLUMNS if c not in rename.values()]
    if missing:
        raise ValueError(
            f"PeMS metadata is missing required columns: {missing}. "
            f"Available: {sorted(df.columns)}"
        )
    return df.rename(columns=rename)


def _extract_route_digits(freeway: Union[str, int]) -> int:
    """Pull the route number out of an `int`, `'210'`, or `'I 210'`-style string."""
    if isinstance(freeway, int):
        return freeway
    s = str(freeway)
    digits = "".join(c for c in s if c.isdigit())
    if not digits:
        raise ValueError(f"could not parse a route number from freeway={freeway!r}")
    return int(digits)


def _build_caltrans_pm_lookup(corridor: Corridor):
    """Return a function `pm_cum -> caltrans_pm` if Caltrans PMs are available."""
    if corridor.caltrans_postmiles is None:
        return None
    # Mirror the linear-interpolation helper in cells.py rather than depend on it
    # to keep the modules from forming a cycle.
    pm_to_caltrans = corridor.caltrans_postmiles
    segs = corridor.mainline_segments

    def lookup(pm_cum: float) -> float:
        for seg in segs:
            if seg.pm_start - 1e-9 <= pm_cum <= seg.pm_end + 1e-9:
                if seg.length <= 0.0:
                    return float(pm_to_caltrans[seg.u])
                t = (pm_cum - seg.pm_start) / seg.length
                u_pm = pm_to_caltrans[seg.u]
                v_pm = pm_to_caltrans[seg.v]
                return float(u_pm + t * (v_pm - u_pm))
        # Outside the corridor: pin to the nearest endpoint.
        if pm_cum < segs[0].pm_start:
            return float(pm_to_caltrans[segs[0].u])
        return float(pm_to_caltrans[segs[-1].v])

    return lookup


def _project_to_corridor(
    pt: Point, corridor: Corridor,
) -> Optional[tuple[int, float, float]]:
    """Find the corridor segment that minimizes the perpendicular distance from
    ``pt`` and return ``(seg_index, fraction_along, lateral_distance_m)``.

    ``fraction_along`` is 0..1 within the chosen segment;
    ``lateral_distance_m`` is the great-circle distance from ``pt`` to its
    projection on the segment.
    """
    best_idx = None
    best_frac = 0.0
    best_lat_m = float("inf")
    for i, seg in enumerate(corridor.mainline_segments):
        line = seg.geometry
        proj_d = line.project(pt)                # WGS84 degrees (small region)
        proj_pt = line.interpolate(proj_d)
        lat_m = _haversine_m(pt.y, pt.x, proj_pt.y, proj_pt.x)
        if lat_m < best_lat_m:
            best_lat_m = lat_m
            best_idx = i
            length_d = line.length
            best_frac = (proj_d / length_d) if length_d > 0 else 0.0
    if best_idx is None:
        return None
    return best_idx, float(best_frac), float(best_lat_m)


def _apply_tiebreaker(
    candidates: pd.DataFrame, pm_mid: float, tiebreaker: _TIEBREAKER,
) -> pd.Series:
    """Pick a single row from ``candidates`` according to ``tiebreaker``."""
    if tiebreaker == "lowest_id":
        return candidates.loc[candidates["vds_id"].idxmin()]
    if tiebreaker == "furthest_downstream":
        return candidates.loc[candidates["pm_cum"].idxmax()]
    # "highest_pm_observed" is a planned hook: once Step-2 time-series quality
    # is plumbed through, switch on a ``pct_observed`` column when present.
    if tiebreaker == "highest_pm_observed" and "pct_observed" in candidates.columns:
        return candidates.loc[candidates["pct_observed"].idxmax()]
    # Default / "midpoint" fallback: closest to the cell's pm midpoint.
    delta = (candidates["pm_cum"] - pm_mid).abs()
    return candidates.loc[delta.idxmin()]


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS-84 points, in meters."""
    r = 6_371_009.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(a)))
