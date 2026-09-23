"""Caltrans State Highway Network postmile lookups for a :class:`Corridor`.

Caltrans publishes a SHN postmile dataset whose primary form is a GeoDataFrame
of POINT geometries -- one per posted postmile marker -- with at least these
columns:

* ``Route`` (int)   -- the state-numbered route (e.g. 210, 134, 101).
* ``PM`` (float)    -- the postmile value at this marker.

Caltrans postmiles run in a route-specific direction (typically increasing
from the southern / western terminus), so for a westbound or southbound
corridor they will *decrease* along the direction of travel. The lookup
helpers here preserve that sign convention: callers downstream (e.g. the
golden tests that compare against Kurzhanskiy 2007's PM 24-39 span on
I-210W) can compare directly.

The integration with :func:`corridor_from_graph` is purely additive: when
``postmiles`` is supplied, each mainline node gets a Caltrans postmile
attached to :attr:`Corridor.caltrans_postmiles`; the corridor's existing
cumulative-length ``pm_start``/``pm_end`` are unchanged so downstream code
that doesn't care about Caltrans coordinates keeps working as-is.

Acquiring the SHN postmile dataset
----------------------------------
Caltrans publishes the **State Highway Network Postmiles Tenth** dataset
(postmiles at 0.1-mile resolution, ~311 k points covering every state-
maintained route) through its ArcGIS Open Data portal:

* Dataset page:  https://gisdata-caltrans.opendata.arcgis.com/datasets/c22341fec9c74c6b9488ee4da23dd967_0
* FeatureServer: https://caltrans-gis.dot.ca.gov/arcgis/rest/services/CHhighway/SHN_Postmiles_Tenth/FeatureServer/0

:func:`download_caltrans_postmiles` fetches this dataset over HTTP and
caches it under ``data/caltrans/`` in the repo by default; pass
``out_path=...`` to write somewhere else. The default file is gitignored.

**Manual download.** If you'd rather grab it by hand: visit the dataset
page above and use *Download → GeoJSON* (whole-state, ~144 MB) or
*Download → Shapefile* (~50 MB zipped). Save it under
``data/caltrans/`` (or anywhere) and point :func:`load_postmiles` at the
file path; the schema is already what we expect.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import requests
from shapely.geometry import Point

from .osm import Corridor, MainlineSegment, _haversine_m, _METERS_PER_MILE

PostmileSource = Union[str, Path, gpd.GeoDataFrame]

# Caltrans SHN Postmiles Tenth dataset, hosted at the Caltrans GIS portal.
# The Hub URL returns the whole-state GeoJSON in a single response; the
# FeatureServer URL is paginated and the entry point we use when the caller
# filters to specific routes.
SHN_POSTMILES_TENTH_HUB_DOWNLOAD = (
    "https://hub.arcgis.com/api/v3/datasets/"
    "c22341fec9c74c6b9488ee4da23dd967_0/downloads/data"
)
SHN_POSTMILES_TENTH_FEATURESERVER = (
    "https://caltrans-gis.dot.ca.gov/arcgis/rest/services/"
    "CHhighway/SHN_Postmiles_Tenth/FeatureServer/0"
)

# Default download target lives in the repo so subsequent runs can find it
# without an internet round-trip. The path is gitignored.
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_POSTMILE_DIR = _REPO_ROOT / "data" / "caltrans"
DEFAULT_POSTMILE_FILE = DEFAULT_POSTMILE_DIR / "shn_postmiles_tenth.geojson"


def download_caltrans_postmiles(
    out_path: Optional[Union[str, Path]] = None,
    *,
    routes: Optional[Sequence[int]] = None,
    overwrite: bool = False,
    progress: bool = True,
    timeout: float = 600.0,
) -> Path:
    """Fetch the Caltrans SHN Postmiles Tenth dataset and return its local path.

    By default writes the full statewide GeoJSON (~144 MB, ~311 k points
    spanning every state route) to
    ``<repo_root>/data/caltrans/shn_postmiles_tenth.geojson``. Pass
    ``out_path`` to write somewhere else; pass ``routes=[210, 880]`` to
    fetch only those routes (uses the FeatureServer query API and paginates
    in 2 k-record chunks -- typically a few hundred KB).

    Parameters
    ----------
    out_path : str or Path, optional
        Where to write the GeoJSON. Defaults to ``DEFAULT_POSTMILE_FILE``
        when ``routes`` is ``None``; otherwise to a route-tagged sibling
        file (so a filtered download doesn't shadow the full statewide
        one).
    routes : sequence of int, optional
        State route numbers to filter to (e.g. ``[210, 880]``). When given,
        we hit the FeatureServer with ``where=Route IN (...)`` and stream
        pages until the server returns less than a full page. When omitted
        we fall back to the Hub direct download for the whole state.
    overwrite : bool, default False
        Skip the download if ``out_path`` already exists (``False``) or
        re-fetch it (``True``).
    progress : bool, default True
        Print a one-line progress indicator to stderr while downloading.
    timeout : float, default 600.0
        Per-request timeout in seconds.

    Returns
    -------
    Path
        Absolute path to the saved GeoJSON. Suitable as the ``postmiles``
        argument to :func:`corridor_from_graph` (it will go through
        :func:`load_postmiles`).

    Notes
    -----
    The Hub direct-download endpoint is generated from a periodically-
    refreshed cache, so it may lag the underlying FeatureServer by a few
    minutes. For our purposes (postmiles change on a yearly cadence) that
    drift is irrelevant.
    """
    out_path = _resolve_out_path(out_path, routes)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        if progress:
            print(f"caltrans postmiles: cache hit at {out_path} "
                  f"({out_path.stat().st_size / 1024 / 1024:.1f} MB) -- "
                  f"pass overwrite=True to refresh", file=sys.stderr)
        return out_path

    if routes is None:
        _download_full_geojson(out_path, progress=progress, timeout=timeout)
    else:
        _download_filtered_geojson(
            out_path, routes=tuple(int(r) for r in routes),
            progress=progress, timeout=timeout,
        )
    return out_path


# ---- Downloader internals -------------------------------------------------


def _resolve_out_path(
    out_path: Optional[Union[str, Path]],
    routes: Optional[Sequence[int]],
) -> Path:
    """Pick the on-disk destination for ``download_caltrans_postmiles``."""
    if out_path is not None:
        return Path(out_path).expanduser().resolve()
    if routes is None:
        return DEFAULT_POSTMILE_FILE
    tag = "_".join(str(int(r)) for r in sorted(set(routes)))
    return DEFAULT_POSTMILE_DIR / f"shn_postmiles_tenth_routes_{tag}.geojson"


def _download_full_geojson(out_path: Path, *, progress: bool, timeout: float) -> None:
    """Stream the whole-state GeoJSON from the Hub endpoint to ``out_path``."""
    params = {"format": "geojson", "spatialRefId": 4326}
    url = SHN_POSTMILES_TENTH_HUB_DOWNLOAD + "?" + urllib.parse.urlencode(params)
    if progress:
        print(f"caltrans postmiles: downloading full statewide GeoJSON "
              f"(~144 MB) to {out_path}", file=sys.stderr)
    tmp = Path(tempfile.mkstemp(prefix=".caltrans_dl_", dir=out_path.parent,
                                suffix=".geojson")[1])
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length") or 0)
            written = 0
            last_pct = -1
            with tmp.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    if progress and total:
                        pct = int(100 * written / total)
                        if pct != last_pct and pct % 5 == 0:
                            print(f"  {pct}%  ({written / 1024 / 1024:.1f} MB)",
                                  file=sys.stderr)
                            last_pct = pct
        shutil.move(str(tmp), str(out_path))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if progress:
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  done: {out_path} ({size_mb:.1f} MB)", file=sys.stderr)


def _download_filtered_geojson(
    out_path: Path, *, routes: tuple[int, ...], progress: bool, timeout: float,
) -> None:
    """Pull only ``routes`` from the FeatureServer (paginated) into a GeoJSON."""
    where = "Route IN (" + ",".join(str(r) for r in routes) + ")"
    base_params = {
        "where": where,
        "outFields": "*",
        "f": "geojson",
        "outSR": 4326,
        "resultRecordCount": 2000,
    }
    if progress:
        print(f"caltrans postmiles: downloading routes {list(routes)} "
              f"(paginated) to {out_path}", file=sys.stderr)

    all_features: list[dict] = []
    offset = 0
    while True:
        params = dict(base_params, resultOffset=offset)
        url = f"{SHN_POSTMILES_TENTH_FEATURESERVER}/query"
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        page = payload.get("features", [])
        all_features.extend(page)
        if progress:
            print(f"  fetched {len(all_features)} features (offset {offset})",
                  file=sys.stderr)
        # FeatureServer signals more pages via "exceededTransferLimit" or by
        # returning a full page of resultRecordCount; either way is safe.
        if len(page) < base_params["resultRecordCount"]:
            break
        offset += base_params["resultRecordCount"]

    out_doc = {
        "type": "FeatureCollection",
        "name": "SHN_Postmiles_Tenth_filtered",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": all_features,
    }
    tmp = Path(tempfile.mkstemp(prefix=".caltrans_dl_", dir=out_path.parent,
                                suffix=".geojson")[1])
    try:
        with tmp.open("w") as f:
            json.dump(out_doc, f)
        shutil.move(str(tmp), str(out_path))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if progress:
        size_kb = out_path.stat().st_size / 1024
        print(f"  done: {out_path} ({size_kb:.1f} KB, {len(all_features)} features)",
              file=sys.stderr)


def extract_route_number(ref: str) -> int:
    """Parse the numeric route from an OSM ``ref`` like ``"I 210"`` or ``"CA 134"``.

    Raises ``ValueError`` if no digit run is present.
    """
    m = re.search(r"\d+", str(ref))
    if not m:
        raise ValueError(f"could not parse a route number from ref={ref!r}")
    return int(m.group(0))


def load_postmiles(source: PostmileSource, *, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Load a SHN postmile GeoDataFrame and reproject to ``target_crs``.

    Accepts either a path to a shapefile / geopackage / geojson, or a
    pre-loaded GeoDataFrame. Output is in ``target_crs`` (defaults to
    EPSG:4326, the osmnx default for graphs).
    """
    if isinstance(source, gpd.GeoDataFrame):
        gdf = source
    else:
        gdf = gpd.read_file(source)
    if gdf.crs is None:
        raise ValueError("postmile GeoDataFrame must declare a CRS")
    if str(gdf.crs).lower() != target_crs.lower():
        gdf = gdf.to_crs(target_crs)
    for col in ("Route", "PM"):
        if col not in gdf.columns:
            raise ValueError(f"postmile GeoDataFrame missing required column {col!r}")
    return gdf


def caltrans_postmiles_for_corridor(
    corridor: Corridor,
    postmiles: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
) -> dict[int, float]:
    """Assign a Caltrans postmile to every mainline node in ``corridor``.

    For each mainline node we find the nearest postmile *marker* of the
    corridor's route, project both onto the mainline polyline, and report
    ``marker_PM ± (distance_marker_to_node_along_mainline / 1609.344 mi)``.
    The sign of the adjustment is set so that postmiles flow monotonically
    along the corridor's travel direction.

    The lookup is O(nodes × markers) -- fine at corridor scale (tens of
    nodes, low hundreds of markers per route).
    """
    route = extract_route_number(corridor.ref)
    route_markers = postmiles.loc[postmiles["Route"] == route].copy()
    if route_markers.empty:
        raise ValueError(
            f"no postmile markers for Route {route} (corridor ref={corridor.ref!r})"
        )

    # Pull marker xy and PM out as plain arrays for vectorized nearest-neighbor.
    marker_xy = np.column_stack(
        [route_markers.geometry.x.to_numpy(), route_markers.geometry.y.to_numpy()]
    )
    marker_pm = route_markers["PM"].to_numpy(dtype=float)

    nodes_in_order = _mainline_nodes_in_order(corridor)
    pm_at: dict[int, float] = {}

    # Cumulative length (in mi) for each mainline node, indexed by position in
    # ``nodes_in_order``. Used to compute "along the corridor" distance between
    # any two mainline nodes.
    cum_mi = _cumulative_lengths(corridor)
    node_idx = {n: i for i, n in enumerate(nodes_in_order)}

    # Sample the corridor's polyline at each mainline node for projection.
    for n in nodes_in_order:
        node_xy = graph.nodes[n]
        nx_, ny_ = node_xy["x"], node_xy["y"]
        # Nearest marker (planar distance in degrees is fine for argmin on the
        # small scale of one route corridor).
        dx = marker_xy[:, 0] - nx_
        dy = marker_xy[:, 1] - ny_
        j = int(np.argmin(dx * dx + dy * dy))
        mx, my = float(marker_xy[j, 0]), float(marker_xy[j, 1])
        base_pm = float(marker_pm[j])

        # Find the corridor's nearest-along-mainline node to the marker and use
        # *that* node's cumulative length as the marker's "where on the
        # corridor it lives" reference. The signed difference between this
        # node and the marker's reference is the postmile adjustment.
        ref_node_idx = _nearest_mainline_idx_to_xy(graph, nodes_in_order, mx, my)
        delta_mi_along = cum_mi[node_idx[n]] - cum_mi[ref_node_idx]
        sign = _route_sign(corridor)
        pm_at[n] = base_pm + sign * delta_mi_along

    return pm_at


# ---- Internals ------------------------------------------------------------


def _mainline_nodes_in_order(corridor: Corridor) -> list[int]:
    """Mainline node IDs ordered upstream → downstream."""
    if not corridor.mainline_segments:
        return []
    seq = [corridor.mainline_segments[0].u]
    seq += [seg.v for seg in corridor.mainline_segments]
    return seq


def _cumulative_lengths(corridor: Corridor) -> list[float]:
    """Cumulative mi from upstream end, parallel to ``_mainline_nodes_in_order``."""
    cum = [0.0]
    for seg in corridor.mainline_segments:
        cum.append(cum[-1] + seg.length)
    return cum


def _nearest_mainline_idx_to_xy(
    graph: nx.MultiDiGraph, nodes_in_order: Iterable[int], x: float, y: float
) -> int:
    """Index (into ``nodes_in_order``) of the mainline node nearest to (x, y)."""
    best = 0
    best_d = float("inf")
    for i, n in enumerate(nodes_in_order):
        nx_, ny_ = graph.nodes[n]["x"], graph.nodes[n]["y"]
        d = (nx_ - x) ** 2 + (ny_ - y) ** 2
        if d < best_d:
            best_d = d
            best = i
    return best


def _route_sign(corridor: Corridor) -> int:
    """+1 if Caltrans PM increases along travel; -1 if it decreases.

    Caltrans postmiles run south-to-north or west-to-east along each route,
    so for north- or east-running corridors the corridor's travel direction
    agrees with PM (+1); for south or west, it opposes (-1). Picks the sign
    from the corridor's target bearing (``[0, 180)`` ⇒ N/E component ⇒ +1).
    Routes with non-standard PM direction (CA 1 along the bending coast,
    e.g.) would need an explicit override -- not yet supported.
    """
    return 1 if 0.0 <= corridor.target_bearing < 180.0 else -1
