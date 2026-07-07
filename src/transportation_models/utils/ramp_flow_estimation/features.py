"""Extract ramp-flow training samples (feature windows + measured flows) per stretch.

Method-agnostic: for a type-(c) stretch, each sliding window of ``window_size``
consecutive 5-min samples yields one training sample:

* **features Z** -- upstream and downstream mainline ``total_flow``,
  ``avg_speed``, ``avg_occupancy`` at each lag ``t-2, t-1, t``, laid out
  oldest-lag-first: ``[up_flow, up_speed, up_occ, down_flow, down_speed,
  down_occ]`` repeated per lag -> ``6 * window_size`` columns.
* **targets** -- the ramp flows ``r, s`` at the prediction step ``t`` (window's
  last sample). By default (``require_both_measured=True``) only windows where
  *both* ramps' detectors report are kept -- clean, directly-measured targets.
  With ``require_both_measured=False`` a window with one unmeasured ramp is also
  kept, recovering it by conservation ``q_up + r - s = q_down`` (the target is
  then partly a function of the mainline inputs).

A window is used only if the mainline is observed across *all* its samples (the
features need every lag). Estimator-specific quantities are *not* computed here:
Kan's alpha targets, capacity bounds, and feasibility live in :mod:`kan` /
:mod:`bounds`. The IO assembler additionally returns a per-stretch *bounds
context* table (weaving capacity ``c_w``, historical ramp peaks) that only
bounds-based methods consume.

The pure extractor takes already-aligned per-station arrays; loading/aligning
the timeseries is the caller's job (kept separate so this stays unit-testable).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .observations import sliding_all

_VARS_PER_STATION = 3          # flow, speed, occupancy
_STATIONS = 2                  # upstream, downstream mainline
FEATURES_PER_LAG = _VARS_PER_STATION * _STATIONS

_TS = "timestamp"
_FLOW = "total_flow_[veh/5-min]"
_SPEED = "avg_speed_[mph]"
_OCC = "avg_occupancy_[%]"
_PCT = "pct_observed"
# PeMS reports flow per 5-min bin; downstream consumers (capacities, physical
# interpretation) work in veh/hr, so flows are scaled to veh/hr on load.
_SAMPLES_PER_HOUR = 12


@dataclass
class Samples:
    """Training samples for one stretch (row-aligned arrays)."""

    X: np.ndarray          # (n, 6 * window_size) features
    r_true: np.ndarray     # (n,) on-ramp flow at t
    s_true: np.ndarray     # (n,) off-ramp flow at t
    q_up: np.ndarray       # (n,) upstream mainline flow at t
    q_down: np.ndarray     # (n,) downstream mainline flow at t
    t_index: np.ndarray    # (n,) prediction sample index into the grid


def mainline_flows(X) -> tuple[np.ndarray, np.ndarray]:
    """``(q_up, q_down)`` at the prediction step ``t``, recovered from the
    last-lag block of the (raw, unscaled) feature matrix."""
    X = np.asarray(X, dtype=float)
    return X[:, -FEATURES_PER_LAG], X[:, -FEATURES_PER_LAG + _VARS_PER_STATION]


def stretch_samples(
    up_id, down_id, on_ids, off_ids, *,
    flow, speed, occ, observed, window_size=3, require_both_measured=True,
) -> Samples:
    n = len(flow[up_id])
    absent = np.zeros(n, dtype=bool)   # mask for a ramp id missing from ``observed``

    def ramp_value(ids, t):
        """(total flow, known?): 0 if structurally absent, the summed detector
        flow if all detectors report, else (None, False)."""
        if not ids:
            return 0.0, True
        if all(bool(observed.get(i, absent)[t]) for i in ids):
            return float(sum(flow[i][t] for i in ids)), True
        return None, False

    main_obs = np.asarray(observed[up_id], bool) & np.asarray(observed[down_id], bool)
    main_win = sliding_all(main_obs, window_size)   # per window-start index

    rows_X = []
    r_list, s_list, qu_list, qd_list, t_list = [], [], [], [], []

    for i in range(main_win.shape[0]):
        if not main_win[i]:
            continue
        t = i + window_size - 1
        q_up = float(flow[up_id][t])
        q_down = float(flow[down_id][t])

        r_val, r_known = ramp_value(on_ids, t)
        s_val, s_known = ramp_value(off_ids, t)
        if require_both_measured:
            if not (r_known and s_known):              # fully-measured targets only
                continue
        elif (not r_known) + (not s_known) > 1:        # > 1 unknown -> not determinable
            continue
        if r_known and s_known:
            r_true, s_true = r_val, s_val
        elif r_known:                                  # off recovered by conservation
            r_true = r_val
            s_true = max(q_up + r_val - q_down, 0.0)
        else:                                          # on recovered by conservation
            s_true = s_val
            r_true = max(q_down + s_val - q_up, 0.0)

        feats = []
        for lag in range(window_size):
            j = i + lag
            feats += [flow[up_id][j], speed[up_id][j], occ[up_id][j],
                      flow[down_id][j], speed[down_id][j], occ[down_id][j]]
        rows_X.append(feats)
        r_list.append(r_true); s_list.append(s_true)
        qu_list.append(q_up); qd_list.append(q_down); t_list.append(t)

    ncols = FEATURES_PER_LAG * window_size
    return Samples(
        X=np.array(rows_X, dtype=float).reshape(-1, ncols),
        r_true=np.array(r_list, dtype=float),
        s_true=np.array(s_list, dtype=float),
        q_up=np.array(qu_list, dtype=float),
        q_down=np.array(qd_list, dtype=float),
        t_index=np.array(t_list, dtype=int),
    )


# --------------------------------------------------------------------------- #
# IO assembler: stretches.csv + timeseries + station metadata -> training data.
# --------------------------------------------------------------------------- #
@dataclass
class TrainingData:
    """Corpus-wide training samples, row-aligned; ``stretch_id`` labels each row
    for stretch-level splits and leave-k-stretches-out validation."""

    X: np.ndarray
    r_true: np.ndarray
    s_true: np.ndarray
    q_up: np.ndarray
    q_down: np.ndarray
    stretch_id: np.ndarray


_CTX_COLUMNS = ["c_w", "r_demand", "s_qmax"]


def _parse_ids(cell) -> list:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []
    out = []
    for tok in s.split(";"):
        tok = tok.strip()
        if tok:
            out.append(int(tok) if tok.lstrip("+-").isdigit() else tok)
    return out


def _as_int(value):
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _empty_corpus(window_size: int) -> tuple[TrainingData, pd.DataFrame]:
    z = lambda: np.zeros(0, dtype=float)
    data = TrainingData(
        X=np.zeros((0, FEATURES_PER_LAG * window_size), dtype=float),
        r_true=z(), s_true=z(), q_up=z(), q_down=z(),
        stretch_id=np.zeros(0, dtype=str),
    )
    ctx = pd.DataFrame(columns=_CTX_COLUMNS, index=pd.Index([], name="stretch_id"))
    return data, ctx


def build_training_data(
    stretches: "pd.DataFrame", timeseries_dir, station_meta: "pd.DataFrame", *,
    pct_floor: float = 0.0, window_size: int = 3, require_both_measured: bool = True,
    require_capacity: bool = True, record_start=None, record_end=None,
) -> tuple[TrainingData, pd.DataFrame]:
    """Assemble training samples from every type-(c) stretch.

    Returns ``(data, stretch_ctx)``: the method-agnostic corpus, and a
    per-stretch bounds-context table (index ``stretch_id``; columns ``c_w``,
    ``r_demand``, ``s_qmax``) consumed only by bounds-based methods (Kan).

    PeMS ``total_flow`` (veh/5-min) is scaled to **veh/hr** on load.
    ``station_meta`` is the calibrated station metadata (``Station ID``,
    per-lane ``capacity``, ``Lanes``); ``c_w = capacity * Lanes`` of the
    stretch's ``up_ml_id``. ``r_demand``/``s_qmax`` are each ramp's historical
    peak *total* observed flow (veh/hr). With ``require_capacity`` (default)
    stretches whose upstream ML lacks a calibrated capacity are skipped, so
    bounds-based and direct estimators see identical rows; with
    ``require_capacity=False`` they are kept and their ``c_w`` is NaN.
    Stretches whose mainline detectors are absent are always skipped.
    """
    timeseries_dir = Path(timeseries_dir)
    cfg = stretches[stretches["config_type"] == "c"]
    meta = station_meta.set_index("Station ID")

    ids: set[int] = set()
    for row in cfg.itertuples(index=False):
        for v in (getattr(row, "up_ml_id"), getattr(row, "down_ml_id")):
            if (i := _as_int(v)) is not None:
                ids.add(i)
        for col in ("on_ids", "off_ids"):
            ids.update(i for i in _parse_ids(getattr(row, col, None)) if isinstance(i, int))

    raw: dict[int, pd.DataFrame] = {}
    for sid in sorted(ids):
        path = timeseries_dir / f"{sid}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=[_TS, _FLOW, _SPEED, _OCC, _PCT], parse_dates=[_TS])
        except (ValueError, KeyError):
            continue
        raw[sid] = df.drop_duplicates(_TS).set_index(_TS).sort_index()

    if record_start is not None and record_end is not None:
        grid = pd.date_range(record_start, record_end, freq="5min", inclusive="left")
    elif raw:
        lo = min(df.index.min() for df in raw.values()).floor("5min")
        hi = max(df.index.max() for df in raw.values()).ceil("5min")
        grid = pd.date_range(lo, hi, freq="5min")
    else:
        return _empty_corpus(window_size)

    data: dict[int, dict] = {}
    for sid, df in raw.items():
        a = df.reindex(grid)
        flow = a[_FLOW].to_numpy(dtype=float) * _SAMPLES_PER_HOUR   # veh/5-min -> veh/hr
        pct = a[_PCT].to_numpy(dtype=float)
        obs_flow = ~np.isnan(flow) & (np.nan_to_num(pct, nan=-1.0) > pct_floor)
        data[sid] = {"flow": flow, "speed": a[_SPEED].to_numpy(dtype=float),
                     "occ": a[_OCC].to_numpy(dtype=float), "obs_flow": obs_flow}

    def hist_peak(ramp_ids) -> float:
        if not ramp_ids or any(i not in data for i in ramp_ids):
            return np.inf
        obs = np.ones(len(grid), dtype=bool)
        total = np.zeros(len(grid), dtype=float)
        for i in ramp_ids:
            obs &= data[i]["obs_flow"]
            total += np.nan_to_num(data[i]["flow"])
        return float(total[obs].max()) if obs.any() else np.inf

    parts: list[TrainingData] = []
    ctx_rows: dict[int, list[float]] = {}
    for row in cfg.itertuples(index=False):
        up, down = _as_int(getattr(row, "up_ml_id")), _as_int(getattr(row, "down_ml_id"))
        if up is None or down is None or up not in data or down not in data:
            continue
        c_w = np.nan
        if up in meta.index:
            cap, lanes = meta.at[up, "capacity"], meta.at[up, "Lanes"]
            if not (pd.isna(cap) or pd.isna(lanes)):
                c_w = float(cap) * float(lanes)
        if require_capacity and np.isnan(c_w):
            continue

        on = [i for i in _parse_ids(getattr(row, "on_ids", None)) if isinstance(i, int)]
        off = [i for i in _parse_ids(getattr(row, "off_ids", None)) if isinstance(i, int)]

        flow = {up: data[up]["flow"], down: data[down]["flow"]}
        speed = {up: data[up]["speed"], down: data[down]["speed"]}
        occ = {up: data[up]["occ"], down: data[down]["occ"]}
        observed = {
            up: data[up]["obs_flow"] & ~np.isnan(data[up]["speed"]) & ~np.isnan(data[up]["occ"]),
            down: data[down]["obs_flow"] & ~np.isnan(data[down]["speed"]) & ~np.isnan(data[down]["occ"]),
        }
        for i in on + off:
            if i in data:
                flow[i] = data[i]["flow"]
                observed[i] = data[i]["obs_flow"]

        s = stretch_samples(
            up, down, on, off, flow=flow, speed=speed, occ=occ, observed=observed,
            window_size=window_size, require_both_measured=require_both_measured,
        )
        n = len(s.r_true)
        if n == 0:
            continue
        sid = getattr(row, "stretch_id")    # opaque label (namespaced string or int)
        parts.append(TrainingData(
            X=s.X, r_true=s.r_true, s_true=s.s_true, q_up=s.q_up, q_down=s.q_down,
            stretch_id=np.full(n, sid),
        ))
        ctx_rows[sid] = [c_w, hist_peak(on), hist_peak(off)]

    if not parts:
        return _empty_corpus(window_size)
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts])
    corpus = TrainingData(
        X=np.concatenate([p.X for p in parts]),
        r_true=cat("r_true"), s_true=cat("s_true"),
        q_up=cat("q_up"), q_down=cat("q_down"), stretch_id=cat("stretch_id"),
    )
    ctx = pd.DataFrame.from_dict(ctx_rows, orient="index", columns=_CTX_COLUMNS)
    ctx.index.name = "stretch_id"
    return corpus, ctx.sort_index()
