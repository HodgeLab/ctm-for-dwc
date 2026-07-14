"""Zhang et al. 2024 corpus: mainline preprocessing + per-stretch samples.

Feature layout (Zhang Definition 1): each window of ``window_size`` (paper
``H = 5``) consecutive 5-min steps yields, per step, the mainline states
``[q_up, q_down, v_up, v_down, rho_up, rho_down]`` plus that step's Time
Feature ``[hour, minute]`` -- density is derived as ``rho = q / v`` (PeMS
reports occupancy, not density). Rows are flattened oldest-step-first, 8
columns per step -> ``8 * window_size`` columns. Flows are veh/hr (PeMS
veh/5-min scaled on load), speeds mph, densities veh/mile.

Mainline preprocessing (project decision; **mainline series only**, applied at
runtime -- the on-disk grid is never modified):

1. samples whose ``pct_observed`` is below a threshold are treated as missing;
2. missing runs of 1-2 steps are filled by persistence (the last observed
   value; a run at the very start of the record falls through to rule 3);
3. longer runs are filled by the detector's historical mean for that
   (day-of-week, time-of-day) slot, computed from its observed samples
   (slots never observed stay NaN and the window is unusable);
4. windows whose mainline data are *entirely* filled are dropped.

Targets ``(r, s)`` are the measured ramp flows at the prediction step ``t``
(all of a ramp's detectors reporting, no filling on ramp series); windows
without fully-measured ramps are dropped. Each row also carries the calendar
date of ``t`` for day-level splits (source validation days, MT survey days).

Following the paper's experiment setup ("we only use the workday data"),
windows whose prediction step falls on a weekend are dropped by default
(``weekdays_only``; public holidays are not excluded).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRAFFIC_PER_STEP = 6           # q_up, q_down, v_up, v_down, rho_up, rho_down
TIME_PER_STEP = 2              # hour, minute
FEATURES_PER_STEP = TRAFFIC_PER_STEP + TIME_PER_STEP
WINDOW_SIZE = 5                # Zhang's H

_TS = "timestamp"
_FLOW = "total_flow_[veh/5-min]"
_SPEED = "avg_speed_[mph]"
_PCT = "pct_observed"
_SAMPLES_PER_HOUR = 12         # veh/5-min -> veh/hr


@dataclass
class Preprocessed:
    """One mainline station after the fill rules."""

    flow: np.ndarray       # veh/hr, filled
    speed: np.ndarray      # mph, filled
    density: np.ndarray    # veh/mile, flow/speed of the filled series
    filled: np.ndarray     # bool: step was missing pre-fill
    usable: np.ndarray     # bool: flow, speed, density all non-NaN post-fill


def _missing_runs(missing: np.ndarray) -> list[np.ndarray]:
    """Maximal runs of consecutive True indices."""
    idx = np.flatnonzero(missing)
    if idx.size == 0:
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)


def _slot_means(values, observed, slot_key) -> np.ndarray:
    """Historical mean per (day-of-week, time-of-day) slot, broadcast back to
    the full record; NaN for slots with no observed sample."""
    obs = pd.Series(values[observed])
    means = obs.groupby(slot_key[observed]).mean()
    return means.reindex(slot_key).to_numpy(dtype=float)


def preprocess_mainline(flow, speed, pct, timestamps, *, pct_threshold) -> Preprocessed:
    """Apply the fill rules to one mainline station's aligned series."""
    flow = np.asarray(flow, dtype=float).copy()
    speed = np.asarray(speed, dtype=float).copy()
    pct = np.asarray(pct, dtype=float)
    ts = pd.DatetimeIndex(timestamps)

    observed = (np.nan_to_num(pct, nan=-1.0) >= pct_threshold) \
        & ~np.isnan(flow) & ~np.isnan(speed)
    flow[~observed] = np.nan
    speed[~observed] = np.nan
    missing = ~observed

    slot_key = (ts.dayofweek * 10_000 + ts.hour * 100 + ts.minute).to_numpy()
    hist = {}   # computed lazily: only needed when a long/leading run exists

    def hist_fill(series, name, idx):
        if name not in hist:
            hist[name] = _slot_means(series, observed, slot_key)
        series[idx] = hist[name][idx]

    for run in _missing_runs(missing):
        if len(run) <= 2 and run[0] > 0:      # persistence from the last observed
            flow[run] = flow[run[0] - 1]
            speed[run] = speed[run[0] - 1]
        else:                                  # historical (dow, time-of-day) mean
            hist_fill(flow, "flow", run)
            hist_fill(speed, "speed", run)

    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(speed > 0, flow / speed, np.nan)
    usable = ~np.isnan(flow) & ~np.isnan(speed) & ~np.isnan(density)
    return Preprocessed(flow=flow, speed=speed, density=density,
                        filled=missing, usable=usable)


@dataclass
class ZhangSamples:
    """Samples for one stretch (row-aligned arrays)."""

    X: np.ndarray          # (n, 8 * window_size) features, oldest step first
    r: np.ndarray          # (n,) measured on-ramp flow at t, veh/hr
    s: np.ndarray          # (n,) measured off-ramp flow at t, veh/hr
    date: np.ndarray       # (n,) datetime64[D] of the prediction step t
    t_index: np.ndarray    # (n,) prediction sample index into the grid


def stretch_samples(
    up: Preprocessed, down: Preprocessed, on_flows, off_flows, timestamps, *,
    window_size=WINDOW_SIZE, weekdays_only=True,
) -> ZhangSamples:
    """Extract windows from preprocessed mainline + measured ramp series.

    ``on_flows``/``off_flows`` are lists of ``(flow, observed)`` array pairs,
    one per ramp detector (empty list = structurally absent ramp, flow 0).
    """
    ts = pd.DatetimeIndex(timestamps)
    hour = ts.hour.to_numpy(dtype=float)
    minute = ts.minute.to_numpy(dtype=float)
    weekend = ts.dayofweek.to_numpy() >= 5
    n = len(up.flow)

    def ramp_at(pairs, t):
        """(total flow, measured?) across a ramp's detectors."""
        if not pairs:
            return 0.0, True
        if all(bool(obs[t]) for _, obs in pairs):
            return float(sum(f[t] for f, _ in pairs)), True
        return None, False

    rows, r_list, s_list, d_list, t_list = [], [], [], [], []
    for i in range(n - window_size + 1):
        t = i + window_size - 1
        if weekdays_only and weekend[t]:
            continue
        w = slice(i, t + 1)
        if not (up.usable[w].all() and down.usable[w].all()):
            continue
        if up.filled[w].all() and down.filled[w].all():   # rule 4: all filled
            continue
        r_val, r_ok = ramp_at(on_flows, t)
        s_val, s_ok = ramp_at(off_flows, t)
        if not (r_ok and s_ok):
            continue
        feats = []
        for j in range(i, t + 1):
            feats += [up.flow[j], down.flow[j], up.speed[j], down.speed[j],
                      up.density[j], down.density[j], hour[j], minute[j]]
        rows.append(feats)
        r_list.append(r_val); s_list.append(s_val)
        d_list.append(ts[t].date()); t_list.append(t)

    return ZhangSamples(
        X=np.array(rows, dtype=float).reshape(-1, FEATURES_PER_STEP * window_size),
        r=np.array(r_list, dtype=float),
        s=np.array(s_list, dtype=float),
        date=np.array(d_list, dtype="datetime64[D]"),
        t_index=np.array(t_list, dtype=int),
    )


def build_stretch_samples(
    stretches: "pd.DataFrame", stretch_id, timeseries_dir, *,
    pct_threshold: float, window_size: int = WINDOW_SIZE,
    ramp_pct_floor: float = 0.0, weekdays_only: bool = True,
    record_start=None, record_end=None,
) -> ZhangSamples:
    """Assemble samples for one stretch of a (namespaced) stretches table.

    Ramp detectors are *not* filled: a ramp sample is measured when its flow
    is present with ``pct_observed > ramp_pct_floor`` (the existing corpus
    convention), so targets stay directly-measured.
    """
    from .features import _parse_ids  # shared "a;b;c" id-list parsing

    match = stretches[stretches["stretch_id"] == stretch_id]
    if len(match) != 1:
        raise ValueError(f"stretch_id {stretch_id!r} matches {len(match)} rows "
                         "in the stretches table (expected exactly 1)")
    row = match.iloc[0]
    up_id, down_id = int(row["up_ml_id"]), int(row["down_ml_id"])
    on_ids = [i for i in _parse_ids(row.get("on_ids")) if isinstance(i, int)]
    off_ids = [i for i in _parse_ids(row.get("off_ids")) if isinstance(i, int)]

    timeseries_dir = Path(timeseries_dir)
    raw: dict[int, pd.DataFrame] = {}
    for sid in {up_id, down_id, *on_ids, *off_ids}:
        path = timeseries_dir / f"{sid}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=[_TS, _FLOW, _SPEED, _PCT], parse_dates=[_TS])
        raw[sid] = df.drop_duplicates(_TS).set_index(_TS).sort_index()
    if up_id not in raw or down_id not in raw:
        raise ValueError(f"stretch {stretch_id!r}: mainline timeseries missing "
                         f"under {timeseries_dir}")

    if record_start is not None and record_end is not None:
        grid = pd.date_range(record_start, record_end, freq="5min", inclusive="left")
    else:
        lo = min(df.index.min() for df in raw.values()).floor("5min")
        hi = max(df.index.max() for df in raw.values()).ceil("5min")
        grid = pd.date_range(lo, hi, freq="5min")

    def series(sid, col):
        vals = raw[sid][col].reindex(grid).to_numpy(dtype=float)
        return vals * _SAMPLES_PER_HOUR if col == _FLOW else vals

    main = {
        sid: preprocess_mainline(series(sid, _FLOW), series(sid, _SPEED),
                                 series(sid, _PCT), grid,
                                 pct_threshold=pct_threshold)
        for sid in (up_id, down_id)
    }

    def ramp_pairs(ids):
        pairs = []
        for i in ids:
            if i not in raw:
                return None          # a detector with no timeseries at all
            flow, pct = series(i, _FLOW), series(i, _PCT)
            obs = ~np.isnan(flow) & (np.nan_to_num(pct, nan=-1.0) > ramp_pct_floor)
            pairs.append((flow, obs))
        return pairs

    on, off = ramp_pairs(on_ids), ramp_pairs(off_ids)
    if on is None or off is None:
        raise ValueError(f"stretch {stretch_id!r}: a ramp detector's timeseries "
                         f"is missing under {timeseries_dir}")
    return stretch_samples(main[up_id], main[down_id], on, off, grid,
                           window_size=window_size, weekdays_only=weekdays_only)
