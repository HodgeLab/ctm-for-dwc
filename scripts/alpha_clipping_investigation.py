#%%
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from transportation_models.utils import constants
from transportation_models.utils.ramp_flow_estimation.features import _as_int, _parse_ids, _TS, _FLOW, _SPEED, _OCC, _PCT, _SAMPLES_PER_HOUR
from transportation_models.utils.ramp_flow_estimation.kan import KanEstimator
from transportation_models.utils.ramp_flow_estimation.validation import run_scenarios
from transportation_models.utils.ramp_flow_estimation.observations import sliding_all
from transportation_models.utils.ramp_flow_estimation import bounds as _bounds

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STRETCHES = _REPO_ROOT / "data/pems/stretches"
_DEFAULT_STATION_META = _REPO_ROOT / "data/pems/calibrated/station_metadata_calibrated.csv"
timeseries_dir = Path("/Volumes/easystore/work/boulder/Caltrans/PeMS/timeseries_data/")
record_start = "2022-01-01 00:00"
record_end = "2022-01-08 00:00"
pct_floor = 0.0

def _load_stretches(path: Path) -> pd.DataFrame:
    csvs = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)

#%%
stretches = _load_stretches(_DEFAULT_STRETCHES)
station_meta = pd.read_csv(_DEFAULT_STATION_META)

stretches

#%%
cfg = stretches.loc[(stretches["config_type"] == "c") & (stretches["up_ml_id"].notna()), :]
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


# %%
grid = pd.date_range(record_start, record_end, freq="5min", inclusive="left")
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


# %%
row = cfg.reset_index(drop=True).iloc[6]
up, down = _as_int(getattr(row, "up_ml_id")), _as_int(getattr(row, "down_ml_id"))
cap, lanes = meta.at[up, "capacity"], meta.at[up, "Lanes"]
c_w = float(cap) * float(lanes)
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
r_demand, s_qmax = hist_peak(on), hist_peak(off)

# %% Spoofing features.stretch_samples()
up_id = up
down_id = down
on_ids = on
off_ids = off
window_size = 3
require_both_measured = True

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

rows_X, a_list = [], []
r_list, s_list, qu_list, qd_list, t_list = [], [], [], [], []

# %%
i = 936
t = i + window_size - 1
q_up = float(flow[up_id][t])
q_down = float(flow[down_id][t])

r_val, r_known = ramp_value(on_ids, t)
s_val, s_known = ramp_value(off_ids, t)

if not (r_known and s_known):              # fully-measured targets only
    print("Cannot calculate")
if r_known and s_known:
    r_true, s_true = r_val, s_val

r_cap = np.inf
s_cap = np.inf
r_min = np.maximum(q_down - q_up, 0.0)
s_min = np.maximum(q_up - q_down, 0.0)
cap_r = np.minimum(np.minimum(c_w - q_up, r_cap), r_demand)     # R_max (A.10)
cap_s = np.minimum(np.minimum(c_w - q_down, s_cap), s_qmax)     # S_max (A.11, q_down)
r_max = np.minimum(q_down - q_up + cap_s, cap_r)               # A.14
s_max = np.minimum(q_up - q_down + cap_r, cap_s)               # A.15
b = _bounds.compute_bounds(q_up, q_down, c_w, r_demand=r_demand, s_qmax=s_qmax)

# %% Spoofing bounds.alpha_target
bounds = b
def predicts_on_ramp(q_up, q_down) -> np.ndarray:
    """True where the on-ramp is the determined-first ramp (``q_down >= q_up``)."""
    return np.asarray(q_down, dtype=float) >= np.asarray(q_up, dtype=float)
    
on = predicts_on_ramp(q_up, q_down)
r_true = np.asarray(r_true, dtype=float)
s_true = np.asarray(s_true, dtype=float)
width_r = bounds.r_max - bounds.r_min
width_s = bounds.s_max - bounds.s_min
with np.errstate(divide="ignore", invalid="ignore"):
    alpha_r = (r_true - bounds.r_min) / width_r
    alpha_s = (s_true - bounds.s_min) / width_s
alpha = np.where(on, alpha_r, alpha_s)
feasible = np.where(on, width_r, width_s) > 0.0
alpha = np.clip(np.where(feasible, alpha, 0.0), 0.0, 1.0)

# %%
