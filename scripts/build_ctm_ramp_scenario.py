"""Step 7: build demand.csv and beta.csv for a CTM simulation window.

Dispatch per ramp VDS:
  1. VDS has any historical data -> load + historical-average fill for gaps
  2. VDS completely absent + type-(a) or -(b) stretch -> conservation
  3. VDS completely absent + type-(c) stretch -> GRU

Aggregates per CTM cell and writes demand.csv + beta.csv at sim cadence,
ready for ``simulate_ctm_corridor.py --demand --beta``.

Run from the repo root::

    python scripts/build_ctm_ramp_scenario.py \\
        --cells     case_studies/I880N_10mi/cells.csv \\
        --stretches data/pems/stretches/880N_stretches.csv \\
        --timeseries-dir data/pems/csv_files \\
        --gru-checkpoint models/gru/best.pt \\
        --manifest   models/gru/manifest.json \\
        --start "2023-06-01 06:00" --end "2023-06-01 10:00" \\
        --dt-seconds 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from transportation_models.utils.ctm import historical_average_fill
from transportation_models.utils.ctm.assembly import parse_ramp_vds_ids
from transportation_models.utils.ramp_flow_estimation.gru import GruEstimator

_FLOW = "total_flow_[veh/5-min]"
_SPEED = "avg_speed_[mph]"
_OCC = "avg_occupancy_[%]"
_TS = "timestamp"
_SAMPLES_PER_HOUR = 12.0
_FIVE_MIN = pd.Timedelta(minutes=5)


# ---- VDS helpers ------------------------------------------------------------

def _has_any_data(vds_id: int, ts_dir: Path) -> bool:
    path = ts_dir / f"{vds_id}.csv"
    if not path.exists():
        return False
    try:
        return pd.read_csv(path, usecols=[_FLOW])[_FLOW].notna().any()
    except Exception:
        return False


def _ramp_flow_veh_hr(vds_id: int, ts_dir: Path, *, start, end) -> np.ndarray:
    """Load ramp timeseries, historical-average fill, slice to [start, end) at 5-min, veh/hr."""
    df = pd.read_csv(ts_dir / f"{vds_id}.csv", parse_dates=[_TS])
    df = historical_average_fill(df, flow_col=_FLOW)
    grid = pd.date_range(start, end, freq="5min", inclusive="left")
    arr = df.set_index(_TS).reindex(grid)[_FLOW].to_numpy(dtype=float)
    return arr * _SAMPLES_PER_HOUR


def _ml_flow_veh_hr(vds_id: int, ts_dir: Path, *, start, end) -> np.ndarray:
    """Load mainline timeseries flow [veh/hr], forward-fill NaN."""
    df = pd.read_csv(ts_dir / f"{vds_id}.csv", parse_dates=[_TS])
    grid = pd.date_range(start, end, freq="5min", inclusive="left")
    arr = df.set_index(_TS).reindex(grid)[_FLOW].ffill().to_numpy(dtype=float)
    return arr * _SAMPLES_PER_HOUR


def _ml_arrays(vds_id: int, ts_dir: Path, grid: pd.DatetimeIndex) -> tuple:
    """Return (flow_veh_hr, speed_mph, occ_pct) on grid, forward-filled."""
    df = pd.read_csv(ts_dir / f"{vds_id}.csv", parse_dates=[_TS])
    a = df.set_index(_TS).reindex(grid).ffill()
    flow = a[_FLOW].to_numpy(dtype=float) * _SAMPLES_PER_HOUR
    speed = a[_SPEED].to_numpy(dtype=float)
    occ = a[_OCC].to_numpy(dtype=float) if _OCC in a.columns else np.zeros(len(grid))
    return flow, speed, occ


# ---- Estimation strategies --------------------------------------------------

def _conservation_flow(stretch: pd.Series, ts_dir: Path, *,
                       start, end, is_on_ramp: bool) -> np.ndarray:
    """Recover ramp flow [veh/hr] from mainline conservation."""
    q_up = _ml_flow_veh_hr(int(stretch["up_ml_id"]), ts_dir, start=start, end=end)
    q_down = _ml_flow_veh_hr(int(stretch["down_ml_id"]), ts_dir, start=start, end=end)
    return np.maximum(q_down - q_up, 0.0) if is_on_ramp else np.maximum(q_up - q_down, 0.0)


def _build_feature_matrix(up_id: int, down_id: int, ts_dir: Path, *,
                           start, end, window_size: int) -> np.ndarray:
    """Build (T, 6*window_size) GRU feature matrix for [start, end) in veh/hr."""
    warmup_start = pd.Timestamp(start) - (window_size - 1) * _FIVE_MIN
    grid = pd.date_range(warmup_start, end, freq="5min", inclusive="left")
    up = _ml_arrays(up_id, ts_dir, grid)
    dn = _ml_arrays(down_id, ts_dir, grid)
    T = (pd.Timestamp(end) - pd.Timestamp(start)) // _FIVE_MIN
    rows = []
    for i in range(T):
        feats = []
        for lag in range(window_size):
            j = i + lag
            feats += [up[0][j], up[1][j], up[2][j], dn[0][j], dn[1][j], dn[2][j]]
        rows.append(feats)
    return np.array(rows, dtype=float)


# ---- Stretch mapping helpers ------------------------------------------------

def _vds_to_stretch_map(stretches_df: pd.DataFrame, col: str) -> dict[int, pd.Series]:
    """Map each VDS id in ``col`` (semicolon-separated) to its stretch row."""
    mapping: dict[int, pd.Series] = {}
    for _, row in stretches_df.iterrows():
        for vid in str(row.get(col, "")).split(";"):
            vid = vid.strip()
            if vid.isdigit():
                mapping[int(vid)] = row
    return mapping


def _furthest_downstream_ml_id(
    off_ids: list[int],
    stretches_df: pd.DataFrame,
    off_to_stretch: dict[int, pd.Series],
) -> int | None:
    """down_ml_id of the furthest-downstream stretch containing any of ``off_ids``."""
    rows = [off_to_stretch[v] for v in off_ids if v in off_to_stretch]
    rows = [r for r in rows if pd.notna(r.get("down_ml_id"))]
    if not rows:
        return None
    direction = str(stretches_df["dir"].iloc[0])
    key = "pm_down"
    chosen = max(rows, key=lambda r: r[key]) if direction in ("N", "E") else min(rows, key=lambda r: r[key])
    return int(chosen["down_ml_id"])


# ---- Resampling -------------------------------------------------------------

def _resample_to_sim(arr_5min: np.ndarray, dt_h: float) -> np.ndarray:
    """Resample a 5-min-cadence array to sim cadence (constant hold)."""
    dt_s = dt_h * 3600.0
    five_min_s = 300.0
    if dt_s > five_min_s + 1e-6:
        raise ValueError(f"dt_s={dt_s} > 5 min; super-5-min demand/beta not supported yet")
    steps_per_sample = round(five_min_s / dt_s)
    return np.repeat(arr_5min, steps_per_sample)


# ---- Main -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", required=True, type=Path)
    parser.add_argument("--stretches", required=True, type=Path)
    parser.add_argument("--timeseries-dir", required=True, type=Path)
    parser.add_argument("--gru-checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--start", required=True, type=pd.Timestamp)
    parser.add_argument("--end", required=True, type=pd.Timestamp)
    parser.add_argument("--dt-seconds", required=True, type=float,
                        help="Sim timestep in seconds (must match simulate_ctm_corridor.py).")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    cells_df = pd.read_csv(args.cells)
    stretches_df = pd.read_csv(args.stretches)
    manifest = json.loads(Path(args.manifest).read_text())
    window_size = int(manifest.get("corpus", {}).get("window_size", 3))
    dt_h = args.dt_seconds / 3600.0
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    ts_dir = args.timeseries_dir
    out_dir = (args.out_dir or args.cells.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== build_ctm_ramp_scenario ===", file=sys.stderr)
    print(f"  window : [{start}, {end})", file=sys.stderr)
    print(f"  dt     : {args.dt_seconds} s", file=sys.stderr)

    gru = GruEstimator.load(args.gru_checkpoint)

    on_to_stretch = _vds_to_stretch_map(stretches_df, "on_ids")
    off_to_stretch = _vds_to_stretch_map(stretches_df, "off_ids")

    # Cache GRU predictions per (up_ml_id, down_ml_id) — avoid re-running the
    # model for stretches that contribute both an on- and off-ramp to a cell.
    gru_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _gru_for(stretch: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        key = (int(stretch["up_ml_id"]), int(stretch["down_ml_id"]))
        if key not in gru_cache:
            X = _build_feature_matrix(key[0], key[1], ts_dir,
                                      start=start, end=end, window_size=window_size)
            gru_cache[key] = gru.predict_flows(X)
        return gru_cache[key]

    def _on_ramp_flow_5min(vds_id: int) -> np.ndarray:
        if _has_any_data(vds_id, ts_dir):
            return _ramp_flow_veh_hr(vds_id, ts_dir, start=start, end=end)
        stretch = on_to_stretch.get(vds_id)
        if stretch is None:
            print(f"  WARN: on-ramp VDS {vds_id} not in stretches; using 0", file=sys.stderr)
            T5 = int((end - start) / _FIVE_MIN)
            return np.zeros(T5)
        config = str(stretch["config_type"])
        if config in ("a", "b"):
            return _conservation_flow(stretch, ts_dir, start=start, end=end, is_on_ramp=True)
        r_hat, _ = _gru_for(stretch)
        return r_hat

    def _off_ramp_flow_5min(vds_id: int) -> np.ndarray:
        if _has_any_data(vds_id, ts_dir):
            return _ramp_flow_veh_hr(vds_id, ts_dir, start=start, end=end)
        stretch = off_to_stretch.get(vds_id)
        if stretch is None:
            print(f"  WARN: off-ramp VDS {vds_id} not in stretches; using 0", file=sys.stderr)
            T5 = int((end - start) / _FIVE_MIN)
            return np.zeros(T5)
        config = str(stretch["config_type"])
        if config in ("a", "b"):
            return _conservation_flow(stretch, ts_dir, start=start, end=end, is_on_ramp=False)
        _, s_hat = _gru_for(stretch)
        return s_hat

    # Build 5-min demand and beta, then resample to sim cadence.
    demand_5min: dict[int, np.ndarray] = {}
    for cell_idx, row in enumerate(cells_df.itertuples(index=False)):
        if not bool(row.on_ramp):
            continue
        on_ids = parse_ramp_vds_ids(getattr(row, "on_ramp_vds_id", None))
        if not on_ids:
            continue
        flows = [_on_ramp_flow_5min(v) for v in on_ids]
        demand_5min[cell_idx] = sum(flows[1:], flows[0])

    beta_5min: dict[int, np.ndarray] = {}
    T5 = int((end - start) / _FIVE_MIN)
    for cell_idx, row in enumerate(cells_df.itertuples(index=False)):
        if not bool(row.off_ramp):
            continue
        off_ids = parse_ramp_vds_ids(getattr(row, "off_ramp_vds_id", None))
        if not off_ids:
            continue
        flows = [_off_ramp_flow_5min(v) for v in off_ids]
        s_i = sum(flows[1:], flows[0])
        ml_id = _furthest_downstream_ml_id(off_ids, stretches_df, off_to_stretch)
        if ml_id is None:
            print(f"  WARN: no downstream ML for off-ramp cell {cell_idx}; beta=0", file=sys.stderr)
            beta_5min[cell_idx] = np.zeros(T5)
            continue
        f_i = _ml_flow_veh_hr(ml_id, ts_dir, start=start, end=end)
        denom = f_i + s_i
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(denom > 0, s_i / denom, 0.0)
        beta_5min[cell_idx] = np.clip(beta, 0.0, np.nextafter(1.0, 0.0))

    demand_df = pd.DataFrame({k: _resample_to_sim(v, dt_h) for k, v in demand_5min.items()})
    beta_df = pd.DataFrame({k: _resample_to_sim(v, dt_h) for k, v in beta_5min.items()})

    demand_df.to_csv(out_dir / "demand.csv", index=False)
    beta_df.to_csv(out_dir / "beta.csv", index=False)
    print(f"  demand : {out_dir}/demand.csv  ({len(demand_5min)} on-ramp cell(s))")
    print(f"  beta   : {out_dir}/beta.csv    ({len(beta_5min)} off-ramp cell(s))")


if __name__ == "__main__":
    main()
