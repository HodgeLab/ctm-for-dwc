"""Tests for ``scripts/build_ctm_qp_inputs.py`` (Level 3 QP input prep).

Covers the dt selection (CFL bound + 5-min-divisor rule) and the bundle
writer (schema, units, direct/tiebreak observed-density extraction).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from build_ctm_qp_inputs import build_bundle, choose_dt_seconds  # noqa: E402

_FW = pd.DataFrame(
    {
        "length": [0.5, 0.4, 0.5],
        "q_max": [4000.0, 4000.0, 4000.0],
        "v_f": [65.0, 65.0, 65.0],
        "w": [11.818, 11.818, 11.818],
        "rho_jam": [400.0, 400.0, 400.0],
        "rho_crit": [61.538, 61.538, 61.538],
        "on_ramp": [False, True, False],
        "off_ramp": [False, False, True],
        "on_ramp_capacity": [np.nan, 1500.0, np.nan],
        "off_ramp_capacity": [np.nan, np.nan, 1200.0],
        "lanes": [2, 2, 2],
    }
)


def _write_ts(directory: Path, vds_id: int, start: str, n: int, flow: float, speed: float):
    ts = pd.date_range(start=start, periods=n, freq="5min")
    pd.DataFrame(
        {
            "timestamp": ts,
            "total_flow_[veh/5-min]": np.full(n, flow, float),
            "avg_speed_[mph]": np.full(n, speed, float),
        }
    ).to_csv(directory / f"{vds_id}.csv", index=False)


def _cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vds_id": [100, 101, 102],
            "vds_source": ["direct", "direct", "direct"],
            "vds_lanes": [2, 2, 2],
            "lanes": [2, 2, 2],
            "on_ramp": [False, True, False],
            "off_ramp": [False, False, True],
            "on_ramp_vds_id": [pd.NA, 900, pd.NA],
            "off_ramp_vds_id": [pd.NA, pd.NA, 800],
        }
    )


# ---- dt selection --------------------------------------------------------


def test_choose_dt_picks_largest_5min_divisor_below_cfl():
    dt_s, advisory = choose_dt_seconds(_FW, None)
    # CFL bound = 0.4 mi / 65 mph = 22.15 s; largest 300/n strictly below it.
    assert dt_s < advisory.binding_dt_max_s
    assert abs(300.0 / dt_s - round(300.0 / dt_s)) < 1e-9  # integer steps/5min
    # No larger divisor (300/(n-1)) fits under the bound.
    n = round(300.0 / dt_s)
    assert 300.0 / (n - 1) >= advisory.binding_dt_max_s


def test_choose_dt_rejects_over_cfl_explicit():
    with pytest.raises(SystemExit, match="CFL"):
        choose_dt_seconds(_FW, 30.0)  # > 22 s bound


def test_choose_dt_rejects_non_divisor_explicit():
    with pytest.raises(SystemExit, match="divide 5 minutes"):
        choose_dt_seconds(_FW, 7.0)  # 300/7 not integer


# ---- bundle writer -------------------------------------------------------


def test_build_bundle_schema_and_units(tmp_path):
    ts = tmp_path / "ts"
    ts.mkdir()
    for vid, flow, speed in [(100, 120, 60), (101, 110, 58), (102, 100, 55)]:
        _write_ts(ts, vid, "2022-04-12 05:00", 36, flow, speed)
    _write_ts(ts, 900, "2022-04-12 05:00", 36, 20, 40)
    _write_ts(ts, 800, "2022-04-12 05:00", 36, 15, 40)

    out = tmp_path / "bundle"
    meta = build_bundle(
        _FW, _cells(), ts,
        start=pd.Timestamp("2022-04-12 06:00"),
        end=pd.Timestamp("2022-04-12 07:00"),
        dt_s=20.0, upstream_vds=100, out_dir=out,
    )

    assert meta["steps_per_5min"] == 15
    assert meta["n_5min"] == 12
    assert meta["T"] == 180
    assert meta["on_ramp_cells"] == [1]
    assert meta["off_ramp_cells"] == [2]
    assert meta["on_ramp_capacity"] == {"1": 1500.0}
    assert meta["off_ramp_capacity"] == {"2": 1200.0}
    assert meta["direct_tiebreak_cells"] == [0, 1, 2]
    assert meta["gamma"] == 1.0

    rho0 = pd.read_csv(out / "rho0.csv")
    assert list(rho0["cell"]) == [0, 1, 2]
    # Observed/initial density = total_flow*12 / speed: cell 0 = 120*12/60 = 24.
    assert rho0["rho0"].iloc[0] == pytest.approx(24.0)

    inflow = pd.read_csv(out / "inflow.csv")
    assert len(inflow) == meta["T"]
    assert inflow["inflow"].iloc[0] == pytest.approx(120 * 12)  # veh/h

    obs = pd.read_csv(out / "observed_density.csv")
    # 3 direct cells x 12 samples, all non-NaN.
    assert len(obs) == 36
    assert set(obs["cell"]) == {0, 1, 2}
    cell0 = obs.loc[obs["cell"] == 0, "rho_obs"]
    assert np.allclose(cell0.to_numpy(), 24.0)


def test_build_bundle_drops_nan_observed_samples(tmp_path):
    ts = tmp_path / "ts"
    ts.mkdir()
    for vid, flow, speed in [(100, 120, 60), (101, 110, 58), (102, 100, 55)]:
        _write_ts(ts, vid, "2022-04-12 05:00", 36, flow, speed)
    _write_ts(ts, 900, "2022-04-12 05:00", 36, 20, 40)
    _write_ts(ts, 800, "2022-04-12 05:00", 36, 15, 40)
    # Punch a NaN into cell 1's mainline VDS inside the sim window.
    df = pd.read_csv(ts / "101.csv")
    df.loc[df["timestamp"] == "2022-04-12 06:15:00", "avg_speed_[mph]"] = np.nan
    df.to_csv(ts / "101.csv", index=False)

    out = tmp_path / "bundle"
    build_bundle(
        _FW, _cells(), ts,
        start=pd.Timestamp("2022-04-12 06:00"),
        end=pd.Timestamp("2022-04-12 07:00"),
        dt_s=20.0, upstream_vds=100, out_dir=out,
    )
    obs = pd.read_csv(out / "observed_density.csv")
    # Cell 1 loses exactly the one NaN sample; cells 0 and 2 keep all 12.
    assert len(obs.loc[obs["cell"] == 1]) == 11
    assert len(obs.loc[obs["cell"] == 0]) == 12
