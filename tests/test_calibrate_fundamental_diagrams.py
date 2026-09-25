"""End-to-end integration tests for
:func:`ctm.fd_calibration.calibrate_fundamental_diagrams`,
``calibrate_ramp_capacities``, the timeseries loader, and the FD-plot
sidecar (:func:`ctm.plots.plot_fundamental_diagram`).

These tests write a small synthetic PeMS dataset under ``tmp_path`` (one
metadata CSV + one per-detector timeseries CSV per station, ~4 weeks of
5-minute rows) where (density, flow) lies on a known triangular FD, then
verify the batch calibration recovers the ground-truth FD parameters and
writes the calibrated metadata CSV. Keeping the synthetic timeseries to
4 weeks bounds the whitening / plotting cost while still leaving enough
samples per (weekday, 5-min-block) cell for the whitening statistics to
be stable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ctm_for_dwc.ctm.fd_calibration import (
    CalibrationCode,
    calibrate_fundamental_diagrams,
    calibrate_ramp_capacities,
    load_detector_timeseries,
    read_station_metadata,
    validate_fd_params,
)


# Same ground-truth FD as in test_fundamental_diagram.py so reading either
# file gives a consistent mental model.
V_F = 60.0
W = 15.0
Q_MAX = 1800.0
RHO_CRIT = 30.0
RHO_JAM = 150.0

# 4 weeks * 7 days * 24 hours * 12 five-minute blocks = 8064 rows / station.
DAYS = 28
PERIODS_PER_DAY = 24 * 12
N_ROWS = DAYS * PERIODS_PER_DAY
LANES = 3


# ---- Synthetic dataset writers ------------------------------------------


def _triangular_fd_flow(density: np.ndarray) -> np.ndarray:
    """Per-lane flow [veh/hr] on the pinned triangular FD."""
    return np.where(
        density <= RHO_CRIT,
        V_F * density,
        np.maximum(0.0, W * (RHO_JAM - density)),
    )


def _write_pems_dataset(
    root: Path,
    *,
    station_ids: list[int],
    n_rows: int = N_ROWS,
    seed: int = 0,
    density_min: float = 1.0,
    density_max: float = RHO_JAM - 1.0,
) -> Path:
    """Write a minimal PeMS-style dataset under ``root``.

    Layout: ``root/metadata/station_metadata.csv`` + one
    ``root/timeseries_data/{station_id}.csv`` per station.
    """
    metadata_dir = root / "metadata"
    ts_dir = root / "timeseries_data"
    metadata_dir.mkdir(parents=True)
    ts_dir.mkdir(parents=True)

    metadata = pd.DataFrame({
        "Station ID": station_ids,
        "Lanes": [LANES] * len(station_ids),
        "Type": ["Mainline"] * len(station_ids),
        "Abs PM": [float(i) for i in range(len(station_ids))],
        "Length": [0.5] * len(station_ids),
    })
    metadata_path = metadata_dir / "station_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    timestamps = pd.date_range(
        start="2022-01-01", periods=n_rows, freq="5min",
    )
    rng = np.random.default_rng(seed)
    for station_id in station_ids:
        # Uniformly sample density across the FD's valid range. Every row
        # corresponds to a different 5-min snapshot, so flow and density
        # vary independently of time-of-day -- the whitening step ends up
        # with a flat per-(weekday, 5-min-block) statistic and the
        # whitened_flow used for outlier rejection stays roughly z-scored.
        density = rng.uniform(density_min, density_max, size=n_rows)
        flow_per_lane = _triangular_fd_flow(density)
        # standardize_timeseries: flow_[veh/hr-lane] = total_flow_5min * 12 / Lanes.
        total_flow_5min = flow_per_lane * LANES / 12.0
        # density_[veh/mi-lane] = flow_per_lane / avg_speed -> avg_speed = flow / density.
        avg_speed = flow_per_lane / density
        rows = pd.DataFrame({
            "timestamp": timestamps,
            "station": station_id,
            "pct_observed": 100.0,
            "total_flow_[veh/5-min]": total_flow_5min,
            "avg_speed_[mph]": avg_speed,
        })
        rows.to_csv(ts_dir / f"{station_id}.csv", index=False)

    return metadata_path


def _calibrate(root: Path, detectors: list[str], **kwargs) -> pd.DataFrame:
    """Run the batch calibration on ``root`` and read back the written CSV."""
    out_csv = root / "station_metadata_calibrated.csv"
    calibrate_fundamental_diagrams(
        root / "metadata" / "station_metadata.csv",
        root / "timeseries_data",
        detectors=detectors,
        output_path=out_csv,
        **kwargs,
    )
    return pd.read_csv(out_csv)


# ---- load_detector_timeseries -------------------------------------------


def test_loader_keeps_contiguous_grid_with_nan_gaps(tmp_path):
    """Rows missing from the CSV come back as NaN rows on a full 5-minute
    grid spanning the observed year(s), not dropped."""
    metadata_path = _write_pems_dataset(tmp_path, station_ids=[0])
    ts0 = tmp_path / "timeseries_data" / "0.csv"
    gap = [100, 101, 102, 500]
    pd.read_csv(ts0).drop(index=gap).to_csv(ts0, index=False)

    df = load_detector_timeseries(
        tmp_path / "timeseries_data", "0", read_station_metadata(metadata_path),
    )

    # Full 2022 grid, strictly contiguous.
    assert len(df) == 365 * PERIODS_PER_DAY
    assert (df["timestamp"].diff().dropna() == pd.Timedelta("5min")).all()

    # Gap rows are NaN in raw-count / pct columns.
    assert df.loc[gap, "total_flow_[veh/5-min]"].isna().all()
    assert df.loc[gap, "avg_speed_[mph]"].isna().all()
    assert df.loc[gap, "pct_observed"].isna().all()

    # Metadata survives the inner merge on reindexed rows.
    assert (df.loc[gap, "Station ID"] == 0).all()
    assert (df.loc[gap, "Lanes"] == LANES).all()


# ---- calibrate_fundamental_diagrams (batch) -----------------------------


def test_batch_calibration_recovers_fd_for_each_station(tmp_path):
    """Two stations with identical synthetic FDs -> both rows in the output
    metadata CSV have the same calibrated parameters, within 1.5% of the
    ground-truth triangular FD."""
    _write_pems_dataset(tmp_path, station_ids=[0, 1])
    df = _calibrate(tmp_path, ["0", "1"])
    expected_cols = {
        "Station ID", "Lanes", "Type", "Abs PM", "Length",
        "capacity", "free_flow_speed", "congestion_wave_speed",
        "jam_density", "critical_density",
    }
    assert expected_cols.issubset(df.columns)

    for station_id in (0, 1):
        row = df.loc[df["Station ID"] == station_id].iloc[0]
        assert row["free_flow_speed"] == pytest.approx(V_F, rel=1.5e-2)
        assert row["congestion_wave_speed"] == pytest.approx(W, rel=1.5e-2)
        assert row["capacity"] == pytest.approx(Q_MAX, rel=1.5e-2)
        assert row["critical_density"] == pytest.approx(RHO_CRIT, rel=1.5e-2)
        assert row["jam_density"] == pytest.approx(RHO_JAM, rel=1.5e-2)
        # A clean fit is flagged as success.
        assert row["calibration_code"] == CalibrationCode.OK
        assert row["calibration_status"] == "ok"


def test_batch_calibration_without_output_path_returns_df_only(tmp_path):
    """Without output_path nothing is written; the result is returned."""
    metadata_path = _write_pems_dataset(tmp_path, station_ids=[0])
    df = calibrate_fundamental_diagrams(
        metadata_path, tmp_path / "timeseries_data", detectors=["0"],
    )
    row = df.loc[df["Station ID"] == 0].iloc[0]
    assert row["calibration_code"] == CalibrationCode.OK
    assert row["free_flow_speed"] == pytest.approx(V_F, rel=1.5e-2)
    assert not list(tmp_path.glob("*.csv"))


def test_batch_calibration_skips_detectors_with_too_few_observed_rows(tmp_path):
    """A station whose every row has pct_observed below the threshold should
    log + skip, leaving its metadata row uncalibrated but not crashing."""
    _write_pems_dataset(tmp_path, station_ids=[0, 1])
    # Knock station 1's pct_observed below the imputation threshold so its
    # post-filter DataFrame is empty -- the calibration loop must continue.
    ts1 = tmp_path / "timeseries_data" / "1.csv"
    df1 = pd.read_csv(ts1)
    df1["pct_observed"] = 50.0
    df1.to_csv(ts1, index=False)

    out = _calibrate(tmp_path, ["0", "1"])
    row_0 = out.loc[out["Station ID"] == 0].iloc[0]
    row_1 = out.loc[out["Station ID"] == 1].iloc[0]
    assert row_0["free_flow_speed"] == pytest.approx(V_F, rel=1.5e-2)
    assert row_0["calibration_code"] == CalibrationCode.OK
    # Station 1 stays uncalibrated: NaN FD params + a no-data-after-filter code.
    assert pd.isna(row_1["free_flow_speed"])
    assert row_1["calibration_code"] == CalibrationCode.NO_DATA_AFTER_FILTER
    assert row_1["calibration_status"] == "no_data_after_filter"


# ---- calibration codes / edge cases -------------------------------------


def test_missing_timeseries_records_missing_code(tmp_path):
    """A detector registered in metadata but with no timeseries CSV gets
    code 1 (missing_timeseries) and NaN FD params, without crashing."""
    _write_pems_dataset(tmp_path, station_ids=[0, 5])
    # Remove station 5's timeseries so load_detector_timeseries raises.
    (tmp_path / "timeseries_data" / "5.csv").unlink()
    out = _calibrate(tmp_path, ["0", "5"])
    row_5 = out.loc[out["Station ID"] == 5].iloc[0]
    assert row_5["calibration_code"] == CalibrationCode.MISSING_TIMESERIES
    assert row_5["calibration_status"] == "missing_timeseries"
    assert pd.isna(row_5["free_flow_speed"])
    # The other station still calibrates fine.
    row_0 = out.loc[out["Station ID"] == 0].iloc[0]
    assert row_0["calibration_code"] == CalibrationCode.OK


def test_no_congestion_points_records_code_3(tmp_path):
    """A detector that only ever observes free-flow (density <= critical)
    has no congestion branch to fit -> code 3, NaN params."""
    _write_pems_dataset(
        tmp_path, station_ids=[0], density_min=1.0, density_max=RHO_CRIT - 1.0,
    )
    out = _calibrate(tmp_path, ["0"])
    row = out.loc[out["Station ID"] == 0].iloc[0]
    assert row["calibration_code"] == CalibrationCode.NO_CONGESTION_POINTS
    assert row["calibration_status"] == "no_congestion_points"
    assert pd.isna(row["capacity"])


def test_bin_knobs_are_threaded_and_recovery_holds(tmp_path):
    """Non-default bin_size / bin_iqr_multiplier flow through to the
    congestion fit without error and still recover the synthetic FD."""
    _write_pems_dataset(tmp_path, station_ids=[0])
    out = _calibrate(tmp_path, ["0"], bin_size=20, bin_iqr_multiplier=2.0)
    row = out.loc[out["Station ID"] == 0].iloc[0]
    assert row["calibration_code"] == CalibrationCode.OK
    assert row["free_flow_speed"] == pytest.approx(V_F, rel=1.5e-2)
    assert row["congestion_wave_speed"] == pytest.approx(W, rel=1.5e-2)


# ---- validate_fd_params (unit) ------------------------------------------


def _good_params() -> dict[str, float]:
    """Params lying exactly on the pinned triangular FD (capacity = v_f*rho_crit
    = w*(rho_jam - rho_crit))."""
    return {
        "capacity": Q_MAX,
        "free_flow_speed": V_F,
        "congestion_wave_speed": W,
        "critical_density": RHO_CRIT,
        "jam_density": RHO_JAM,
    }


def test_validate_fd_params_accepts_clean_triangle():
    code, reason = validate_fd_params(_good_params())
    assert code == CalibrationCode.OK
    assert reason == "ok"


@pytest.mark.parametrize("key, value", [
    ("free_flow_speed", -1.0),
    ("congestion_wave_speed", -5.0),
    ("capacity", 0.0),
])
def test_validate_fd_params_rejects_non_positive(key, value):
    params = _good_params()
    params[key] = value
    code, _ = validate_fd_params(params)
    assert code == CalibrationCode.VALIDATION_FAILED


def test_validate_fd_params_rejects_wave_faster_than_free_flow():
    params = _good_params()
    params["congestion_wave_speed"] = params["free_flow_speed"] + 5.0
    code, reason = validate_fd_params(params)
    assert code == CalibrationCode.VALIDATION_FAILED
    assert "free_flow_speed" in reason


def test_validate_fd_params_rejects_bad_density_ordering():
    params = _good_params()
    # Swap so critical >= jam.
    params["critical_density"] = RHO_JAM
    params["jam_density"] = RHO_CRIT
    code, _ = validate_fd_params(params)
    assert code == CalibrationCode.VALIDATION_FAILED


def test_validate_fd_params_rejects_triangle_inconsistency():
    params = _good_params()
    # Knock capacity ~40% off while leaving the branch params alone.
    params["capacity"] = Q_MAX * 0.6
    code, reason = validate_fd_params(params)
    assert code == CalibrationCode.VALIDATION_FAILED
    assert "triangle" in reason


def test_validate_fd_params_triangle_rtol_is_honored():
    params = _good_params()
    params["capacity"] = Q_MAX * 0.95  # 5% off
    # Default 0.05 tolerance: 5% off is at the edge -> rejected.
    assert validate_fd_params(params)[0] == CalibrationCode.VALIDATION_FAILED
    # A looser tolerance accepts it.
    assert validate_fd_params(params, triangle_rtol=0.1)[0] == CalibrationCode.OK


# ---- calibrate_ramp_capacities ------------------------------------------


def _write_ramp_timeseries(
    root: Path,
    *,
    station_id: int,
    capacity_veh_hr: float,
    n_rows: int = N_ROWS,
    seed: int = 0,
    spike: bool = False,
) -> None:
    """Write a single ramp-VDS timeseries CSV.

    Models a ramp whose 5-min flow is uniformly distributed in
    ``[0, capacity_veh_hr]``; the 99th percentile of the raw flow is
    therefore ~``0.99 * capacity_veh_hr``. When ``spike=True``, a few
    rows are inflated well above ``capacity_veh_hr`` to verify the
    outlier filter rejects them.
    """
    ts_dir = root / "timeseries_data"
    ts_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range(
        start="2022-01-01", periods=n_rows, freq="5min",
    )
    rng = np.random.default_rng(seed)
    flow_veh_hr = rng.uniform(0.0, capacity_veh_hr, size=n_rows)
    if spike:
        # 10 unrealistic spikes 3x the capacity -- whitening + IQR filter
        # should drop them before the 99th-percentile takes effect.
        flow_veh_hr[:10] = capacity_veh_hr * 3.0
    total_flow_5min = flow_veh_hr / 12.0
    rows = pd.DataFrame({
        "timestamp": timestamps,
        "station": station_id,
        "pct_observed": 100.0,
        "total_flow_[veh/5-min]": total_flow_5min,
        # avg_speed_[mph]: ramps don't report speed but the test fixture
        # carries the column so load_detector_timeseries reads it the same
        # way as on the mainline path.
        "avg_speed_[mph]": np.nan,
    })
    rows.to_csv(ts_dir / f"{station_id}.csv", index=False)


def _write_metadata_with_ramp(root: Path, ramp_station_ids: list[int]) -> Path:
    """Minimal metadata file that registers ramp VDSs with Type='On Ramp'."""
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame({
        "Station ID": ramp_station_ids,
        "Lanes": [1] * len(ramp_station_ids),
        "Type": ["On Ramp"] * len(ramp_station_ids),
        "Abs PM": [float(i) for i in range(len(ramp_station_ids))],
        "Length": [0.1] * len(ramp_station_ids),
    })
    metadata_path = metadata_dir / "station_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    return metadata_path


def _calibrate_ramps(root: Path, detectors: list[str], **kwargs) -> pd.DataFrame:
    return calibrate_ramp_capacities(
        detectors,
        root / "metadata" / "station_metadata.csv",
        root / "timeseries_data",
        **kwargs,
    )


def test_calibrate_ramp_capacities_recovers_99th_percentile(tmp_path):
    """A synthetic ramp with uniform[0, C] flow has ~0.99*C as its 99th pctl."""
    capacity = 1200.0
    _write_metadata_with_ramp(tmp_path, [100])
    _write_ramp_timeseries(tmp_path, station_id=100, capacity_veh_hr=capacity)
    out_csv = tmp_path / "ramp_metadata_calibrated.csv"
    ramp_df = _calibrate_ramps(tmp_path, ["100"], output_path=out_csv)
    # In-memory result has the expected schema and one row.
    assert list(ramp_df.columns) == [
        "Station ID", "ramp_capacity_[veh/hr]", "n_observations",
        "calibration_code", "calibration_status",
    ]
    assert len(ramp_df) == 1
    assert ramp_df.iloc[0]["ramp_capacity_[veh/hr]"] == pytest.approx(
        0.99 * capacity, rel=5e-2,
    )
    assert ramp_df.iloc[0]["calibration_code"] == CalibrationCode.OK
    assert ramp_df.iloc[0]["calibration_status"] == "ok"
    # CSV got written to output_path.
    assert out_csv.exists()
    on_disk = pd.read_csv(out_csv)
    assert on_disk.iloc[0]["ramp_capacity_[veh/hr]"] == pytest.approx(
        ramp_df.iloc[0]["ramp_capacity_[veh/hr]"]
    )


def test_calibrate_ramp_capacities_outlier_spikes_get_filtered(tmp_path):
    """A handful of 3x-capacity spikes shouldn't drag the 99th percentile up.

    The whiten + IQR filter should remove the bad rows before the
    quantile is taken, so the returned capacity stays near the genuine
    distribution's 99th percentile (~0.99 * capacity).
    """
    capacity = 1200.0
    _write_metadata_with_ramp(tmp_path, [100])
    _write_ramp_timeseries(
        tmp_path, station_id=100, capacity_veh_hr=capacity, spike=True,
    )
    ramp_df = _calibrate_ramps(tmp_path, ["100"])
    assert ramp_df.iloc[0]["ramp_capacity_[veh/hr]"] == pytest.approx(
        0.99 * capacity, rel=5e-2,
    )


def test_calibrate_ramp_capacities_emits_nan_for_missing_timeseries(tmp_path):
    """A detector with no timeseries file should record NaN, not crash."""
    _write_metadata_with_ramp(tmp_path, [100])
    # Don't write the timeseries CSV -- the loader will FileNotFoundError.
    ramp_df = _calibrate_ramps(tmp_path, ["100"])
    assert len(ramp_df) == 1
    assert pd.isna(ramp_df.iloc[0]["ramp_capacity_[veh/hr]"])
    assert ramp_df.iloc[0]["n_observations"] == 0
    assert ramp_df.iloc[0]["calibration_code"] == CalibrationCode.MISSING_TIMESERIES
    assert ramp_df.iloc[0]["calibration_status"] == "missing_timeseries"


def test_calibrate_ramp_capacities_quantile_arg_is_honored(tmp_path):
    """Asking for the median (q=0.5) should return ~capacity/2, not ~0.99*capacity."""
    capacity = 1200.0
    _write_metadata_with_ramp(tmp_path, [100])
    _write_ramp_timeseries(tmp_path, station_id=100, capacity_veh_hr=capacity)
    median = _calibrate_ramps(
        tmp_path, ["100"], quantile=0.5,
    ).iloc[0]["ramp_capacity_[veh/hr]"]
    assert median == pytest.approx(0.5 * capacity, rel=5e-2)


# ---- plot_fundamental_diagram (smoke test) ------------------------------


def test_plot_fundamental_diagram_writes_png(tmp_path):
    """The plotter shouldn't crash and should write a PNG when save_path is set."""
    metadata_path = _write_pems_dataset(tmp_path, station_ids=[0])
    out_dir = tmp_path / "plots"
    calibrate_fundamental_diagrams(
        metadata_path, tmp_path / "timeseries_data",
        detectors=["0"], plot_dir=out_dir,
    )
    png = out_dir / "0.png"
    assert png.exists() and png.stat().st_size > 0
