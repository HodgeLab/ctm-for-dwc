"""End-to-end integration tests for
:meth:`PeMSDataProcessor.calibrate_fundamental_diagrams` and the FD-plot
sidecar (:meth:`plot_fundamental_diagram`).

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

from transportation_models.utils.data_processing import PeMSDataProcessor


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
) -> Path:
    """Write a minimal PeMS-style dataset under ``root``.

    Layout matches :func:`_write_mock_dataset` in test_data_processing.py:
    ``root/metadata/station_metadata.csv`` + one
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
        density = rng.uniform(1.0, RHO_JAM - 1.0, size=n_rows)
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


def _make_processor(root: Path, metadata_path: Path) -> PeMSDataProcessor:
    return PeMSDataProcessor(
        root_directory=str(root),
        metadata_filepath=str(metadata_path),
        save_directory=str(root / "processed"),
    )


# ---- calibrate_fundamental_diagrams (batch) -----------------------------


def test_batch_calibration_recovers_fd_for_each_station(tmp_path):
    """Two stations with identical synthetic FDs -> both rows in the output
    metadata CSV have the same calibrated parameters, within 1.5% of the
    ground-truth triangular FD."""
    metadata_path = _write_pems_dataset(tmp_path, station_ids=[0, 1])
    proc = _make_processor(tmp_path, metadata_path)
    proc.calibrate_fundamental_diagrams(
        detectors=["0", "1"],
        make_plots=False,
        save_params=True,
    )
    out_csv = tmp_path / "processed" / "station_metadata_calibrated.csv"
    assert out_csv.exists(), "calibrated metadata CSV not written"
    df = pd.read_csv(out_csv)
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


def test_batch_calibration_skips_when_save_params_false(tmp_path):
    """Without save_params, the calibrated CSV must not be written."""
    metadata_path = _write_pems_dataset(tmp_path, station_ids=[0])
    proc = _make_processor(tmp_path, metadata_path)
    proc.calibrate_fundamental_diagrams(
        detectors=["0"],
        make_plots=False,
        save_params=False,
    )
    assert not (tmp_path / "processed" / "station_metadata_calibrated.csv").exists()


def test_batch_calibration_honors_explicit_output_path(tmp_path):
    """An explicit output_metadata_path overrides the save_directory default."""
    metadata_path = _write_pems_dataset(tmp_path, station_ids=[0])
    proc = _make_processor(tmp_path, metadata_path)
    out_dir = tmp_path / "custom_out"
    out_dir.mkdir()
    explicit = out_dir / "my_calibrated.csv"
    proc.calibrate_fundamental_diagrams(
        detectors=["0"],
        make_plots=False,
        save_params=True,
        output_metadata_path=str(explicit),
    )
    assert explicit.exists()
    # And the default path was NOT written.
    assert not (tmp_path / "processed" / "station_metadata_calibrated.csv").exists()


def test_batch_calibration_skips_detectors_with_too_few_observed_rows(tmp_path):
    """A station whose every row has pct_observed below the threshold should
    log + skip, leaving its metadata row uncalibrated but not crashing."""
    metadata_path = _write_pems_dataset(tmp_path, station_ids=[0, 1])
    # Knock station 1's pct_observed below the imputation threshold so its
    # post-filter DataFrame is empty -- the calibration loop must continue.
    ts1 = tmp_path / "timeseries_data" / "1.csv"
    df1 = pd.read_csv(ts1)
    df1["pct_observed"] = 50.0
    df1.to_csv(ts1, index=False)

    proc = _make_processor(tmp_path, metadata_path)
    proc.calibrate_fundamental_diagrams(
        detectors=["0", "1"],
        make_plots=False,
        save_params=True,
    )
    out = pd.read_csv(tmp_path / "processed" / "station_metadata_calibrated.csv")
    row_0 = out.loc[out["Station ID"] == 0].iloc[0]
    row_1 = out.loc[out["Station ID"] == 1].iloc[0]
    assert row_0["free_flow_speed"] == pytest.approx(V_F, rel=1.5e-2)
    # Station 1 stays uncalibrated (the column either doesn't exist in the
    # input metadata or holds NaN after the loop skipped its row).
    assert pd.isna(row_1.get("free_flow_speed"))


# ---- plot_fundamental_diagram (smoke test) ------------------------------


def test_plot_fundamental_diagram_writes_png(tmp_path):
    """The plotter shouldn't crash and should write a PNG when save_path is set."""
    metadata_path = _write_pems_dataset(tmp_path, station_ids=[0])
    proc = _make_processor(tmp_path, metadata_path)
    out_dir = tmp_path / "plots"
    out_dir.mkdir()
    proc.calibrate_fundamental_diagrams(
        detectors=["0"],
        make_plots=True,
        saved_plot_dir=out_dir,
        save_params=False,
    )
    png = out_dir / "0.png"
    assert png.exists() and png.stat().st_size > 0
