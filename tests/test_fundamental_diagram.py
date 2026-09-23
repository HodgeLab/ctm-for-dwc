"""Tests for the fundamental-diagram calibration path of
:class:`PeMSDataProcessor` (Step 4 of ``docs/ctm_module.md``).

Strategy
--------
1. **Primitive tests** (this layer) build small synthetic DataFrames in
   memory, where (density, flow) lies on a known triangular FD with a
   pinned set of parameters. They verify that
   :meth:`estimate_free_flow_speed`,
   :meth:`estimate_congestion_wave_speed`,
   :meth:`find_fundamental_diagram_peak`, and
   :meth:`extract_fundamental_diagram_params_from_timeseries` recover the
   ground-truth FD to within numerical tolerance. No filesystem.

2. **Batch / plot tests** (in :mod:`tests.test_calibrate_fundamental_diagrams`)
   exercise the full :meth:`calibrate_fundamental_diagrams` pipeline against
   ``tmp_path`` CSVs.

Ground-truth FD pinned across the file:
    v_f      = 60     mi/h          (free-flow speed)
    w        = 15     mi/h          (congestion wave speed)
    q_max    = 1800   veh/h-lane    (= v_f * rho_crit)
    rho_crit = 30     veh/mi-lane
    rho_jam  = 150    veh/mi-lane   (= rho_crit + q_max/w)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ctm_for_dwc.utils.data_processing import (
    CalibrationCode,
    PeMSDataProcessor,
)


# Ground-truth FD parameters used throughout this file.
V_F = 60.0
W = 15.0
Q_MAX = 1800.0
RHO_CRIT = 30.0
RHO_JAM = 150.0


# ---- Synthetic-FD generators --------------------------------------------


def _free_flow_points(n: int = 50, *, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """``n`` (density, flow) points on the free-flow side of the FD.

    ``flow = V_F * density`` with optional Gaussian noise. Densities span
    ``(0, RHO_CRIT)``.
    """
    rng = np.random.default_rng(seed)
    density = np.linspace(1.0, RHO_CRIT - 0.5, n)
    flow = V_F * density + rng.normal(0.0, noise, n)
    return pd.DataFrame({
        "density_[veh/mi-lane]": density,
        "flow_[veh/hr-lane]": flow,
    })


def _congestion_points(
    n: int = 100, *, noise: float = 0.0, seed: int = 0,
) -> pd.DataFrame:
    """``n`` (density, flow) points on the congestion side of the FD.

    ``flow = -W * (density - RHO_JAM)`` -- i.e. line from
    ``(RHO_CRIT, Q_MAX)`` down to ``(RHO_JAM, 0)``. Densities span
    ``(RHO_CRIT, RHO_JAM)``. Carries a ``whitened_flow`` column so
    :meth:`estimate_congestion_wave_speed`'s IQR-fence step has the column it
    expects (we set it to a z-scored copy of flow so it's well-defined).
    """
    rng = np.random.default_rng(seed)
    density = np.linspace(RHO_CRIT + 0.5, RHO_JAM - 0.5, n)
    flow = -W * (density - RHO_JAM) + rng.normal(0.0, noise, n)
    whitened = (flow - flow.mean()) / (flow.std(ddof=0) or 1.0)
    return pd.DataFrame({
        "density_[veh/mi-lane]": density,
        "flow_[veh/hr-lane]": flow,
        "whitened_flow": whitened,
    })


def _triangular_fd_points(
    n: int = 2000, *, noise: float = 0.0, seed: int = 0,
) -> pd.DataFrame:
    """``n`` points covering both halves of the triangular FD.

    The free-flow / congestion split is 50/50. The ``whitened_flow`` column
    is added so the bin-and-IQR pipeline in
    :meth:`estimate_congestion_wave_speed` has its input.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    rho_free = np.linspace(1.0, RHO_CRIT - 0.5, half)
    rho_cong = np.linspace(RHO_CRIT + 0.5, RHO_JAM - 0.5, n - half)
    density = np.concatenate([rho_free, rho_cong])
    flow_free = V_F * rho_free
    flow_cong = -W * (rho_cong - RHO_JAM)
    flow = np.concatenate([flow_free, flow_cong]) + rng.normal(0.0, noise, n)
    whitened = (flow - flow.mean()) / (flow.std(ddof=0) or 1.0)
    # Ensure the row with the peak flow lands exactly at RHO_CRIT/Q_MAX so the
    # critical-density estimate in
    # `extract_fundamental_diagram_params_from_timeseries` seeds correctly.
    peak_row = pd.DataFrame({
        "density_[veh/mi-lane]": [RHO_CRIT],
        "flow_[veh/hr-lane]": [Q_MAX],
        "whitened_flow": [(Q_MAX - flow.mean()) / (flow.std(ddof=0) or 1.0)],
    })
    df = pd.DataFrame({
        "density_[veh/mi-lane]": density,
        "flow_[veh/hr-lane]": flow,
        "whitened_flow": whitened,
    })
    return pd.concat([df, peak_row], ignore_index=True)


# ---- Minimal PeMSDataProcessor (skip the live metadata read) ------------


def _minimal_processor(tmp_path) -> PeMSDataProcessor:
    """Build a :class:`PeMSDataProcessor` against a tiny tmp_path scaffold.

    The constructor reads ``station_metadata.csv`` to populate
    ``self.metadata_df``; the primitive tests don't touch metadata, but the
    object still needs to be constructible.
    """
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(parents=True)
    pd.DataFrame({
        "Station ID": [0], "Lanes": [3], "Type": ["Mainline"],
        "Abs PM": [0.0], "Length": [0.5],
    }).to_csv(metadata_dir / "station_metadata.csv", index=False)
    return PeMSDataProcessor(
        root_directory=str(tmp_path),
        metadata_filepath=str(metadata_dir / "station_metadata.csv"),
    )


# ---- estimate_free_flow_speed -------------------------------------------


def test_estimate_free_flow_speed_recovers_ground_truth_slope(tmp_path):
    proc = _minimal_processor(tmp_path)
    df = _free_flow_points(n=50)
    v_f_hat, model = proc.estimate_free_flow_speed(
        timeseries_df=df,
        critical_density_estimate=RHO_CRIT,
        test_points=[0.0, RHO_CRIT],
    )
    assert v_f_hat == pytest.approx(V_F, rel=1e-6)
    # The returned model is what `find_fundamental_diagram_peak` will compose.
    assert model.coef_[0] == pytest.approx(V_F, rel=1e-6)


def test_estimate_free_flow_speed_drops_congestion_points(tmp_path):
    """Points above the critical-density cutoff must not influence the fit."""
    proc = _minimal_processor(tmp_path)
    free = _free_flow_points(n=30)
    # Inject 10 noisy congestion points that would steepen the slope if kept.
    congested = pd.DataFrame({
        "density_[veh/mi-lane]": np.linspace(RHO_CRIT + 5, RHO_JAM - 5, 10),
        "flow_[veh/hr-lane]": np.linspace(500, 0, 10),    # off-line garbage
    })
    df = pd.concat([free, congested], ignore_index=True)
    v_f_hat, _ = proc.estimate_free_flow_speed(
        timeseries_df=df,
        critical_density_estimate=RHO_CRIT,
        test_points=[0.0],
    )
    assert v_f_hat == pytest.approx(V_F, rel=1e-6)


# ---- estimate_congestion_wave_speed -------------------------------------


def test_estimate_congestion_wave_speed_recovers_wave_and_jam(tmp_path):
    """With bin_size=1 each bin holds one point, so the max-flow-per-bin trick
    pairs ``(density, flow)`` exactly and the linear fit recovers the line."""
    proc = _minimal_processor(tmp_path)
    df = _congestion_points(n=100)
    slope, jam, model, bin_df = proc.estimate_congestion_wave_speed(
        timeseries_df=df,
        critical_density_estimate=RHO_CRIT,
        test_points=[RHO_CRIT, RHO_JAM],
        bin_size=1,
    )
    # The line drops from (RHO_CRIT, Q_MAX) to (RHO_JAM, 0) so its slope is -W.
    assert slope == pytest.approx(-W, rel=1e-3)
    # Jam density = -intercept / slope; intercept = W * RHO_JAM, so jam = RHO_JAM.
    assert jam == pytest.approx(RHO_JAM, rel=1e-3)
    assert len(bin_df) == 100
    assert {"BinDensity", "BinFlow"}.issubset(bin_df.columns)


def test_estimate_congestion_wave_speed_drops_free_flow_points(tmp_path):
    """Points below the critical-density cutoff must not pollute the bins."""
    proc = _minimal_processor(tmp_path)
    cong = _congestion_points(n=100)
    free = _free_flow_points(n=50)
    free["whitened_flow"] = 0.0
    df = pd.concat([free, cong], ignore_index=True)
    slope, jam, _model, _bins = proc.estimate_congestion_wave_speed(
        timeseries_df=df,
        critical_density_estimate=RHO_CRIT,
        test_points=[RHO_CRIT],
        bin_size=1,
    )
    assert slope == pytest.approx(-W, rel=1e-3)
    assert jam == pytest.approx(RHO_JAM, rel=1e-3)


def test_estimate_congestion_wave_speed_bin_size_changes_bin_count(tmp_path):
    """100 points / bin_size = expected number of bins."""
    proc = _minimal_processor(tmp_path)
    df = _congestion_points(n=100)
    _slope, _jam, _model, bins_5 = proc.estimate_congestion_wave_speed(
        timeseries_df=df, critical_density_estimate=RHO_CRIT,
        test_points=[], bin_size=5,
    )
    _slope, _jam, _model, bins_20 = proc.estimate_congestion_wave_speed(
        timeseries_df=df, critical_density_estimate=RHO_CRIT,
        test_points=[], bin_size=20,
    )
    assert len(bins_5) == 20
    assert len(bins_20) == 5


# ---- find_fundamental_diagram_peak --------------------------------------


def test_find_peak_intersects_free_flow_and_congestion_lines(tmp_path):
    """Composite test: pipe both fits into the peak finder, recover (rho_crit, q_max)."""
    proc = _minimal_processor(tmp_path)
    _v_f, ff_model = proc.estimate_free_flow_speed(
        timeseries_df=_free_flow_points(n=50),
        critical_density_estimate=RHO_CRIT,
        test_points=[],
    )
    _slope, _jam, cong_model, _ = proc.estimate_congestion_wave_speed(
        timeseries_df=_congestion_points(n=100),
        critical_density_estimate=RHO_CRIT,
        test_points=[],
        bin_size=1,
    )
    rho_crit_hat, q_max_hat = proc.find_fundamental_diagram_peak(
        free_flow_model=ff_model, congestion_model=cong_model,
    )
    assert rho_crit_hat == pytest.approx(RHO_CRIT, rel=1e-3)
    assert q_max_hat == pytest.approx(Q_MAX, rel=1e-3)


def test_find_peak_handles_arbitrary_slopes_and_intercept(tmp_path):
    """Sanity-check the intersection math against a hand-derived case."""
    from sklearn.linear_model import LinearRegression

    proc = _minimal_processor(tmp_path)
    # Free-flow: flow = 50 * density (no intercept).
    ff = LinearRegression(fit_intercept=False)
    ff.fit(np.array([[1.0], [2.0]]), np.array([50.0, 100.0]))
    # Congestion: flow = -10 * density + 600  ->  intersects free flow at (10, 500).
    cong = LinearRegression()
    cong.fit(np.array([[40.0], [60.0]]), np.array([200.0, 0.0]))
    rho, q = proc.find_fundamental_diagram_peak(ff, cong)
    assert rho == pytest.approx(10.0, rel=1e-6)
    assert q == pytest.approx(500.0, rel=1e-6)


# ---- extract_fundamental_diagram_params_from_timeseries -----------------


def test_extract_recovers_all_five_fd_parameters(tmp_path):
    """End-to-end recovery on a high-resolution triangular FD.

    The default ``bin_size=10`` inside :meth:`estimate_congestion_wave_speed`
    introduces a small systematic shift on perfectly monotonic synthetic data
    (each bin's max flow lands at the bin's *smallest* density), so we use
    2000 points (1000 congestion side, density spacing ~0.12, bin range ~1.2)
    to keep the bias below 1%. Tolerance set to 1.5% accordingly.
    """
    proc = _minimal_processor(tmp_path)
    df = _triangular_fd_points(n=2000)
    params, code = proc.extract_fundamental_diagram_params_from_timeseries(
        timeseries_df=df,
        detector_id="0",
        make_plots=False,
    )
    assert code == CalibrationCode.OK
    assert set(params) == {
        "Station ID", "capacity", "free_flow_speed",
        "congestion_wave_speed", "jam_density", "critical_density",
    }
    assert params["Station ID"] == 0
    assert params["free_flow_speed"] == pytest.approx(V_F, rel=2e-3)
    # `congestion_wave_speed` is flipped sign internally so it matches CTM
    # convention (positive going downstream); ground truth is +W here.
    assert params["congestion_wave_speed"] == pytest.approx(W, rel=2e-3)
    assert params["capacity"] == pytest.approx(Q_MAX, rel=1.5e-2)
    assert params["critical_density"] == pytest.approx(RHO_CRIT, rel=1.5e-2)
    assert params["jam_density"] == pytest.approx(RHO_JAM, rel=1.5e-2)


def test_extract_returns_int_station_id(tmp_path):
    """`Station ID` is coerced to int -- downstream metadata join keys depend on it."""
    proc = _minimal_processor(tmp_path)
    params, _ = proc.extract_fundamental_diagram_params_from_timeseries(
        timeseries_df=_triangular_fd_points(n=2000),
        detector_id="42",
        make_plots=False,
    )
    assert isinstance(params["Station ID"], int)
    assert params["Station ID"] == 42
