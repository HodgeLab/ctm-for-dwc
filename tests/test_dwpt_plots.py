"""Tests for the DWPT P6 sandbox: position_m property, corridor_from_ctm,
and the plot helpers.

Plots are verified as smoke tests (run without error, write a non-empty
file) under the Agg backend; the numeric helpers get real assertions. The
demo entrypoint (``scripts/run_dwpt_demand_demo.py``) is not unit-tested,
matching the CTM convention for runnable scripts.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from transportation_models.utils.ctm.engine import simulate
from transportation_models.utils.ctm.examples import (
    four_cell_freeway,
    four_cell_scenario,
)
from transportation_models.utils.dwpt import plots
from transportation_models.utils.dwpt.adapter import corridor_from_ctm
from transportation_models.utils.dwpt.model import CorridorSpec, DemandResult, PadSpec


def _pad() -> PadSpec:
    return PadSpec(
        alpha=20.0, delta=20.0, lambda_gap=10.0,
        beta=150_000.0, beta_prime=150_000.0,
    )


# ---------------------------------------------------------------------------
# CorridorSpec.position_m and corridor_from_ctm
# ---------------------------------------------------------------------------


def test_corridor_spec_position_m_is_center_reference():
    pad = PadSpec(alpha=2.0, delta=3.0, lambda_gap=1.0, beta=1.0, beta_prime=1.0)
    corridor = CorridorSpec(cell_lengths_m=(10.0,), pad=pad, dx_grid=1.0)
    off = (corridor.delta_grid - 1) // 2
    expected = (np.arange(corridor.m_traversal) - off) * corridor.dx_grid
    np.testing.assert_allclose(corridor.position_m, expected)
    assert corridor.position_m.shape == (corridor.m_traversal,)


def test_corridor_from_ctm_snaps_cell_lengths():
    result = simulate(four_cell_freeway(), four_cell_scenario(steps=5))
    corridor = corridor_from_ctm(result, _pad(), dx_grid=10.0)
    assert len(corridor.cell_lengths_m) == 4
    # each 1-mile cell (1609.344 m) snaps to 161 positions at dx=10
    assert corridor.positions_per_cell == (161, 161, 161, 161)


# ---------------------------------------------------------------------------
# Plot helpers (smoke)
# ---------------------------------------------------------------------------


def test_plot_P_y_writes_file(tmp_path):
    corridor = CorridorSpec(cell_lengths_m=(200.0,), pad=_pad(), dx_grid=10.0)
    out = tmp_path / "p_y.png"
    plots.plot_P_y(corridor.build_P_y(), corridor.position_m, out_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_demand_heatmap_writes_file(tmp_path):
    result = DemandResult(
        E=np.arange(12, dtype=float).reshape(3, 4),
        position_m=np.array([0.0, 1.0, 2.0, 3.0]),
        timesteps=np.arange(3),
    )
    out = tmp_path / "heat.png"
    plots.plot_demand_heatmap(result, dt_h=1.0 / 360.0, out_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_demand_heatmap_y_axis_is_physical_time(tmp_path, monkeypatch):
    """The y-axis spans physical hours (n_t * dt_h), not timestep indices."""
    n_t, dt_h = 3, 10.0 / 3600.0  # 10-second timesteps
    result = DemandResult(
        E=np.arange(12, dtype=float).reshape(n_t, 4),
        position_m=np.array([0.0, 1.0, 2.0, 3.0]),
        timesteps=np.arange(n_t),
    )
    # Capture the figure instead of letting the plotter close it.
    captured = {}
    monkeypatch.setattr(plots.plt, "close", lambda fig: captured.setdefault("fig", fig))
    plots.plot_demand_heatmap(result, dt_h=dt_h, out_path=tmp_path / "heat.png")

    ax = captured["fig"].axes[0]
    assert ax.get_ylabel() == "time [h]"
    np.testing.assert_allclose(ax.get_ylim(), (0.0, n_t * dt_h))
    plots.plt.close(captured["fig"])


def _ramp_demand() -> DemandResult:
    """Two bins whose loads peak at different timesteps.

    Positions 0-1 fall in bin [0, 2); positions 2-3 in bin [2, 4). Bin 0 peaks
    at t=0 and bin 1 at t=2, so no single timestep holds both peaks.
    """
    E = np.array([
        [4.0, 4.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 5.0, 5.0],
    ])
    return DemandResult(
        E=E,
        position_m=np.array([0.0, 1.0, 2.0, 3.0]),
        timesteps=np.arange(3),
    )


def _captured_bars(monkeypatch, plot_fn, **kwargs):
    """Run a profile plotter and return (bar heights, ylabel) from its axes."""
    captured = {}
    monkeypatch.setattr(plots.plt, "close", lambda fig: captured.setdefault("fig", fig))
    plot_fn(**kwargs)
    ax = captured["fig"].axes[0]
    heights = np.array([patch.get_height() for patch in ax.patches])
    ylabel = ax.get_ylabel()
    plots.plt.close(captured["fig"])
    return heights, ylabel


def test_absolute_peak_profile_is_non_coincident(tmp_path, monkeypatch):
    """Each bin peaks at its own worst timestep, not at the corridor peak."""
    result = _ramp_demand()
    dt_h, seg = 1.0, 2.0
    peak, ylabel = _captured_bars(
        monkeypatch, plots.plot_absolute_peak_profile, result=result, dt_h=dt_h,
        segment_m=seg, out_path=tmp_path / "abs_peak.png",
    )
    at_peak, _ = _captured_bars(
        monkeypatch, plots.plot_peak_load_profile, result=result, dt_h=dt_h,
        segment_m=seg, out_path=tmp_path / "peak.png",
    )
    # Bin 0 peaks at t=0 (8 W -> 8e-6 MW), bin 1 at t=2 (10 W).
    np.testing.assert_allclose(peak, np.array([8.0, 10.0]) / 1e6)
    # The corridor peaks at t=2 (10 W), where bin 0 carries nothing.
    np.testing.assert_allclose(at_peak, np.array([0.0, 10.0]) / 1e6)
    assert np.all(peak >= at_peak) and peak.sum() > at_peak.sum()
    assert "peak power per 2 m segment [MW]" == ylabel


def test_average_load_profile_is_energy_over_duration(tmp_path, monkeypatch):
    """A bin's bar is its total energy divided by the simulated duration."""
    result = _ramp_demand()
    dt_h = 0.5
    mean, ylabel = _captured_bars(
        monkeypatch, plots.plot_average_load_profile, result=result, dt_h=dt_h,
        segment_m=2.0, out_path=tmp_path / "avg.png",
    )
    duration_h = result.E.shape[0] * dt_h
    expected = np.array([10.0, 12.0]) / duration_h / 1e6  # bin energies [Wh]
    np.testing.assert_allclose(mean, expected)
    assert "mean power per 2 m segment [MW]" == ylabel


def test_load_factor_profile_is_mean_over_peak(tmp_path, monkeypatch):
    """Load factor is each bin's own mean/peak, in [0, 1]; empty bins read 0."""
    result = _ramp_demand()
    factor, ylabel = _captured_bars(
        monkeypatch, plots.plot_load_factor_profile, result=result, dt_h=0.5,
        segment_m=2.0, out_path=tmp_path / "lf.png",
    )
    np.testing.assert_allclose(factor, [(10.0 / 3) / 8.0, (12.0 / 3) / 10.0])
    assert np.all((factor >= 0.0) & (factor <= 1.0))
    assert "load factor per 2 m segment [-]" == ylabel

    # A corridor bin that never sees load divides 0/0 -> 0, not NaN.
    empty = DemandResult(
        E=np.zeros((3, 4)), position_m=result.position_m, timesteps=np.arange(3),
    )
    zero, _ = _captured_bars(
        monkeypatch, plots.plot_load_factor_profile, result=empty, dt_h=0.5,
        segment_m=2.0, out_path=tmp_path / "lf0.png",
    )
    np.testing.assert_array_equal(zero, np.zeros(2))


def test_cell_binning_matches_cell_edges(tmp_path, monkeypatch):
    """With --aggregate=cell the bins are the given cell edges, labelled 'cell'."""
    result = _ramp_demand()
    peak, ylabel = _captured_bars(
        monkeypatch, plots.plot_absolute_peak_profile, result=result, dt_h=1.0,
        cell_edges_m=np.array([0.0, 3.0, 4.0]), out_path=tmp_path / "cells.png",
    )
    # Positions 0,1,2 fall in cell 0 (peaking at t=0, 4+4+0 W); position 3 in
    # cell 1 (peaking at t=2, 5 W).
    np.testing.assert_allclose(peak, np.array([8.0, 5.0]) / 1e6)
    assert "peak power per cell [MW]" == ylabel
