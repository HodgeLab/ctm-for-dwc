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


_MI = 1609.344  # meters per mile, matching plots._METERS_PER_MILE


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
    """Run a profile plotter and return (bar heights, ylabel) from its axes.

    Bars carry a length-normalized density, so expected values below are the
    bin's load in MW divided by that bin's width in miles.
    """
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
    # Bin 0 peaks at t=0 (8 W), bin 1 at t=2 (10 W); bins are 2 m wide.
    np.testing.assert_allclose(peak, np.array([8.0, 10.0]) / 1e6 / (seg / _MI))
    # The corridor peaks at t=2 (10 W), where bin 0 carries nothing.
    np.testing.assert_allclose(at_peak, np.array([0.0, 10.0]) / 1e6 / (seg / _MI))
    assert np.all(peak >= at_peak) and peak.sum() > at_peak.sum()
    assert "peak power density [MW/mi]" == ylabel


def test_average_load_profile_is_energy_over_duration(tmp_path, monkeypatch):
    """A bin's bar is its total energy divided by the simulated duration."""
    result = _ramp_demand()
    dt_h = 0.5
    mean, ylabel = _captured_bars(
        monkeypatch, plots.plot_average_load_profile, result=result, dt_h=dt_h,
        segment_m=2.0, out_path=tmp_path / "avg.png",
    )
    duration_h = result.E.shape[0] * dt_h
    # bin energies [Wh], over the duration and the 2 m bin width
    expected = np.array([10.0, 12.0]) / duration_h / 1e6 / (2.0 / _MI)
    np.testing.assert_allclose(mean, expected)
    assert "mean power density [MW/mi]" == ylabel


def test_load_factor_profile_is_mean_over_peak(tmp_path, monkeypatch):
    """Load factor is each bin's own mean/peak, in [0, 1]; empty bins read 0."""
    result = _ramp_demand()
    factor, ylabel = _captured_bars(
        monkeypatch, plots.plot_load_factor_profile, result=result, dt_h=0.5,
        segment_m=2.0, out_path=tmp_path / "lf.png",
    )
    # Bin length cancels in mean/peak, so these are unnormalized ratios.
    np.testing.assert_allclose(factor, [(10.0 / 3) / 8.0, (12.0 / 3) / 10.0])
    assert np.all((factor >= 0.0) & (factor <= 1.0))
    assert "load factor [-]" == ylabel

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
    # Positions 0,1,2 fall in cell 0 (peaking at t=0, 4+4+0 W over 3 m);
    # position 3 in cell 1 (peaking at t=2, 5 W over 1 m). The unequal cell
    # lengths are exactly what the MW/mi normalization has to divide out.
    np.testing.assert_allclose(
        peak, np.array([8.0 / (3.0 / _MI), 5.0 / (1.0 / _MI)]) / 1e6
    )
    assert "peak power density [MW/mi]" == ylabel


def test_load_distribution_profile_boxes_track_bins(tmp_path, monkeypatch):
    """One box per bin, at the bin center, summarizing that bin over time."""
    result = _ramp_demand()
    captured = {}
    monkeypatch.setattr(plots.plt, "close", lambda fig: captured.setdefault("fig", fig))
    # Cell 0 holds positions 0-2, so its load over time is [8, 3, 5] W;
    # cell 1 holds position 3 only: [0, 1, 5] W.
    plots.plot_load_distribution_profile(
        result, dt_h=1.0, cell_edges_m=np.array([0.0, 3.0, 4.0]),
        out_path=tmp_path / "dist.png",
    )
    ax = captured["fig"].axes[0]

    # Each box contributes three horizontal segments -- the two whisker caps
    # and the median between them. Group them by the bin center they sit on.
    levels: dict[float, list[float]] = {}
    for line in ax.lines:
        x, y = line.get_xdata(), line.get_ydata()
        if len(y) == 2 and y[0] == y[1] and x[0] != x[1]:
            levels.setdefault(round(float(np.mean(x)), 12), []).append(float(y[0]))

    # Bin centers are 1.5 m and 3.5 m, plotted on a mile axis.
    centers = sorted(levels)
    np.testing.assert_allclose(centers, np.array([1.5, 3.5]) / _MI)
    # (low whisker, median, high whisker) per bin, as MW over the cell's own
    # length (3 m and 1 m in miles); no point falls outside the whiskers.
    np.testing.assert_allclose(
        sorted(levels[centers[0]]), np.array([3.0, 5.0, 8.0]) / 1e6 / (3.0 / _MI)
    )
    np.testing.assert_allclose(
        sorted(levels[centers[1]]), np.array([0.0, 1.0, 5.0]) / 1e6 / (1.0 / _MI)
    )
    assert ax.get_ylabel() == "power density [MW/mi]"
    plots.plt.close(captured["fig"])


def test_peak_profile_density_integrates_to_corridor_peak(tmp_path, monkeypatch):
    """Density bars integrate -- not sum -- back to the peak corridor power.

    This is the invariant that replaced "bars sum to the corridor peak" when
    the profiles switched to MW/mi, and it must hold for unequal bin widths.
    """
    result = _ramp_demand()
    dt_h = 0.25
    edges = np.array([0.0, 3.0, 4.0])  # cells of 3 m and 1 m
    density, _ = _captured_bars(
        monkeypatch, plots.plot_peak_load_profile, result=result, dt_h=dt_h,
        cell_edges_m=edges, out_path=tmp_path / "peak.png",
    )
    t_star = int(np.argmax(result.E.sum(axis=1)))
    corridor_MW = result.E[t_star].sum() / dt_h / 1e6
    widths_mi = np.diff(edges) / _MI
    np.testing.assert_allclose((density * widths_mi).sum(), corridor_MW)
