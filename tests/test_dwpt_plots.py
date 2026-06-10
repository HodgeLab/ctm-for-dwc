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
    plots.plot_demand_heatmap(result, out_path=out)
    assert out.exists() and out.stat().st_size > 0
