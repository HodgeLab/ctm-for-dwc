"""Tests for the CTMSIM .mat adapter in utils/ctm/io.py.

Uses the I-210W ``w060412.mat`` config that ships with CTMSIM v1.1 as a
fixture. The file lives outside the repo; tests skip cleanly if it's
absent so the suite stays green in any environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ctm_for_dwc.utils.ctm import (
    ctmsim_demand_at_sim_steps,
    ctmsim_initial_densities,
    freeway_from_ctmsim_mat,
)

CTMSIM_MAT = Path("/Users/robstallman/projects/ctmsim_v1_1/ctmsim_v1_1/i210/w060412.mat")


@pytest.fixture(scope="module")
def mat_path() -> Path:
    if not CTMSIM_MAT.exists():
        pytest.skip(f"CTMSIM .mat file not found at {CTMSIM_MAT}")
    return CTMSIM_MAT


@pytest.fixture(scope="module")
def freeway(mat_path: Path):
    # Validation (FD consistency + CFL) runs inside freeway_from_ctmsim_mat.
    return freeway_from_ctmsim_mat(mat_path)


# ---- Freeway adapter ------------------------------------------------------


def test_freeway_has_40_cells_and_correct_dt(freeway):
    assert freeway.n_cells == 40
    # CTMSIM TS = 10 s = 1/360 h for this config.
    assert freeway.dt == pytest.approx(1.0 / 360.0, abs=1e-12)


def test_cell_0_matches_ctmsim_struct(freeway):
    """Cell 0 of w060412 is 'ML Vernon' upstream entrance (4 lanes, FDfmax=8390)."""
    c = freeway.cells[0]
    # Geometry / FD: lifted directly from the .mat struct.
    assert c.length == pytest.approx(abs(38.7802 - 38.9696), abs=1e-3)
    assert c.q_max == pytest.approx(8390.0)
    assert c.rho_crit == pytest.approx(131.4)
    assert c.rho_jam == pytest.approx(655.77)
    # Triangular FD: v_f = q_max / rho_crit, w = q_max / (rho_jam - rho_crit).
    assert c.v_f == pytest.approx(8390.0 / 131.4)
    assert c.w == pytest.approx(8390.0 / (655.77 - 131.4))
    # Cell 0 carries the upstream "ML Vernon" on-ramp and has no off-ramp.
    assert c.on_ramp is True
    assert c.off_ramp is False
    assert c.on_ramp_capacity == pytest.approx(8390.0)
    # CTMSIM defaults gamma = xi = 1.
    assert c.gamma == pytest.approx(1.0)
    assert c.xi == pytest.approx(1.0)


def test_ramp_pattern_matches_w060412(freeway):
    """w060412 has 22 on-ramps and 18 off-ramps; specific cell indices match."""
    on_ramp_idx = [i for i, c in enumerate(freeway.cells) if c.on_ramp]
    off_ramp_idx = [i for i, c in enumerate(freeway.cells) if c.off_ramp]
    # The exact indices we saw when inspecting the .mat file.
    assert on_ramp_idx == [0, 1, 3, 4, 7, 8, 11, 12, 14, 16, 18, 19, 21, 22,
                           24, 25, 26, 28, 30, 31, 34, 36]
    assert off_ramp_idx == [1, 4, 5, 8, 9, 12, 14, 16, 19, 22, 26, 28, 31, 32,
                            34, 36, 37, 38]
    assert len(on_ramp_idx) == 22
    assert len(off_ramp_idx) == 18


def test_freeway_passes_cfl_on_every_cell(freeway):
    """Validation runs inside the adapter, but assert the rule explicitly too."""
    for i, c in enumerate(freeway.cells):
        assert c.v_f * freeway.dt <= c.length, (
            f"CFL violated for cell {i}: v_f*dt={c.v_f * freeway.dt:.4f} > "
            f"length={c.length:.4f}"
        )


def test_fd_consistency_holds_for_every_cell(freeway):
    """v_f and w are derived from the FD identity; residuals should be ~0."""
    for c in freeway.cells:
        free, cong = c.fd_residuals()
        assert free < 1e-12
        assert cong < 1e-12


# ---- Initial densities ----------------------------------------------------


def test_initial_densities_have_correct_shape_and_range(mat_path, freeway):
    rho0 = ctmsim_initial_densities(mat_path)
    assert rho0.shape == (freeway.n_cells,)
    # Initial state is light (max ~16 veh/mi seen during inspection); definitely
    # within each cell's jam density.
    rho_jam = np.array([c.rho_jam for c in freeway.cells])
    assert (rho0 >= 0).all()
    assert (rho0 <= rho_jam).all()


# ---- Demand profile expansion ---------------------------------------------


def test_demand_expansion_shape_and_zero_order_hold(mat_path, freeway):
    """demandProfile (288 samples x 42 cols) -> (n_cells, T) with T = 288 * 30."""
    demand = ctmsim_demand_at_sim_steps(mat_path)
    assert demand.shape[0] == freeway.n_cells
    # plotTS / TS = 300 s / 10 s = 30 steps per sample.
    assert demand.shape[1] == 288 * 30
    # 24 hours = 86400 s; T * TS should equal that.
    assert demand.shape[1] * freeway.dt == pytest.approx(24.0, abs=1e-9)


def test_demand_zero_on_cells_without_an_on_ramp(mat_path, freeway):
    demand = ctmsim_demand_at_sim_steps(mat_path)
    for i, c in enumerate(freeway.cells):
        if not c.on_ramp:
            assert demand[i, :].max() == 0.0, (
                f"cell {i} has on_ramp=False but non-zero demand"
            )


def test_demand_first_step_matches_first_sample(mat_path, freeway):
    """First 30 sim steps should hold the first plotTS sample constant.

    The expected value is the raw ``demandProfile`` sample **times** each
    cell's ``ORknob`` -- CTMSIM applies the knob during effective-demand
    construction, and our adapter mirrors that.
    """
    import scipy.io as sio
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    n = freeway.n_cells
    or_knobs = np.array([c.ORknob for c in mat["celldata"]], dtype=float)
    raw_first = mat["demandProfile"][0, 1 : n + 1] * or_knobs
    raw_second = mat["demandProfile"][1, 1 : n + 1] * or_knobs

    demand = ctmsim_demand_at_sim_steps(mat_path)
    # Steps 0..29 are the first 5-minute window.
    np.testing.assert_array_equal(demand[:, 0], raw_first)
    np.testing.assert_array_equal(demand[:, 29], raw_first)
    # Step 30 should switch to the second sample.
    np.testing.assert_array_equal(demand[:, 30], raw_second)


def test_demand_applies_ORknob_multiplier(mat_path, freeway):
    """Every cell where ORknob != 1 must have its demand scaled accordingly."""
    import scipy.io as sio
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    or_knobs = np.array([c.ORknob for c in mat["celldata"]], dtype=float)
    nontrivial = np.where((or_knobs != 1.0) & (or_knobs != 0.0))[0]
    assert len(nontrivial) > 0, (
        "w060412 should have at least one cell with ORknob outside {0, 1}; "
        "if not, this test fixture has drifted."
    )
    raw_per_period = mat["demandProfile"][:, 1 : freeway.n_cells + 1]
    expected = raw_per_period * or_knobs[None, :]            # (n_samples, n_cells)
    # Downsample our (n_cells, T) back to (n_samples, n_cells).
    P = 30
    demand = ctmsim_demand_at_sim_steps(mat_path)
    actual = demand[:, ::P].T
    np.testing.assert_array_equal(actual, expected)
