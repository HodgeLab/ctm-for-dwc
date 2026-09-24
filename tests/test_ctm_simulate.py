"""Whole-trajectory invariant tests for the CTM ``simulate()`` (P2, layer 2).

The deep invariant here is mass conservation, which must hold by construction
because eq. 4.7 *is* a discrete conservation law. Drift signals an engine bug.
Structural bounds (density, flow, queue, speed) are checked on a stress
scenario chosen to exercise them.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ctm_for_dwc.ctm import examples, simulate
from ctm_for_dwc.ctm.model import Scenario
from ctm_for_dwc.ctm.results import SimulationResult


# ---- fixtures (module-scoped: same trajectory shared across tests) ---------


@pytest.fixture(scope="module")
def empty_run():
    fwy = examples.example_1_freeway()
    scn = examples.example_1_scenario(steps=600, start="empty")
    return simulate(fwy, scn)


@pytest.fixture(scope="module")
def jam_run():
    fwy = examples.example_1_freeway()
    scn = examples.example_1_scenario(steps=600, start="jam")
    return simulate(fwy, scn)


# ---- smoke -----------------------------------------------------------------


def test_simulate_returns_correct_shapes(empty_run):
    res = empty_run
    assert res.density.shape == (2, 601)
    assert res.queue.shape == (2, 601)
    assert res.mainline_flow.shape == (2, 600)
    assert res.off_ramp.shape == (2, 600)
    assert res.on_ramp.shape == (2, 600)
    assert res.speed.shape == (2, 600)
    assert res.boundary_inflow.shape == (600,)
    assert res.n_cells == 2
    assert res.n_steps == 600


def test_simulate_initial_state_matches_scenario(empty_run):
    res = empty_run
    np.testing.assert_array_equal(res.density[:, 0], res.scenario.rho0)
    np.testing.assert_array_equal(res.queue[:, 0], res.scenario.q0)


def test_simulate_reaches_uncongested_equilibrium(empty_run):
    # From-empty in Example 1 converges to rho^u = (80, 100) (diss. p. 46).
    np.testing.assert_allclose(empty_run.density[:, -1], [80.0, 100.0], atol=1e-9)


def test_simulate_reaches_most_congested_equilibrium(jam_run):
    # From-jam in Example 1 converges to rho^con = (160, 160) (diss. p. 46).
    np.testing.assert_allclose(jam_run.density[:, -1], [160.0, 160.0], atol=1e-9)


# ---- structural bounds -----------------------------------------------------


def test_density_within_jam_bounds(jam_run):
    rho_jam = np.array([c.rho_jam for c in jam_run.freeway.cells])
    assert (jam_run.density >= 0.0).all()
    assert (jam_run.density <= rho_jam[:, None] + 1e-9).all()


def test_flows_within_capacity(jam_run):
    q_max = np.array([c.q_max for c in jam_run.freeway.cells])
    assert (jam_run.mainline_flow >= 0.0).all()
    assert (jam_run.mainline_flow <= q_max[:, None] + 1e-9).all()
    assert (jam_run.on_ramp >= 0.0).all()
    assert (jam_run.off_ramp >= 0.0).all()
    assert (jam_run.boundary_inflow >= 0.0).all()


def test_queue_non_negative(jam_run):
    assert (jam_run.queue >= 0.0).all()


def test_speed_within_freeflow(jam_run):
    v_f = np.array([c.v_f for c in jam_run.freeway.cells])
    assert (jam_run.speed >= 0.0).all()
    assert (jam_run.speed <= v_f[:, None] + 1e-9).all()


# ---- mass conservation (eq. 4.7 is a discrete conservation law) ------------


def _balance_terms(res):
    """Return (delta_stored, expected_delta) per-step for the conservation check."""
    L = np.array([c.length for c in res.freeway.cells])
    veh_freeway = (res.density * L[:, None]).sum(axis=0)        # (T+1,)
    queue_total = res.queue.sum(axis=0)                          # (T+1,)
    stored = veh_freeway + queue_total
    delta = np.diff(stored)                                       # (T,)
    # Inputs to the freeway + on-ramp queue system, veh/h:
    inflow = res.boundary_inflow + res.scenario.demand.sum(axis=0)
    # Outputs (sink at the downstream end + off-ramps):
    outflow = res.mainline_flow[-1] + res.off_ramp.sum(axis=0)
    expected = (inflow - outflow) * res.freeway.dt
    return delta, expected


def test_mass_conservation_per_step(jam_run):
    delta, expected = _balance_terms(jam_run)
    np.testing.assert_allclose(delta, expected, atol=1e-9, rtol=1e-10)


def test_mass_conservation_aggregate(jam_run):
    delta, expected = _balance_terms(jam_run)
    np.testing.assert_allclose(delta.sum(), expected.sum(), atol=1e-9)


def test_mass_conservation_holds_from_empty_too(empty_run):
    delta, expected = _balance_terms(empty_run)
    np.testing.assert_allclose(delta, expected, atol=1e-9, rtol=1e-10)


# ---- equilibrium is a fixed point (stretch) -------------------------------


def test_uncongested_equilibrium_is_a_fixed_point():
    fwy = examples.example_1_freeway()
    demand = np.zeros((2, 60))
    demand[1, :] = 1200.0
    scn = Scenario.build(
        fwy,
        steps=60,
        rho0=np.array([80.0, 100.0]),
        inflow=4800.0,
        demand=demand,
        beta=0.0,
    )
    res = simulate(fwy, scn)
    np.testing.assert_allclose(
        res.density,
        np.broadcast_to([[80.0], [100.0]], res.density.shape),
        atol=1e-9,
    )


# ---- to_dataframes output --------------------------------------------------


def test_to_dataframes_shapes_and_indices(empty_run):
    frames = empty_run.to_dataframes()
    n_cells = empty_run.n_cells
    n_steps = empty_run.n_steps
    # State frames: T+1 rows, cell columns.
    for name in ("density", "queue"):
        df = frames[name]
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (n_steps + 1, n_cells)
        assert df.index.name == "time_h"
        assert list(df.columns) == list(range(n_cells))
        assert df.columns.name == "cell"
    # Flow frames: T rows.
    for name in ("mainline_flow", "off_ramp", "on_ramp", "speed"):
        df = frames[name]
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (n_steps, n_cells)
        assert df.index.name == "time_h"
    # boundary_inflow is a 1-D time series.
    b = frames["boundary_inflow"]
    assert isinstance(b, pd.Series)
    assert b.shape == (n_steps,)
    assert b.name == "boundary_inflow"
    assert b.index.name == "time_h"


def test_to_dataframes_values_match_arrays(empty_run):
    res = empty_run
    frames = res.to_dataframes()
    np.testing.assert_array_equal(frames["density"].to_numpy().T, res.density)
    np.testing.assert_array_equal(frames["queue"].to_numpy().T, res.queue)
    np.testing.assert_array_equal(frames["mainline_flow"].to_numpy().T, res.mainline_flow)
    np.testing.assert_array_equal(frames["off_ramp"].to_numpy().T, res.off_ramp)
    np.testing.assert_array_equal(frames["on_ramp"].to_numpy().T, res.on_ramp)
    np.testing.assert_array_equal(frames["speed"].to_numpy().T, res.speed)
    np.testing.assert_array_equal(frames["boundary_inflow"].to_numpy(), res.boundary_inflow)


def test_to_dataframes_time_index_uses_hours(empty_run):
    res = empty_run
    frames = res.to_dataframes()
    # State index covers [0, T*dt] in T+1 steps; flow index covers [0, (T-1)*dt].
    np.testing.assert_allclose(frames["density"].index.to_numpy(), res.time_state)
    np.testing.assert_allclose(frames["mainline_flow"].index.to_numpy(), res.time_flow)


# ---- to_npz / from_npz round-trip ------------------------------------------


@pytest.fixture(scope="module")
def ramp_run():
    # Exercises on-ramp config (finite on_ramp_capacity, gamma/xi) alongside
    # plain cells with the default inf capacity, so the freeway round-trip is
    # tested on both ramp and no-ramp cells.
    fwy = examples.metered_on_ramp_freeway()
    scn = examples.metered_on_ramp_scenario(steps=120)
    return simulate(fwy, scn)


def test_to_npz_round_trips_arrays(ramp_run, tmp_path):
    res = ramp_run
    res.to_npz(tmp_path / "run.npz")
    back = SimulationResult.from_npz(tmp_path / "run.npz")
    for name in (
        "density", "queue", "mainline_flow", "off_ramp",
        "on_ramp", "speed", "boundary_inflow",
    ):
        np.testing.assert_array_equal(getattr(back, name), getattr(res, name))


def test_to_npz_round_trips_freeway(ramp_run, tmp_path):
    res = ramp_run
    res.to_npz(tmp_path / "run.npz")
    back = SimulationResult.from_npz(tmp_path / "run.npz")
    assert back.freeway.dt == res.freeway.dt
    assert back.n_cells == res.n_cells
    # At least one on-ramp (finite cap) and one plain (inf cap) cell, so the
    # inf-preservation and bool round-trip below are non-trivial.
    assert any(c.on_ramp for c in res.freeway.cells)
    assert any(not c.on_ramp for c in res.freeway.cells)
    for orig, got in zip(res.freeway.cells, back.freeway.cells):
        for field in (
            "length", "q_max", "v_f", "w", "rho_jam", "rho_crit",
            "on_ramp", "off_ramp", "on_ramp_capacity", "off_ramp_capacity",
            "gamma", "xi",
        ):
            assert getattr(got, field) == getattr(orig, field), field


def test_to_npz_round_trips_scenario(ramp_run, tmp_path):
    res = ramp_run
    res.to_npz(tmp_path / "run.npz")
    back = SimulationResult.from_npz(tmp_path / "run.npz")
    for name in ("rho0", "inflow", "demand", "beta", "q0"):
        np.testing.assert_array_equal(
            getattr(back.scenario, name), getattr(res.scenario, name)
        )


def test_simulate_leaves_start_unset(ramp_run):
    # The engine works in step/hour terms from t=0 and has no notion of a
    # wall-clock anchor; scripts stamp --start on afterward (the replace() in
    # simulate_ctm_corridor.py). So a fresh run's start must be None.
    assert ramp_run.start is None


def test_to_npz_round_trips_start(ramp_run, tmp_path):
    ts = pd.Timestamp("2023-06-01 06:00")
    res = replace(ramp_run, start=ts)
    res.to_npz(tmp_path / "run.npz")
    back = SimulationResult.from_npz(tmp_path / "run.npz")
    assert back.start == ts


def test_npz_omits_start_when_unset(ramp_run, tmp_path):
    # Backward compat: a result with no start must not write the meta key, and
    # must reload as start=None (so pre-existing .npz files keep loading).
    ramp_run.to_npz(tmp_path / "run.npz")
    assert "meta__start" not in np.load(tmp_path / "run.npz")
    assert SimulationResult.from_npz(tmp_path / "run.npz").start is None


def test_from_npz_revalidates(ramp_run, tmp_path):
    # A corrupted bundle (rho0 driven above jam density) must fail loudly on
    # load rather than yield an invalid SimulationResult.
    res = ramp_run
    res.to_npz(tmp_path / "run.npz")
    data = dict(np.load(tmp_path / "run.npz"))
    data["scenario__rho0"] = data["scenario__rho0"] + 1e6
    np.savez(tmp_path / "bad.npz", **data)
    with pytest.raises(ValueError):
        SimulationResult.from_npz(tmp_path / "bad.npz")
