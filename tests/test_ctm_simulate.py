"""Whole-trajectory invariant tests for the CTM ``simulate()`` (P2, layer 2).

The deep invariant here is mass conservation, which must hold by construction
because eq. 4.7 *is* a discrete conservation law. Drift signals an engine bug.
Structural bounds (density, flow, queue, speed) are checked on a stress
scenario chosen to exercise them.
"""

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ctm import examples, simulate
from transportation_models.utils.ctm.model import Scenario


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
