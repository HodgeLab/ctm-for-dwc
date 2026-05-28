"""Performance-metric tests (P4): VHT, VMT, delay, productivity loss.

Hand-computes the eqs. 4.9-4.17 quantities at the dissertation's Example 1
equilibria (uncongested and most congested) and asserts the engine reproduces
them. Both equilibria support the same throughput (cell 1 at capacity), so
VMT matches across them; VHT, delay, productivity loss, and travel time
diverge.
"""

import numpy as np
import pytest

from transportation_models.utils.ctm import examples, simulate
from transportation_models.utils.ctm.metrics import compute_metrics

DT = 1.0 / 120.0  # Example 1 time step [h]


@pytest.fixture(scope="module")
def empty_metrics():
    fwy = examples.example_1_freeway()
    scn = examples.example_1_scenario(steps=120, start="empty")  # 1 h: well past convergence
    return compute_metrics(simulate(fwy, scn))


@pytest.fixture(scope="module")
def jam_metrics():
    fwy = examples.example_1_freeway()
    scn = examples.example_1_scenario(steps=600, start="jam")    # 5 h: well past convergence
    return compute_metrics(simulate(fwy, scn))


# ---- shape / total consistency --------------------------------------------


def test_metrics_arrays_have_expected_shapes(empty_metrics):
    n_cells, T = 2, 120
    m = empty_metrics
    assert m.travel_time.shape == (T,)
    assert m.vht_per_cell.shape == (n_cells, T)
    assert m.vmt_per_cell.shape == (n_cells, T)
    assert m.delay_per_cell.shape == (n_cells, T)
    assert m.productivity_loss_per_cell.shape == (n_cells, T)


def test_total_metrics_equal_sum_over_cells(empty_metrics):
    m = empty_metrics
    np.testing.assert_array_equal(m.vht, m.vht_per_cell.sum(axis=0))
    np.testing.assert_array_equal(m.vmt, m.vmt_per_cell.sum(axis=0))
    np.testing.assert_array_equal(m.delay, m.delay_per_cell.sum(axis=0))
    np.testing.assert_array_equal(m.productivity_loss, m.productivity_loss_per_cell.sum(axis=0))


# ---- uncongested equilibrium (rho^u = (80, 100), V = (60, 60)) ------------


def test_vht_at_uncongested_equilibrium(empty_metrics):
    # VHT_i = (rho_i * l_i + q_i) * dt; queues are 0 at the Example-1 equilibrium.
    np.testing.assert_allclose(
        empty_metrics.vht_per_cell[:, -1], [80.0 * DT, 100.0 * DT], atol=1e-9
    )


def test_vmt_at_uncongested_equilibrium(empty_metrics):
    # VMT_i = rho_i * V_i * dt * l_i; V_i = v_f = 60 mph at the uncongested eq.
    np.testing.assert_allclose(
        empty_metrics.vmt_per_cell[:, -1],
        [80.0 * 60.0 * DT, 100.0 * 60.0 * DT],
        atol=1e-9,
    )


def test_delay_zero_at_uncongested_equilibrium(empty_metrics):
    np.testing.assert_allclose(empty_metrics.delay_per_cell[:, -1], 0.0, atol=1e-12)


def test_productivity_loss_zero_at_uncongested_equilibrium(empty_metrics):
    np.testing.assert_allclose(empty_metrics.productivity_loss_per_cell[:, -1], 0.0, atol=1e-12)


def test_travel_time_at_uncongested_equilibrium(empty_metrics):
    # Two 1-mi cells at v_f = 60 mph -> TT = 2/60 h = 2 min.
    np.testing.assert_allclose(empty_metrics.travel_time[-1], 2.0 / 60.0, atol=1e-9)


# ---- most-congested equilibrium (rho^con = (160, 160)) --------------------
#
# At rho^con, both cells are congested. Flows match the bottleneck:
#   f_0 = r_0 = 4800 veh/h, f_1 = q_max = 6000 veh/h.
# Speeds collapse to flow/density:
#   V_0 = 4800/160 = 30 mph, V_1 = 6000/160 = 37.5 mph.


def test_vht_at_congested_equilibrium(jam_metrics):
    # VHT_i = rho_i * l_i * dt with queues drained (= 0 at this equilibrium).
    np.testing.assert_allclose(
        jam_metrics.vht_per_cell[:, -1], [160.0 * DT, 160.0 * DT], atol=1e-9
    )


def test_vmt_at_congested_equilibrium(jam_metrics):
    # VMT_i = rho_i * V_i * dt * l_i: throughput equals the uncongested case
    # (cell 0: 160*30 = 4800 = uncong. 80*60; cell 1: 160*37.5 = 6000).
    np.testing.assert_allclose(
        jam_metrics.vmt_per_cell[:, -1],
        [160.0 * 30.0 * DT, 160.0 * 37.5 * DT],
        atol=1e-9,
    )


def test_delay_positive_at_congested_equilibrium(jam_metrics):
    # D_i = VHT_i - VMT_i/v_f when congested.
    # cell 0: 160*DT - (160*30*DT)/60 = (160 - 80)*DT = 80*DT
    # cell 1: 160*DT - (160*37.5*DT)/60 = (160 - 100)*DT = 60*DT
    np.testing.assert_allclose(
        jam_metrics.delay_per_cell[:, -1], [80.0 * DT, 60.0 * DT], atol=1e-9
    )


def test_productivity_loss_at_congested_equilibrium(jam_metrics):
    # PL_i = (1 - f_i/q_max) * dt * l_i when congested.
    # cell 0: (1 - 4800/6000) * DT * 1 = 0.2 * DT (operating below capacity)
    # cell 1: (1 - 6000/6000) * DT * 1 = 0    (bottleneck IS at capacity)
    np.testing.assert_allclose(
        jam_metrics.productivity_loss_per_cell[:, -1], [0.2 * DT, 0.0], atol=1e-9
    )


def test_travel_time_at_congested_equilibrium(jam_metrics):
    # TT = l_0/V_0 + l_1/V_1 = 1/30 + 1/37.5 h.
    expected = 1.0 / 30.0 + 1.0 / 37.5
    np.testing.assert_allclose(jam_metrics.travel_time[-1], expected, atol=1e-9)


# ---- gating: delay/PL only accrue in congestion ---------------------------


def test_delay_and_pl_gate_on_congestion(empty_metrics, jam_metrics):
    """Empty-start (uncongested everywhere) -> all delay and PL are 0;
    jam-start (both cells congested at equilibrium) -> all delay and PL > 0."""
    # Empty start: never congested in steady state -> zero everywhere along
    # the trajectory's tail (well after the first few transient steps).
    np.testing.assert_allclose(empty_metrics.delay_per_cell[:, 60:], 0.0, atol=1e-12)
    np.testing.assert_allclose(empty_metrics.productivity_loss_per_cell[:, 60:], 0.0, atol=1e-12)
    # Jam start: at equilibrium both cells are congested -> delay > 0 in both.
    assert (jam_metrics.delay_per_cell[:, -1] > 0).all()
    # Productivity loss: cell 0 below capacity (>0), cell 1 at capacity (=0).
    assert jam_metrics.productivity_loss_per_cell[0, -1] > 0
    assert jam_metrics.productivity_loss_per_cell[1, -1] == pytest.approx(0.0, abs=1e-12)
