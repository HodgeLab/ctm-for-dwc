"""End-to-end tests for the CTM -> DWC adapter (P5).

``from_ctm`` builds a ``DemandResult`` from a CTM ``SimulationResult``:
mainline (queue-free) VHT, cell lengths snapped to the position grid. Uses a
tiny four-cell CTM run (Kurzhanskiy diss. §3.4) as the smoke test.
"""

import numpy as np

from ctm_for_dwc.ctm.engine import simulate
from ctm_for_dwc.ctm.examples import (
    four_cell_freeway,
    four_cell_scenario,
    metered_on_ramp_freeway,
    metered_on_ramp_scenario,
)
from ctm_for_dwc.ctm.metrics import compute_metrics
from ctm_for_dwc.dwc.adapter import (
    corridor_from_ctm,
    from_ctm,
    mainline_vht,
)
from ctm_for_dwc.dwc.demand import compute
from ctm_for_dwc.dwc.model import PadSpec

_METERS_PER_MILE = 1609.344


def _tiny_ctm_result(steps: int = 20, r_4: float = 1300.0):
    """A tiny four-cell CTM run; r_4=1300 is infeasible, so on-ramp queues build."""
    return simulate(four_cell_freeway(), four_cell_scenario(steps=steps, r_4=r_4))


def _coarse_pad() -> PadSpec:
    # Coarse pad so a dense T_R over four 1-mile cells stays small at dx=10 m.
    return PadSpec(
        alpha=20.0, delta=20.0, lambda_gap=10.0,
        beta=150_000.0, beta_prime=150_000.0,
    )


def test_mainline_vht_matches_ctm_metrics_minus_queue():
    """On a real CTM run, mainline VHT == metrics VHT minus the queue term."""
    result = _tiny_ctm_result()
    metrics = compute_metrics(result)
    queue_vht = result.queue[:, 1:] * result.freeway.dt
    np.testing.assert_allclose(mainline_vht(result), metrics.vht_per_cell - queue_vht)


def test_mainline_vht_excludes_queue_contribution():
    """With a nonzero on-ramp queue, mainline VHT drops the queue term.

    A controlled unit-level check; ``test_from_ctm_demand_excludes_queue_vht``
    below confirms the same exclusion end-to-end on a real queuing run.
    """
    import types

    rho = np.array([[2.0, 3.0], [1.0, 0.0]])  # (2 cells, T=2) post-step density
    queue = np.array([[5.0, 4.0], [0.0, 2.0]])  # nonzero on-ramp queue
    L = np.array([0.5, 1.0])
    dt = 0.1
    result = types.SimpleNamespace(
        density=np.column_stack([np.zeros(2), rho]),  # (2, T+1); [:,1:] is rho
        freeway=types.SimpleNamespace(
            cells=[types.SimpleNamespace(length=L[0]),
                   types.SimpleNamespace(length=L[1])],
            dt=dt,
        ),
    )

    ml = mainline_vht(result)
    np.testing.assert_allclose(ml, rho * L[:, None] * dt)
    total = (rho * L[:, None] + queue) * dt  # the CTM's full VHT (eq. 4.10)
    assert np.all(ml <= total)
    assert np.any(ml < total)  # strictly less wherever the queue is nonzero


def test_from_ctm_smoke_produces_valid_demand_result():
    result = _tiny_ctm_result()
    dx_grid = 10.0
    dr = from_ctm(result, _coarse_pad(), dx_grid=dx_grid, eta_EV=0.3)

    # 4 cells x 1 mile snapped at dx=10 -> 161 positions each; delta_grid = 2.
    n = 4 * round(_METERS_PER_MILE / dx_grid)
    m = n + 2 - 1
    T = result.mainline_flow.shape[1]
    assert dr.E.shape == (T, m)
    assert dr.position_m.shape == (m,)
    assert dr.timesteps.shape == (T,)
    assert np.all(dr.E >= 0.0)


def test_from_ctm_to_dataframe_round_trips():
    dr = from_ctm(_tiny_ctm_result(), _coarse_pad(), dx_grid=10.0, eta_EV=0.3)
    df = dr.to_dataframe()
    assert list(df.columns) == ["timestep", "position_m", "energy_wh"]
    assert len(df) == dr.E.size


# ---------------------------------------------------------------------------
# Queuing scenario: confirm demand excludes on-ramp queue VHT end-to-end
# ---------------------------------------------------------------------------


def _queuing_ctm_result(steps: int = 40):
    """A two-cell run whose metered on-ramp actually builds (then drains) a queue."""
    return simulate(
        metered_on_ramp_freeway(), metered_on_ramp_scenario(steps=steps)
    )


def test_metered_on_ramp_actually_builds_queue():
    """Sanity: unlike the four-cell example, this scenario accumulates queue."""
    result = _queuing_ctm_result()
    assert np.any(result.queue[:, 1:] > 0)


def test_mainline_vht_below_total_on_real_queuing_run():
    """On a run that truly queues, mainline VHT is strictly below the CTM's VHT."""
    result = _queuing_ctm_result()
    metrics = compute_metrics(result)
    queue_vht = result.queue[:, 1:] * result.freeway.dt
    ml = mainline_vht(result)
    np.testing.assert_allclose(ml, metrics.vht_per_cell - queue_vht)
    assert np.any(ml < metrics.vht_per_cell)  # strictly less wherever queue > 0


def test_from_ctm_demand_excludes_queue_vht():
    """End-to-end: DWC demand omits exactly the on-ramp queue's contribution.

    By linearity of ``compute`` in VHT, the queue-inclusive demand splits as
    mainline demand + queue-only demand; ``from_ctm`` keeps only the first.
    """
    result = _queuing_ctm_result()
    metrics = compute_metrics(result)
    pad, dx, eta = _coarse_pad(), 10.0, 0.3
    corridor = corridor_from_ctm(result, pad, dx)

    E_mainline = from_ctm(result, pad, dx_grid=dx, eta_EV=eta).E
    queue_vht = result.queue[:, 1:] * result.freeway.dt
    E_with_queue = compute(metrics.vht_per_cell, corridor, eta).E
    E_queue_only = compute(queue_vht, corridor, eta).E

    np.testing.assert_allclose(E_with_queue, E_mainline + E_queue_only)
    assert E_queue_only.sum() > 0.0  # the queue carries real energy...
    assert E_mainline.sum() < E_with_queue.sum()  # ...which the demand excludes
