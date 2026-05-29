"""Compare our CTM engine against CTMSIM v1.1 reference output on I-210W.

For each day in ``ctmsim_configs/`` we have a matching directory in
``ctmsim_results/`` with CSV exports of CTMSIM's trajectory (density, flows)
and per-period aggregate metrics. We feed CTMSIM's *effective* on-ramp demand
and off-ramp split-ratio (i.e. what its solver actually used after any internal
metering / capacity logic) back into our engine, run a 24-hour simulation, and
assert the trajectories and aggregate metrics line up.

The match is essentially exact on density and on-ramp flow; off-ramp and
mainline flow are tight but pick up small numerical drift around the downstream
boundary (cell ``N-1``: ``frflowProfile`` ramp-flow override + ``outflow`` cap),
so we set per-sample tolerances generously and check the rms across the run.

CTMSIM CSV layout (per User Guide §8, transposed to wide format here):
  ``K + 1`` rows = plotting samples (0 = initial, t = sim step t * 30);
  ``N + 1`` cols where the *last* col is the to-be-ignored placeholder.
  ``mainline_flow`` is special: col 0 = upstream boundary inflow, col ``k`` for
  ``k = 1..N`` = flow entering cell k (= our ``f_{k-1}``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from transportation_models.utils.ctm import (
    Scenario,
    compute_metrics,
    ctmsim_initial_densities,
    freeway_from_ctmsim_mat,
    simulate,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "src/transportation_models/utils/ctm/ctmsim_configs"
RESULTS = REPO / "src/transportation_models/utils/ctm/ctmsim_results"

# CTMSIM I-210W defaults (User Guide §4): plotTS = 5 min, TS = 10 s -> 30 sim
# steps per plotting sample over a 24-hour horizon.
SIM_STEPS_PER_PLOT_SAMPLE = 30
PLOT_SAMPLES_PER_DAY = 288


# ---- Helpers --------------------------------------------------------------


def _ctm_csv(day: str, name: str, *, drop_last_col: bool = True) -> np.ndarray:
    arr = np.loadtxt(RESULTS / day / f"{name}.csv", delimiter=",")
    return arr[:, :-1] if drop_last_col else arr


def _expand(plot_arr: np.ndarray, n_steps: int) -> np.ndarray:
    """Zero-order-hold expansion from (K, N) plot samples to (N, T) sim steps."""
    expanded = np.repeat(plot_arr, SIM_STEPS_PER_PLOT_SAMPLE, axis=0)[:n_steps, :]
    return expanded.T


def _build_scenario_from_ctmsim(day: str):
    """Build a Scenario that replays CTMSIM's effective inputs for ``day``.

    Uses ``effective_demand`` and ``effective_beta`` from CTMSIM's CSV output
    (the values its solver actually applied each plotting period) rather than
    the raw ``demandProfile`` / ``betaProfile`` — this isolates the *engine
    update* from CTMSIM's internal demand/ramp shaping for an apples-to-apples
    comparison.
    """
    mat_path = CONFIGS / f"{day}.mat"
    fwy = freeway_from_ctmsim_mat(mat_path)
    n = fwy.n_cells
    n_steps = PLOT_SAMPLES_PER_DAY * SIM_STEPS_PER_PLOT_SAMPLE

    rho0 = ctmsim_initial_densities(mat_path)

    # Upstream mainline inflow lives in demandProfile column 0; expand to sim
    # steps with zero-order hold to match plotTS = 5 min.
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    inflow_plot = mat["demandProfile"][:, 0]
    inflow = np.repeat(inflow_plot, SIM_STEPS_PER_PLOT_SAMPLE)[:n_steps]

    # Drop the trailing initial-state row of the effective_* CSVs (those have
    # K + 1 rows including the t = 0 snapshot) and the to-be-ignored last col.
    demand_plot = _ctm_csv(day, "effective_demand")[:-1, :]
    beta_plot = _ctm_csv(day, "effective_beta")[:-1, :]

    scn = Scenario(
        rho0=rho0,
        inflow=inflow,
        demand=_expand(demand_plot, n_steps),
        beta=_expand(beta_plot, n_steps),
        q0=np.zeros(n),
    )
    return fwy, scn.validate(fwy)


def _sample_state(arr: np.ndarray) -> np.ndarray:
    """Sample (N, T+1) state array at plot-sample boundaries -> (N, K+1)."""
    cols = np.arange(PLOT_SAMPLES_PER_DAY + 1) * SIM_STEPS_PER_PLOT_SAMPLE
    return arr[:, cols]


def _sample_flow(arr: np.ndarray) -> np.ndarray:
    """Sample (N, T) flow array at plot-sample boundaries.

    Matches CTMSIM's convention that row t (t = 0..K-1) holds the flow at sim
    step ``t * 30``; the t = K row is the trailing junk sample and is excluded
    by callers.
    """
    cols = np.arange(PLOT_SAMPLES_PER_DAY) * SIM_STEPS_PER_PLOT_SAMPLE
    return arr[:, cols]


def _aggregate_to_periods(per_step: np.ndarray) -> np.ndarray:
    """Sum per-sim-step series into per-plotting-period totals (T -> K)."""
    return per_step.reshape(PLOT_SAMPLES_PER_DAY, SIM_STEPS_PER_PLOT_SAMPLE).sum(axis=1)


# ---- Fixtures -------------------------------------------------------------

DAY = "w060412"  # 24-hour I-210 West run, 04/12/2006


@pytest.fixture(scope="module")
def result():
    if not (CONFIGS / f"{DAY}.mat").exists():
        pytest.skip(f"CTMSIM config missing: {DAY}.mat")
    fwy, scn = _build_scenario_from_ctmsim(DAY)
    return simulate(fwy, scn)


# ---- Trajectory comparison ------------------------------------------------


def test_density_matches_ctmsim_at_plot_samples(result):
    """Density at every 5-min boundary should match CTMSIM very tightly."""
    ours = _sample_state(result.density)              # (N, K+1)
    ctm = _ctm_csv(DAY, "density").T                  # (N, K+1)
    diff = ours - ctm
    # In free flow our engine reproduces CTMSIM's state to floating-point
    # precision; during morning peak congestion (~07:40-10:15) the small
    # mainline-flow drift around the downstream boundary feeds back into a sub-
    # veh/mi density gap. rms < 0.05 keeps the typical sample numerical-noise
    # tight while max < 1 covers the peak.
    assert np.sqrt(np.mean(diff ** 2)) < 0.05
    assert np.max(np.abs(diff)) < 1.0


def test_on_ramp_flow_matches_ctmsim(result):
    """Driven by effective_demand, on-ramp flow should match exactly."""
    ours = _sample_flow(result.on_ramp)               # (N, K)
    ctm = _ctm_csv(DAY, "on_ramp_flow")[:-1, :].T     # (N, K)
    assert np.array_equal(ours, ctm)


def test_off_ramp_flow_matches_ctmsim(result):
    """Off-ramp flow tracks closely; cell N-1 picks up small numerical drift."""
    ours = _sample_flow(result.off_ramp)
    ctm = _ctm_csv(DAY, "off_ramp_flow")[:-1, :].T
    diff = ours - ctm
    # Overall trajectory matches tightly. A handful of (cell, t) pairs near the
    # downstream boundary drift by < 2 vph (CTMSIM's frflowProfile override on
    # off-ramp flow + capacity interactions our engine handles via beta only).
    assert np.sqrt(np.mean(diff ** 2)) < 0.1
    assert np.max(np.abs(diff)) < 2.0


def test_mainline_flow_matches_ctmsim(result):
    """Mainline flow matches; tolerance allows small downstream-boundary drift."""
    ours = _sample_flow(result.mainline_flow)                  # (N, K), our f_i
    # CTMSIM mainline_flow col 0 = upstream boundary inflow, col k (k>=1) =
    # flow entering cell k = our f_{k-1}. Trim the trailing junk row.
    ctm_full = _ctm_csv(DAY, "mainline_flow", drop_last_col=False)[:-1, :]
    ctm = ctm_full[:, 1:].T                                    # (N, K)
    diff = ours - ctm
    # Most (cell, t) pairs agree to <0.1 vph; downstream-boundary cells pick up
    # occasional 100-300 vph drift during peak congestion (see module docstring).
    assert np.median(np.abs(diff)) < 1e-6
    assert np.sqrt(np.mean(diff ** 2)) < 10.0
    # 99th-percentile mismatch stays well below the per-cell capacity scale.
    assert np.quantile(np.abs(diff), 0.99) < 50.0


def test_ctm_boundary_inflow_matches(result):
    """Upstream boundary inflow (CTMSIM mainline_flow col 0) should match ours."""
    ctm_boundary = _ctm_csv(DAY, "mainline_flow", drop_last_col=False)[:-1, 0]
    ours = _sample_flow(result.boundary_inflow[None, :])[0]
    assert np.allclose(ours, ctm_boundary, atol=1e-9)


# ---- Aggregate metrics ----------------------------------------------------


def test_aggregate_metrics_match_ctmsim(result):
    """VHT, VMT, delay, productivity_loss per plotting period align with CTMSIM.

    CTMSIM's aggregate_metrics.csv row t (t = 1..K) holds totals over plotting
    period t (= our sim steps (t-1)*30 .. t*30-1). Row 0 is the initial-state
    snapshot and is skipped.
    """
    m = compute_metrics(result)
    ctm = np.loadtxt(RESULTS / DAY / "aggregate_metrics.csv", delimiter=",")
    ref = ctm[1 : PLOT_SAMPLES_PER_DAY + 1, :]                    # (K, 4)

    for label, ours, col, atol_max, rtol_total in [
        ("vht",   _aggregate_to_periods(m.vht),                0, 0.1,  1e-3),
        ("vmt",   _aggregate_to_periods(m.vmt),                1, 50.0, 1e-3),
        ("delay", _aggregate_to_periods(m.delay),              2, 1.0,  5e-3),
        ("ploss", _aggregate_to_periods(m.productivity_loss),  3, 0.05, 5e-2),
    ]:
        diff = ours - ref[:, col]
        assert np.max(np.abs(diff)) < atol_max, (
            f"{label}: max|Δ|={np.max(np.abs(diff)):.4g} exceeds {atol_max}"
        )
        # Day-total agreement is the more useful sanity check.
        total_ours, total_ref = ours.sum(), ref[:, col].sum()
        if total_ref != 0:
            rel = abs(total_ours - total_ref) / abs(total_ref)
            assert rel < rtol_total, (
                f"{label}: 24h total {total_ours:.4g} vs CTMSIM {total_ref:.4g} "
                f"(rel.err {rel:.4g} > {rtol_total})"
            )
