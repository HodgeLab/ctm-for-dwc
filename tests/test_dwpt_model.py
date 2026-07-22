"""Validation and construction tests for the DWPT module's data model (P0).

Each test pins one property of ``PadSpec`` or ``CorridorSpec`` from
``docs/dwpt_demand_module.md`` (spec) and
``docs/dwpt_demand_implementation_plan.md`` (plan, P0 row).
"""

import numpy as np
import pytest

from transportation_models.utils.dwpt.model import (
    CorridorSpec,
    DemandResult,
    PadSpec,
)


# ---------------------------------------------------------------------------
# PadSpec
# ---------------------------------------------------------------------------


def _newbolt_pad() -> PadSpec:
    return PadSpec(
        alpha=2.0,
        delta=1.5,
        lambda_gap=0.1524,
        beta=150_000.0,
        beta_prime=150_000.0,
    )


def test_pad_spec_holds_field_values():
    pad = _newbolt_pad()
    assert pad.alpha == 2.0
    assert pad.delta == 1.5
    assert pad.lambda_gap == pytest.approx(0.1524)
    assert pad.beta == 150_000.0
    assert pad.beta_prime == 150_000.0


def test_pad_spec_gamma_is_min_of_capacities():
    """gamma = min(beta, beta_prime), spec §"Spatial Dependence"."""
    pad_rx_limited = PadSpec(
        alpha=2.0, delta=1.5, lambda_gap=0.1,
        beta=100_000.0, beta_prime=200_000.0,
    )
    pad_tx_limited = PadSpec(
        alpha=2.0, delta=1.5, lambda_gap=0.1,
        beta=200_000.0, beta_prime=100_000.0,
    )
    pad_equal = _newbolt_pad()
    assert pad_rx_limited.gamma == 100_000.0
    assert pad_tx_limited.gamma == 100_000.0
    assert pad_equal.gamma == 150_000.0


@pytest.mark.parametrize(
    "field,bad",
    [
        ("alpha", 0.0), ("alpha", -1.0),
        ("delta", 0.0), ("delta", -1.0),
        ("lambda_gap", 0.0), ("lambda_gap", -1.0),
        ("beta", 0.0), ("beta", -1.0),
        ("beta_prime", 0.0), ("beta_prime", -1.0),
    ],
)
def test_pad_spec_rejects_non_positive(field, bad):
    kwargs = dict(
        alpha=2.0, delta=1.5, lambda_gap=0.1,
        beta=150_000.0, beta_prime=150_000.0,
    )
    kwargs[field] = bad
    with pytest.raises(ValueError, match=field):
        PadSpec(**kwargs)


# ---------------------------------------------------------------------------
# CorridorSpec
# ---------------------------------------------------------------------------


def test_corridor_spec_grid_properties_match_newbolt_dims():
    spec = CorridorSpec(
        cell_lengths_m=(6.4572,), pad=_newbolt_pad(), dx_grid=0.0001,
    )
    assert spec.corridor_length_m == pytest.approx(6.4572)
    assert spec.n_corridor == 64_572
    assert spec.alpha_grid == 20_000
    assert spec.delta_grid == 15_000
    assert spec.lambda_grid == 1_524
    assert spec.positions_per_cell == (64_572,)


def test_corridor_spec_m_traversal_extends_by_delta_minus_one():
    """m = n + delta_grid - 1: Rx kernel can enter and exit the corridor."""
    spec = CorridorSpec(
        cell_lengths_m=(6.4572,), pad=_newbolt_pad(), dx_grid=0.0001,
    )
    assert spec.m_traversal == 64_572 + 15_000 - 1


def test_corridor_spec_multi_cell_positions_per_cell_and_n_corridor():
    pad = PadSpec(
        alpha=1.0, delta=1.0, lambda_gap=1.0, beta=1.0, beta_prime=1.0,
    )
    spec = CorridorSpec(
        cell_lengths_m=(3.0, 5.0), pad=pad, dx_grid=1.0,
    )
    assert spec.positions_per_cell == (3, 5)
    assert spec.n_corridor == 8


def test_corridor_spec_rejects_misaligned_pad_dim():
    """Pad dims must be integer multiples of dx_grid (split case deferred)."""
    with pytest.raises(ValueError, match="lambda_gap"):
        CorridorSpec(
            cell_lengths_m=(6.0,), pad=_newbolt_pad(), dx_grid=0.01,
        )


def test_corridor_spec_rejects_misaligned_cell_length():
    pad = PadSpec(
        alpha=2.0, delta=1.0, lambda_gap=0.1, beta=1.0, beta_prime=1.0,
    )
    with pytest.raises(ValueError, match="cell_lengths_m"):
        CorridorSpec(cell_lengths_m=(6.05,), pad=pad, dx_grid=0.1)


def test_corridor_spec_rejects_nonpositive_dx_grid():
    with pytest.raises(ValueError, match="dx_grid"):
        CorridorSpec(
            cell_lengths_m=(1.0,), pad=_newbolt_pad(), dx_grid=0.0,
        )


def test_corridor_spec_rejects_empty_cells():
    with pytest.raises(ValueError, match="cell_lengths_m"):
        CorridorSpec(
            cell_lengths_m=(), pad=_newbolt_pad(), dx_grid=0.0001,
        )


def test_corridor_spec_rejects_nonpositive_cell_length():
    with pytest.raises(ValueError, match="cell_lengths_m"):
        CorridorSpec(
            cell_lengths_m=(1.0, 0.0), pad=_newbolt_pad(), dx_grid=0.0001,
        )


# ---------------------------------------------------------------------------
# Newbolt example builder
# ---------------------------------------------------------------------------


def test_newbolt_small_scale_pad_matches_paper_table_1():
    from transportation_models.utils.dwpt.examples import newbolt_small_scale

    spec = newbolt_small_scale()
    assert spec.pad.alpha == 2.0
    assert spec.pad.delta == 1.5
    assert spec.pad.lambda_gap == pytest.approx(0.1524)
    assert spec.pad.beta == 150_000.0
    assert spec.pad.beta_prime == 150_000.0


def test_newbolt_small_scale_corridor_holds_three_tiles():
    from transportation_models.utils.dwpt.examples import newbolt_small_scale

    spec = newbolt_small_scale()
    assert len(spec.cell_lengths_m) == 1
    assert spec.corridor_length_m == pytest.approx(3 * (2.0 + 0.1524))


# ---------------------------------------------------------------------------
# DemandResult
# ---------------------------------------------------------------------------


def _demand_result() -> DemandResult:
    return DemandResult(
        E=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # (T=2, m=3)
        position_m=np.array([0.0, 0.5, 1.0]),
        timesteps=np.array([0, 1]),
    )


def test_demand_result_holds_fields():
    r = _demand_result()
    assert r.E.shape == (2, 3)
    np.testing.assert_array_equal(r.position_m, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(r.timesteps, [0, 1])


def test_demand_result_rejects_timestep_length_mismatch():
    with pytest.raises(ValueError, match="timesteps"):
        DemandResult(
            E=np.zeros((2, 3)), position_m=np.zeros(3), timesteps=np.zeros(5),
        )


def test_demand_result_rejects_position_length_mismatch():
    with pytest.raises(ValueError, match="position_m"):
        DemandResult(
            E=np.zeros((2, 3)), position_m=np.zeros(4), timesteps=np.zeros(2),
        )


def test_demand_result_to_dataframe_is_long_format():
    df = _demand_result().to_dataframe()
    assert list(df.columns) == ["timestep", "position_m", "energy_wh"]
    assert len(df) == 2 * 3
    np.testing.assert_array_equal(df["timestep"].to_numpy(), [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(
        df["position_m"].to_numpy(), [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
    )
    np.testing.assert_array_equal(df["energy_wh"].to_numpy(), [1, 2, 3, 4, 5, 6])
