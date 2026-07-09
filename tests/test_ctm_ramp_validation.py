"""Tests for ramp-flow gap-fill accuracy validation (Step 9, ramp).

Covers the synthetic hold-out sweep core (:func:`_collect_fill_residuals`
+ :func:`_summarize`) and the two public entry points
(:func:`onramp_fill_accuracy`, :func:`offramp_split_accuracy`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ctm import (
    offramp_split_accuracy,
    onramp_fill_accuracy,
)
from transportation_models.utils.ctm.ramp_validation import (
    DEFAULT_STRATEGIES,
    _collect_fill_residuals,
    _summarize,
)

_PERSISTENCE = {"persistence": DEFAULT_STRATEGIES["persistence"]}

_FLOW_COL = "total_flow_[veh/5-min]"


def _series(start: str, flows: np.ndarray) -> pd.DataFrame:
    """5-min-grid VDS frame from a flow array."""
    ts = pd.date_range(start=start, periods=len(flows), freq="5min")
    return pd.DataFrame({"timestamp": ts, _FLOW_COL: np.asarray(flows, dtype=float)})


def _write_vds(directory, vds_id: int, start: str, flows: np.ndarray) -> None:
    _series(start, flows).to_csv(directory / f"{vds_id}.csv", index=False)


# ---- core sweep ----------------------------------------------------------


def test_persistence_perfect_on_constant_series():
    """Persistence fills a constant series exactly -> zero error everywhere."""
    series = _series("2022-01-03", np.full(2016, 42.0))  # 1 week
    resid = _collect_fill_residuals(
        series,
        flow_col=_FLOW_COL,
        gap_lengths_min=(5, 30, 60),
        strategies=_PERSISTENCE,
        gap_spacing_min=30.0,
    )
    summ = _summarize(
        resid,
        group_cols=["gap_length_min", "strategy"],
        value_specs=[("flow", "truth", "pred")],
    )
    assert (summ["flow_rmse"] == 0.0).all()
    assert (summ["flow_n"] > 0).all()
    assert (summ["flow_n_unfilled"] == 0).all()


def test_masked_stats_historical_recovers_bin_constant_series():
    """Historical avg recovers a per-(weekday,tod) constant even with the
    held-out sample removed from its own bin (masked-stats regime)."""
    # Two full weeks; flow depends only on time-of-day (same each day), so
    # every (weekday, tod) bin holds 2 identical samples. Masking one leaves
    # the other -> the bin mean is unchanged and the fill is exact.
    n = 2016 * 2
    tod = np.arange(n) % 288
    flows = 100.0 + tod.astype(float)
    series = _series("2022-01-03", flows)  # a Monday
    resid = _collect_fill_residuals(
        series,
        flow_col=_FLOW_COL,
        gap_lengths_min=(15, 60),
        strategies={"historical_average": DEFAULT_STRATEGIES["historical_average"]},
        gap_spacing_min=60.0,
    )
    summ = _summarize(
        resid,
        group_cols=["gap_length_min", "strategy"],
        value_specs=[("flow", "truth", "pred")],
    )
    assert summ["flow_n"].sum() > 0
    assert summ["flow_rmse"].max() == pytest.approx(0.0, abs=1e-9)


def test_persistence_error_grows_with_gap_length():
    """On a strictly rising series, longer gaps drift further from truth."""
    series = _series("2022-01-03", np.arange(2016, dtype=float))
    resid = _collect_fill_residuals(
        series,
        flow_col=_FLOW_COL,
        gap_lengths_min=(5, 15, 30, 60),
        strategies=_PERSISTENCE,
        gap_spacing_min=30.0,
    )
    summ = _summarize(
        resid,
        group_cols=["gap_length_min", "strategy"],
        value_specs=[("flow", "truth", "pred")],
    ).sort_values("gap_length_min")
    rmse = summ["flow_rmse"].to_numpy()
    assert np.all(np.diff(rmse) > 0)
    # Persistence under-predicts a rising series -> negative bias.
    assert (summ["flow_bias"] < 0).all()


def test_collect_residuals_skips_blocks_with_nan_truth():
    """A masked block overlapping a real NaN is never scored."""
    flows = np.full(2016, 10.0)
    flows[500:520] = np.nan  # a real outage in the middle
    series = _series("2022-01-03", flows)
    resid = _collect_fill_residuals(
        series,
        flow_col=_FLOW_COL,
        gap_lengths_min=(15,),
        strategies=_PERSISTENCE,
        gap_spacing_min=30.0,
    )
    # No scored position may land on a real-NaN index.
    assert not resid["pos"].isin(range(500, 520)).any()
    assert resid["truth"].notna().all()


def test_no_gaps_fit_returns_empty():
    """A series too short for any block yields empty residuals + summary."""
    series = _series("2022-01-03", np.full(5, 10.0))
    resid = _collect_fill_residuals(
        series,
        flow_col=_FLOW_COL,
        gap_lengths_min=(60,),
        strategies=_PERSISTENCE,
        gap_spacing_min=30.0,
    )
    assert resid.empty
    summ = _summarize(
        resid,
        group_cols=["gap_length_min", "strategy"],
        value_specs=[("flow", "truth", "pred")],
    )
    assert summ.empty
    assert "flow_rmse" in summ.columns


# ---- on-ramp entry point -------------------------------------------------


def _onramp_cells(rows):
    return pd.DataFrame(rows)


def test_onramp_fill_accuracy_pools_and_dedups(tmp_path):
    """Two cells sharing one on-ramp VDS score it once; pooled + per_vds
    tables are both populated."""
    _write_vds(tmp_path, 900, "2022-01-03", np.full(2016, 30.0))
    cells = _onramp_cells(
        [
            {"on_ramp": True, "on_ramp_vds_id": 900},
            {"on_ramp": False, "on_ramp_vds_id": pd.NA},
            {"on_ramp": True, "on_ramp_vds_id": 900},  # duplicate VDS
        ]
    )
    acc = onramp_fill_accuracy(
        cells,
        tmp_path,
        gap_lengths_min=(5, 30),
        strategies=_PERSISTENCE,
        gap_spacing_min=30.0,
    )
    # Deduped: only one VDS in per_vds.
    assert acc.per_vds["vds_id"].nunique() == 1
    assert (acc.per_vds["vds_id"] == "900").all()
    # Constant series -> persistence exact.
    assert (acc.pooled["flow_rmse"] == 0.0).all()
    assert set(acc.pooled["gap_length_min"]) == {5, 30}


# ---- off-ramp split entry point ------------------------------------------


def test_offramp_split_accuracy_perfect_on_constant_flows(tmp_path):
    """Constant off-ramp s and mainline f -> reconstructed beta is exact."""
    _write_vds(tmp_path, 800, "2022-01-03", np.full(2016, 25.0))  # off-ramp s
    _write_vds(tmp_path, 100, "2022-01-03", np.full(2016, 75.0))  # downstream f
    cells = pd.DataFrame(
        [
            {"off_ramp": True, "off_ramp_vds_id": 800, "vds_id": pd.NA},
            {"off_ramp": False, "off_ramp_vds_id": pd.NA, "vds_id": 100},
        ]
    )
    acc = offramp_split_accuracy(
        cells,
        tmp_path,
        gap_lengths_min=(5, 30),
        strategies=_PERSISTENCE,
        gap_spacing_min=30.0,
    )
    assert (acc.pooled["flow_rmse"] == 0.0).all()
    assert (acc.pooled["beta_rmse"] == 0.0).all()
    # Sanity: truth beta = 25/(25+75) = 0.25, so a wrong fill would move it.
    assert {"beta_rmse", "beta_mape", "beta_bias"} <= set(acc.pooled.columns)


def test_offramp_tail_cell_skipped_with_warning(tmp_path):
    """An off-ramp on the corridor's last cell has no downstream f -> skip."""
    _write_vds(tmp_path, 800, "2022-01-03", np.full(2016, 25.0))
    cells = pd.DataFrame(
        [{"off_ramp": True, "off_ramp_vds_id": 800, "vds_id": pd.NA}]
    )
    with pytest.warns(UserWarning, match="last cell"):
        acc = offramp_split_accuracy(
            cells,
            tmp_path,
            gap_lengths_min=(5,),
            strategies=_PERSISTENCE,
        )
    assert acc.pooled.empty
    assert acc.per_vds.empty
