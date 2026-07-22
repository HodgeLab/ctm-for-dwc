"""Tests for ``compute_flow_geh`` + ``plot_geh_heatmap`` -- flow GEH statistic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transportation_models.utils.ctm import GEHResult, compute_flow_geh
from transportation_models.utils.ctm.plots import plot_geh_heatmap


# ---- Fixture helpers -----------------------------------------------------


@dataclass
class _Cell:
    length: float = 0.5


@dataclass
class _Freeway:
    dt: float
    cells: list[_Cell]

    @property
    def n_cells(self) -> int:
        return len(self.cells)


@dataclass
class _Result:
    freeway: _Freeway
    mainline_flow: np.ndarray   # (n_cells, T) veh/h

    @property
    def n_steps(self) -> int:
        return self.mainline_flow.shape[1]


def _result_from_5min(flow_5min: np.ndarray, *, steps_per_5min: int = 30) -> _Result:
    """Build a result whose per-5-min mean mainline flow equals ``flow_5min``.

    ``flow_5min`` is ``(n_cells, n_5min)`` in veh/h; each value is held
    constant across its 5-min block of sim steps.
    """
    flow_5min = np.asarray(flow_5min, float)
    n_cells = flow_5min.shape[0]
    mainline = np.repeat(flow_5min, steps_per_5min, axis=1)
    return _Result(
        freeway=_Freeway(
            dt=(5.0 / 60.0) / steps_per_5min,
            cells=[_Cell() for _ in range(n_cells)],
        ),
        mainline_flow=mainline,
    )


def _write_vds_csv(
    path: Path, *, start: pd.Timestamp,
    flows_5min: np.ndarray, station_id: int, speed_mph: float = 60.0,
) -> None:
    ts = pd.date_range(start=start, periods=len(flows_5min), freq="5min")
    pd.DataFrame({
        "timestamp": ts,
        "station": station_id,
        "pct_observed": 100.0,
        "total_flow_[veh/5-min]": flows_5min,
        "avg_speed_[mph]": speed_mph,
    }).to_csv(path, index=False)


def _cells(vds_ids: list[int], sources: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"vds_id": vds_ids, "vds_source": sources})


START = pd.Timestamp("2022-04-12 06:00")


# ---- Core GEH numbers ----------------------------------------------------


def test_perfect_match_geh_zero(tmp_path):
    """Sim 1200 veh/h, observed 100 veh/5min (= 1200 veh/h) -> GEH 0."""
    _write_vds_csv(
        tmp_path / "100.csv", start=START, station_id=100,
        flows_5min=np.full(12, 100.0),
    )
    result = _result_from_5min(np.full((1, 12), 1200.0))
    res = compute_flow_geh(
        result, _cells([100], ["direct"]), tmp_path, start=START,
    )
    assert isinstance(res, GEHResult)
    row = res.per_cell.iloc[0]
    assert row["n_geh_hours"] == 1
    assert row["flow_geh_median"] == pytest.approx(0.0)
    assert row["flow_geh_pct_under5"] == pytest.approx(100.0)
    assert res.corridor_pct_under5 == pytest.approx(100.0)
    assert res.n_geh_samples == 1
    assert res.hourly_geh.shape == (1, 1)


def test_recovers_known_geh(tmp_path):
    """M=1200, C=1000 hourly -> GEH = sqrt(2*200^2/2200) = 6.0302, fails <5."""
    _write_vds_csv(
        tmp_path / "100.csv", start=START, station_id=100,
        flows_5min=np.full(12, 1000.0 / 12.0),  # -> 1000 veh/h observed
    )
    result = _result_from_5min(np.full((1, 12), 1200.0))
    res = compute_flow_geh(
        result, _cells([100], ["direct"]), tmp_path, start=START,
    )
    row = res.per_cell.iloc[0]
    assert row["flow_geh_median"] == pytest.approx(6.0302, abs=1e-3)
    assert row["flow_geh_pct_under5"] == pytest.approx(0.0)
    assert res.corridor_pct_under5 == pytest.approx(0.0)


def test_pct_under5_over_multiple_hours(tmp_path):
    """Hour 0 matches (GEH 0 < 5), hour 1 is far off (GEH > 5) -> 50% pass."""
    obs = np.concatenate([np.full(12, 100.0), np.full(12, 1000.0 / 12.0)])
    _write_vds_csv(
        tmp_path / "100.csv", start=START, station_id=100, flows_5min=obs,
    )
    result = _result_from_5min(np.full((1, 24), 1200.0))  # 2 hours @ 1200
    res = compute_flow_geh(
        result, _cells([100], ["direct"]), tmp_path, start=START,
    )
    row = res.per_cell.iloc[0]
    assert row["n_geh_hours"] == 2
    assert row["flow_geh_pct_under5"] == pytest.approx(50.0)
    assert res.hourly_geh.shape == (1, 2)


# ---- Cell selection ------------------------------------------------------


def test_only_direct_and_tiebreak_cells_scored(tmp_path):
    """A nearest_upstream cell is excluded; direct + tiebreak are kept."""
    for vds in (100, 200):
        _write_vds_csv(
            tmp_path / f"{vds}.csv", start=START, station_id=vds,
            flows_5min=np.full(12, 100.0),
        )
    result = _result_from_5min(np.full((3, 12), 1200.0))
    cells = _cells([100, 999, 200], ["direct", "nearest_upstream", "direct_tiebreak"])
    res = compute_flow_geh(result, cells, tmp_path, start=START)
    assert list(res.per_cell["cell"]) == [0, 2]
    assert res.hourly_geh.shape == (2, 1)


def test_vds_counted_once_across_cells(tmp_path):
    _write_vds_csv(
        tmp_path / "100.csv", start=START, station_id=100,
        flows_5min=np.full(12, 100.0),
    )
    result = _result_from_5min(np.full((2, 12), 1200.0))
    cells = _cells([100, 100], ["direct", "direct_tiebreak"])
    res = compute_flow_geh(result, cells, tmp_path, start=START)
    assert len(res.per_cell) == 1


def test_nan_observed_hour_excluded(tmp_path):
    """An hour with any NaN observed 5-min sample drops out of the GEH."""
    obs = np.concatenate([np.full(12, 100.0), np.full(12, 100.0)])
    obs[15] = np.nan  # hour 1 incomplete
    _write_vds_csv(
        tmp_path / "100.csv", start=START, station_id=100, flows_5min=obs,
    )
    result = _result_from_5min(np.full((1, 24), 1200.0))
    res = compute_flow_geh(
        result, _cells([100], ["direct"]), tmp_path, start=START,
    )
    row = res.per_cell.iloc[0]
    assert row["n_geh_hours"] == 1            # only hour 0 survives
    assert np.isnan(res.hourly_geh[0, 1])     # hour 1 is NaN
    assert res.n_geh_samples == 1


# ---- Error paths ---------------------------------------------------------


def test_non_whole_hour_window_raises(tmp_path):
    _write_vds_csv(
        tmp_path / "100.csv", start=START, station_id=100,
        flows_5min=np.full(6, 100.0),  # 30 min, not a whole hour
    )
    result = _result_from_5min(np.full((1, 6), 1200.0))
    with pytest.raises(ValueError, match="whole-hour"):
        compute_flow_geh(
            result, _cells([100], ["direct"]), tmp_path, start=START,
        )


def test_missing_vds_source_raises(tmp_path):
    result = _result_from_5min(np.full((1, 12), 1200.0))
    with pytest.raises(KeyError, match="vds_source"):
        compute_flow_geh(
            result, pd.DataFrame({"vds_id": [100]}), tmp_path, start=START,
        )


# ---- Plot smoke test -----------------------------------------------------


def test_plot_geh_heatmap_writes_file(tmp_path):
    hourly = np.array([[1.0, 6.0, np.nan], [3.0, 4.0, 2.0]])
    out = tmp_path / "geh_heatmap.png"
    plot_geh_heatmap(
        hourly,
        row_labels=["cell 0 (VDS 100)", "cell 2 (VDS 200)"],
        hour_starts=pd.DatetimeIndex(
            [START, START + pd.Timedelta(hours=1), START + pd.Timedelta(hours=2)]
        ),
        title="test", out_path=out,
    )
    assert out.exists() and out.stat().st_size > 0
