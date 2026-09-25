"""Tests for ``compare_corridor_aggregates`` -- sim vs PeMS corridor VMT/VHT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ctm_for_dwc.ctm import (
    CorridorAggregates, compare_corridor_aggregates,
)


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
    """Minimal SimulationResult stand-in carrying density + speed."""
    freeway: _Freeway
    density: np.ndarray   # (n_cells, T+1)
    speed: np.ndarray     # (n_cells, T)

    @property
    def n_steps(self) -> int:
        return self.speed.shape[1]


def _result_constant(
    densities: list[float], speeds: list[float], *,
    n_5min: int, steps_per_5min: int = 30,
    cell_lengths: list[float] | None = None,
) -> _Result:
    """Constant per-cell density and speed over the horizon.

    ``cell_lengths`` defaults to 0.5 mi per cell; the sim VMT/VHT totals
    are measured over it, so tests that want a "perfect match" give each
    cell the ``Length`` of its own station.
    """
    n_cells = len(densities)
    n_steps = n_5min * steps_per_5min
    dt_h = (5.0 / 60.0) / steps_per_5min
    lengths = [0.5] * n_cells if cell_lengths is None else cell_lengths
    density = np.repeat(
        np.asarray(densities, float)[:, None], n_steps + 1, axis=1,
    )
    speed = np.repeat(np.asarray(speeds, float)[:, None], n_steps, axis=1)
    return _Result(
        freeway=_Freeway(dt=dt_h, cells=[_Cell(length=L) for L in lengths]),
        density=density, speed=speed,
    )


def _write_vds_csv(
    path: Path, *, start: pd.Timestamp,
    flows_5min: np.ndarray, speeds_mph: np.ndarray, station_id: int,
) -> None:
    ts = pd.date_range(start=start, periods=len(flows_5min), freq="5min")
    pd.DataFrame({
        "timestamp": ts,
        "station": station_id,
        "pct_observed": 100.0,
        "total_flow_[veh/5-min]": flows_5min,
        "avg_speed_[mph]": speeds_mph,
    }).to_csv(path, index=False)


def _station_metadata(lengths: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"ID": list(lengths), "Length": list(lengths.values())}
    )


START = pd.Timestamp("2022-04-12 06:00")


# ---- Exact-match cases ---------------------------------------------------


def test_perfect_match_sums_per_unique_direct_vds(tmp_path):
    """Two direct/tiebreak VDS, one excluded nearest_upstream cell.

    VDS 100 (L=0.5): obs 1200 veh/h @ 60 mph -> rho 20; sim rho 20, v 60.
    VDS 200 (L=1.0): obs 2400 veh/h @ 48 mph -> rho 50; sim rho 50, v 48.
    Per-window VMT = flow*L*(5/60); summed over 2 windows. Each cell is
    given its own station's Length so the sim totals (cell length) and
    the observed totals (station Length) coincide.
    """
    n_5min = 2
    _write_vds_csv(
        tmp_path / "100.csv", start=START, station_id=100,
        flows_5min=np.full(n_5min, 100.0), speeds_mph=np.full(n_5min, 60.0),
    )
    _write_vds_csv(
        tmp_path / "200.csv", start=START, station_id=200,
        flows_5min=np.full(n_5min, 200.0), speeds_mph=np.full(n_5min, 48.0),
    )
    # cell 1 is nearest_upstream -> excluded; its state is irrelevant.
    result = _result_constant(
        densities=[20.0, 999.0, 50.0], speeds=[60.0, 1.0, 48.0], n_5min=n_5min,
        cell_lengths=[0.5, 0.5, 1.0],
    )
    cells = pd.DataFrame({
        "vds_id": [100, 100, 200],
        "vds_source": ["direct", "nearest_upstream", "direct_tiebreak"],
    })
    agg = compare_corridor_aggregates(
        result, cells, tmp_path, _station_metadata({100: 0.5, 200: 1.0}),
        start=START,
    )
    assert isinstance(agg, CorridorAggregates)
    assert agg.n_vds == 2
    # VMT: VDS100 = 1200*(1/12)*0.5*2 = 100; VDS200 = 2400*(1/12)*1.0*2 = 400.
    assert agg.sim_vmt == pytest.approx(500.0)
    assert agg.obs_vmt == pytest.approx(500.0)
    # VHT: VDS100 = 20*(1/12)*0.5*2 = 1.6667; VDS200 = 50*(1/12)*1.0*2 = 8.3333.
    assert agg.sim_vht == pytest.approx(10.0)
    assert agg.obs_vht == pytest.approx(10.0)
    assert agg.vmt_pct_diff == pytest.approx(0.0)
    assert agg.vht_pct_diff == pytest.approx(0.0)


def test_each_vds_counted_once_when_assigned_to_multiple_cells(tmp_path):
    """A direct VDS on two cells contributes a single term (first cell)."""
    n_5min = 2
    _write_vds_csv(
        tmp_path / "100.csv", start=START, station_id=100,
        flows_5min=np.full(n_5min, 100.0), speeds_mph=np.full(n_5min, 60.0),
    )
    result = _result_constant(
        densities=[20.0, 20.0], speeds=[60.0, 60.0], n_5min=n_5min,
    )
    cells = pd.DataFrame({
        "vds_id": [100, 100],
        "vds_source": ["direct", "direct_tiebreak"],
    })
    agg = compare_corridor_aggregates(
        result, cells, tmp_path, _station_metadata({100: 0.5}), start=START,
    )
    assert agg.n_vds == 1
    assert agg.sim_vmt == pytest.approx(100.0)   # single VDS, not doubled


def test_no_direct_cells_yields_zero_totals(tmp_path):
    result = _result_constant(densities=[20.0], speeds=[60.0], n_5min=2)
    cells = pd.DataFrame({"vds_id": [100], "vds_source": ["nearest_upstream"]})
    agg = compare_corridor_aggregates(
        result, cells, tmp_path, _station_metadata({100: 0.5}), start=START,
    )
    assert agg.n_vds == 0
    assert agg.sim_vmt == 0.0 and agg.obs_vmt == 0.0
    assert np.isnan(agg.vmt_pct_diff)


# ---- Error / mismatch cases ----------------------------------------------


def test_known_percent_diff_when_sim_high(tmp_path):
    """Sim density and speed scaled so rho*v is +10% and rho is +10%."""
    n_5min = 2
    _write_vds_csv(
        tmp_path / "100.csv", start=START, station_id=100,
        flows_5min=np.full(n_5min, 100.0), speeds_mph=np.full(n_5min, 60.0),
    )
    # rho 22 (+10%), v 66 -> rho*v = 1452 (+21% vs 1200). Keep it simple:
    # rho 22 (+10%) and v 60 -> rho*v = 1320 (+10% VMT), rho +10% VHT.
    result = _result_constant(
        densities=[22.0], speeds=[60.0], n_5min=n_5min,
    )
    agg = compare_corridor_aggregates(
        result, pd.DataFrame({"vds_id": [100], "vds_source": ["direct"]}),
        tmp_path, _station_metadata({100: 0.5}), start=START,
    )
    assert agg.vmt_pct_diff == pytest.approx(10.0)
    assert agg.vht_pct_diff == pytest.approx(10.0)


def test_nan_observed_window_dropped_from_both_sides(tmp_path):
    """A NaN observed 5-min sample drops that window from sim and obs alike."""
    n_5min = 2
    _write_vds_csv(
        tmp_path / "100.csv", start=START, station_id=100,
        flows_5min=np.array([100.0, np.nan]),
        speeds_mph=np.array([60.0, np.nan]),
    )
    result = _result_constant(densities=[20.0], speeds=[60.0], n_5min=n_5min)
    agg = compare_corridor_aggregates(
        result, pd.DataFrame({"vds_id": [100], "vds_source": ["direct"]}),
        tmp_path, _station_metadata({100: 0.5}), start=START,
    )
    # Only the first window survives: VMT = 1200*(1/12)*0.5 = 50 (not 100).
    assert agg.sim_vmt == pytest.approx(50.0)
    assert agg.obs_vmt == pytest.approx(50.0)


def test_missing_station_length_raises(tmp_path):
    n_5min = 2
    _write_vds_csv(
        tmp_path / "100.csv", start=START, station_id=100,
        flows_5min=np.full(n_5min, 100.0), speeds_mph=np.full(n_5min, 60.0),
    )
    result = _result_constant(densities=[20.0], speeds=[60.0], n_5min=n_5min)
    with pytest.raises(ValueError, match="no row in station_metadata"):
        compare_corridor_aggregates(
            result, pd.DataFrame({"vds_id": [100], "vds_source": ["direct"]}),
            tmp_path, _station_metadata({999: 0.5}), start=START,
        )


def test_missing_vds_source_column_raises(tmp_path):
    result = _result_constant(densities=[20.0], speeds=[60.0], n_5min=2)
    with pytest.raises(KeyError, match="vds_source"):
        compare_corridor_aggregates(
            result, pd.DataFrame({"vds_id": [100]}),
            tmp_path, _station_metadata({100: 0.5}), start=START,
        )


def test_row_count_mismatch_raises(tmp_path):
    result = _result_constant(densities=[20.0, 20.0], speeds=[60.0, 60.0], n_5min=2)
    with pytest.raises(ValueError, match="one row per cell"):
        compare_corridor_aggregates(
            result, pd.DataFrame({"vds_id": [100], "vds_source": ["direct"]}),
            tmp_path, _station_metadata({100: 0.5}), start=START,
        )


# ---- Sim totals are measured over the cell's own length -----------------


def _single_vds(tmp_path, *, n_5min: int = 2) -> pd.DataFrame:
    """One direct VDS reading 1200 veh/h @ 60 mph (-> rho 20 veh/mi)."""
    _write_vds_csv(
        tmp_path / "100.csv", start=START, station_id=100,
        flows_5min=np.full(n_5min, 100.0), speeds_mph=np.full(n_5min, 60.0),
    )
    return pd.DataFrame({"vds_id": [100], "vds_source": ["direct"]})


def test_sim_totals_use_cell_length_observed_uses_station_length(tmp_path):
    """Halving the cell length halves sim VMT *and* sim VHT.

    The sim side is the model's own vehicle-miles/-hours over its cell;
    the observed side stays on the station's Length. So length does not
    cancel: a 0.25 mi cell against a 0.5 mi station reads -50% on both
    even though flow and density match exactly.
    """
    cells = _single_vds(tmp_path)
    result = _result_constant(
        densities=[20.0], speeds=[60.0], n_5min=2, cell_lengths=[0.25],
    )
    agg = compare_corridor_aggregates(
        result, cells, tmp_path, _station_metadata({100: 0.5}), start=START,
    )
    # VMT: rho*v = 1200 over 0.25 mi, vs observed 1200 over 0.5 mi.
    assert agg.sim_vmt == pytest.approx(1200.0 / 12.0 * 0.25 * 2)
    assert agg.obs_vmt == pytest.approx(1200.0 / 12.0 * 0.5 * 2)
    assert agg.vmt_pct_diff == pytest.approx(-50.0)
    # VHT: rho = 20 over 0.25 mi, vs observed 20 over 0.5 mi.
    assert agg.sim_vht == pytest.approx(20.0 / 12.0 * 0.25 * 2)
    assert agg.obs_vht == pytest.approx(20.0 / 12.0 * 0.5 * 2)
    assert agg.vht_pct_diff == pytest.approx(-50.0)
