"""Gap-fill accuracy validation for ramp VDS timeseries.

The Step-7 scenario builder reconstructs missing ramp samples with
:func:`utils.ctm.ramp_flow.historical_average_fill` before they feed CTM
demand/beta, so the filler's accuracy bounds how well the simulation can
match reality -- yet the real gaps it fills have no ground truth to score
against.

This module measures fill accuracy by **synthetic hold-out**: for each ramp
detector listed in a *stretches* CSV (the PeMS-native topology from
``scripts/build_pems_stretches.py``), mask a seeded-random subset of the
observed samples at a given ``mask_rate``, run ``historical_average_fill``
once on the masked copy, and score the filled values against the held-out
truth. Each ramp detector is treated in isolation (no summing of stretch
siblings) and classified ``on``/``off`` by whether it came from the
stretch's ``on_ids`` or ``off_ids``.

Metrics per VDS (truth-minus-fill sign convention, matching
``ramp_flow_estimation.validation.flow_metrics``):

* ``rmse``  -- root-mean-square error [veh/5-min]
* ``nrmse`` -- ``sqrt(sum(err^2) / sum(truth^2))`` (Kan Eq. 12 form)
* ``bias``  -- ``mean(truth - fill)``
* ``nbias`` -- ``sum(truth - fill) / sum(truth)``

Run the multi-seed trial average via ``scripts/validate_ramp_flow.py``.

:func:`conservation_fill_accuracy` benchmarks an alternative fill strategy
-- inferring the ramp flow from the mainline pair via flow conservation --
on the type (a)/(b) stretches where that inference is fully determined. Run
it via ``scripts/validate_ramp_flow_conservation.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .ramp_flow import historical_average_fill

_FLOW_COL = "total_flow_[veh/5-min]"
_TIME_COL = "timestamp"

_TABLE_COLUMNS = [
    "vds_id", "ramp_type", "n_observed", "n_masked", "n_scored",
    "n_unfilled", "rmse", "nrmse", "bias", "nbias",
]

_CONSERVATION_COLUMNS = [
    "stretch_id", "config_type", "ramp_type", "up_ml_id", "down_ml_id",
    "ramp_ids", "n_scored", "rmse", "nrmse", "bias", "nbias",
]

_POINT_COLUMNS = ["stretch_id", "ramp_type", "measured", "estimated"]


def _parse_stretch_ramp_ids(value) -> list[int]:
    """Parse a stretches-CSV ramp id cell (semicolon-separated).

    Integer tokens only: virtual ids (e.g. ``'v400001'``, ramps that
    physically exist but are absent from PeMS) have no timeseries to mask
    and are dropped here.
    """
    ids: list[int] = []
    for tok in str(value if value is not None else "").split(";"):
        tok = tok.strip()
        if tok and tok.isdigit():
            ids.append(int(tok))
    return ids


def _ramp_roster(stretches_df: pd.DataFrame) -> list[tuple[int, str]]:
    """``(vds_id, ramp_type)`` for every unique measured ramp detector.

    A detector appearing in several stretches is scored once; one that
    appears as both an on- and off-ramp keeps its first classification and
    is flagged with a ``UserWarning``.
    """
    roster: dict[int, str] = {}
    for row in stretches_df.itertuples(index=False):
        for col, ramp_type in (("on_ids", "on"), ("off_ids", "off")):
            for vid in _parse_stretch_ramp_ids(getattr(row, col, None)):
                prior = roster.get(vid)
                if prior is None:
                    roster[vid] = ramp_type
                elif prior != ramp_type:
                    warnings.warn(
                        f"VDS {vid} appears as both on- and off-ramp across "
                        f"stretches; keeping {prior!r}.",
                        UserWarning, stacklevel=3,
                    )
    return sorted(roster.items())


def _fill_metrics(truth: np.ndarray, fill: np.ndarray) -> dict:
    """RMSE / NRMSE / BIAS / NBIAS over the scored (non-NaN fill) samples."""
    scored = ~np.isnan(fill)
    t, f = truth[scored], fill[scored]
    err = t - f
    out = {"n_scored": int(scored.sum()), "n_unfilled": int((~scored).sum())}
    if err.size == 0:
        return {**out, "rmse": np.nan, "nrmse": np.nan,
                "bias": np.nan, "nbias": np.nan}
    ss_truth = float(np.sum(t ** 2))
    sum_truth = float(np.sum(t))
    out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["nrmse"] = (float(np.sqrt(np.sum(err ** 2) / ss_truth))
                    if ss_truth > 0 else np.nan)
    out["bias"] = float(np.mean(err))
    out["nbias"] = float(np.sum(err) / sum_truth) if sum_truth > 0 else np.nan
    return out


def ramp_fill_accuracy(
    stretches_df: pd.DataFrame,
    timeseries_dir: Union[Path, str],
    *,
    mask_rate: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Historical-average gap-fill accuracy, one row per ramp VDS.

    For each measured ramp detector in ``stretches_df`` (union of
    ``on_ids`` / ``off_ids``): load its timeseries, mask a seeded-random
    ``mask_rate`` fraction of the observed (non-NaN) samples, fill the
    masked copy with :func:`historical_average_fill` in a single pass, and
    score the fill against the held-out truth at the masked positions.

    Parameters
    ----------
    stretches_df : pandas.DataFrame
        Stretch inventory (``scripts/build_pems_stretches.py``); only the
        ``on_ids`` / ``off_ids`` columns are consumed.
    timeseries_dir : Path or str
        Directory of per-VDS timeseries CSVs (``<vds_id>.csv``). Detectors
        without a file are skipped with a ``UserWarning``.
    mask_rate : float, default 0.2
        Fraction of each detector's observed samples to hold out
        (``floor(mask_rate * n_observed)``, at least 1 when any samples
        are observed).
    seed : int, default 0
        Seed for the random hold-out draw. Run several seeds and average
        for multi-trial estimates (``scripts/validate_ramp_flow.py``).

    Returns
    -------
    pandas.DataFrame
        One row per ramp VDS: ``vds_id``, ``ramp_type`` (``on``/``off``),
        ``n_observed``, ``n_masked``, ``n_scored``, ``n_unfilled`` (masked
        samples the filler left NaN -- their (weekday, time-of-day) bin had
        no other history), ``rmse``, ``nrmse``, ``bias``, ``nbias``.
    """
    if not 0.0 < mask_rate < 1.0:
        raise ValueError(f"mask_rate must be in (0, 1); got {mask_rate}")
    timeseries_dir = Path(timeseries_dir)
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    for vds_id, ramp_type in _ramp_roster(stretches_df):
        path = timeseries_dir / f"{vds_id}.csv"
        if not path.exists():
            warnings.warn(
                f"VDS {vds_id} ({ramp_type}-ramp) has no timeseries at "
                f"{path}; skipping.",
                UserWarning, stacklevel=2,
            )
            continue
        df = (
            pd.read_csv(path, usecols=[_TIME_COL, _FLOW_COL],
                        parse_dates=[_TIME_COL])
            .sort_values(_TIME_COL).reset_index(drop=True)
        )
        truth = df[_FLOW_COL].to_numpy(dtype=float)
        observed = np.flatnonzero(~np.isnan(truth))
        n_masked = int(mask_rate * len(observed))
        if len(observed):
            n_masked = max(n_masked, 1)
        row = {"vds_id": vds_id, "ramp_type": ramp_type,
               "n_observed": int(len(observed)), "n_masked": n_masked}
        if n_masked == 0:
            rows.append({**row, **_fill_metrics(np.empty(0), np.empty(0))})
            continue

        masked_pos = rng.choice(observed, size=n_masked, replace=False)
        masked = df.copy()
        masked.loc[masked_pos, _FLOW_COL] = np.nan
        with warnings.catch_warnings():
            # The filler warns on bins with no other history; expected here
            # and surfaced via n_unfilled.
            warnings.simplefilter("ignore", RuntimeWarning)
            filled = historical_average_fill(masked, flow_col=_FLOW_COL)
        fill = filled[_FLOW_COL].to_numpy(dtype=float)[masked_pos]
        rows.append({**row, **_fill_metrics(truth[masked_pos], fill)})

    return pd.DataFrame(rows, columns=_TABLE_COLUMNS)


def conservation_fill_accuracy(
    stretches_df: pd.DataFrame,
    timeseries_dir: Union[Path, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flow-conservation gap-fill accuracy, one row per type (a)/(b) stretch.

    Conservation across a stretch reads ``q_up + r = q_down + s``. On a
    type (a) stretch (on-ramps only) the ramp flow is fully determined by
    the mainline pair, ``r = q_down - q_up``; on a type (b) stretch
    (off-ramps only), ``s = q_up - q_down``. This function scores that
    estimate against the measured ramp flow -- summed over the stretch's
    ramps when there are several, since conservation only constrains their
    total -- at every timestamp where the upstream ML, downstream ML, and
    all measured ramps are simultaneously observed. The estimate never
    consults the ramp's own record, so unlike :func:`ramp_fill_accuracy` no
    hold-out masking is needed and the pass is deterministic.

    The raw estimate goes negative whenever counting noise or unmetered
    flow makes the mainline difference point the wrong way. It is **clipped
    at zero before scoring**: this benchmarks conservation as a *fill*
    strategy, and a real fill would never write a negative flow into the
    timeseries. Clipping only ever raises the estimate, so systematic
    conservation violations still surface as negative ``bias``/``nbias``.

    Type (c) stretches are excluded by construction (with both on- and
    off-ramps the conservation equation is underdetermined). A stretch is
    skipped with a ``UserWarning`` when its ramp set includes a virtual id
    (physically present but absent from PeMS, so there is no measured truth)
    or when any of its detectors lacks a timeseries CSV.

    Parameters
    ----------
    stretches_df : pandas.DataFrame
        Stretch inventory (``scripts/build_pems_stretches.py``); consumes
        ``stretch_id``, ``config_type``, ``up_ml_id``, ``down_ml_id`` and
        the ``on_ids`` / ``off_ids`` column matching the stretch type.
    timeseries_dir : Path or str
        Directory of per-VDS timeseries CSVs (``<vds_id>.csv``).

    Returns
    -------
    (table, points) : tuple of pandas.DataFrame
        ``table``: one row per scored stretch -- ``stretch_id``,
        ``config_type``, ``ramp_type`` (``on`` for (a), ``off`` for (b)),
        ``up_ml_id``, ``down_ml_id``, ``ramp_ids`` (semicolon-joined),
        ``n_scored``, ``rmse``, ``nrmse``, ``bias``, ``nbias`` (same
        truth-minus-estimate conventions as :func:`ramp_fill_accuracy`).
        ``points``: every scored sample -- ``stretch_id``, ``ramp_type``,
        ``measured``, ``estimated`` [veh/5-min] -- for pooled
        distribution (QQ) plots.
    """
    timeseries_dir = Path(timeseries_dir)
    cache: dict[int, "pd.Series | None"] = {}

    def _flow_series(vds_id: int):
        if vds_id not in cache:
            path = timeseries_dir / f"{vds_id}.csv"
            if not path.exists():
                cache[vds_id] = None
            else:
                df = pd.read_csv(path, usecols=[_TIME_COL, _FLOW_COL],
                                 parse_dates=[_TIME_COL])
                cache[vds_id] = df.set_index(_TIME_COL)[_FLOW_COL].sort_index()
        return cache[vds_id]

    rows: list[dict] = []
    points: list[pd.DataFrame] = []
    for row in stretches_df.itertuples(index=False):
        config_type = getattr(row, "config_type", None)
        if config_type not in ("a", "b"):
            continue
        ramp_col, ramp_type = (("on_ids", "on") if config_type == "a"
                               else ("off_ids", "off"))
        cell = getattr(row, ramp_col, None)
        if pd.isna(cell):
            tokens = []
        elif isinstance(cell, (int, float)):
            # A ramp column whose cells are all single ids round-trips
            # through read_csv as float; str() would yield e.g. '3.0'.
            tokens = [str(int(cell))]
        else:
            tokens = [t.strip() for t in str(cell).split(";") if t.strip()]
        if not tokens:
            continue
        if not all(t.isdigit() for t in tokens):
            warnings.warn(
                f"stretch {row.stretch_id}: ramp set {cell!r} includes a "
                "virtual id (no measured truth); skipping.",
                UserWarning, stacklevel=2,
            )
            continue
        if pd.isna(row.up_ml_id) or pd.isna(row.down_ml_id):
            warnings.warn(
                f"stretch {row.stretch_id}: missing a mainline id; skipping.",
                UserWarning, stacklevel=2,
            )
            continue
        ramp_ids = [int(t) for t in tokens]
        up_id, down_id = int(row.up_ml_id), int(row.down_ml_id)
        series = {vid: _flow_series(vid)
                  for vid in (up_id, down_id, *ramp_ids)}
        missing = [vid for vid, s in series.items() if s is None]
        if missing:
            warnings.warn(
                f"stretch {row.stretch_id}: no timeseries for VDS {missing}; "
                "skipping.",
                UserWarning, stacklevel=2,
            )
            continue
        aligned = pd.concat(series.values(), axis=1, keys=list(series),
                            join="inner").dropna()
        q_up = aligned[up_id].to_numpy(dtype=float)
        q_down = aligned[down_id].to_numpy(dtype=float)
        truth = aligned[ramp_ids].to_numpy(dtype=float).sum(axis=1)
        raw = q_down - q_up if config_type == "a" else q_up - q_down
        est = np.clip(raw, 0.0, None)
        metrics = _fill_metrics(truth, est)
        metrics.pop("n_unfilled")           # est is never NaN after dropna
        rows.append({
            "stretch_id": row.stretch_id, "config_type": config_type,
            "ramp_type": ramp_type, "up_ml_id": up_id, "down_ml_id": down_id,
            "ramp_ids": ";".join(map(str, ramp_ids)), **metrics,
        })
        points.append(pd.DataFrame({
            "stretch_id": row.stretch_id, "ramp_type": ramp_type,
            "measured": truth, "estimated": est,
        }))

    table = pd.DataFrame(rows, columns=_CONSERVATION_COLUMNS)
    points_df = (pd.concat(points, ignore_index=True) if points
                 else pd.DataFrame(columns=_POINT_COLUMNS))
    return table, points_df
