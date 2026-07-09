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
