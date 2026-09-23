"""Step 4: calibrate fundamental diagrams and ramp capacities from PeMS data.

Two calibrations live here, both driven by
``scripts/calibrate_fundamental_diagrams.py``:

* :func:`calibrate_fundamental_diagrams` -- fit the five-parameter
  triangular FD (capacity, free-flow speed, congestion wave speed, jam and
  critical density) for each mainline VDS from its (density, flow) cloud,
  and write the results into ``station_metadata_calibrated.csv``.
* :func:`calibrate_ramp_capacities` -- ramp VDSs report flow but not
  speed, so their capacity is a high quantile of the cleaned total flow,
  written to ``ramp_metadata_calibrated.csv``.

Every detector gets a :class:`CalibrationCode` so failed fits are flagged
rather than silently dropped. See Step 4 of ``docs/ctm_module.md``.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .plots import plot_fundamental_diagram

logger = logging.getLogger(__name__)


class CalibrationCode(IntEnum):
    """Outcome of a single detector's fundamental-diagram calibration.

    Written to the ``calibration_code`` column of the calibrated metadata
    (with the matching :data:`CALIBRATION_STATUS` string in
    ``calibration_status``) so results can be triaged. ``0`` is the only
    success; every other code marks a distinct failure path and leaves the
    FD parameter columns NaN.
    """

    # Clean success: params computed and passed validation.
    OK = 0
    # The timeseries file could not be found / read.
    MISSING_TIMESERIES = 1
    # No rows survived the pct_observed threshold or the IQR row filter.
    NO_DATA_AFTER_FILTER = 2
    # No points in the congestion regime (all density <= the critical-density
    # seed), so the congestion branch has nothing to fit.
    NO_CONGESTION_POINTS = 3
    # Fit produced physically invalid parameters (see validate_fd_params).
    VALIDATION_FAILED = 4


# Human-readable status string for each CalibrationCode, mirrored into the
# calibration_status column alongside the integer calibration_code.
CALIBRATION_STATUS: dict[int, str] = {
    CalibrationCode.OK: "ok",
    CalibrationCode.MISSING_TIMESERIES: "missing_timeseries",
    CalibrationCode.NO_DATA_AFTER_FILTER: "no_data_after_filter",
    CalibrationCode.NO_CONGESTION_POINTS: "no_congestion_points",
    CalibrationCode.VALIDATION_FAILED: "validation_failed",
}

# FD parameter columns produced by calibration; blanked to NaN on any
# non-OK calibration code.
FD_PARAM_COLS: list[str] = [
    "capacity",
    "free_flow_speed",
    "congestion_wave_speed",
    "jam_density",
    "critical_density",
]

# Columns read from each per-VDS timeseries CSV (Step 3 output).
TIMESERIES_COLS: list[str] = [
    "timestamp", "station", "pct_observed",
    "total_flow_[veh/5-min]", "avg_speed_[mph]",
]


# ---- Loading and cleaning --------------------------------------------------


def read_station_metadata(metadata_path: Path | str) -> pd.DataFrame:
    """Read a PeMS station metadata CSV, normalizing the key column names.

    Raw PeMS metadata names its key ``ID`` (and postmile ``Abs_PM``); these
    are renamed to ``Station ID`` / ``Abs PM`` so the file joins against the
    timeseries and against a previously calibrated metadata CSV alike.
    """
    metadata = pd.read_csv(metadata_path)
    if "ID" in metadata.columns:
        metadata = metadata.rename(columns={"ID": "Station ID", "Abs_PM": "Abs PM"})
    return metadata


def available_detectors(
    metadata: pd.DataFrame,
    timeseries_dir: Path | str,
    detector_type: str | None = None,
) -> list[str]:
    """VDS ids present in both ``metadata`` and ``timeseries_dir``.

    Parameters
    ----------
    metadata : pd.DataFrame
        Station metadata from :func:`read_station_metadata`.
    timeseries_dir : Path | str
        Directory of ``{station_id}.csv`` timeseries files.
    detector_type : str, optional
        Keep only this metadata ``Type`` (e.g. ``"ML"``); None keeps all.

    Returns
    -------
    list[str]
        Sorted VDS ids.
    """
    if detector_type is None:
        mask = pd.Series(True, index=metadata.index)
    else:
        mask = metadata["Type"] == detector_type
    in_metadata = set(metadata.loc[mask, "Station ID"].astype(str))
    with_timeseries = {p.stem for p in Path(timeseries_dir).glob("*.csv")}
    return sorted(in_metadata & with_timeseries)


def load_detector_timeseries(
    timeseries_dir: Path | str,
    station_id: str,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Load one VDS's timeseries on a contiguous 5-minute grid.

    Reindexes the timeseries to a complete 5-minute grid spanning the year
    range observed in the file -- gaps become NaN rows rather than being
    dropped -- then inner-joins the station's ``Lanes`` from ``metadata`` on
    ``Station ID``.

    Parameters
    ----------
    timeseries_dir : Path | str
        Directory holding ``{station_id}.csv``.
    station_id : str
        VDS station ID.
    metadata : pd.DataFrame
        Station metadata from :func:`read_station_metadata`.

    Returns
    -------
    pd.DataFrame
        Timeseries columns plus ``Station ID`` and ``Lanes``.

    Raises
    ------
    FileNotFoundError
        If the timeseries CSV does not exist.
    """
    filepath = Path(timeseries_dir) / f"{station_id}.csv"
    timeseries_df = pd.read_csv(filepath, usecols=TIMESERIES_COLS)
    timeseries_df["timestamp"] = pd.to_datetime(timeseries_df["timestamp"])

    # Ensure complete indices
    years_in_df = timeseries_df["timestamp"].dt.year.unique()
    full_date_range = pd.date_range(
        start=f"{years_in_df.min()}-01-01 00:00",
        end=f"{years_in_df.max()}-12-31 23:59",
        freq="5min",
    )
    df = timeseries_df.set_index("timestamp").reindex(full_date_range)
    df = df.reset_index(names="timestamp")

    # Broadcast station onto reindexed gap rows so they survive the inner merge below.
    df["station"] = int(station_id)

    return df.rename(columns={"station": "Station ID"}).merge(
        right=metadata.loc[:, ["Station ID", "Lanes"]], how="inner", on="Station ID",
    )


def standardize_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw PeMS counts to per-lane hourly flow and density.

    Adds ``flow_[veh/hr-lane]`` (5-minute station-wide flow x 12 / lanes)
    and ``density_[veh/mi-lane]`` (flow / average speed), and drops the raw
    ``total_flow_[veh/5-min]`` and ``avg_speed_[mph]`` columns.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_detector_timeseries`.

    Returns
    -------
    pd.DataFrame
        The standardized DataFrame.
    """
    # Number of 5-minute periods per hour (used for normalization)
    periods_per_hour = 12

    # Scale the flow: [veh/5-min] * [5-min/hr] * [1/lanes] = [veh/hr*lanes]
    df["flow_[veh/hr-lane]"] = (
        df["total_flow_[veh/5-min]"] * periods_per_hour
    ) / df["Lanes"]

    # Scale the density: [veh/5-min] * [5-min/h] * [1/lanes] / [mi/h] = [veh/mi*lanes]
    df["density_[veh/mi-lane]"] = df["flow_[veh/hr-lane]"] / df["avg_speed_[mph]"]

    return df.drop(columns=["total_flow_[veh/5-min]", "avg_speed_[mph]"])


def whiten_flow(df: pd.DataFrame, flow_col: str) -> pd.DataFrame:
    """Add a ``whitened_flow`` column z-scored by (weekday, 5-minute block).

    Subtracts the per-(weekday, 5-minute-of-day) mean of ``flow_col`` and
    divides by the per-(weekday, 5-minute-of-day) sample standard deviation,
    so the outlier filter isn't fooled by morning-peak vs midnight-shoulder
    differences.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``timestamp`` and ``flow_col`` columns.
    flow_col : str
        Flow column to whiten.

    Returns
    -------
    pd.DataFrame
        Copy of the input with ``whitened_flow`` added.
    """
    out = df.copy()
    weekday = out["timestamp"].dt.weekday.rename("weekday")
    block = ((out["timestamp"].dt.minute / 60) + out["timestamp"].dt.hour).rename(
        "5min_block"
    )
    grouped = out.groupby([weekday, block])[flow_col]
    out["whitened_flow"] = (
        (out[flow_col] - grouped.transform("mean")) / grouped.transform("std")
    )
    return out


def filter_flow_outliers(
    df: pd.DataFrame,
    flow_col: str = "whitened_flow",
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Drop rows whose ``flow_col`` falls outside an IQR-based fence.

    Keeps rows in ``[Q1 - k*IQR, Q3 + k*IQR]``, where ``k = iqr_multiplier``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``flow_col``.
    flow_col : str, optional
        Column used to identify outliers, by default ``"whitened_flow"``.
    iqr_multiplier : float, optional
        Fence multiplier; larger values are more permissive, by default 1.5.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with the index reset.
    """
    q1 = df[flow_col].quantile(0.25)
    q3 = df[flow_col].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - iqr_multiplier * iqr
    upper_fence = q3 + iqr_multiplier * iqr
    mask = df[flow_col].between(lower_fence, upper_fence)
    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.debug(
            f"Outlier filter removed {n_removed} rows "
            f"({flow_col} outside [{lower_fence:.1f}, {upper_fence:.1f}])"
        )
    return df.loc[mask].reset_index(drop=True)


# ---- Fundamental-diagram fit -----------------------------------------------


def estimate_free_flow_speed(
    df: pd.DataFrame, critical_density_estimate: float,
) -> tuple[float, LinearRegression]:
    """Fit a no-intercept line to the free-flow regime of an FD.

    Keeps rows with density <= ``critical_density_estimate`` and fits
    ``flow = v_f * density``; the slope is the free-flow speed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``flow_[veh/hr-lane]`` and ``density_[veh/mi-lane]``.
    critical_density_estimate : float
        Density boundary below which points are considered free-flow.

    Returns
    -------
    tuple[float, LinearRegression]
        Free-flow speed (the fitted slope) and the fitted model.
    """
    free_flow_points = df.loc[
        df["density_[veh/mi-lane]"] <= critical_density_estimate, :
    ]
    X = free_flow_points["density_[veh/mi-lane]"].to_numpy().reshape(-1, 1)
    y = free_flow_points["flow_[veh/hr-lane]"].to_numpy()

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    logger.debug(f"Free flow speed model fit score: {model.score(X, y)}")

    return model.coef_[0], model


def estimate_congestion_wave_speed(
    df: pd.DataFrame,
    critical_density_estimate: float,
    bin_size: int = 10,
    iqr_multiplier: float = 1.0,
) -> tuple[float, float, LinearRegression, pd.DataFrame]:
    """Fit the congestion regime of an FD via binned peak-flow regression.

    Restricts to rows with density > ``critical_density_estimate``, sorts
    by density, and partitions into non-overlapping bins of ``bin_size``
    rows. Within each bin, drops whitened-flow outliers (above
    ``Q3 + iqr_multiplier * IQR``) and records the bin's mean density and
    max flow. Fits ``flow = slope * density + b`` to those bin points;
    ``slope`` is the (negative) congestion-wave slope and ``-b/slope`` is
    the jam density.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``flow_[veh/hr-lane]``, ``density_[veh/mi-lane]``,
        and ``whitened_flow`` columns.
    critical_density_estimate : float
        Density boundary above which points are considered congested.
    bin_size : int, optional
        Number of points per non-overlapping density bin, by default 10.
    iqr_multiplier : float, optional
        IQR-fence multiplier applied per bin to reject high-flow outliers,
        by default 1.0.

    Returns
    -------
    tuple[float, float, LinearRegression, pd.DataFrame]
        ``(slope, jam_density, model, bin_df)`` where ``bin_df`` has columns
        ``BinDensity`` and ``BinFlow``.
    """
    congested = df.loc[df["density_[veh/mi-lane]"] > critical_density_estimate, :]
    congested = congested.sort_values(by="density_[veh/mi-lane]").reset_index(
        drop=True
    )

    num_bins = len(congested) // bin_size
    bin_densities = []
    bin_flows = []
    for i in range(num_bins):
        bin_data = congested.iloc[i * bin_size : (i + 1) * bin_size]
        flows = bin_data["flow_[veh/hr-lane]"].values
        densities = bin_data["density_[veh/mi-lane]"].values
        whitened_flows = bin_data["whitened_flow"].values

        # IQR fence on whitened_flow, applied to both flows and densities
        q1 = np.percentile(whitened_flows, 25)
        q3 = np.percentile(whitened_flows, 75)
        valid_mask = whitened_flows <= q3 + iqr_multiplier * (q3 - q1)
        if not valid_mask.any():
            continue

        bin_densities.append(np.mean(densities[valid_mask]))
        bin_flows.append(np.max(flows[valid_mask]))

    bin_df = pd.DataFrame({"BinDensity": bin_densities, "BinFlow": bin_flows})

    X = bin_df["BinDensity"].values.reshape(-1, 1)
    y = bin_df["BinFlow"].values
    model = LinearRegression()
    model.fit(X, y)
    logger.debug(f"Congestion wave model fit score: {model.score(X, y)}")

    slope = model.coef_[0]
    # x-intercept: flow = 0 => density = -intercept / slope
    jam_density = -model.intercept_ / slope

    return slope, jam_density, model, bin_df


def find_fundamental_diagram_peak(
    free_flow_model: LinearRegression,
    congestion_model: LinearRegression,
) -> tuple[float, float]:
    """Intersect the free-flow and congestion fit lines.

    Free flow:  ``flow = ff_slope * density`` (no intercept).
    Congestion: ``flow = cw_slope * density + cw_intercept``.
    Intersection: ``density = cw_intercept / (ff_slope - cw_slope)``.

    Returns
    -------
    tuple[float, float]
        ``(critical_density, capacity)``.
    """
    ff_slope = free_flow_model.coef_[0]
    cw_slope = congestion_model.coef_[0]
    cw_intercept = congestion_model.intercept_

    critical_density = cw_intercept / (ff_slope - cw_slope)
    capacity = ff_slope * critical_density
    return critical_density, capacity


def validate_fd_params(
    params: dict[str, float], *, triangle_rtol: float = 0.05,
) -> tuple[CalibrationCode, str]:
    """Check that a calibrated triangular FD is physically valid.

    Pure function (no I/O) so it can be unit-tested independently of the
    calibration pipeline. Returns :attr:`CalibrationCode.OK` with an
    ``"ok"`` reason when every gate passes, otherwise
    :attr:`CalibrationCode.VALIDATION_FAILED` and a short human-readable
    reason naming the first failing gate.

    Gates:
      1. Positivity -- ``free_flow_speed``, ``congestion_wave_speed`` and
         ``capacity`` are finite and strictly positive.
      2. ``congestion_wave_speed < free_flow_speed`` (required for a
         backward-bending congestion branch).
      3. Density ordering -- ``0 < critical_density < jam_density``.
      4. Triangle consistency (defensive) -- ``capacity`` agrees with both
         ``free_flow_speed * critical_density`` and
         ``congestion_wave_speed * (jam_density - critical_density)`` to
         within ``triangle_rtol`` relative error. Holds by construction in
         the current estimator, so this guards against future regressions.

    Parameters
    ----------
    params : dict[str, float]
        Must contain ``capacity``, ``free_flow_speed``,
        ``congestion_wave_speed``, ``jam_density``, ``critical_density``.
    triangle_rtol : float, optional
        Relative tolerance for the triangle-consistency gate, by default
        0.05.

    Returns
    -------
    tuple[CalibrationCode, str]
        ``(CalibrationCode.OK, "ok")`` or
        ``(CalibrationCode.VALIDATION_FAILED, <reason>)``.
    """
    v_f = params["free_flow_speed"]
    w = params["congestion_wave_speed"]
    capacity = params["capacity"]
    rho_crit = params["critical_density"]
    rho_jam = params["jam_density"]

    def _fail(reason: str) -> tuple[CalibrationCode, str]:
        return CalibrationCode.VALIDATION_FAILED, reason

    # 1. Positivity (also rejects NaN/inf, since comparisons with NaN are False).
    if not (np.isfinite(v_f) and v_f > 0):
        return _fail(f"free_flow_speed not positive ({v_f})")
    if not (np.isfinite(w) and w > 0):
        return _fail(f"congestion_wave_speed not positive ({w})")
    if not (np.isfinite(capacity) and capacity > 0):
        return _fail(f"capacity not positive ({capacity})")

    # 2. Congestion wave strictly slower than free-flow speed.
    if not (w < v_f):
        return _fail(f"congestion_wave_speed ({w}) >= free_flow_speed ({v_f})")

    # 3. Density ordering.
    if not (np.isfinite(rho_crit) and np.isfinite(rho_jam)):
        return _fail(f"non-finite density (crit={rho_crit}, jam={rho_jam})")
    if not (0 < rho_crit < rho_jam):
        return _fail(
            f"density ordering violated (0 < {rho_crit} < {rho_jam} is False)"
        )

    # 4. Triangle consistency.
    free_branch_cap = v_f * rho_crit
    cong_branch_cap = w * (rho_jam - rho_crit)
    for label, expected in (
        ("free_flow_speed * critical_density", free_branch_cap),
        ("congestion_wave_speed * (jam_density - critical_density)", cong_branch_cap),
    ):
        if abs(capacity - expected) > triangle_rtol * abs(capacity):
            return _fail(
                f"triangle inconsistency: capacity ({capacity:.2f}) != "
                f"{label} ({expected:.2f}) within rtol {triangle_rtol}"
            )

    return CalibrationCode.OK, "ok"


def _nan_fd_params(detector_id: str) -> dict[str, float]:
    """FD parameter dict with all values NaN (used for failed detectors)."""
    params = {"Station ID": int(detector_id)}
    params.update({col: np.nan for col in FD_PARAM_COLS})
    return params


def fit_fundamental_diagram(
    df: pd.DataFrame,
    detector_id: str,
    *,
    bin_size: int = 10,
    bin_iqr_multiplier: float = 1.0,
    triangle_rtol: float = 0.05,
    plot_path: Path | None = None,
) -> tuple[dict[str, float], CalibrationCode]:
    """Derive the full set of FD parameters for a single detector.

    Seeds the critical-density estimate from the row with the highest
    observed flow, fits both regimes, and recovers capacity and critical
    density from their intersection. The fit is then checked with
    :func:`validate_fd_params`. If the congestion regime has too few points
    to fit, or the fit is physically invalid, the FD parameters are
    returned as NaN and a non-OK :class:`CalibrationCode` flags the reason.

    Parameters
    ----------
    df : pd.DataFrame
        Whitened and outlier-filtered DataFrame for a single detector.
    detector_id : str
        VDS station ID associated with the data.
    bin_size : int, optional
        Points per congestion density bin, by default 10.
    bin_iqr_multiplier : float, optional
        IQR-fence multiplier applied per congestion bin, by default 1.0.
    triangle_rtol : float, optional
        Relative tolerance for the triangle-consistency gate, by default
        0.05.
    plot_path : Path, optional
        If given and the fit is valid, write an FD plot here via
        :func:`utils.ctm.plots.plot_fundamental_diagram`.

    Returns
    -------
    tuple[dict[str, float], CalibrationCode]
        The parameter dict (keys ``Station ID`` and :data:`FD_PARAM_COLS`;
        FD values are NaN on failure) and the outcome code.
    """
    # Estimate critical density based on peak observed flow
    capacity_idx = df["flow_[veh/hr-lane]"].idxmax()
    critical_density_estimate = df.loc[capacity_idx, "density_[veh/mi-lane]"]

    # Bail out if the congestion regime can't support a line fit. We need
    # at least two non-overlapping bins of points above the seed density.
    n_congested = int((df["density_[veh/mi-lane]"] > critical_density_estimate).sum())
    if n_congested // bin_size < 2:
        logger.warning(
            f"VDS {detector_id}: only {n_congested} congestion-regime points "
            f"(< 2 bins of {bin_size}); cannot fit congestion branch."
        )
        return _nan_fd_params(detector_id), CalibrationCode.NO_CONGESTION_POINTS

    free_flow_speed, free_flow_model = estimate_free_flow_speed(
        df, critical_density_estimate,
    )
    congestion_slope, jam_density, congestion_model, bin_df = (
        estimate_congestion_wave_speed(
            df,
            critical_density_estimate,
            bin_size=bin_size,
            iqr_multiplier=bin_iqr_multiplier,
        )
    )
    critical_density, capacity = find_fundamental_diagram_peak(
        free_flow_model, congestion_model,
    )

    params = {
        "Station ID": int(detector_id),
        "capacity": capacity,
        "free_flow_speed": free_flow_speed,
        "congestion_wave_speed": -1 * congestion_slope,
        "jam_density": jam_density,
        "critical_density": critical_density,
    }

    # Validate before accepting; reject physically invalid fits.
    code, reason = validate_fd_params(params, triangle_rtol=triangle_rtol)
    if code != CalibrationCode.OK:
        logger.warning(f"VDS {detector_id}: FD validation failed: {reason}")
        return _nan_fd_params(detector_id), code

    if plot_path is not None:
        plot_fundamental_diagram(
            df, bin_df, free_flow_model, congestion_model, params,
            out_path=plot_path,
        )

    return params, code


# ---- Batch drivers ---------------------------------------------------------


def calibrate_fundamental_diagrams(
    metadata_path: Path | str,
    timeseries_dir: Path | str,
    *,
    detectors: list[str] | None = None,
    imputation_threshold: float = 100.0,
    iqr_multiplier: float = 1.0,
    bin_iqr_multiplier: float = 1.0,
    bin_size: int = 10,
    triangle_rtol: float = 0.05,
    plot_dir: Path | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Calibrate the triangular FD for each detector and record the results.

    For each detector: load its timeseries, standardize to per-lane flow and
    density, drop rows with ``pct_observed < imputation_threshold``, whiten
    flow, drop outliers, and fit via :func:`fit_fundamental_diagram`. Every
    detector gets a :class:`CalibrationCode` and ``calibration_status``;
    failures leave the FD parameter columns NaN rather than aborting the run.

    Parameters
    ----------
    metadata_path : Path | str
        PeMS station metadata CSV.
    timeseries_dir : Path | str
        Directory of per-VDS ``{station_id}.csv`` timeseries (Step 3).
    detectors : list[str], optional
        Detectors to calibrate. Defaults to every mainline (``Type == "ML"``)
        detector present in both the metadata and ``timeseries_dir``.
    imputation_threshold : float, optional
        Minimum ``pct_observed`` for a row to be used, by default 100.0.
    iqr_multiplier : float, optional
        IQR multiplier for the row-level outlier filter, by default 1.0.
    bin_iqr_multiplier : float, optional
        IQR multiplier for the per-bin congestion filter, by default 1.0.
    bin_size : int, optional
        Points per congestion density bin, by default 10.
    triangle_rtol : float, optional
        Relative tolerance for the triangle-consistency gate, by default
        0.05.
    plot_dir : Path, optional
        If given, write one ``{station_id}.png`` FD plot per valid fit here.
    output_path : Path | str, optional
        If given, write the calibrated metadata CSV here. An existing file is
        updated incrementally: detectors not in this run keep their previous
        calibration.

    Returns
    -------
    pd.DataFrame
        The station metadata with :data:`FD_PARAM_COLS`, ``calibration_code``
        and ``calibration_status`` filled in for the calibrated detectors.
    """
    metadata = read_station_metadata(metadata_path)
    if detectors is None:
        detectors = available_detectors(metadata, timeseries_dir, detector_type="ML")
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    def _record(detector: str, params: dict[str, float], code: CalibrationCode) -> None:
        mask = metadata["Station ID"] == int(detector)
        for key, value in params.items():
            metadata.loc[mask, key] = value
        metadata.loc[mask, "calibration_code"] = int(code)
        metadata.loc[mask, "calibration_status"] = CALIBRATION_STATUS[code]

    for idx, detector in enumerate(detectors):
        logger.info(f"Calibrating detector {idx + 1}/{len(detectors)}: VDS {detector}")

        # A missing/unreadable timeseries is a tracked outcome, not a crash.
        try:
            df = load_detector_timeseries(timeseries_dir, detector, metadata)
        except (FileNotFoundError, OSError) as e:
            logger.warning(
                f"VDS {detector}: timeseries could not be loaded ({e}); "
                f"recording {CALIBRATION_STATUS[CalibrationCode.MISSING_TIMESERIES]}."
            )
            _record(detector, _nan_fd_params(detector), CalibrationCode.MISSING_TIMESERIES)
            continue

        df = standardize_timeseries(df)
        len_original = len(df)
        df = df.loc[df["pct_observed"] >= imputation_threshold, :].reset_index(drop=True)
        logger.debug(
            f"{(((len_original - len(df)) / len_original) * 100):.2f} percent of"
            f" datapoints removed from dataframe of length {len_original}:"
            f" pct_observed < {imputation_threshold}"
        )
        if df.empty:
            logger.warning(
                f"Calibration skipped for VDS {detector}: 0 of {len_original} rows "
                f"had pct_observed >= {imputation_threshold}. "
                f"Lower the imputation threshold to retain partially-observed rows."
            )
            _record(detector, _nan_fd_params(detector), CalibrationCode.NO_DATA_AFTER_FILTER)
            continue

        df = filter_flow_outliers(
            whiten_flow(df, "flow_[veh/hr-lane]"), iqr_multiplier=iqr_multiplier,
        )
        if df.empty:
            logger.warning(
                f"VDS {detector}: outlier filter dropped all rows; "
                f"recording {CALIBRATION_STATUS[CalibrationCode.NO_DATA_AFTER_FILTER]}."
            )
            _record(detector, _nan_fd_params(detector), CalibrationCode.NO_DATA_AFTER_FILTER)
            continue

        params, code = fit_fundamental_diagram(
            df,
            detector,
            bin_size=bin_size,
            bin_iqr_multiplier=bin_iqr_multiplier,
            triangle_rtol=triangle_rtol,
            plot_path=plot_dir / f"{detector}.png" if plot_dir is not None else None,
        )
        if code == CalibrationCode.OK:
            params_str = "\n".join(f"  {k}: {v:.3f}" for k, v in params.items())
            logger.info(f"Calibrated parameters:\n{params_str}")
        else:
            logger.info(
                f"VDS {detector}: calibration code {int(code)} "
                f"({CALIBRATION_STATUS[code]}); FD parameters left NaN."
            )
        _record(detector, params, code)

    if output_path is not None:
        # Merge with any prior calibration so this run is incremental: restore
        # previously-calibrated params for detectors in the output file that
        # were NOT processed this run. Without this, writing out would wipe
        # the calibration of every detector not in this run.
        processed_ids = {int(d) for d in detectors}
        if Path(output_path).exists():
            prev = pd.read_csv(output_path).set_index("Station ID")
            keep_mask = ~metadata["Station ID"].isin(processed_ids)
            for col in FD_PARAM_COLS + ["calibration_code", "calibration_status"]:
                if col in prev.columns:
                    prev_vals = metadata["Station ID"].map(prev[col])
                    metadata.loc[keep_mask, col] = prev_vals[keep_mask]

        metadata.to_csv(output_path, index=False)
        logger.info(f"Calibrated metadata saved to: {output_path}")

    return metadata


def calibrate_ramp_capacities(
    detectors: list[str],
    metadata_path: Path | str,
    timeseries_dir: Path | str,
    *,
    imputation_threshold: float = 100.0,
    quantile: float = 0.99,
    iqr_multiplier: float = 1.5,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Estimate per-VDS ramp capacity from cleaned 5-minute total-flow data.

    Ramp detectors report total flow [veh/5-min] but not speed, so we
    can't fit a triangular FD for them. Instead we take a high quantile
    of the cleaned, total-flow timeseries as the ramp's capacity in
    veh/hr -- this is the value that slots into a Cell's
    ``on_ramp_capacity`` (R_i) or ``off_ramp_capacity`` (S_i) at
    assembly time (Step 6).

    Per-VDS pipeline:
      1. Load the 5-minute timeseries via :func:`load_detector_timeseries`.
      2. Drop rows below ``imputation_threshold`` on ``pct_observed``.
      3. Convert ``total_flow_[veh/5-min] -> flow_[veh/hr]`` (x 12).
         We use total flow (not per-lane) because R_i / S_i are the
         ramp's total inflow / outflow capacity in the CTM.
      4. Whiten ``flow_[veh/hr]`` against its per-(weekday, 5-min-block)
         mean and stddev.
      5. Drop whitened-flow outliers via an IQR fence.
      6. Return ``quantile`` of the cleaned ``flow_[veh/hr]`` (default
         99th percentile -- robust to a single noisy spike while
         preserving the genuine observed peak).

    Parameters
    ----------
    detectors : list[str]
        Explicit list of ramp VDS Station IDs (typically the union of
        ``on_ramp_vds_id`` and ``off_ramp_vds_id`` from a Step-1
        ``cells.csv``), so the function is agnostic to the metadata
        schema's ramp-type spelling.
    metadata_path : Path | str
        PeMS station metadata CSV.
    timeseries_dir : Path | str
        Directory of per-VDS ``{station_id}.csv`` timeseries (Step 3).
    imputation_threshold : float, default 100.0
        Minimum ``pct_observed`` for a row to be used.
    quantile : float, default 0.99
        Quantile of cleaned ``flow_[veh/hr]`` to report as capacity.
    iqr_multiplier : float, default 1.5
        IQR-fence width on ``whitened_flow`` for the outlier filter.
    output_path : Path | str, optional
        If given, write the result CSV here.

    Returns
    -------
    pd.DataFrame
        One row per detector with columns:

        ============================ ============================================
        column                       meaning
        ============================ ============================================
        ``Station ID``               VDS id (int)
        ``ramp_capacity_[veh/hr]``   capacity estimate, in veh/hr; NaN if no
                                     timeseries was available or the cleaned
                                     set was empty (assembly will treat NaN
                                     as :math:`R_i = \\infty` via the CTM
                                     :class:`Cell` defaults).
        ``n_observations``           count of 5-min samples used for the
                                     quantile (after imputation + outlier
                                     filtering)
        ``calibration_code``         :class:`CalibrationCode` int: 0 = ok,
                                     1 = missing timeseries, 2 = no data
                                     after filtering, 4 = non-positive
                                     capacity
        ``calibration_status``       human-readable status string for the
                                     code (see :data:`CALIBRATION_STATUS`)
        ============================ ============================================
    """
    metadata = read_station_metadata(metadata_path)
    rows: list[dict] = []
    for idx, detector in enumerate(detectors):
        logger.info(
            f"Calibrating ramp capacity {idx + 1}/{len(detectors)}: VDS {detector}"
        )

        def _ramp_row(
            code: CalibrationCode, capacity: float = np.nan, n_obs: int = 0,
        ) -> dict:
            return {
                "Station ID": int(detector),
                "ramp_capacity_[veh/hr]": capacity,
                "n_observations": n_obs,
                "calibration_code": int(code),
                "calibration_status": CALIBRATION_STATUS[code],
            }

        try:
            df = load_detector_timeseries(timeseries_dir, detector, metadata)
        except (FileNotFoundError, OSError):
            logger.warning(
                f"No timeseries file for ramp VDS {detector}; "
                f"emitting NaN capacity (CTM will default to infinity)."
            )
            rows.append(_ramp_row(CalibrationCode.MISSING_TIMESERIES))
            continue

        df = df.loc[df["pct_observed"] >= imputation_threshold, :].reset_index(drop=True)
        if df.empty:
            logger.warning(
                f"Ramp VDS {detector}: no rows survived pct_observed >= "
                f"{imputation_threshold}; emitting NaN capacity."
            )
            rows.append(_ramp_row(CalibrationCode.NO_DATA_AFTER_FILTER))
            continue

        # Total flow per hour (not per-lane). Ramp capacity in the CTM is
        # the ramp's total inflow capacity in veh/h.
        df["flow_[veh/hr]"] = df["total_flow_[veh/5-min]"] * 12.0
        df_clean = filter_flow_outliers(
            whiten_flow(df, "flow_[veh/hr]"), iqr_multiplier=iqr_multiplier,
        )
        if df_clean.empty:
            logger.warning(
                f"Ramp VDS {detector}: outlier filter dropped all rows; "
                f"emitting NaN capacity."
            )
            rows.append(_ramp_row(CalibrationCode.NO_DATA_AFTER_FILTER))
            continue

        capacity = float(df_clean["flow_[veh/hr]"].quantile(quantile))
        # A ramp capacity must be a positive, finite veh/hr value.
        if not (np.isfinite(capacity) and capacity > 0):
            logger.warning(
                f"Ramp VDS {detector}: non-positive capacity estimate "
                f"({capacity}); recording "
                f"{CALIBRATION_STATUS[CalibrationCode.VALIDATION_FAILED]}."
            )
            rows.append(_ramp_row(
                CalibrationCode.VALIDATION_FAILED, n_obs=int(len(df_clean)),
            ))
            continue
        rows.append(_ramp_row(
            CalibrationCode.OK, capacity=capacity, n_obs=int(len(df_clean)),
        ))

    ramp_df = pd.DataFrame(rows, columns=[
        "Station ID", "ramp_capacity_[veh/hr]", "n_observations",
        "calibration_code", "calibration_status",
    ])

    if output_path is not None:
        ramp_df.to_csv(output_path, index=False)
        logger.info(f"Ramp capacity calibration saved to: {output_path}")

    return ramp_df
