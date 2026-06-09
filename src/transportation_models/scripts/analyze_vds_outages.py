"""
Script for analyzing and comparing PeMS VDS sensor outages by detector type.

For each Mainline / On Ramp / Off Ramp vehicle detector station (VDS), detect
outages -- runs of consecutive 5-minute intervals that are missing or
low-quality -- then summarize and visualize outage durations, and compare
the three detector types head-to-head.

An interval is considered MISSING if the row is absent from the complete
5-minute grid (NaN ``pct_observed``) OR ``pct_observed < PCT_OBSERVED_THRESHOLD``.

Produces, under ``OUTPUT_DIR``:
    - vds_outages.csv                      : one row per outage (Type-tagged)
    - vds_outage_summary.csv               : per-VDS + per-type + overall stats
    - vds_monthly_uptime.csv               : per-VDS x month uptime ratio
    - vds_outage_duration_histogram.png    : outage counts per duration bucket,
                                             grouped by detector type
    - vds_uptime_distribution_by_type.png  : per-station uptime % distribution,
                                             one box per detector type
    - vds_monthly_uptime_heatmap.png       : station x month uptime heatmap,
                                             one panel per detector type
"""

# ----- Setup ----- #

# 3rd party imports
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
from transportation_models.utils import constants
from transportation_models.utils.data_processing import PeMSDataProcessor
from transportation_models.utils.logs import make_logger

# ----- Config ----- #
ROOT_DIR = str(constants.ROOT_PATH)
METADATA_FILEPATH = str(constants.CALTRANS_VDS_METADATA_FILEPATH)
OUTPUT_DIR = os.path.join(ROOT_DIR, "analysis", "outage_analysis")
DETECTOR_TYPES = ["Mainline", "On Ramp", "Off Ramp"]
TYPE_COLORS = {
    "Mainline": "#2c7fb8",
    "On Ramp": "#41b6c4",
    "Off Ramp": "#7fcdbb",
}
PCT_OBSERVED_THRESHOLD = 100.0
INTERVAL_MINUTES = 5
MINUTES_PER_YEAR = 60 * 24 * 365

# Duration bucket edges in minutes (last bucket is open-ended).
DURATION_BUCKET_EDGES = [0, 15, 30, 60, 120, 180, 240, 300, 360, 1440, 10080, np.inf]
DURATION_BUCKET_LABELS = [
    "0-15 min",
    "15-30 min",
    "30 min-1 hr",
    "1-2 hr",
    "2-3 hr",
    "3-4 hr",
    "4-5 hr",
    "5-6 hr",
    "6-24 hr",
    "1-7 days",
    ">7 days",
]

# Logging setup
logger = make_logger(include_stdout=True, log_prefix="vds_outage_analysis")


# ----- Outage detection ----- #
OUTAGE_COLS = ["Station ID", "Type", "start", "end", "duration_minutes", "n_intervals"]


def detect_outages_for_station(
    df: pd.DataFrame, station_id: str, station_type: str
) -> pd.DataFrame:
    """
    Detect outages for a single VDS from its reindexed 5-minute grid.

    The grid is first clipped to the station's *observed span* (first to last
    timestamp with a non-NaN ``pct_observed``) so that pre-deployment /
    post-decommission stretches are not counted as outages. Within that span,
    consecutive missing intervals are collapsed into single outage events.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``PeMSDataProcessor.load_data_by_id`` -- a complete 5-minute
        grid with columns ``timestamp`` and ``pct_observed``.
    station_id : str
        VDS station ID (for labelling the records).
    station_type : str
        Detector type (e.g., ``"Mainline"``, ``"On Ramp"``, ``"Off Ramp"``).

    Returns
    -------
    pd.DataFrame
        One row per outage with columns ``Station ID``, ``Type``, ``start``,
        ``end``, ``duration_minutes``, ``n_intervals``. Empty if the station
        never reported or had no outages.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Observed span: clip to the first/last timestamp the station actually reported.
    observed = df["pct_observed"].notna()
    if not observed.any():
        logger.warning(f"Station {station_id}: no observed data; skipping.")
        return pd.DataFrame(columns=OUTAGE_COLS)

    first_idx = observed.idxmax()
    last_idx = observed[::-1].idxmax()
    span = df.loc[first_idx:last_idx].reset_index(drop=True)

    # An interval is missing if absent (NaN) OR below the quality threshold.
    missing = span["pct_observed"].isna() | (
        span["pct_observed"] < PCT_OBSERVED_THRESHOLD
    )

    # Run-length grouping: a new group starts whenever the missing flag flips.
    group_id = missing.ne(missing.shift()).cumsum()
    outage_groups = span.loc[missing].groupby(group_id[missing])

    records = []
    for _, grp in outage_groups:
        n_intervals = len(grp)
        records.append(
            {
                "Station ID": station_id,
                "Type": station_type,
                "start": grp["timestamp"].iloc[0],
                "end": grp["timestamp"].iloc[-1],
                "duration_minutes": n_intervals * INTERVAL_MINUTES,
                "n_intervals": n_intervals,
            }
        )

    return pd.DataFrame(records, columns=OUTAGE_COLS)


def summarize_station(
    station_id: str, station_type: str, df: pd.DataFrame, outages: pd.DataFrame
) -> dict:
    """
    Compute per-VDS summary statistics over the station's observed span.

    Parameters
    ----------
    station_id : str
        VDS station ID.
    station_type : str
        Detector type (e.g., ``"Mainline"``, ``"On Ramp"``, ``"Off Ramp"``).
    df : pd.DataFrame
        The station's reindexed 5-minute grid (from ``load_data_by_id``).
    outages : pd.DataFrame
        This station's outage records (from ``detect_outages_for_station``).

    Returns
    -------
    dict
        Summary row, or ``None`` if the station never reported.
    """
    observed = df["pct_observed"].notna()
    if not observed.any():
        return None

    df = df.sort_values("timestamp").reset_index(drop=True)
    first_idx = observed.idxmax()
    last_idx = observed[::-1].idxmax()
    span_intervals = last_idx - first_idx + 1
    span_minutes = span_intervals * INTERVAL_MINUTES
    span_years = span_minutes / MINUTES_PER_YEAR

    downtime_intervals = int(outages["n_intervals"].sum()) if not outages.empty else 0
    downtime_minutes = downtime_intervals * INTERVAL_MINUTES
    n_outages = len(outages)

    return {
        "Station ID": station_id,
        "Type": station_type,
        "span_days": round(span_minutes / (60 * 24), 2),
        "span_years": round(span_years, 3),
        "n_outages": n_outages,
        "outages_per_year": round(n_outages / span_years, 2) if span_years else np.nan,
        "mean_outage_minutes": (
            round(outages["duration_minutes"].mean(), 1) if n_outages else 0.0
        ),
        "median_outage_minutes": (
            round(outages["duration_minutes"].median(), 1) if n_outages else 0.0
        ),
        "total_downtime_hours": round(downtime_minutes / 60, 2),
        "uptime_pct": round(
            100 * (span_intervals - downtime_intervals) / span_intervals, 3
        ),
    }


def compute_monthly_uptime(
    station_id: str, station_type: str, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute per-month uptime ratio for a single VDS, restricted to its
    observed span.

    Within the span, an interval counts as UP iff ``pct_observed`` is present
    and >= ``PCT_OBSERVED_THRESHOLD``. ``uptime_pct`` is the percentage of
    such intervals among all 5-minute slots in the month that fall inside
    the span (so months that are only partially covered by the span are still
    scored over their covered portion).

    Parameters
    ----------
    station_id : str
        VDS station ID.
    station_type : str
        Detector type.
    df : pd.DataFrame
        The station's reindexed 5-minute grid (from ``load_data_by_id``).

    Returns
    -------
    pd.DataFrame
        Columns ``Station ID``, ``Type``, ``month`` (period[M] as Timestamp),
        ``uptime_pct``, ``n_intervals``. Empty if the station never reported.
    """
    observed = df["pct_observed"].notna()
    if not observed.any():
        return pd.DataFrame(
            columns=["Station ID", "Type", "month", "uptime_pct", "n_intervals"]
        )

    df = df.sort_values("timestamp").reset_index(drop=True)
    first_idx = observed.idxmax()
    last_idx = observed[::-1].idxmax()
    span = df.loc[first_idx:last_idx, ["timestamp", "pct_observed"]].copy()
    span["is_up"] = span["pct_observed"].notna() & (
        span["pct_observed"] >= PCT_OBSERVED_THRESHOLD
    )
    span["month"] = span["timestamp"].dt.to_period("M").dt.to_timestamp()

    monthly = span.groupby("month", as_index=False).agg(
        uptime_pct=("is_up", "mean"), n_intervals=("is_up", "size")
    )
    monthly["uptime_pct"] = monthly["uptime_pct"] * 100.0
    monthly["Station ID"] = station_id
    monthly["Type"] = station_type
    return monthly[["Station ID", "Type", "month", "uptime_pct", "n_intervals"]]


# ----- Plotting ----- #
PLOT_RC = {
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
}


def plot_duration_histogram(
    outages: pd.DataFrame, stations_per_type: dict, save_path: str
) -> None:
    """Grouped bar chart of outage counts per duration bucket, by detector type.

    Counts are normalized to outages-per-station so the three types can be
    compared on equal footing (Mainline usually has more stations than ramps).
    Denominator is *all* stations with observed data of that type, not just
    stations that produced an outage.
    """
    outages = outages.copy()
    outages["bucket"] = pd.cut(
        outages["duration_minutes"],
        bins=DURATION_BUCKET_EDGES,
        labels=DURATION_BUCKET_LABELS,
        right=False,
    )

    counts = (
        outages.groupby(["Type", "bucket"], observed=False)
        .size()
        .unstack("bucket")
        .reindex(index=DETECTOR_TYPES, columns=DURATION_BUCKET_LABELS, fill_value=0)
    )

    plt.rcParams.update(PLOT_RC)
    fig, (ax_raw, ax_norm) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)

    x = np.arange(len(DURATION_BUCKET_LABELS))
    width = 0.8 / len(DETECTOR_TYPES)
    for i, dtype in enumerate(DETECTOR_TYPES):
        offset = (i - (len(DETECTOR_TYPES) - 1) / 2) * width
        raw_vals = counts.loc[dtype].values
        n_stations = stations_per_type.get(dtype, 0)
        norm_vals = raw_vals / n_stations if n_stations else np.zeros_like(raw_vals)
        label = f"{dtype} (n={n_stations})"
        ax_raw.bar(x + offset, raw_vals, width, color=TYPE_COLORS[dtype], label=label)
        ax_norm.bar(x + offset, norm_vals, width, color=TYPE_COLORS[dtype], label=label)

    ax_raw.set_ylabel("Number of outages")
    ax_raw.set_title(
        "PeMS VDS Outage Duration Distribution by Detector Type\n"
        f"(missing = absent OR pct_observed < {PCT_OBSERVED_THRESHOLD:g})"
    )
    ax_raw.legend()

    ax_norm.set_ylabel("Outages per station")
    ax_norm.set_xlabel("Outage duration")
    ax_norm.set_xticks(x)
    ax_norm.set_xticklabels(DURATION_BUCKET_LABELS, rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Duration histogram saved to: {save_path}")


def plot_uptime_distribution_by_type(summary_df: pd.DataFrame, save_path: str) -> None:
    """Boxplot of per-station uptime % distribution, one box per detector type.

    Each point underlying the box is one VDS's overall uptime % over its
    observed span. Individual stations are overlaid as jittered scatter so
    outliers are visible.
    """
    plt.rcParams.update(PLOT_RC)
    fig, ax = plt.subplots(figsize=(12, 8))

    data = [
        summary_df.loc[summary_df["Type"] == dtype, "uptime_pct"].dropna().values
        for dtype in DETECTOR_TYPES
    ]
    positions = np.arange(len(DETECTOR_TYPES)) + 1

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )
    for patch, dtype in zip(bp["boxes"], DETECTOR_TYPES):
        patch.set_facecolor(TYPE_COLORS[dtype])
        patch.set_alpha(0.6)

    rng = np.random.default_rng(seed=0)
    for pos, dtype, vals in zip(positions, DETECTOR_TYPES, data):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full_like(vals, pos, dtype=float) + jitter,
            vals,
            s=12,
            alpha=0.5,
            color=TYPE_COLORS[dtype],
            edgecolor="black",
            linewidth=0.3,
        )

    labels = [f"{dtype}\n(n={len(vals)})" for dtype, vals in zip(DETECTOR_TYPES, data)]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-station uptime (%)")
    ax.set_title(
        "PeMS VDS Per-Station Uptime by Detector Type\n"
        f"(uptime = % of intervals with pct_observed >= {PCT_OBSERVED_THRESHOLD:g})"
    )
    ax.set_ylim(0, 101)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Uptime distribution plot saved to: {save_path}")


def plot_monthly_uptime_heatmap(monthly_df: pd.DataFrame, save_path: str) -> None:
    """Heatmap of monthly uptime % per station, one panel per detector type.

    Rows = stations (sorted by mean uptime within the type, worst on top),
    columns = months across the full observed period. Cells where the station
    was outside its observed span are left blank (NaN).
    """
    if monthly_df.empty:
        logger.warning("No monthly uptime data; skipping heatmap.")
        return

    plt.rcParams.update(PLOT_RC)
    all_months = pd.DatetimeIndex(sorted(monthly_df["month"].unique()))

    fig, axes = plt.subplots(
        len(DETECTOR_TYPES),
        1,
        figsize=(16, 4 + 2.5 * len(DETECTOR_TYPES)),
        sharex=True,
        constrained_layout=True,
    )
    if len(DETECTOR_TYPES) == 1:
        axes = [axes]

    im = None
    for ax, dtype in zip(axes, DETECTOR_TYPES):
        sub = monthly_df[monthly_df["Type"] == dtype]
        if sub.empty:
            ax.set_title(f"{dtype} -- no data")
            ax.set_axis_off()
            continue

        pivot = sub.pivot(
            index="Station ID", columns="month", values="uptime_pct"
        ).reindex(columns=all_months)
        # Sort worst-uptime-on-top so the eye is drawn to the problem stations.
        pivot = (
            pivot.assign(_mean=pivot.mean(axis=1))
            .sort_values("_mean", ascending=True)
            .drop(columns="_mean")
        )

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            interpolation="nearest",
        )
        ax.set_ylabel(f"{dtype}\n(n={pivot.shape[0]} stations)")
        ax.set_yticks([])

    # Shared month axis on bottom panel.
    bottom_ax = axes[-1]
    month_labels = [m.strftime("%Y-%m") for m in all_months]
    # Show ~12 evenly spaced ticks to keep the axis readable.
    n_ticks = min(12, len(all_months))
    if n_ticks > 0:
        tick_idx = np.linspace(0, len(all_months) - 1, n_ticks, dtype=int)
        bottom_ax.set_xticks(tick_idx)
        bottom_ax.set_xticklabels(
            [month_labels[i] for i in tick_idx], rotation=45, ha="right"
        )
    bottom_ax.set_xlabel("Month")

    fig.suptitle(
        "PeMS VDS Monthly Uptime % by Detector Type\n"
        f"(uptime = % of 5-min intervals in month with pct_observed >= {PCT_OBSERVED_THRESHOLD:g})",
        fontsize=22,
    )
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label("Monthly uptime (%)")

    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Monthly uptime heatmap saved to: {save_path}")


# ----- Aggregation ----- #
def build_type_aggregate(
    label: str, type_summary: pd.DataFrame, type_outages: pd.DataFrame
) -> dict:
    """Roll a per-station ``summary`` slice up into a single aggregate row."""
    return {
        "Station ID": label,
        "Type": type_summary["Type"].iloc[0] if not type_summary.empty else "",
        "span_days": round(type_summary["span_days"].mean(), 2),
        "span_years": round(type_summary["span_years"].mean(), 3),
        "n_outages": int(type_summary["n_outages"].sum()),
        "outages_per_year": round(type_summary["outages_per_year"].mean(), 2),
        "mean_outage_minutes": (
            round(type_outages["duration_minutes"].mean(), 1)
            if not type_outages.empty
            else 0.0
        ),
        "median_outage_minutes": (
            round(type_outages["duration_minutes"].median(), 1)
            if not type_outages.empty
            else 0.0
        ),
        "total_downtime_hours": round(type_summary["total_downtime_hours"].sum(), 2),
        "uptime_pct": round(type_summary["uptime_pct"].mean(), 3),
    }


# ----- Main ----- #
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processor = PeMSDataProcessor(
        root_directory=ROOT_DIR,
        metadata_filepath=METADATA_FILEPATH,
    )

    all_outages: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    monthly_frames: list[pd.DataFrame] = []

    for dtype in DETECTOR_TYPES:
        detectors = processor.retrieve_detectors(detector_type=dtype)
        logger.info(f"Found {len(detectors)} {dtype} detectors to analyze.")

        for i, station_id in enumerate(detectors, start=1):
            try:
                df = processor.load_data_by_id(station_id)
            except Exception as exc:  # noqa: BLE001 - skip unreadable stations
                logger.warning(
                    f"Station {station_id} ({dtype}): failed to load ({exc}); skipping."
                )
                continue

            outages = detect_outages_for_station(df, station_id, dtype)
            summary = summarize_station(station_id, dtype, df, outages)
            if summary is None:
                continue

            all_outages.append(outages)
            summary_rows.append(summary)
            monthly_frames.append(compute_monthly_uptime(station_id, dtype, df))

            if i % 25 == 0 or i == len(detectors):
                logger.info(f"  [{dtype}] processed {i}/{len(detectors)} detectors.")

    if not summary_rows:
        logger.error("No detectors yielded any observed data; nothing to write.")
        return

    outages_df = pd.concat(all_outages, ignore_index=True)
    per_station_summary = pd.DataFrame(summary_rows)
    monthly_df = (
        pd.concat(monthly_frames, ignore_index=True)
        if monthly_frames
        else pd.DataFrame(
            columns=["Station ID", "Type", "month", "uptime_pct", "n_intervals"]
        )
    )

    # Per-type station counts (used for aggregate rows, plot normalization,
    # and the headline log). Captured here, before we append synthetic
    # "ALL ..." rows whose Type matches a real detector type.
    stations_per_type = (
        per_station_summary.groupby("Type")["Station ID"].nunique().to_dict()
    )

    # Per-type aggregate rows + an overall ALL row, appended to the summary.
    aggregate_rows = []
    for dtype in DETECTOR_TYPES:
        type_summary = per_station_summary[per_station_summary["Type"] == dtype]
        type_outages = outages_df[outages_df["Type"] == dtype]
        if type_summary.empty:
            continue
        aggregate_rows.append(
            build_type_aggregate(f"ALL ({dtype})", type_summary, type_outages)
        )
    overall = build_type_aggregate("ALL", per_station_summary, outages_df)
    overall["Type"] = "ALL"
    aggregate_rows.append(overall)
    summary_df = pd.concat(
        [per_station_summary, pd.DataFrame(aggregate_rows)], ignore_index=True
    )

    # Write outputs.
    outages_path = os.path.join(OUTPUT_DIR, "vds_outages.csv")
    summary_path = os.path.join(OUTPUT_DIR, "vds_outage_summary.csv")
    monthly_path = os.path.join(OUTPUT_DIR, "vds_monthly_uptime.csv")
    hist_path = os.path.join(OUTPUT_DIR, "vds_outage_duration_histogram.png")
    uptime_dist_path = os.path.join(OUTPUT_DIR, "vds_uptime_distribution_by_type.png")
    heatmap_path = os.path.join(OUTPUT_DIR, "vds_monthly_uptime_heatmap.png")

    outages_df.to_csv(outages_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)
    logger.info(f"Outage records ({len(outages_df)}) saved to: {outages_path}")
    logger.info(f"Summary table saved to: {summary_path}")
    logger.info(f"Monthly uptime ({len(monthly_df)} rows) saved to: {monthly_path}")

    # Headline log: side-by-side comparison across types.
    logger.info("Per-type headline stats (averaged across stations):")
    logger.info(
        "  %-10s  %5s  %14s  %14s  %10s",
        "Type",
        "n",
        "outages/yr",
        "mean dur (min)",
        "uptime %",
    )
    for row in aggregate_rows:
        type_label = row["Type"]
        n_stations = (
            len(per_station_summary)
            if type_label == "ALL"
            else stations_per_type.get(type_label, 0)
        )
        logger.info(
            "  %-10s  %5d  %14.2f  %14.1f  %9.3f%%",
            type_label,
            n_stations,
            row["outages_per_year"],
            row["mean_outage_minutes"],
            row["uptime_pct"],
        )

    if not outages_df.empty:
        plot_duration_histogram(outages_df, stations_per_type, hist_path)
    else:
        logger.warning("No outages detected; skipping duration histogram.")

    plot_uptime_distribution_by_type(per_station_summary, uptime_dist_path)
    plot_monthly_uptime_heatmap(monthly_df, heatmap_path)


if __name__ == "__main__":
    main()
