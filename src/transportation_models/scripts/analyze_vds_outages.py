"""
Script for analyzing PeMS VDS sensor outages.

For each Mainline vehicle detector station (VDS), detect outages -- runs of
consecutive 5-minute intervals that are missing or low-quality -- then
summarize and visualize outage durations.

An interval is considered MISSING if the row is absent from the complete
5-minute grid (NaN ``pct_observed``) OR ``pct_observed < PCT_OBSERVED_THRESHOLD``.

Produces, under ``OUTPUT_DIR``:
    - vds_outages.csv                  : one row per outage
    - vds_outage_summary.csv           : per-VDS + aggregate summary stats
    - vds_outage_duration_histogram.png: outage counts per duration bucket
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
DETECTOR_TYPE = "Mainline"
PCT_OBSERVED_THRESHOLD = 50.0
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
def detect_outages_for_station(df: pd.DataFrame, station_id: str) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
        One row per outage with columns ``Station ID``, ``start``, ``end``,
        ``duration_minutes``, ``n_intervals``. Empty if the station never
        reported or had no outages.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Observed span: clip to the first/last timestamp the station actually reported.
    observed = df["pct_observed"].notna()
    if not observed.any():
        logger.warning(f"Station {station_id}: no observed data; skipping.")
        return pd.DataFrame(
            columns=["Station ID", "start", "end", "duration_minutes", "n_intervals"]
        )

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
                "start": grp["timestamp"].iloc[0],
                "end": grp["timestamp"].iloc[-1],
                "duration_minutes": n_intervals * INTERVAL_MINUTES,
                "n_intervals": n_intervals,
            }
        )

    return pd.DataFrame(
        records,
        columns=["Station ID", "start", "end", "duration_minutes", "n_intervals"],
    )


def summarize_station(
    station_id: str, df: pd.DataFrame, outages: pd.DataFrame
) -> dict:
    """
    Compute per-VDS summary statistics over the station's observed span.

    Parameters
    ----------
    station_id : str
        VDS station ID.
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
        "span_days": round(span_minutes / (60 * 24), 2),
        "span_years": round(span_years, 3),
        "n_outages": n_outages,
        "outages_per_year": round(n_outages / span_years, 2) if span_years else np.nan,
        "mean_outage_minutes": round(outages["duration_minutes"].mean(), 1)
        if n_outages
        else 0.0,
        "median_outage_minutes": round(outages["duration_minutes"].median(), 1)
        if n_outages
        else 0.0,
        "total_downtime_hours": round(downtime_minutes / 60, 2),
        "uptime_pct": round(
            100 * (span_intervals - downtime_intervals) / span_intervals, 3
        ),
    }


# ----- Plotting ----- #
def plot_duration_histogram(outages: pd.DataFrame, save_path: str) -> None:
    """Bar chart of outage counts per duration bucket, pooled across all VDS."""
    buckets = pd.cut(
        outages["duration_minutes"],
        bins=DURATION_BUCKET_EDGES,
        labels=DURATION_BUCKET_LABELS,
        right=False,
    )
    counts = buckets.value_counts().reindex(DURATION_BUCKET_LABELS, fill_value=0)

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 24,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(counts.index.astype(str), counts.values, color="#2c7fb8")
    ax.set_xlabel("Outage duration")
    ax.set_ylabel("Number of outages")
    ax.set_title(
        f"PeMS {DETECTOR_TYPE} VDS Outage Duration Distribution\n"
        f"(missing = absent OR pct_observed < {PCT_OBSERVED_THRESHOLD:g})"
    )
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Histogram saved to: {save_path}")


# ----- Main ----- #
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processor = PeMSDataProcessor(
        root_directory=ROOT_DIR,
        metadata_filepath=METADATA_FILEPATH,
    )

    detectors = processor.retrieve_detectors(detector_type=DETECTOR_TYPE)
    logger.info(f"Found {len(detectors)} {DETECTOR_TYPE} detectors to analyze.")

    all_outages = []
    summary_rows = []
    for i, station_id in enumerate(detectors, start=1):
        try:
            df = processor.load_data_by_id(station_id)
        except Exception as exc:  # noqa: BLE001 - skip unreadable stations, keep going
            logger.warning(f"Station {station_id}: failed to load ({exc}); skipping.")
            continue

        outages = detect_outages_for_station(df, station_id)
        summary = summarize_station(station_id, df, outages)
        if summary is None:
            continue

        all_outages.append(outages)
        summary_rows.append(summary)

        if i % 25 == 0 or i == len(detectors):
            logger.info(f"Processed {i}/{len(detectors)} detectors.")

    if not summary_rows:
        logger.error("No detectors yielded any observed data; nothing to write.")
        return

    outages_df = pd.concat(all_outages, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    # Aggregate (headline) row: averages across all Mainline VDS.
    aggregate = {
        "Station ID": "ALL",
        "span_days": round(summary_df["span_days"].mean(), 2),
        "span_years": round(summary_df["span_years"].mean(), 3),
        "n_outages": int(summary_df["n_outages"].sum()),
        "outages_per_year": round(summary_df["outages_per_year"].mean(), 2),
        "mean_outage_minutes": round(outages_df["duration_minutes"].mean(), 1)
        if not outages_df.empty
        else 0.0,
        "median_outage_minutes": round(outages_df["duration_minutes"].median(), 1)
        if not outages_df.empty
        else 0.0,
        "total_downtime_hours": round(summary_df["total_downtime_hours"].sum(), 2),
        "uptime_pct": round(summary_df["uptime_pct"].mean(), 3),
    }
    summary_df = pd.concat(
        [summary_df, pd.DataFrame([aggregate])], ignore_index=True
    )

    # Write outputs.
    outages_path = os.path.join(OUTPUT_DIR, "vds_outages.csv")
    summary_path = os.path.join(OUTPUT_DIR, "vds_outage_summary.csv")
    hist_path = os.path.join(OUTPUT_DIR, "vds_outage_duration_histogram.png")

    outages_df.to_csv(outages_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Outage records ({len(outages_df)}) saved to: {outages_path}")
    logger.info(f"Summary table saved to: {summary_path}")

    logger.info(
        "Headline stats (averaged across %d %s VDS):\n"
        "  avg outages/year      = %.2f\n"
        "  avg outage duration   = %.1f min\n"
        "  avg uptime            = %.3f%%",
        len(summary_rows),
        DETECTOR_TYPE,
        aggregate["outages_per_year"],
        aggregate["mean_outage_minutes"],
        aggregate["uptime_pct"],
    )

    if not outages_df.empty:
        plot_duration_histogram(outages_df, hist_path)
    else:
        logger.warning("No outages detected; skipping histogram.")


if __name__ == "__main__":
    main()
