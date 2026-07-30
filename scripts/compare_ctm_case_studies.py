"""Step 10: compare CTM validation metrics across case studies.

Builds on the single-study validation in
``scripts/validate_ctm_corridor.py``: for each study listed in a manifest
CSV it recomputes the same corridor-pooled metrics (flow/density
RMSE/MAPE, VMT/VHT percent difference, and the share of flow GEH<5)
straight from that study's ``result.npz``, then renders how each metric
moves with corridor length and ramp-fill strategy.

A *case study* here is one CTM sim of a length-subset of a corridor under
one ramp-fill strategy. The manifest declares, one row per study, the
same inputs the single-study script takes plus two label columns:

    sim_dir,cells,timeseries_dir,station_metadata,start,distance_mi,ramp_strategy

``station_metadata`` and ``start`` are optional (blank cells allowed):
``station_metadata`` defaults to ``data/pems/station_metadata.csv`` and
``start`` falls back to the start baked into ``result.npz``.

Artifacts are written to ``--out-dir`` (default: a ``cross_study/``
folder next to the manifest):

* ``summary.csv``           -- one tidy row per study, all metrics.
* ``<metric>_vs_distance.png`` (x7) -- metric vs corridor length, a line
  per ramp strategy.
* ``flow_qq.png`` / ``density_qq.png`` -- overlaid corridor-pooled
  sim-vs-observed QQ curves, color = distance, line style = strategy.

Run from the repo root::

    python scripts/compare_ctm_case_studies.py --manifest studies.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transportation_models.utils.ctm import (
    compare_corridor_aggregates,
    compute_flow_geh,
    compute_qq_samples,
    corridor_rmse_mape,
)
from transportation_models.utils.ctm.plots import (
    plot_metric_vs_distance,
    plot_qq_cross_study,
)
from transportation_models.utils.ctm.results import SimulationResult

_DEFAULT_STATION_METADATA = Path("data/pems/station_metadata.csv")

# Line styles cycled across ramp strategies in the QQ overlay.
_QQ_LINESTYLES = ["-", "--", "-.", ":"]

# (column, y-axis label, plot title) for the seven per-metric line plots.
_LINE_METRICS = [
    ("flow_rmse", "flow RMSE [veh/h]", "Flow RMSE"),
    ("flow_mape", "flow MAPE [%]", "Flow MAPE"),
    ("density_rmse", "density RMSE [veh/mi]", "Density RMSE"),
    ("density_mape", "density MAPE [%]", "Density MAPE"),
    ("vmt_pct_diff", "VMT % diff (sim - obs)", "VMT percent difference"),
    ("vht_pct_diff", "VHT % diff (sim - obs)", "VHT percent difference"),
    ("geh_pct_under5", "flow GEH<5 [%]", "Share of flow GEH < 5"),
]


def resolve_start(
    manifest_start: pd.Timestamp | None, baked_start: pd.Timestamp | None,
) -> pd.Timestamp:
    """Pick the validation window's start: prefer the baked-in value.

    Mirrors ``validate_ctm_corridor.resolve_start``: falls back to the
    manifest ``start`` for older npz files that predate start-persistence,
    and raises :class:`ValueError` if neither is available or if a given
    manifest start disagrees with the baked-in one.
    """
    if manifest_start is None and baked_start is None:
        raise ValueError(
            "result.npz has no persisted start; provide a start column "
            "(or rebuild the CTM to bake it in)."
        )
    if (
        manifest_start is not None and baked_start is not None
        and manifest_start != baked_start
    ):
        raise ValueError(
            f"manifest start {manifest_start} disagrees with the start baked "
            f"into result.npz ({baked_start}); leave it blank to use the "
            "baked-in value."
        )
    return manifest_start if manifest_start is not None else baked_start


def _read_manifest(path: Path) -> pd.DataFrame:
    """Load and validate the study manifest."""
    manifest = pd.read_csv(path)
    required = {
        "sim_dir", "cells", "timeseries_dir", "distance_mi", "ramp_strategy",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"manifest is missing required column(s): {sorted(missing)}. "
            f"Required: {sorted(required)}."
        )
    return manifest


def _opt(row: pd.Series, column: str) -> str | None:
    """Return a stripped optional string cell, or None when blank/absent."""
    if column not in row or pd.isna(row[column]):
        return None
    text = str(row[column]).strip()
    return text or None


def _score_study(row: pd.Series) -> tuple[dict, dict, dict]:
    """Recompute one study's pooled metrics and pooled QQ samples.

    Returns ``(summary_row, flow_curve, density_curve)`` where the curves
    carry the corridor-pooled paired sim/observed arrays for the QQ plots.
    """
    sim_dir = Path(row["sim_dir"])
    result = SimulationResult.from_npz(sim_dir / "result.npz")
    cells = pd.read_csv(row["cells"])
    timeseries_dir = Path(row["timeseries_dir"])

    manifest_start = _opt(row, "start")
    start = resolve_start(
        pd.Timestamp(manifest_start) if manifest_start else None,
        result.start,
    )

    station_metadata = pd.read_csv(
        Path(_opt(row, "station_metadata") or _DEFAULT_STATION_METADATA)
    )

    qq = compute_qq_samples(result, cells, timeseries_dir, start=start)
    errors = corridor_rmse_mape(qq)
    aggregates = compare_corridor_aggregates(
        result, cells, timeseries_dir, station_metadata, start=start,
    )

    # GEH needs whole-hour windows; leave a gap rather than failing the run.
    try:
        geh_pct = compute_flow_geh(
            result, cells, timeseries_dir, start=start,
        ).corridor_pct_under5
    except ValueError as e:
        print(
            f"  skipping GEH for {row['ramp_strategy']} @ "
            f"{row['distance_mi']} mi: {e}",
            file=sys.stderr,
        )
        geh_pct = float("nan")

    distance = float(row["distance_mi"])
    strategy = str(row["ramp_strategy"])
    summary_row = {
        "distance_mi": distance,
        "ramp_strategy": strategy,
        "flow_rmse": errors["flow"][1],
        "flow_mape": errors["flow"][2],
        "density_rmse": errors["density"][1],
        "density_mape": errors["density"][2],
        "vmt_pct_diff": aggregates.vmt_pct_diff,
        "vht_pct_diff": aggregates.vht_pct_diff,
        "geh_pct_under5": geh_pct,
    }
    flow_sim, flow_obs = qq.pooled("flow")
    density_sim, density_obs = qq.pooled("density")
    flow_curve = {
        "distance": distance, "strategy": strategy,
        "sim": flow_sim, "obs": flow_obs,
    }
    density_curve = {
        "distance": distance, "strategy": strategy,
        "sim": density_sim, "obs": density_obs,
    }
    return summary_row, flow_curve, density_curve


def _strategy_colors(strategies: list[str]) -> dict[str, str]:
    """Stable color per ramp strategy (consistent across the line plots)."""
    palette = plt.get_cmap("tab10")
    return {s: palette(i % 10) for i, s in enumerate(sorted(set(strategies)))}


def _distance_styles(distances: list[float]) -> dict[float, str]:
    """Line style per distance for the QQ overlay."""
    uniq = sorted(set(distances))
    return {d: _QQ_LINESTYLES[i % len(_QQ_LINESTYLES)] for i, d in enumerate(uniq)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest", required=True, type=Path,
        help="CSV listing the studies to compare (see module docstring).",
    )
    parser.add_argument(
        "--out-dir", default=None, type=Path,
        help=("Directory for summary.csv and the plots. Defaults to a "
              "cross_study/ folder next to the manifest."),
    )
    args = parser.parse_args()

    manifest = _read_manifest(args.manifest)

    summary_rows: list[dict] = []
    flow_curves: list[dict] = []
    density_curves: list[dict] = []
    for _, row in manifest.iterrows():
        print(
            f"scoring {row['ramp_strategy']} @ {row['distance_mi']} mi "
            f"({row['sim_dir']})",
            file=sys.stderr,
        )
        summary_row, flow_curve, density_curve = _score_study(row)
        summary_rows.append(summary_row)
        flow_curves.append(flow_curve)
        density_curves.append(density_curve)

    summary = pd.DataFrame(summary_rows)

    out_dir = (
        args.out_dir if args.out_dir is not None
        else args.manifest.parent / "cross_study"
    )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)

    strategy_colors = _strategy_colors(summary["ramp_strategy"].tolist())
    for metric, ylabel, title in _LINE_METRICS:
        plot_metric_vs_distance(
            summary, metric=metric, ylabel=ylabel, title=title,
            out_path=out_dir / f"{metric}_vs_distance.png",
            strategy_colors=strategy_colors,
        )

    distance_styles = _distance_styles(summary["distance_mi"].tolist())
    for quantity, curves in (
        ("flow", flow_curves), ("density", density_curves),
    ):
        plot_qq_cross_study(
            curves, quantity=quantity,
            out_path=out_dir / f"{quantity}_qq.png",
            strategy_colors=strategy_colors, distance_styles=distance_styles,
        )

    print()
    print(f"Compared {len(summary)} studies; wrote artifacts to {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
