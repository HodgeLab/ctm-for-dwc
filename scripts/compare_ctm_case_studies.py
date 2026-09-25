"""Compare CTM case studies across corridor lengths.

Reads the summary text files the per-study scripts already wrote -- no
simulation or metric is recomputed here:

* ``ctm_validation_summary.txt`` (``scripts/validate_ctm_corridor.py``)
  supplies the corridor-pooled flow/density RMSE and MAPE and the VMT/VHT
  percent errors.
* ``dwc_summary.txt`` (``scripts/generate_dwc_demand.py``) supplies the
  energy delivered and the peak corridor power. It is read from the parent
  of the validation directory, i.e. the sim dir that both scripts wrote to.

A *case study* here is one CTM sim of a length-subset of a corridor under
one ramp-fill strategy. The manifest declares one row per study:

    distance_mi,ramp_strategy,validation_summary

where ``validation_summary`` is the path to that study's
``ctm_validation_summary.txt``.

Artifacts are written to ``--out-dir`` (default: a ``cross_study/``
folder next to the manifest):

* ``summary.csv``           -- one tidy row per study, all metrics.
* ``<metric>_vs_distance.png`` (x8) -- metric vs corridor length, a line
  per ramp strategy.

Run from the repo root::

    python scripts/compare_ctm_case_studies.py --manifest studies.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ctm_for_dwc.ctm.plots import plot_metric_vs_distance

# Patterns matching the summary lines the two per-study scripts emit.
# The error/aggregate lines come from validate_ctm_corridor._print_summary;
# the DWC lines from generate_dwc_demand's summary_lines (which writes
# them in scientific notation).
_NUM = r"([-+]?[\d.]+(?:[eE][-+]?\d+)?|nan)"
_FLOW_ERROR = re.compile(rf"flow\s+RMSE\s+{_NUM}\s*veh/h\s*,\s*MAPE\s+{_NUM}%")
_DENSITY_ERROR = re.compile(
    rf"density\s+RMSE\s+{_NUM}\s*veh/mi\s*,\s*MAPE\s+{_NUM}%"
)
_VMT_DIFF = re.compile(rf"VMT \[veh\*mi\].*diff\s*{_NUM}%")
_VHT_DIFF = re.compile(rf"VHT \[veh\*h\].*diff\s*{_NUM}%")
_ENERGY = re.compile(rf"Energy delivered\s*:\s*{_NUM}\s*Wh")
_PEAK_POWER = re.compile(rf"Peak corridor power\s*:\s*{_NUM}\s*W")

# (column, y-axis label, plot title) for the eight per-metric line plots.
_LINE_METRICS = [
    ("flow_rmse", "flow RMSE [veh/h]", "Flow RMSE"),
    ("flow_mape", "flow MAPE [%]", "Flow MAPE"),
    ("density_rmse", "density RMSE [veh/mi]", "Density RMSE"),
    ("density_mape", "density MAPE [%]", "Density MAPE"),
    ("vmt_pct_diff", "VMT % error (sim - obs)", "VMT percent error"),
    ("vht_pct_diff", "VHT % error (sim - obs)", "VHT percent error"),
    ("energy_delivered_wh", "energy delivered [Wh]", "DWC energy delivered"),
    ("peak_corridor_power_w", "peak corridor power [W]",
     "DWC peak corridor power"),
]


def _read_manifest(path: Path) -> pd.DataFrame:
    """Load and validate the study manifest."""
    manifest = pd.read_csv(path)
    required = {"distance_mi", "ramp_strategy", "validation_summary"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"manifest is missing required column(s): {sorted(missing)}. "
            f"Required: {sorted(required)}."
        )
    return manifest


def _extract(
    pattern: re.Pattern, text: str, path: Path, what: str,
) -> list[float]:
    """Pull the numbers out of the one summary line matching ``pattern``."""
    match = pattern.search(text)
    if match is None:
        raise ValueError(
            f"{path}: no {what} line found -- is this the summary written by "
            "validate_ctm_corridor.py / generate_dwc_demand.py?"
        )
    return [float(g) for g in match.groups()]


def _read_study(row: pd.Series) -> dict:
    """Parse one study's two summary files into a tidy metric row.

    ``generate_dwc_demand.py`` writes ``dwc_summary.txt`` to the sim dir,
    which is the parent of the ``validation/`` dir holding the validation
    summary -- so the manifest only has to name the latter.
    """
    validation_path = Path(row["validation_summary"])
    dwc_path = validation_path.parent.parent / "dwc_summary.txt"
    validation = validation_path.read_text()
    dwc = dwc_path.read_text()

    flow_rmse, flow_mape = _extract(
        _FLOW_ERROR, validation, validation_path, "corridor-pooled flow error",
    )
    density_rmse, density_mape = _extract(
        _DENSITY_ERROR, validation, validation_path,
        "corridor-pooled density error",
    )
    (vmt_pct_diff,) = _extract(_VMT_DIFF, validation, validation_path, "VMT")
    (vht_pct_diff,) = _extract(_VHT_DIFF, validation, validation_path, "VHT")
    (energy_wh,) = _extract(_ENERGY, dwc, dwc_path, "energy delivered")
    (peak_power_w,) = _extract(
        _PEAK_POWER, dwc, dwc_path, "peak corridor power",
    )

    return {
        "distance_mi": float(row["distance_mi"]),
        "ramp_strategy": str(row["ramp_strategy"]),
        "flow_rmse": flow_rmse,
        "flow_mape": flow_mape,
        "density_rmse": density_rmse,
        "density_mape": density_mape,
        "vmt_pct_diff": vmt_pct_diff,
        "vht_pct_diff": vht_pct_diff,
        "energy_delivered_wh": energy_wh,
        "peak_corridor_power_w": peak_power_w,
    }


def _strategy_colors(strategies: list[str]) -> dict[str, str]:
    """Stable color per ramp strategy (consistent across the line plots)."""
    palette = plt.get_cmap("tab10")
    return {s: palette(i % 10) for i, s in enumerate(sorted(set(strategies)))}


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
    for _, row in manifest.iterrows():
        print(
            f"reading {row['ramp_strategy']} @ {row['distance_mi']} mi "
            f"({row['validation_summary']})",
            file=sys.stderr,
        )
        summary_rows.append(_read_study(row))

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

    print()
    print(f"Compared {len(summary)} studies; wrote artifacts to {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
