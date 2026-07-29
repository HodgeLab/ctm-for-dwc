"""Plot CTM simulation runtime and peak RSS vs corridor length.

Each row of the manifest CSV is one CTM sim of a corridor of a given
length (``simulate_ctm_corridor.py`` reports ``runtime`` and ``peak RSS``
in its ``summary.txt``). The manifest gathers those into four columns --
``length_mi``, ``n_cells``, ``runtime_s``, ``peakRSS_gb`` -- one row per
run. This renders two figures showing how each cost metric scales with
corridor length, with x-tick labels annotating both the length and the
cell count of every run.

Run from the repo root::

    python scripts/compare_ctm_runtimes.py --manifest runtimes.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_manifest(path: Path) -> pd.DataFrame:
    """Load and validate the simulation manifest, sorted by length."""
    manifest = pd.read_csv(path)
    required = {"length_mi", "n_cells", "runtime_s", "peakRSS_gb"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"manifest is missing required column(s): {sorted(missing)}. "
            f"Required: {sorted(required)}."
        )
    return manifest.sort_values("length_mi").reset_index(drop=True)


def _plot_vs_length(
    manifest: pd.DataFrame, *,
    metric: str, ylabel: str, title: str, out_path: Path,
) -> None:
    """One cost metric vs corridor length, one marker per run.

    x ticks sit at each run's length and are labelled with both the
    corridor length and its cell count so the two axes of scale read
    together.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        manifest["length_mi"], manifest[metric],
        marker="o", color="tab:blue",
    )
    ax.set_xticks(manifest["length_mi"])
    ax.set_xticklabels([
        f"{length:g} mi\n{cells:g} cells"
        for length, cells in zip(manifest["length_mi"], manifest["n_cells"])
    ])
    ax.set_xlabel("corridor length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest", required=True, type=Path,
        help=("CSV listing the simulations to compare.")
    )
    parser.add_argument(
        "--out-dir", default=None, type=Path,
        help=("Directory for comparison plots.")
    )
    args = parser.parse_args()

    manifest = _read_manifest(args.manifest)

    out_dir = (
        args.out_dir if args.out_dir is not None
        else args.manifest.parent / "runtime_comparison"
    )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_vs_length(
        manifest, metric="runtime_s",
        ylabel="runtime [s]", title="CTM simulation runtime vs corridor length",
        out_path=out_dir / "runtime_vs_length.png",
    )
    _plot_vs_length(
        manifest, metric="peakRSS_gb",
        ylabel="peak RSS [GB]", title="CTM simulation peak RSS vs corridor length",
        out_path=out_dir / "peakRSS_vs_length.png",
    )

    print(f"Compared {len(manifest)} runs; wrote plots to {out_dir}")


if __name__ == "__main__":
    main()