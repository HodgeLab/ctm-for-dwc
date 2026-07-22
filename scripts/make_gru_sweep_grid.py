#!/usr/bin/env python3
"""Enumerate the GRU hyperparameter sweep grid, one flag-line per trial.

Writes a text file where line ``i`` holds the ``train_ramp_flow_gru.py`` flags
for slurm array task ``i``. Grid: target_transform {zscore, log1p} x
num_layers {1, 2, 3} x hidden_size {64, 128, 256} x dropout {0, 0.25, 0.5} x
beta1 {0.8, 0.9, 0.99} x weight_decay {0, 1e-4} -- pruned so num_layers=1 runs
only at dropout 0 (inter-layer dropout is undefined for a single layer):
252 trials.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

TARGET_TRANSFORMS = ["zscore", "log1p"]
NUM_LAYERS = [1, 2, 3]
HIDDEN_SIZES = [64, 128, 256]
DROPOUTS = [0.0, 0.25, 0.5]
BETA1S = [0.8, 0.9, 0.99]
WEIGHT_DECAYS = [0.0, 1e-4]


def grid() -> list[str]:
    lines = []
    for tt, nl, hs, do, b1, wd in itertools.product(
            TARGET_TRANSFORMS, NUM_LAYERS, HIDDEN_SIZES, DROPOUTS,
            BETA1S, WEIGHT_DECAYS):
        if nl == 1 and do > 0:
            continue
        lines.append(f"--target-transform {tt} --num-layers {nl} "
                     f"--hidden-size {hs} --dropout {do} --beta1 {b1} "
                     f"--weight-decay {wd}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "scripts/output/gru_sweep/sweep_grid.txt")
    args = ap.parse_args(argv)

    lines = grid()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} trials -> {args.out}")
    print(f"slurm array spec: --array=0-{len(lines) - 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
