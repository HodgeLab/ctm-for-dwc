#!/usr/bin/env python3
"""Rank GRU sweep trials by a validation metric from their result.json files.

Walks ``<sweep-dir>/trial_*/result.json`` (written by train_ramp_flow_gru.py),
writes a CSV ranked ascending by ``val_<metric>`` (default NRMSE -- the
raw-space metric comparable across target transforms), and prints the top of
the table. The test stretch remains untouched; evaluate only the winner on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_KNOBS = ["target_transform", "num_layers", "hidden_size", "dropout",
          "beta1", "weight_decay"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep-dir", type=Path, required=True,
                    help="Directory holding trial_*/result.json.")
    ap.add_argument("--metric", default="nrmse",
                    help="Validation metric to rank by, ascending (default: nrmse).")
    ap.add_argument("--top", type=int, default=15,
                    help="How many leading rows to print.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Ranked CSV path (default: <sweep-dir>/ranking.csv).")
    args = ap.parse_args(argv)

    rows = []
    for rj in sorted(args.sweep_dir.glob("trial_*/result.json")):
        d = json.loads(rj.read_text())
        rows.append({
            "trial": rj.parent.name,
            **{k: d["config"].get(k) for k in _KNOBS},
            **{f"val_{k}": v for k, v in d["val_metrics"].items()},
            "wandb_run_id": d.get("wandb_run_id"),
        })
    if not rows:
        print(f"no trial_*/result.json found under {args.sweep_dir}")
        return 1

    df = pd.DataFrame(rows).sort_values(f"val_{args.metric}").reset_index(drop=True)
    out = args.out if args.out is not None else args.sweep_dir / "ranking.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} trials ranked by val_{args.metric} -> {out}\n")
    show = ["trial"] + _KNOBS + [f"val_{args.metric}", "val_r2", "val_bias"]
    show = [c for c in show if c in df.columns]
    print(df[show].head(args.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
