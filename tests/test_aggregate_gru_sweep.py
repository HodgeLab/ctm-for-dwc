"""Tests for scripts/aggregate_gru_sweep.py (sweep ranking)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from aggregate_gru_sweep import main  # noqa: E402


def _write_result(sweep_dir, trial, nrmse, **config):
    d = sweep_dir / trial
    d.mkdir(parents=True)
    cfg = {"target_transform": "zscore", "num_layers": 1, "hidden_size": 64,
           "dropout": 0.0, "beta1": 0.9, "weight_decay": 0.0, **config}
    (d / "result.json").write_text(json.dumps({
        "config": cfg,
        "val_metrics": {"nrmse": nrmse, "rmse": nrmse * 100, "r2": 1 - nrmse,
                        "bias": 0.0, "nbias": 0.0},
        "wandb_run_id": f"run-{trial}",
    }))


def test_ranks_trials_ascending_by_val_nrmse(tmp_path):
    _write_result(tmp_path, "trial_0", 0.9)
    _write_result(tmp_path, "trial_1", 0.3, target_transform="log1p")
    _write_result(tmp_path, "trial_2", 0.6)
    rc = main(["--sweep-dir", str(tmp_path)])
    assert rc == 0
    df = pd.read_csv(tmp_path / "ranking.csv")
    assert list(df["trial"]) == ["trial_1", "trial_2", "trial_0"]
    assert df.iloc[0]["target_transform"] == "log1p"
    assert "val_nrmse" in df.columns and "wandb_run_id" in df.columns


def test_empty_sweep_dir_returns_error(tmp_path):
    assert main(["--sweep-dir", str(tmp_path)]) == 1
