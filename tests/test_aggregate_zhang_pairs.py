"""Tests for scripts/aggregate_zhang_pairs.py (offline pair-sweep evaluation)
over hand-built result.json fixtures matching the pairwise driver's schema."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from aggregate_zhang_pairs import METRICS, STAGES  # noqa: E402
from aggregate_zhang_pairs import main as aggregate_main  # noqa: E402


def _fake_result(out_dir, source, target, mmd_dda, dda_nrmse):
    metrics = lambda v: {"nrmse": v, "r2": 1 - v, "nbias": v / 10, "rmse": 100.0}
    out_dir.mkdir(parents=True)
    (out_dir / "result.json").write_text(json.dumps({
        "config": {"source_stretch": source, "target_stretch": target},
        "source_target_mmd": {"backbone": mmd_dda + 0.1, "dda": mmd_dda},
        "target_metrics": {
            "backbone": metrics(dda_nrmse + 0.2),
            "dda": metrics(dda_nrmse),
            "dda_mt": {
                "per_survey_day": [],
                "mean": metrics(dda_nrmse - 0.05),
                "std": {m: 0.01 for m in (*METRICS, "rmse")},
            },
        },
    }))


def test_aggregator_builds_csv_and_plots(tmp_path):
    results = tmp_path / "sweep"
    _fake_result(results / "a__b", "880_N:3", "880_N:11", mmd_dda=0.3, dda_nrmse=0.5)
    _fake_result(results / "a__c", "880_N:3", "880_N:13", mmd_dda=0.8, dda_nrmse=0.7)
    rc = aggregate_main(["--results-dir", str(results)])
    assert rc == 0

    out = results / "aggregate"
    pairs = pd.read_csv(out / "pairs.csv")
    assert len(pairs) == 2 * len(STAGES)
    dda = pairs[pairs["stage"] == "dda"]
    mt = pairs[pairs["stage"] == "dda_mt"]
    # DDA rows carry the stage value with no std; MT rows carry mean + std
    assert dda["nrmse_std"].isna().all()
    assert (mt["nrmse_std"] == 0.01).all()
    row = mt[mt["target"] == "880_N:11"].iloc[0]
    assert row["nrmse"] == 0.45 and row["mmd_dda"] == 0.3
    for metric in METRICS:
        assert (out / f"{metric}_vs_mmd.png").exists()


def test_aggregator_skips_incomplete_results(tmp_path, capsys):
    results = tmp_path / "sweep"
    _fake_result(results / "a__b", "880_N:3", "880_N:11", mmd_dda=0.3, dda_nrmse=0.5)
    bad = results / "a__c"; bad.mkdir(parents=True)
    (bad / "result.json").write_text(json.dumps({"config": {}, "arms": {}}))
    rc = aggregate_main(["--results-dir", str(results)])
    assert rc == 0
    assert "skipping" in capsys.readouterr().out
    pairs = pd.read_csv(results / "aggregate" / "pairs.csv")
    assert len(pairs) == len(STAGES)                  # only the complete pair


def test_aggregator_empty_dir_fails_cleanly(tmp_path):
    assert aggregate_main(["--results-dir", str(tmp_path)]) == 1
