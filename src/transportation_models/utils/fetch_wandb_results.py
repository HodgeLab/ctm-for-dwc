"""Fetch model training results from Weights & Biases for analysis."""

import os

import pandas as pd
import wandb
from dotenv import load_dotenv

WANDB_ENTITY = "transportation-models"
WANDB_PROJECT = "PeMS-Imputation"


def fetch_runs(
    entity: str = WANDB_ENTITY,
    project: str = WANDB_PROJECT,
    filters: dict | None = None,
) -> pd.DataFrame:
    """Fetch all runs from a W&B project and return as a DataFrame.

    Each row contains the run metadata, full config, and summary metrics
    (which include the final-epoch values for all logged metrics).

    Args:
        entity: W&B entity (team or user).
        project: W&B project name.
        filters: Optional MongoDB-style filter dict passed to the W&B API
            (e.g. ``{"state": "finished"}``).

    Returns:
        DataFrame with one row per run. Columns are prefixed:
        - ``config/`` for hyperparameters
        - ``summary/`` for summary metrics
        - unprefixed for run metadata (name, state, tags, etc.)
    """
    load_dotenv()
    api = wandb.Api(overrides={"entity": entity, "project": project})

    runs = api.runs(f"{entity}/{project}", filters=filters or {})

    records = []
    for run in runs:
        record = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "tags": run.tags,
            "created_at": run.created_at,
            "notes": run.notes,
            "url": run.url,
        }
        for k, v in run.config.items():
            if not k.startswith("_"):
                record[f"config/{k}"] = v
        for k, v in run.summary.items():
            if not k.startswith("_"):
                record[f"summary/{k}"] = v
        records.append(record)

    return pd.DataFrame(records)


def best_flow_rmse_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return runs ranked by best (minimum) validation flow RMSE.

    Includes all config columns so you can see how hyperparameters relate
    to performance.
    """
    metric_col = "summary/val_flow_rmse"
    if metric_col not in df.columns:
        raise KeyError(
            f"'{metric_col}' not found in results. "
            f"Available summary columns: {[c for c in df.columns if c.startswith('summary/')]}"
        )

    config_cols = [c for c in df.columns if c.startswith("config/")]
    keep_cols = [
        "run_name",
        "state",
        "created_at",
        metric_col,
        "summary/val_flow_mae",
        "summary/val_flow_mape",
        "summary/validation_loss",
        *config_cols,
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    result = df[keep_cols].copy()
    result[metric_col] = pd.to_numeric(result[metric_col], errors="coerce")
    return result.sort_values(metric_col).reset_index(drop=True)


if __name__ == "__main__":
    df = fetch_runs(filters={"state": "finished"})
    print(f"Fetched {len(df)} finished runs.\n")

    if len(df) == 0:
        print("No finished runs found.")
    else:
        summary = best_flow_rmse_summary(df)
        print(summary.to_string(index=False))
