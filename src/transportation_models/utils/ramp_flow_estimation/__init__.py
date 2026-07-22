"""Missing ramp-flow estimation.

Method-agnostic pieces (feature/corpus extraction, validation, the split
manifest) are kept separate from the estimators. Estimators share the
interface ``fit(X, r, s, ctx=None)`` / ``predict_flows(X, ctx=None)``:
Kan et al. 2021 (RF/GBM over an alpha-between-bounds formulation, with
``bounds`` holding its physical band math) and a direct ``(r, s)`` GRU.

See ``docs/ramp_flow_estimation_module.md``.
"""
from __future__ import annotations

from .bounds import (
    Bounds,
    alpha_out_of_band,
    alpha_target,
    compute_bounds,
    feasible_band,
    predicts_on_ramp,
    reconstruct_pair,
)
from .features import (
    Samples,
    TrainingData,
    build_training_data,
    mainline_flows,
    stretch_samples,
)

__all__ = [
    "Bounds",
    "alpha_out_of_band",
    "alpha_target",
    "compute_bounds",
    "feasible_band",
    "predicts_on_ramp",
    "reconstruct_pair",
    "Samples",
    "TrainingData",
    "build_training_data",
    "mainline_flows",
    "stretch_samples",
]
