"""Missing ramp-flow estimation.

Method-agnostic pieces (feature/corpus extraction, validation, the split
manifest) are kept separate from the estimator. The estimator is a direct
``(r, s)`` GRU exposing ``fit(X, r, s, ctx=None)`` /
``predict_flows(X, ctx=None)``.

See ``docs/ramp_flow_estimation_module.md``.
"""
from __future__ import annotations

from .features import (
    Samples,
    TrainingData,
    build_training_data,
    mainline_flows,
    stretch_samples,
)

__all__ = [
    "Samples",
    "TrainingData",
    "build_training_data",
    "mainline_flows",
    "stretch_samples",
]
