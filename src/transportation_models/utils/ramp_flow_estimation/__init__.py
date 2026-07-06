"""Missing ramp-flow estimation.

Named generally so alternative methods can be added over time; the first
implementation is Kan et al. 2021 (RF/GBM over an alpha-between-bounds
formulation). Method-agnostic pieces (``bounds``, feature extraction,
validation) are kept separate from the Kan-specific estimator.

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
from .features import Samples, TrainingData, build_training_data, stretch_samples

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
    "stretch_samples",
]
