"""DWPT demand module.

Adapts the mCONV method of Newbolt 2024a to consume macroscopic CTM
outputs and produce a spatiotemporal DWPT demand profile.

- Spec: ``docs/dwpt_demand_module.md``
- Plan: ``docs/dwpt_demand_implementation_plan.md``
"""

from .model import CorridorSpec, PadSpec

__all__ = ["CorridorSpec", "PadSpec"]
