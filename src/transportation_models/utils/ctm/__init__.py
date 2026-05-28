"""Cell Transmission Model (CTMSIM) engine.

Implements the canonical CTMSIM state-update equations documented in
``docs/ctm_module.md`` (Kurzhanskiy 2007, UCB/EECS-2007-148, sec. 4.2.1), in
dimensional units: length [mi], time [h], speed [mi/h], flow [veh/h], density
[veh/mi], queue [veh].
"""

from .engine import StepResult, step
from .model import Cell, Freeway, FreewayArrays, Scenario
from . import examples

__all__ = [
    "Cell",
    "Freeway",
    "FreewayArrays",
    "Scenario",
    "step",
    "StepResult",
    "examples",
]
