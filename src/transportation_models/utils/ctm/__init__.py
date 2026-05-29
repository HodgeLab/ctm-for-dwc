"""Cell Transmission Model (CTMSIM) engine.

Implements the canonical CTMSIM state-update equations documented in
``docs/ctm_module.md`` (Kurzhanskiy 2007, UCB/EECS-2007-148, sec. 4.2.1), in
dimensional units: length [mi], time [h], speed [mi/h], flow [veh/h], density
[veh/mi], queue [veh].
"""

from .engine import StepResult, simulate, step
from .io import (
    ctmsim_demand_at_sim_steps,
    ctmsim_initial_densities,
    freeway_from_csv,
    freeway_from_ctmsim_mat,
    freeway_from_dataframe,
    freeway_to_dataframe,
    scenario_from_dataframes,
    scenario_to_dataframes,
)
from .caltrans import (
    caltrans_postmiles_for_corridor,
    extract_route_number,
    load_postmiles,
)
from .cells import cells_from_corridor
from .metrics import Metrics, compute_metrics
from .model import Cell, Freeway, FreewayArrays, Scenario
from .osm import (
    Corridor,
    MainlineSegment,
    RampJunction,
    corridor_from_graph,
    summarize_refs,
)
from .results import SimulationResult
from . import examples

__all__ = [
    "Cell",
    "Freeway",
    "FreewayArrays",
    "Scenario",
    "step",
    "StepResult",
    "simulate",
    "SimulationResult",
    "Metrics",
    "compute_metrics",
    "freeway_from_dataframe",
    "freeway_to_dataframe",
    "freeway_from_csv",
    "scenario_from_dataframes",
    "scenario_to_dataframes",
    "freeway_from_ctmsim_mat",
    "ctmsim_initial_densities",
    "ctmsim_demand_at_sim_steps",
    "Corridor",
    "MainlineSegment",
    "RampJunction",
    "corridor_from_graph",
    "cells_from_corridor",
    "summarize_refs",
    "extract_route_number",
    "load_postmiles",
    "caltrans_postmiles_for_corridor",
    "examples",
]
