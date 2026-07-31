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
    DEFAULT_POSTMILE_FILE,
    caltrans_postmiles_for_corridor,
    download_caltrans_postmiles,
    extract_route_number,
    load_postmiles,
)
from .assembly import (
    CFLAdvisory,
    FREEWAY_COLUMNS,
    assemble_freeway_table,
    compute_cfl_advisory,
    format_cfl_advisory,
    parse_ramp_vds_ids,
)
from .cells import (
    cells_from_corridor,
    cells_to_geodataframe,
    corridor_from_artifacts,
    corridor_to_artifacts,
    flag_cell_length_warnings,
    mainline_segments_to_geodataframe,
    ramp_junctions_to_geodataframe,
)
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
from .ramp_flow import (
    gap_aware_fill,
    historical_average_fill,
    persistence_fill,
    stochastic_historical_fill,
)
from .ramp_validation import ramp_fill_accuracy
from .scenario import (
    FillStrategy,
    beta_from_off_ramp_vds,
    demand_from_ramp_vds,
    inflow_from_vds,
    initial_state_from_vds,
)
from .validation import (
    CorridorAggregates,
    CTMSIMValidation,
    GEHResult,
    ObservedGrid,
    QQCellSamples,
    QQResult,
    compare_against_ctmsim,
    compare_against_historical,
    compare_corridor_aggregates,
    compute_flow_geh,
    compute_observed_grid,
    compute_qq_samples,
    corridor_rmse_mape,
    restrict_to_direct_tiebreak,
)
from .vds import (
    DEFAULT_PEMS_DIR,
    assign_ramp_vds_to_cells,
    assign_vds_to_cells,
    download_pems_station_metadata,
    load_pems_station_metadata,
    project_vds_to_corridor,
)
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
    "cells_to_geodataframe",
    "corridor_from_artifacts",
    "corridor_to_artifacts",
    "flag_cell_length_warnings",
    "mainline_segments_to_geodataframe",
    "ramp_junctions_to_geodataframe",
    "FREEWAY_COLUMNS",
    "assemble_freeway_table",
    "CFLAdvisory",
    "compute_cfl_advisory",
    "format_cfl_advisory",
    "parse_ramp_vds_ids",
    "inflow_from_vds",
    "initial_state_from_vds",
    "FillStrategy",
    "demand_from_ramp_vds",
    "beta_from_off_ramp_vds",
    "persistence_fill",
    "historical_average_fill",
    "stochastic_historical_fill",
    "gap_aware_fill",
    "ramp_fill_accuracy",
    "compare_against_historical",
    "restrict_to_direct_tiebreak",
    "CTMSIMValidation",
    "compare_against_ctmsim",
    "CorridorAggregates",
    "compare_corridor_aggregates",
    "GEHResult",
    "compute_flow_geh",
    "QQCellSamples",
    "QQResult",
    "compute_qq_samples",
    "ObservedGrid",
    "compute_observed_grid",
    "corridor_rmse_mape",
    "summarize_refs",
    "extract_route_number",
    "load_postmiles",
    "caltrans_postmiles_for_corridor",
    "download_caltrans_postmiles",
    "DEFAULT_POSTMILE_FILE",
    "load_pems_station_metadata",
    "project_vds_to_corridor",
    "assign_vds_to_cells",
    "assign_ramp_vds_to_cells",
    "download_pems_station_metadata",
    "DEFAULT_PEMS_DIR",
    "examples",
]
