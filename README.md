# transportation-models

Code to download, process, and calibrate
**Caltrans PeMS** freeway traffic data, assemble it into a Cell Transmission
Model (CTM) of a study corridor, simulate that corridor, and convert the result
into **dynamic wireless charging (DWC)** charging demand.

The pipeline, end to end:

```
PeMS data -> calibrated CTM corridor -> ramp flows -> CTM simulation -> DWC demand
```

The package lives in `src/transportation_models/` (installed as
`transportation-models`, imported as `transportation_models`). Python >= 3.10,
hatchling build.

## Layout

`utils/` is library code (import from here). `scripts/` are entrypoints and
one-offs (run these, don't import them).

| Module | Role |
|---|---|
| `utils/ctm/` | The core module: CTM data model, engine, and the whole OSM-to-simulation pipeline. See `docs/ctm_module.md`. |
| `utils/dwc/` | DWC demand module: CTM `VHT` -> adapted mCONV -> spatiotemporal charging demand (`E`). See `docs/dwc_demand_module.md`. |
| `utils/ramp_flow_estimation/` | Estimating missing ramp flows from mainline data (feature extraction, the GRU estimator, split manifest, scoring). See `docs/ramp_flow_estimation_module.md`. |
| `utils/data_processing.py` | `PeMSDataProcessor` (preprocessing + fundamental-diagram calibration) and `TargetNormalization`. |
| `utils/data_downloading.py` | `PeMSDownloader` / `PeMSExtractor` — scrape the PeMS clearinghouse. Adapted from Seb-Good/caltrans-pems. |
| `utils/pems_settings.py` | PeMS URLs and district list for the downloader. |
| `utils/validation.py` | DataFrame structure/value checks and diagnostic plots. |
| `utils/plot_style.py` | Shared matplotlib style for paper figures (`PAPER_RC`, `COLUMN_W`, `PAPER_DPI`). |
| `utils/constants.py` | Filesystem paths to the local PeMS data store. |
| `utils/logs.py` | `make_logger()` — timestamped file logger under `./logs/`. |

Inside `utils/ctm/`, the pipeline stages map to modules: `osm.py` + `cells.py`
(Step 1), `vds.py` (Step 2), `assembly.py` (Step 6), `ramp_fill.py` +
`scenario.py` (Steps 7-8), `model.py` + `engine.py` + `results.py` (the
simulator), `validation.py` + `metrics.py` (Step 9), and `plots.py` for
figures. `diffsim.py` is a differentiable (PyTorch) reimplementation of the
engine step, used to optimize ramp flows.

## Pipeline

Steps are numbered as in [docs/ctm_module.md](docs/ctm_module.md), and most
scripts name their step in the first line of their docstring.

**Corridor build-out** (done once per case study; dormant between new corridors):

```
list_osm_refs.py                 # pick the --ref for the next step
build_ctm_from_osm.py            # Steps 1+2: OSM -> Corridor -> cell table, VDS -> cell
download_pems_station_metadata.py
download_pems_timeseries.py      # Step 3
calibrate_fundamental_diagrams.py  # Step 4: writes station_metadata_calibrated.csv
regenerate_cell_artifacts.py     # Step 5 helper: after manual edits to cells.csv
assemble_ctm_freeway.py          # Step 6: -> freeway.csv
```

Assembled corridors live in `case_studies/<corridor>/`.

**Ramp flow estimation** (Step 7) — mainline detectors are well covered, ramps
are not, so the missing ramp flows are estimated and then refined:

```
build_pems_stretches.py          # the stretch corpus the estimator trains on
train_ramp_flow_gru.py           # GRU estimator -> checkpoint
build_ctm_ramp_scenario.py       # --estimator gru -> demand.csv, beta.csv
build_ctm_diffsim_inputs.py      # -> <case>/diffsim_inputs/ bundle
augment_observed_grid.py         # impute unobserved cells of the bundle's grids
optimize_ramp_flows_diffsim.py   # refine the ramp profiles through the exact CTM
```

The GRU result seeds the optimizer; `optimize_ramp_flows_diffsim.py` produces
the ramp flows used for simulation.

**Simulation and demand** (Steps 8-10):

```
simulate_ctm_corridor.py         # Step 8
validate_ctm_corridor.py         # Step 9: score against historical PeMS
generate_dwc_demand.py          # CTM result -> DWC charging demand + plots
compare_ctm_case_studies.py      # Step 10: compare across corridor lengths
```

Demos and profiling: `run_ctm_demo.py` (standalone, no data required),
`run_ctm_ctmsim_demo.py` (from a CTMSIM `.mat` config),
`run_dwc_demand_demo.py`, `profile_ctm_dwc_pipeline.py`,
`benchmark_dwc_convolution.py`, `analyze_dwc_temporal_aggregation.py`.

### FD calibration outcome codes

`calibrate_fundamental_diagrams.py` tags every detector with a
`calibration_code` (int) + `calibration_status` (string) column in the
calibrated metadata, so failed fits are flagged rather than silently dropped.
Only `0` is a success; any non-zero code leaves the FD parameter columns NaN.
The same scheme is mirrored onto `ramp_metadata_calibrated.csv`.

| code | status | meaning |
|---|---|---|
| 0 | `ok` | params computed and passed validation |
| 1 | `missing_timeseries` | timeseries CSV absent / unreadable |
| 2 | `no_data_after_filter` | no rows survived the `pct_observed` threshold or the IQR filter |
| 3 | `no_congestion_points` | too few points above the critical-density seed to fit the congestion branch (mainline only) |
| 4 | `validation_failed` | fit rejected by `validate_fd_params` (non-positive speeds/capacity, `w >= v_f`, bad density ordering, or triangle inconsistency) |

Validation strictness is tunable from the CLI: `--iqr-multiplier` (row-level
outlier filter), `--bin-iqr-multiplier` and `--bin-size` (congestion-branch
binning), and `--triangle-rtol` (triangle-consistency tolerance).

## Conventions

- Every file opens with a module docstring describing its role; public
  functions use NumPy-style docstrings.
- Scripts use `# %%` cell markers (Jupyter / VS Code interactive) and configure
  logging to stdout inline; library modules use `logging.getLogger(__name__)`.
- Preprocessing intentionally **preserves a contiguous 5-minute grid** — gaps
  become NaN rows rather than being dropped. Don't "clean" these out.
- `pct_observed` is the data-quality field; `PCT_OBSERVED_THRESHOLD` gates
  missing / low-quality intervals.

## Running

- Tests: `pytest` (config in `pyproject.toml`; `testpaths=["tests"]`). Install
  test dependencies via the `test` extra: `pip install -e ".[test]"`.
  Network-hitting and slow tests are deselected by default; run them with
  `pytest -m network` / `pytest -m slow`.
- Secrets are read from a gitignored `.env`: PeMS credentials
  (`PEMS_USERNAME` / `PEMS_PASSWORD`) and `WANDB_*`. Loaded via `python-dotenv`.
- **Data is not in the repo.** Paths in `constants.py` point at an external
  drive (`/Volumes/easystore/...`); some scripts hardcode HPC paths
  (`/projects/rost5691/...`). Expect to repoint these for the current machine.
- `requirements.txt` is a full `pip freeze` snapshot and is not the source of
  truth for direct dependencies — `pyproject.toml` is.
