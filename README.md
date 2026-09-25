# ctm-for-dwc

This repository contains code to accompany the paper "Scalable Load Models for Dynamic Wireless Charging of Electric Vehicles in Power Systems." The project uses **Caltrans PeMS** data to calibrate a **Cell Transmission Model (CTM)** for freeway traffic flow simulation, which feeds a projection of the energy demand from **dynamic wireless charging (DWC)** of electric vehicles on the freeway.


## Layout

Library code lives in `src/ctm_for_dwc/`. 
Entrypoint scripts for the library live in `scripts/`.

| Module | Role |
|---|---|
| `ctm/` | The CTM module: model construction, calibration, and simulation engine. See `docs/ctm_module.md`. |
| `dwc/` | DWC demand module: CTM results -> DWC demand. See `docs/dwc_demand_module.md`. |
| `pems/` | PeMS clearinghouse access: `PeMSDownloader` (download files) and `PeMSExtractor` (`.gz` dumps → per-VDS CSVs). Downloader adapted from [Seb-Good/caltrans-pems](https://github.com/Seb-Good/caltrans-pems). |
| `plot_style.py` | Shared matplotlib style for paper figures (`PAPER_RC`, `COLUMN_W`, `PAPER_DPI`). |


## Pipeline

Steps are numbered as in [docs/ctm_module.md](docs/ctm_module.md), and most
scripts name their step in the first line of their docstring.

**Corridor build-out** (done once per case study; dormant between new corridors):

```
list_osm_refs.py                 # pick the --ref for the next step
build_ctm_from_osm.py            # Steps 1+2: OSM -> Corridor -> cell table, VDS -> cell
download_pems_station_metadata.py
download_pems_timeseries.py      # Step 3: clearinghouse -> .gz files
extract_pems_timeseries.py       # Step 3: .gz files -> one CSV per VDS
calibrate_fundamental_diagrams.py  # Step 4: writes station_metadata_calibrated.csv
regenerate_cell_artifacts.py     # Step 5 helper: after manual edits to cells.csv
assemble_ctm_freeway.py          # Step 6: -> freeway.csv
```

Assembled corridors live in `case_studies/<corridor>/`.

**Ramp flows** (Step 7) — mainline detectors are well-covered in PeMS, but ramps are not,
so a starting ramp profile is built from the data and then refined through the
CTM:

```
build_pems_stretches.py          # stretch table (ramp configuration types)
build_ctm_ramp_scenario.py       # ramp flow imputation seeding -> demand.csv, beta.csv
build_ctm_diffsim_inputs.py      # bundled inputs for model-based ramp flow calibration
augment_observed_grid.py         # impute unobserved cells of the bundle's grids (optional)
optimize_ramp_flows_diffsim.py   # model-based calibration of ramp profiles
```

`build_ctm_ramp_scenario.py` output seeds the optimizer (absent ramps with no
conservation relation start at a fraction of mainline flow, and the optimizer
floors every starting demand above 0); `optimize_ramp_flows_diffsim.py`
produces the ramp flows used for simulation.

**Simulation and demand** (Steps 8-10):

```
simulate_ctm_corridor.py         # Step 8
validate_ctm_corridor.py         # Step 9: score against historical PeMS
generate_dwc_demand.py           # Step 10: CTM result -> DWC demand + plots
```

Demos and profiling: `run_ctm_demo.py` (standalone, no data required),
`run_ctm_ctmsim_demo.py` (from a CTMSIM `.mat` config),
`run_dwc_demand_demo.py`, `profile_ctm_dwc_pipeline.py`,
`benchmark_dwc_convolution.py`, `analyze_dwc_temporal_aggregation.py`.



## Conventions

- Every file opens with a module docstring describing its role; public
  functions use NumPy-style docstrings.
- Scripts are `argparse` CLIs with a `main()` behind an
  `if __name__ == "__main__":` guard and print progress to stderr. Library modules that log
  use `logging.getLogger(__name__)` (except `PeMSDownloader`, which has its
  own stdout `scraper` logger); scripts that want those records on screen call
  `logging.basicConfig`, as `assemble_ctm_freeway.py` and
  `calibrate_fundamental_diagrams.py` do.
- PeMS data preprocessing intentionally **preserves a contiguous 5-minute grid** — gaps
  become NaN rows rather than being dropped. Don't "clean" these out.
- `pct_observed` is the data-quality field; `PCT_OBSERVED_THRESHOLD` gates
  missing / low-quality intervals.

## Running

**Install** (Python 3.10 or newer), from the repo root:

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

This installs the package in editable mode, so code changes take effect without
reinstalling. The dependencies come from `pyproject.toml`; `requirements.txt` is
only a `pip freeze` snapshot of one working environment.

**Run a script** from the repo root. This demo needs no data and finishes in a
few seconds:

```
python scripts/run_ctm_demo.py
```

Every other script takes `--help` to list its options:

```
python scripts/simulate_ctm_corridor.py --help
```

**Data and credentials.** Data is not in the repo. Scripts read and write under
the gitignored `data/` directory by default; point their path flags (such as
`--timeseries-dir`) elsewhere if your data lives somewhere else. Scripts that
download from PeMS need an account: put `PEMS_USERNAME` and `PEMS_PASSWORD` in a
`.env` file at the repo root (it is gitignored). `optimize_ramp_flows_diffsim.py`
logs to Weights & Biases; pass `--no-wandb` if you don't use it.

**Tests:** run `pytest`. Tests that use the network or run slowly are skipped by
default; run them with `pytest -m network` or `pytest -m slow`.
