# transportation-models

Models for automobile transportation. Code to download, process, validate, and
model **Caltrans PeMS** freeway traffic data for a study corridor (postmiles
6.7–13.7, Alameda County, both directions).

Two modeling threads:

1. **Imputation model** — a GRU (`SimpleGRU`) that fills masked flow/density
   values, trained with a combined MSE + fundamental-diagram physics loss.
   Experiments are tracked in Weights & Biases (entity `transportation-models`,
   project `PeMS-Imputation`).
2. **CTM representation** — a graph (`Freeway`) that fuses GMNS links/nodes,
   PeMS detector metadata, and postmile/boundary shapefiles into a
   Cell-Transmission-Model-ready NetworkX network, including split-ratio
   computation at diverging nodes.

The package lives in `src/transportation_models/` (installed as
`transportation-models`, imported as `transportation_models`). Python ≥ 3.10,
hatchling build.

## Layout

`utils/` is library code (import from here). `scripts/` are entrypoints and
one-offs (run these, don't import them).

| Module | Role |
|---|---|
| `utils/constants.py` | Filesystem paths, corridor bounds, and hand-curated GMNS-link→PeMS-station overrides (`NB/SB_LINK_TO_STATION_OVERRIDE`). Edit here when spatial matching picks the wrong detector. |
| `utils/data_downloading.py` | `PeMSDownloader` / `PeMSExtractor` — scrape the PeMS clearinghouse. Adapted from Seb-Good/caltrans-pems. |
| `utils/pems_settings.py` | PeMS URLs and district list for the downloader. |
| `utils/data_processing.py` | `PeMSDataProcessor` (core preprocessing pipeline) + `TargetNormalization`. Load → unit-standardize → normalize → FD calibration → timestamp deconstruction → encode/scale → sequence/imputation dataset. |
| `utils/representation.py` | `Freeway` class + helpers for OSM→GMNS conversion, link classification/direction, and split ratios. Largest module (~2.6k lines). |
| `utils/validation.py` | DataFrame structure/value checks and diagnostic plots. |
| `utils/fetch_wandb_results.py` | Pull W&B runs into a DataFrame for analysis. |
| `utils/logs.py` | `make_logger()` — timestamped file logger under `./logs/`. |
| `utils/dwpt/` | DWPT demand module: CTM `VHT` → adapted mCONV → spatiotemporal charging demand (`E`). See `docs/dwpt_demand_module.md`. |
| `utils/microsim/` | N-lane Newbolt modified-Gipps microsimulation (with lane-changing) used to validate the macroscopic DWPT pipeline (`gipps`, `lanechange`, `seeding`, `simulate`, `aggregate` (Edie), `charging`). See `docs/dwpt_validation_spec.md`. |

Typical pipeline:

```
download_pems_data.py            # fetch raw PeMS data
  → calibrate_fundamental_diagrams.py   # writes station_metadata_calibrated.csv
  → generate_long_df.py / generate_gmns_links_and_nodes.py / build_freeway.py
  → model_prototyping.py         # train the GRU
  → analyze_vds_outages.py, plot_*   # analysis / QA
```

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
- Secrets are read from a gitignored `.env`: PeMS credentials
  (`PEMS_USERNAME` / `PEMS_PASSWORD`) and `WANDB_*`. Loaded via `python-dotenv`.
- **Data is not in the repo.** Paths in `constants.py` point at an external
  drive (`/Volumes/easystore/...`); some scripts hardcode HPC paths
  (`/projects/rost5691/...`). Expect to repoint these for the current machine.
