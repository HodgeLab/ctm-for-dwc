"""Step 7 Level 3: build the inverse-CTM QP input bundle (Python prep).

The model-based ramp-flow imputer (`julia/ramp_qp/solve_ramp_qp.jl`) solves a
convex QP that fits the CTM's mainline density to observed PeMS density, with
on-ramp admitted flow ``r_i(k)`` and off-ramp flow ``s_i(k)`` as the free
variables (see the Level 3 subsection of docs/ctm_module.md). This script
prepares the solver's inputs from the existing Python pipeline and writes a
self-contained CSV/JSON bundle the Julia side consumes -- no runtime
Python<->Julia bridge.

Bundle contents (written to ``--out-dir``):

* ``freeway.csv``          -- copied Step-6 per-cell FD params.
* ``rho0.csv``             -- ``cell, rho0`` initial density [veh/mi].
* ``inflow.csv``           -- ``k, inflow`` upstream boundary [veh/h], length T.
* ``observed_density.csv`` -- ``cell, m_5min, rho_obs`` for direct/tiebreak
                              cells only, NaN samples dropped.
* ``meta.json``            -- dt, horizon, 5-min stride, ramp structure +
                              capacities, direct/tiebreak cells, gamma/xi.

Run from the repo root::

    python scripts/build_ctm_qp_inputs.py \\
        --freeway scripts/output/ctm_corridor/<case>/freeway.csv \\
        --cells   scripts/output/ctm_corridor/<case>/cells.csv \\
        --timeseries-dir data/pems/csv_files \\
        --start "2022-04-12 06:00" --end "2022-04-12 09:00"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from transportation_models.utils.ctm import (
    compute_cfl_advisory,
    format_cfl_advisory,
    inflow_from_vds,
    initial_state_from_vds,
    restrict_to_direct_tiebreak,
)
# Reused for the exact observed-density definition (total_flow*12/avg_speed)
# and contiguity check -- the same extraction Step-9 validation scores against.
from transportation_models.utils.ctm.validation import _load_observed_window

_FIVE_MIN_S = 300.0


def _divides_5min(dt_s: float) -> bool:
    """True iff ``dt_s`` evenly divides 5 minutes (integer steps per 5-min)."""
    ratio = _FIVE_MIN_S / dt_s
    return abs(ratio - round(ratio)) < 1e-9 and round(ratio) >= 1


def choose_dt_seconds(
    freeway_df: pd.DataFrame, explicit: float | None,
) -> tuple[float, object]:
    """Pick the QP sim ``dt`` [s]: the largest CFL-stable 5-min divisor.

    Any ``dt = 300/n`` (integer ``n``) aligns the sim grid to the PeMS 5-min
    grid; we take the smallest ``n`` (largest ``dt``) strictly below the CFL
    bound to minimise the step count ``T``. ``--dt-seconds`` overrides, but is
    validated against both the CFL bound and the 5-min-divisor rule.
    """
    advisory = compute_cfl_advisory(freeway_df)
    bound_s = advisory.binding_dt_max_s
    if explicit is not None:
        if explicit > bound_s + 1e-9:
            raise SystemExit(
                f"--dt-seconds {explicit} exceeds the CFL bound "
                f"{bound_s:.4g} s (binding cell {advisory.binding_cell_idx}); "
                "pick a smaller dt."
            )
        if not _divides_5min(explicit):
            raise SystemExit(
                f"--dt-seconds {explicit} must evenly divide 5 minutes "
                "(300 s) so the sim grid aligns with PeMS."
            )
        return float(explicit), advisory
    n = math.ceil(_FIVE_MIN_S / bound_s)
    if _FIVE_MIN_S / n >= bound_s:  # keep dt strictly below the CFL bound
        n += 1
    return _FIVE_MIN_S / n, advisory


def _capacity_or_none(value: object) -> float | None:
    """Map a freeway-table ramp capacity to a JSON value (None == no bound)."""
    if value is None:
        return None
    f = float(value)
    return None if (math.isnan(f) or math.isinf(f)) else f


def build_bundle(
    freeway_df: pd.DataFrame,
    cells_df: pd.DataFrame,
    timeseries_dir: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dt_s: float,
    upstream_vds: int,
    out_dir: Path,
) -> dict:
    """Assemble + write the QP bundle; return the meta dict (also serialized)."""
    if len(freeway_df) != len(cells_df):
        raise SystemExit(
            f"freeway.csv has {len(freeway_df)} rows but cells.csv has "
            f"{len(cells_df)}; they must be one row per cell in the same order."
        )
    n_cells = len(cells_df)
    dt_h = dt_s / 3600.0
    steps_per_5min = int(round(_FIVE_MIN_S / dt_s))

    total_s = (end - start).total_seconds()
    if total_s <= 0 or abs(total_s % _FIVE_MIN_S) > 1e-6:
        raise SystemExit(
            f"window [{start}, {end}) must be a positive whole number of "
            "5-minute intervals."
        )
    n_5min = int(round(total_s / _FIVE_MIN_S))
    T = n_5min * steps_per_5min

    # Initial density (all cells) and upstream boundary inflow (length T).
    rho0 = initial_state_from_vds(cells_df, timeseries_dir, at_time=start)
    inflow = inflow_from_vds(
        timeseries_dir / f"{upstream_vds}.csv", dt=dt_h, start=start, end=end,
    )

    # Observed density at direct/tiebreak cells only (one cell per unique VDS).
    restricted = restrict_to_direct_tiebreak(cells_df)
    scored_cells = [
        i for i, v in enumerate(restricted["vds_id"].to_list()) if not pd.isna(v)
    ]
    cache: dict[int, pd.DataFrame] = {}
    obs_rows: list[dict] = []
    for cell_idx in scored_cells:
        vds_id = int(restricted["vds_id"].iloc[cell_idx])
        _flow, density = _load_observed_window(
            timeseries_dir, vds_id, start=start, end=end,
            n_5min=n_5min, cache=cache,
        )
        for m, rho_obs in enumerate(density):
            if not np.isnan(rho_obs):
                obs_rows.append({"cell": cell_idx, "m_5min": m, "rho_obs": rho_obs})

    # Ramp structure + capacities, keyed by cell index (None == no bound/inf).
    on_ramp_cells, off_ramp_cells = [], []
    on_cap, off_cap = {}, {}
    for i in range(n_cells):
        if bool(freeway_df["on_ramp"].iloc[i]):
            on_ramp_cells.append(i)
            on_cap[i] = _capacity_or_none(freeway_df["on_ramp_capacity"].iloc[i])
        if bool(freeway_df["off_ramp"].iloc[i]):
            off_ramp_cells.append(i)
            off_cap[i] = _capacity_or_none(freeway_df["off_ramp_capacity"].iloc[i])

    out_dir.mkdir(parents=True, exist_ok=True)
    freeway_df.to_csv(out_dir / "freeway.csv", index=False)
    pd.DataFrame({"cell": range(n_cells), "rho0": rho0}).to_csv(
        out_dir / "rho0.csv", index=False,
    )
    pd.DataFrame({"k": range(T), "inflow": inflow}).to_csv(
        out_dir / "inflow.csv", index=False,
    )
    pd.DataFrame(
        obs_rows, columns=["cell", "m_5min", "rho_obs"],
    ).to_csv(out_dir / "observed_density.csv", index=False)

    meta = {
        "dt_s": dt_s,
        "dt_h": dt_h,
        "steps_per_5min": steps_per_5min,
        "n_5min": n_5min,
        "T": T,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_cells": n_cells,
        "gamma": 1.0,
        "xi": 1.0,
        "upstream_vds": int(upstream_vds),
        "on_ramp_cells": on_ramp_cells,
        "off_ramp_cells": off_ramp_cells,
        "on_ramp_capacity": {str(k): v for k, v in on_cap.items()},
        "off_ramp_capacity": {str(k): v for k, v in off_cap.items()},
        "direct_tiebreak_cells": scored_cells,
        "n_observed_samples": len(obs_rows),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeway", required=True, type=Path)
    parser.add_argument("--cells", required=True, type=Path)
    parser.add_argument("--timeseries-dir", required=True, type=Path)
    parser.add_argument("--start", required=True, type=pd.Timestamp)
    parser.add_argument("--end", required=True, type=pd.Timestamp)
    parser.add_argument(
        "--dt-seconds", default=None, type=float,
        help="Sim dt in seconds. Default: largest CFL-stable 5-min divisor.",
    )
    parser.add_argument(
        "--upstream-vds", default=None, type=int,
        help="Upstream boundary VDS. Default: cells.csv first row's vds_id.",
    )
    parser.add_argument(
        "--out-dir", default=None, type=Path,
        help="Bundle directory. Default: <freeway parent>/qp_inputs.",
    )
    args = parser.parse_args()

    freeway_df = pd.read_csv(args.freeway)
    cells_df = pd.read_csv(args.cells)

    dt_s, advisory = choose_dt_seconds(freeway_df, args.dt_seconds)
    print(format_cfl_advisory(advisory), file=sys.stderr)

    if args.upstream_vds is not None:
        upstream_vds = args.upstream_vds
    else:
        raw = cells_df.iloc[0]["vds_id"]
        if pd.isna(raw):
            raise SystemExit(
                "cells.csv first row has no vds_id for the upstream boundary; "
                "pass --upstream-vds."
            )
        upstream_vds = int(raw)

    out_dir = (
        args.out_dir if args.out_dir is not None
        else args.freeway.resolve().parent / "qp_inputs"
    )

    meta = build_bundle(
        freeway_df, cells_df, args.timeseries_dir,
        start=args.start, end=args.end, dt_s=dt_s,
        upstream_vds=upstream_vds, out_dir=out_dir,
    )

    print(
        f"\nQP bundle written to {out_dir}\n"
        f"  dt            : {meta['dt_s']:.4g} s ({meta['steps_per_5min']} "
        f"steps / 5 min)\n"
        f"  horizon       : {meta['n_5min']} x 5 min = {meta['T']} sim steps\n"
        f"  cells         : {meta['n_cells']} "
        f"({len(meta['on_ramp_cells'])} on-ramp, "
        f"{len(meta['off_ramp_cells'])} off-ramp)\n"
        f"  scored cells  : {len(meta['direct_tiebreak_cells'])} "
        f"direct/tiebreak\n"
        f"  observed pts  : {meta['n_observed_samples']} (non-NaN 5-min "
        "density samples)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
