"""Step 7 Level 3: verify the inverse-CTM QP solution by forward round-trip.

The QP (`julia/ramp_qp/solve_ramp_qp.jl`) relaxes the CTM `min` operators to
`<=`, so its fitted trajectory only equals a true CTM run when those
relaxations are *tight*. This script closes the loop: it feeds the solver's
`demand.csv` / `beta.csv` back through the exact forward simulator (γ=1, the
engine default) and reports

  1. the **round-trip density gap** -- max/RMS difference between the
     forward-sim density and the QP's fitted density (the global tightness
     symptom), and
  2. a **per-operator relaxation-tightness map** -- for each cell/step, the
     slack between the QP's mainline flow `f` and the smallest of its CTM upper
     bounds (sending, downstream receiving, capacity). Slack ≈ 0 means tight
     (the engine would reproduce that flow); slack > 0 localizes where a future
     MIQP pass would add a big-M binary.

Run from the repo root after the Julia solver has written its outputs into the
bundle dir::

    python scripts/verify_ramp_qp.py --bundle scripts/output/ctm_corridor/<case>/qp_inputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from transportation_models.utils.ctm import (
    freeway_from_dataframe,
    scenario_from_dataframes,
    simulate,
)


def _wide(path: Path, n_cells: int) -> np.ndarray:
    """Read a wide T×N solver CSV (string cell-index headers) as (n_cells, T)."""
    df = pd.read_csv(path)
    df.columns = df.columns.astype(int)
    return df.reindex(columns=range(n_cells), fill_value=0.0).to_numpy(dtype=float).T


def round_trip(bundle: Path) -> dict:
    """Forward-simulate the QP outputs and compare to the fitted trajectory."""
    meta = json.loads((bundle / "meta.json").read_text())
    dt_h = float(meta["dt_h"])
    n_cells = int(meta["n_cells"])
    gamma = float(meta["gamma"])

    freeway_df = pd.read_csv(bundle / "freeway.csv")
    freeway = freeway_from_dataframe(freeway_df, dt=dt_h)

    rho0 = pd.read_csv(bundle / "rho0.csv")["rho0"].to_numpy(dtype=float)
    inflow = pd.read_csv(bundle / "inflow.csv")["inflow"].to_numpy(dtype=float)

    demand = pd.read_csv(bundle / "demand.csv")
    beta = pd.read_csv(bundle / "beta.csv")
    demand.columns = demand.columns.astype(int)
    beta.columns = beta.columns.astype(int)

    scenario = scenario_from_dataframes(
        freeway, inflow=inflow, demand=demand, beta=beta, rho0=rho0, q0=0.0,
    )
    result = simulate(freeway, scenario)

    # --- (1) round-trip density gap ---
    rho_fwd = result.density                                    # (n_cells, T+1)
    rho_qp = pd.read_csv(bundle / "rho_fitted.csv").to_numpy(dtype=float).T
    gap = rho_fwd - rho_qp
    density_gap_max = float(np.nanmax(np.abs(gap)))
    density_gap_rms = float(np.sqrt(np.nanmean(gap**2)))

    # --- (2) per-operator relaxation tightness ---
    l = freeway_df["length"].to_numpy(dtype=float)
    v_f = freeway_df["v_f"].to_numpy(dtype=float)
    w = freeway_df["w"].to_numpy(dtype=float)
    rho_jam = freeway_df["rho_jam"].to_numpy(dtype=float)
    q_max = freeway_df["q_max"].to_numpy(dtype=float)

    f_qp = _wide(bundle / "mainline_flow_fitted.csv", n_cells)     # (n_cells, T)
    s_qp = _wide(bundle / "off_ramp_flow_fitted.csv", n_cells)
    r_qp = _wide(bundle / "demand.csv", n_cells)
    rho_state = rho_qp[:, :-1]                                     # states at k=0..T-1

    # gamma=1 on-ramp-blended density, per cell/step.
    rho_eff = rho_state + gamma * r_qp * (dt_h / l)[:, None]
    bound_send = v_f[:, None] * rho_eff - s_qp
    bound_qmax = np.broadcast_to(q_max[:, None], f_qp.shape)
    bound_recv = np.full_like(f_qp, np.inf)
    bound_recv[:-1, :] = w[1:, None] * (rho_jam[1:, None] - rho_eff[1:, :])
    binding = np.minimum.reduce([bound_send, bound_recv, bound_qmax])
    slack = binding - f_qp                                         # (n_cells, T)

    tol = 1e-6
    loose = slack > tol
    loose_cells = sorted({int(i) for i in np.where(loose.any(axis=1))[0]})

    return {
        "n_cells": n_cells,
        "T": int(meta["T"]),
        "density_gap_max": density_gap_max,
        "density_gap_rms": density_gap_rms,
        "slack_max": float(np.max(slack)),
        "n_loose_operators": int(loose.sum()),
        "frac_loose": float(loose.mean()),
        "loose_cells": loose_cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", required=True, type=Path,
        help="QP bundle dir holding meta.json, the inputs, and the solver's "
             "demand.csv/beta.csv/*_fitted.csv outputs.",
    )
    args = parser.parse_args()

    report = round_trip(args.bundle)
    print(
        f"=== Level-3 QP round-trip ({args.bundle}) ===\n"
        f"  cells x steps        : {report['n_cells']} x {report['T']}\n"
        f"  density gap (max/RMS): {report['density_gap_max']:.4g} / "
        f"{report['density_gap_rms']:.4g} veh/mi\n"
        f"  relaxation tightness : {report['n_loose_operators']} loose "
        f"mainline operators ({100*report['frac_loose']:.2f}%), "
        f"max slack {report['slack_max']:.4g} veh/h\n"
        f"  loose cells          : {report['loose_cells'] or 'none (relaxation tight)'}"
    )
    if report["density_gap_max"] > 1e-3:
        print(
            "\n  NOTE: round-trip gap is non-trivial -- the relaxation is loose "
            "at the cells above; the forward sim does not reproduce the fitted\n"
            "  density there. Consider the MIQP escalation (big-M on those "
            "operators) per the plan's deferred item.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
