"""Step 7 Level 3: differentiable forward-sim ramp-flow optimizer ("option B").

Rather than the relaxed convex QP (`solve_ramp_qp.jl`), this fits the ramp
inputs by running the *exact* CTM forward simulator each iteration and
descending the mainline density+flow error into the ramp `demand`/`beta` via
autograd (`utils.ctm.diffsim`). Because it never leaves the exact dynamics, the
recovered inputs round-trip by construction -- the transfer failure that made
the relaxed QP degrade on longer corridors cannot occur (see
docs/ramp_flow_estimation_module.md, "density transfers, flow does not").

Decision variables (free, all ramps -- observed ramp data is held out as the
test set): on-ramp-cell `demand` (>= 0 via clamp) and off-ramp-cell `beta`
(in (0,1) via sigmoid), placed at the ramp cells and zero elsewhere. They are
**piecewise-constant over 5-min blocks** (the PeMS cadence, matching the QP's
`r_blk`/`s_blk`): per-step ramps would be ~sp5x more unknowns than 5-min
observations and badly underdetermined. Objective: scale-normalized MSE of
mainline density AND flow at the direct/tiebreak cells, on the 5-min grid,
sampled exactly as `validate_ctm_corridor` does (density at 5-min boundaries,
flow block-averaged). No regularization in v1.

Consumes the bundle from `build_ctm_qp_inputs.py` (now including
`observed_flow.csv`); writes the same wide `demand.csv` / `beta.csv` the QP
solver and `simulate_ctm_corridor.py` use, plus `optimize_report.json`.

    python scripts/optimize_ramp_flows_diffsim.py \
        --bundle case_studies/I880N_5mi/qp_inputs \
        --init-demand case_studies/I880N_5mi/histAve_ramps/demand.csv \
        --init-beta   case_studies/I880N_5mi/histAve_ramps/beta.csv \
        --iters 3000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transportation_models.utils.ctm import diffsim, freeway_from_dataframe

_EPS = 1e-4  # keeps logit()/sigmoid() away from the beta = 0/1 singularities


def _wide(path: Path, n_cells: int, T: int) -> np.ndarray:
    """Read a wide T x N cell-indexed CSV as ``(n_cells, T)`` (missing cols 0)."""
    df = pd.read_csv(path)
    df.columns = df.columns.astype(int)
    arr = df.reindex(columns=range(n_cells), fill_value=0.0).to_numpy(dtype=float).T
    if arr.shape[1] != T:
        raise SystemExit(
            f"{path.name} has {arr.shape[1]} rows but the bundle horizon is "
            f"T={T}; the init profile must cover the same window/dt."
        )
    return arr


def _observed_grid(path: Path, value_col: str, n_cells: int, n_5min: int):
    """Dense ``(n_cells, n_5min)`` observed grid + finite-sample mask."""
    obs = torch.zeros((n_cells, n_5min), dtype=torch.float64)
    mask = torch.zeros((n_cells, n_5min), dtype=torch.float64)
    df = pd.read_csv(path)
    for cell, m, val in zip(df["cell"], df["m_5min"], df[value_col]):
        obs[int(cell), int(m)] = float(val)
        mask[int(cell), int(m)] = 1.0
    return obs, mask


def optimize_bundle(
    bundle: Path,
    *,
    init_demand: Path | None = None,
    init_beta: Path | None = None,
    iters: int = 2000,
    lr_demand: float = 20.0,
    lr_beta: float = 0.1,
    rho_scale: float = 50.0,
    flow_scale: float = 2000.0,
    w_rho: float = 1.0,
    w_flow: float = 1.0,
    beta_init: float = 0.05,
    log_every: int = 0,
    out_dir: Path | None = None,
) -> dict:
    """Optimize ramp demand/beta to fit mainline density+flow; write the CSVs.

    Returns a report dict (also written to ``optimize_report.json``).
    """
    bundle = Path(bundle)
    out_dir = Path(out_dir) if out_dir is not None else bundle
    meta = json.loads((bundle / "meta.json").read_text())
    dt_h = float(meta["dt_h"])
    n, T = int(meta["n_cells"]), int(meta["T"])
    sp5, n_5min = int(meta["steps_per_5min"]), int(meta["n_5min"])

    freeway_df = pd.read_csv(bundle / "freeway.csv")
    params = diffsim.freeway_arrays_to_torch(freeway_from_dataframe(freeway_df, dt=dt_h).arrays())
    to_t = lambda a: torch.as_tensor(np.asarray(a, dtype=float), dtype=torch.float64)  # noqa: E731
    rho0 = to_t(pd.read_csv(bundle / "rho0.csv")["rho0"].to_numpy())
    q0 = torch.zeros(n, dtype=torch.float64)
    inflow = to_t(pd.read_csv(bundle / "inflow.csv")["inflow"].to_numpy())

    obs_rho, mask_rho = _observed_grid(bundle / "observed_density.csv", "rho_obs", n, n_5min)
    obs_flow, mask_flow = _observed_grid(bundle / "observed_flow.csv", "flow_obs", n, n_5min)

    on_idx = torch.tensor(meta["on_ramp_cells"], dtype=torch.long)
    off_idx = torch.tensor(meta["off_ramp_cells"], dtype=torch.long)

    # --- initial profiles, down-sampled to 5-min block means ---
    def to_blocks(arr):  # (n, T) -> (n, n_5min)
        return arr.reshape(n, n_5min, sp5).mean(axis=2)

    d0 = to_blocks(_wide(Path(init_demand), n, T) if init_demand else np.zeros((n, T)))
    if init_beta:
        b0 = to_blocks(_wide(Path(init_beta), n, T))
    else:
        b0 = np.zeros((n, n_5min))
        b0[meta["off_ramp_cells"]] = beta_init
    b0 = np.clip(b0, _EPS, 1.0 - _EPS)

    # --- free variables at ramp cells only, one value per 5-min block ---
    param_groups, demand_param, beta_logit = [], None, None
    if len(on_idx):
        demand_param = torch.tensor(d0[meta["on_ramp_cells"]], dtype=torch.float64, requires_grad=True)
        param_groups.append({"params": [demand_param], "lr": lr_demand})
    if len(off_idx):
        beta_logit = torch.logit(to_t(b0[meta["off_ramp_cells"]])).requires_grad_(True)
        param_groups.append({"params": [beta_logit], "lr": lr_beta})
    if not param_groups:
        raise SystemExit("bundle has no on- or off-ramp cells; nothing to optimize.")
    opt = torch.optim.Adam(param_groups)

    zeros_nt = torch.zeros((n, T), dtype=torch.float64)

    def build_inputs():
        # Expand each (n_ramp, n_5min) block value to (n_ramp, T) per-step.
        demand = zeros_nt
        if demand_param is not None:
            d_step = torch.clamp(demand_param, min=0.0).repeat_interleave(sp5, dim=1)
            demand = zeros_nt.index_copy(0, on_idx, d_step)
        beta = zeros_nt
        if beta_logit is not None:
            b_step = torch.sigmoid(beta_logit).repeat_interleave(sp5, dim=1)
            beta = zeros_nt.index_copy(0, off_idx, b_step)
        return demand, beta

    def masked_terms(traj):
        sim_rho5 = traj.density[:, : n_5min * sp5 : sp5]
        sim_flow5 = traj.mainline_flow[:, : n_5min * sp5].reshape(n, n_5min, sp5).mean(dim=2)
        lr_ = ((sim_rho5 - obs_rho) ** 2 * mask_rho).sum() / mask_rho.sum()
        lf_ = ((sim_flow5 - obs_flow) ** 2 * mask_flow).sum() / mask_flow.sum()
        return lr_, lf_

    for it in range(iters):
        opt.zero_grad()
        demand, beta = build_inputs()
        traj = diffsim.simulate(params, dt_h, rho0, q0, demand, beta, inflow)
        mse_rho, mse_flow = masked_terms(traj)
        loss = w_rho * mse_rho / rho_scale**2 + w_flow * mse_flow / flow_scale**2
        loss.backward()
        opt.step()
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"  iter {it:5d}  loss {loss.item():.6g}  "
                  f"rmse_rho {mse_rho.sqrt().item():.4g}  rmse_flow {mse_flow.sqrt().item():.4g}")

    with torch.no_grad():
        demand, beta = build_inputs()
        traj = diffsim.simulate(params, dt_h, rho0, q0, demand, beta, inflow)
        mse_rho, mse_flow = masked_terms(traj)
        demand_np = demand.numpy()
        beta_np = beta.numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [str(i) for i in range(n)]
    pd.DataFrame(demand_np.T, columns=cols).to_csv(out_dir / "demand.csv", index=False)
    pd.DataFrame(beta_np.T, columns=cols).to_csv(out_dir / "beta.csv", index=False)

    report = {
        "iters": iters,
        "n_on_ramp_cells": int(len(on_idx)),
        "n_off_ramp_cells": int(len(off_idx)),
        "density_rmse": float(mse_rho.sqrt()),   # veh/mi, scored cells, 5-min grid
        "flow_rmse": float(mse_flow.sqrt()),     # veh/h
        "final_loss": float(w_rho * mse_rho / rho_scale**2 + w_flow * mse_flow / flow_scale**2),
        "lr_demand": lr_demand,
        "lr_beta": lr_beta,
        "w_rho": w_rho,
        "w_flow": w_flow,
        "rho_scale": rho_scale,
        "flow_scale": flow_scale,
        "init_demand": str(init_demand) if init_demand else None,
        "init_beta": str(init_beta) if init_beta else None,
    }
    (out_dir / "optimize_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True, type=Path,
                   help="Bundle dir from build_ctm_qp_inputs.py (+ observed_flow.csv).")
    p.add_argument("--init-demand", type=Path, default=None,
                   help="Wide demand.csv to initialize from (else zeros).")
    p.add_argument("--init-beta", type=Path, default=None,
                   help="Wide beta.csv to initialize from (else beta-init at off-ramps).")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lr-demand", type=float, default=20.0)
    p.add_argument("--lr-beta", type=float, default=0.1)
    p.add_argument("--rho-scale", type=float, default=50.0)
    p.add_argument("--flow-scale", type=float, default=2000.0)
    p.add_argument("--w-rho", type=float, default=1.0)
    p.add_argument("--w-flow", type=float, default=1.0)
    p.add_argument("--beta-init", type=float, default=0.05,
                   help="Off-ramp beta initial value when --init-beta is absent.")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out-dir", type=Path, default=None, help="Default: the bundle dir.")
    args = p.parse_args()

    report = optimize_bundle(
        args.bundle, init_demand=args.init_demand, init_beta=args.init_beta,
        iters=args.iters, lr_demand=args.lr_demand, lr_beta=args.lr_beta,
        rho_scale=args.rho_scale, flow_scale=args.flow_scale,
        w_rho=args.w_rho, w_flow=args.w_flow, beta_init=args.beta_init,
        log_every=args.log_every, out_dir=args.out_dir,
    )
    print(
        f"\nOptimized ramp flows written to {args.out_dir or args.bundle}\n"
        f"  cells         : {report['n_on_ramp_cells']} on-ramp, "
        f"{report['n_off_ramp_cells']} off-ramp\n"
        f"  density RMSE  : {report['density_rmse']:.4g} veh/mi (scored cells, 5-min)\n"
        f"  flow RMSE     : {report['flow_rmse']:.4g} veh/h\n"
        f"  final loss    : {report['final_loss']:.6g}"
    )


if __name__ == "__main__":
    main()
