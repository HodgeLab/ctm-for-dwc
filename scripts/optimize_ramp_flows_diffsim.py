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

Logs per-iteration loss / density_rmse / flow_rmse to Weights & Biases
(project `ramp-flow-estimation`, `WANDB_MODE=offline` honored; `--no-wandb` to
disable) so hyperparameter sweeps are inspectable, and early-stops on the loss
(relative `--min-delta` improvement over `--patience` iters) with best-iterate
restore so a sweep run doesn't burn iterations after it converges.

    python scripts/optimize_ramp_flows_diffsim.py \
        --bundle case_studies/I880N_5mi/qp_inputs \
        --init-demand case_studies/I880N_5mi/histAve_ramps/demand.csv \
        --init-beta   case_studies/I880N_5mi/histAve_ramps/beta.csv \
        --iters 3000
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
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
    """Dense ``(n_cells, n_5min)`` observed grid + finite-sample mask.

    Returns ``(obs, weight, genuine)`` where ``weight`` (the loss weighting) and
    ``genuine`` (the reporting mask over real observations) are both the 0/1
    sample mask -- imputed grids override ``weight`` in :func:`_imputed_grid`.
    """
    obs = torch.zeros((n_cells, n_5min), dtype=torch.float64)
    mask = torch.zeros((n_cells, n_5min), dtype=torch.float64)
    df = pd.read_csv(path)
    for cell, m, val in zip(df["cell"], df["m_5min"], df[value_col]):
        obs[int(cell), int(m)] = float(val)
        mask[int(cell), int(m)] = 1.0
    return obs, mask, mask.clone()


def _imputed_grid(path: Path, n_cells: int, n_5min: int, imputed_weight: float):
    """Augmented ``cell,m_5min,value,source`` grid from ``augment_observed_grid``.

    Returns ``(obs, weight, genuine)``: ``weight`` is 1.0 at genuine
    observations and ``imputed_weight`` at hist/knn-filled cells (the loss
    weighting); ``genuine`` marks only ``source == "observed"`` so reported
    RMSE/MAPE stay on real measurements.
    """
    obs = torch.zeros((n_cells, n_5min), dtype=torch.float64)
    weight = torch.zeros((n_cells, n_5min), dtype=torch.float64)
    genuine = torch.zeros((n_cells, n_5min), dtype=torch.float64)
    df = pd.read_csv(path)
    for cell, m, val, src in zip(df["cell"], df["m_5min"], df["value"], df["source"]):
        i, j = int(cell), int(m)
        obs[i, j] = float(val)
        is_obs = src == "observed"
        weight[i, j] = 1.0 if is_obs else imputed_weight
        genuine[i, j] = 1.0 if is_obs else 0.0
    return obs, weight, genuine


def _masked_mape(sim, obs, mask):
    """MAPE [%] over observed (mask=1) *and* non-zero samples (repo convention).

    The zero-observed denominator is guarded (``safe_obs``) so the excluded
    ``obs == 0`` entries never form ``|sim|/0 * 0 = 0 * inf = NaN`` in the sum.
    """
    valid = mask * (obs != 0.0).to(mask.dtype)
    denom = valid.sum()
    if denom.item() == 0:
        return torch.tensor(float("nan"), dtype=sim.dtype)
    safe_obs = torch.where(obs != 0.0, obs, torch.ones_like(obs))
    return (torch.abs(sim - obs) / torch.abs(safe_obs) * valid).sum() / denom * 100.0


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
    imputed_dir: Path | None = None,
    imputed_weight: float = 0.3,
    patience: int | None = None,
    min_delta: float = 1e-4,
    log_every: int = 0,
    log_fn: Callable[[dict], None] | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Optimize ramp demand/beta to fit mainline density+flow; write the CSVs.

    ``patience`` (iters without a ``min_delta`` *relative* loss improvement)
    enables early stopping with best-iterate restore; ``None`` runs all
    ``iters`` and uses the final iterate. ``log_fn`` receives a per-iteration
    metrics dict (``iter``/``loss``/``density_rmse``/``flow_rmse``/
    ``density_mape``/``flow_mape``) every ``log_every`` iters -- the W&B seam,
    kept out of this function so it stays import-light and testable. Returns a
    report dict (also written to ``optimize_report.json``).
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

    if imputed_dir is not None:
        imputed_dir = Path(imputed_dir)
        obs_rho, w_rho_grid, gen_rho = _imputed_grid(
            imputed_dir / "observed_density.csv", n, n_5min, imputed_weight)
        obs_flow, w_flow_grid, gen_flow = _imputed_grid(
            imputed_dir / "observed_flow.csv", n, n_5min, imputed_weight)
    else:
        obs_rho, w_rho_grid, gen_rho = _observed_grid(
            bundle / "observed_density.csv", "rho_obs", n, n_5min)
        obs_flow, w_flow_grid, gen_flow = _observed_grid(
            bundle / "observed_flow.csv", "flow_obs", n, n_5min)

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
        param_groups.append({"name": "demand", "params": [demand_param], "lr": lr_demand})
    if len(off_idx):
        beta_logit = torch.logit(to_t(b0[meta["off_ramp_cells"]])).requires_grad_(True)
        param_groups.append({"name": "beta", "params": [beta_logit], "lr": lr_beta})
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

    def masked_metrics(traj):
        """Weighted MSE (drives the loss) + genuine-observation RMSE/MAPE.

        ``loss_rho``/``loss_flow`` weight imputed cells by ``imputed_weight``;
        ``rep_rho``/``rep_flow`` (reported as RMSE) and the MAPEs score only
        genuine observations, so the report stays comparable across runs.
        """
        sim_rho5 = traj.density[:, : n_5min * sp5 : sp5]
        sim_flow5 = traj.mainline_flow[:, : n_5min * sp5].reshape(n, n_5min, sp5).mean(dim=2)
        loss_rho = ((sim_rho5 - obs_rho) ** 2 * w_rho_grid).sum() / w_rho_grid.sum()
        loss_flow = ((sim_flow5 - obs_flow) ** 2 * w_flow_grid).sum() / w_flow_grid.sum()
        rep_rho = ((sim_rho5 - obs_rho) ** 2 * gen_rho).sum() / gen_rho.sum()
        rep_flow = ((sim_flow5 - obs_flow) ** 2 * gen_flow).sum() / gen_flow.sum()
        mape_rho = _masked_mape(sim_rho5, obs_rho, gen_rho)
        mape_flow = _masked_mape(sim_flow5, obs_flow, gen_flow)
        return loss_rho, loss_flow, rep_rho, rep_flow, mape_rho, mape_flow

    def snapshot():
        s = {}
        if demand_param is not None:
            s["demand"] = demand_param.detach().clone()
        if beta_logit is not None:
            s["beta"] = beta_logit.detach().clone()
        return s

    early_stop = patience is not None
    best_loss, best_state, best_iter, since_improve = float("inf"), None, -1, 0
    stopped_early, iters_run = False, iters
    for it in range(iters):
        opt.zero_grad()
        demand, beta = build_inputs()
        traj = diffsim.simulate(params, dt_h, rho0, q0, demand, beta, inflow)
        loss_rho, loss_flow, rep_rho, rep_flow, mape_rho, mape_flow = masked_metrics(traj)
        loss = w_rho * loss_rho / rho_scale**2 + w_flow * loss_flow / flow_scale**2
        cur = loss.item()

        # Best-iterate tracking uses the loss at the *current* (pre-step) params,
        # so the snapshot matches the reported value.
        if early_stop:
            if cur < best_loss * (1.0 - min_delta):
                best_loss, best_iter, since_improve = cur, it, 0
                best_state = snapshot()
            else:
                since_improve += 1

        if log_every and (it % log_every == 0 or it == iters - 1):
            row = {"iter": it, "loss": cur,
                   "density_rmse": rep_rho.sqrt().item(),
                   "flow_rmse": rep_flow.sqrt().item(),
                   "density_mape": mape_rho.item(),
                   "flow_mape": mape_flow.item()}
            if log_fn is not None:
                log_fn(row)
            print(f"  iter {it:5d}  loss {cur:.6g}  "
                  f"rmse_rho {row['density_rmse']:.4g}  rmse_flow {row['flow_rmse']:.4g}  "
                  f"mape_rho {row['density_mape']:.3g}%  mape_flow {row['flow_mape']:.3g}%")

        loss.backward()
        opt.step()

        if early_stop and since_improve >= patience:
            stopped_early, iters_run = True, it + 1
            break

    if early_stop and best_state is not None:  # restore the best iterate
        with torch.no_grad():
            if demand_param is not None:
                demand_param.copy_(best_state["demand"])
            if beta_logit is not None:
                beta_logit.copy_(best_state["beta"])

    with torch.no_grad():
        demand, beta = build_inputs()
        traj = diffsim.simulate(params, dt_h, rho0, q0, demand, beta, inflow)
        loss_rho, loss_flow, rep_rho, rep_flow, mape_rho, mape_flow = masked_metrics(traj)
        demand_np = demand.numpy()
        beta_np = beta.numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [str(i) for i in range(n)]
    pd.DataFrame(demand_np.T, columns=cols).to_csv(out_dir / "demand.csv", index=False)
    pd.DataFrame(beta_np.T, columns=cols).to_csv(out_dir / "beta.csv", index=False)

    report = {
        "iters": iters,
        "iters_run": iters_run,
        "stopped_early": stopped_early,
        "best_iter": best_iter if early_stop else iters_run - 1,
        "patience": patience,
        "min_delta": min_delta,
        "n_on_ramp_cells": int(len(on_idx)),
        "n_off_ramp_cells": int(len(off_idx)),
        "density_rmse": float(rep_rho.sqrt()),   # veh/mi, genuine obs, 5-min grid
        "flow_rmse": float(rep_flow.sqrt()),     # veh/h
        "density_mape": float(mape_rho),         # %, non-zero observed samples
        "flow_mape": float(mape_flow),           # %
        "final_loss": float(w_rho * loss_rho / rho_scale**2 + w_flow * loss_flow / flow_scale**2),
        "imputed_dir": str(imputed_dir) if imputed_dir is not None else None,
        "imputed_weight": imputed_weight if imputed_dir is not None else None,
        "n_scored_density": int(gen_rho.sum()),   # genuine observations
        "n_scored_flow": int(gen_flow.sum()),
        "n_imputed_density": int((w_rho_grid > 0).sum() - gen_rho.sum()),
        "n_imputed_flow": int((w_flow_grid > 0).sum() - gen_flow.sum()),
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
    # Underscore aliases so a W&B sweep (which emits --lr_demand=...) drives the
    # same flags the repo writes with hyphens.
    p.add_argument("--lr-demand", "--lr_demand", dest="lr_demand", type=float, default=20.0)
    p.add_argument("--lr-beta", "--lr_beta", dest="lr_beta", type=float, default=0.1)
    p.add_argument("--rho-scale", "--rho_scale", dest="rho_scale", type=float, default=50.0)
    p.add_argument("--flow-scale", "--flow_scale", dest="flow_scale", type=float, default=2000.0)
    p.add_argument("--w-rho", "--w_rho", dest="w_rho", type=float, default=1.0)
    p.add_argument("--w-flow", "--w_flow", dest="w_flow", type=float, default=1.0)
    p.add_argument("--beta-init", type=float, default=0.05,
                   help="Off-ramp beta initial value when --init-beta is absent.")
    p.add_argument("--imputed-dir", "--imputed_dir", dest="imputed_dir", type=Path, default=None,
                   help="augment_observed_grid.py output dir (e.g. .../imputed/k2_d1.0); "
                        "scores imputed cells too. Default: bundle's observed_*.csv only.")
    p.add_argument("--imputed-weight", "--imputed_weight", dest="imputed_weight",
                   type=float, default=0.3,
                   help="Loss weight of imputed cells vs 1.0 for observations.")
    p.add_argument("--patience", type=int, default=200,
                   help="Early-stop after this many iters without a --min-delta "
                        "relative loss improvement (--no-early-stop disables).")
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="Min relative loss improvement that resets patience.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--wandb-project", default="ramp-flow-estimation")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: the bundle dir.")
    args = p.parse_args()

    patience = None if args.no_early_stop else args.patience

    log_fn, run = None, None
    if not args.no_wandb:
        import wandb
        config = {
            "bundle": str(args.bundle), "iters": args.iters,
            "lr_demand": args.lr_demand, "lr_beta": args.lr_beta,
            "rho_scale": args.rho_scale, "flow_scale": args.flow_scale,
            "w_rho": args.w_rho, "w_flow": args.w_flow, "beta_init": args.beta_init,
            "imputed_dir": str(args.imputed_dir) if args.imputed_dir else None,
            "imputed_weight": args.imputed_weight,
            "patience": patience, "min_delta": args.min_delta,
            "init_demand": str(args.init_demand) if args.init_demand else None,
            "init_beta": str(args.init_beta) if args.init_beta else None,
        }
        run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=config)
        log_fn = lambda row: wandb.log(  # noqa: E731
            {k: v for k, v in row.items() if k != "iter"}, step=row["iter"])

    # In a sweep every trial reads the one shared bundle but must write to its
    # own dir, or the trials clobber each other (and the bundle's own CSVs).
    out_dir = args.out_dir
    if out_dir is None and run is not None:
        out_dir = args.bundle / "sweep_runs" / run.id

    report = optimize_bundle(
        args.bundle, init_demand=args.init_demand, init_beta=args.init_beta,
        iters=args.iters, lr_demand=args.lr_demand, lr_beta=args.lr_beta,
        rho_scale=args.rho_scale, flow_scale=args.flow_scale,
        w_rho=args.w_rho, w_flow=args.w_flow, beta_init=args.beta_init,
        imputed_dir=args.imputed_dir, imputed_weight=args.imputed_weight,
        patience=patience, min_delta=args.min_delta,
        log_every=args.log_every, log_fn=log_fn, out_dir=out_dir,
    )

    if run is not None:
        run.summary.update({
            "density_rmse": report["density_rmse"], "flow_rmse": report["flow_rmse"],
            "density_mape": report["density_mape"], "flow_mape": report["flow_mape"],
            "final_loss": report["final_loss"], "best_iter": report["best_iter"],
            "iters_run": report["iters_run"], "stopped_early": report["stopped_early"],
        })
        run.finish()

    stop_note = (f"early-stopped at iter {report['iters_run']} (best {report['best_iter']})"
                 if report["stopped_early"] else f"ran {report['iters_run']} iters")
    print(
        f"\nOptimized ramp flows written to {out_dir or args.bundle}\n"
        f"  cells         : {report['n_on_ramp_cells']} on-ramp, "
        f"{report['n_off_ramp_cells']} off-ramp\n"
        f"  {stop_note}\n"
        f"  density RMSE  : {report['density_rmse']:.4g} veh/mi (scored cells, 5-min)\n"
        f"  flow RMSE     : {report['flow_rmse']:.4g} veh/h\n"
        f"  density MAPE  : {report['density_mape']:.3g} %\n"
        f"  flow MAPE     : {report['flow_mape']:.3g} %\n"
        f"  final loss    : {report['final_loss']:.6g}"
    )


if __name__ == "__main__":
    main()
