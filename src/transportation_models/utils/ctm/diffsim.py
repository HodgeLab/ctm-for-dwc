"""Differentiable (PyTorch) reimplementation of the canonical CTM step.

:func:`step` mirrors :func:`utils.ctm.engine.step` exactly -- the same six
update equations of ``docs/ctm_module.md`` (Kurzhanskiy 2007, sec. 4.2.1) -- but
in torch, so the mainline density/flow trajectory is differentiable w.r.t. the
ramp inputs (``demand``, ``beta``). A numerical parity test
(``tests/test_ctm_diffsim.py``) pins this to the numpy engine.

This is the engine for the Step-7 Level-3 differentiable forward-sim ramp
optimizer ("option B" in docs/ramp_flow_estimation_module.md): backprop a
density+flow error into ``demand``/``beta`` through the *exact* dynamics, so --
unlike the relaxed QP -- the recovered inputs round-trip by construction.

The array-index shifts (``recv_down``, ``f_up``) are built with ``torch.cat``
rather than in-place assignment so autograd sees a functional graph.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from .model import FreewayArrays

# torch params are the same 9-tuple as FreewayArrays, as tensors.
TorchParams = tuple[Tensor, ...]


class TorchStepResult(NamedTuple):
    """Torch counterpart of :class:`engine.StepResult` (all fields tensors)."""

    rho: Tensor            # density at k+1 [veh/mi]
    queue: Tensor          # on-ramp queue at k+1 [veh]
    mainline_flow: Tensor  # f_i, cell i -> i+1 [veh/h]
    off_ramp: Tensor       # s_i [veh/h]
    on_ramp: Tensor        # r_i [veh/h]
    speed: Tensor          # V_i mean speed [mi/h]
    boundary_inflow: Tensor  # admitted f_0 [veh/h] (scalar)


def freeway_arrays_to_torch(
    arr: FreewayArrays, *, dtype: torch.dtype = torch.float64, device=None,
) -> TorchParams:
    """Compile a numpy :class:`FreewayArrays` to a tuple of tensors (float64).

    float64 keeps the parity with the numpy engine tight; ``inf`` ramp
    capacities carry through ``torch.as_tensor`` unchanged.
    """
    return tuple(
        torch.as_tensor(np.asarray(a, dtype=float), dtype=dtype, device=device)
        for a in arr
    )


def step(
    params: TorchParams,
    dt: float,
    rho: Tensor,
    queue: Tensor,
    demand: Tensor,
    beta: Tensor,
    inflow: Tensor | float,
) -> TorchStepResult:
    """Advance one time step (torch). Equation numbers refer to docs/ctm_module.md.

    ``params`` is :func:`freeway_arrays_to_torch` output; ``rho``, ``queue``,
    ``demand``, ``beta`` are ``(n_cells,)`` tensors; ``inflow`` is a scalar. The
    beta edge cases (``beta = 0`` -> no off-ramp restriction, ``beta = 1`` ->
    full diversion) match the numpy engine; interior ``beta`` (the optimizer's
    sigmoid range) is differentiable throughout.
    """
    length, q_max, v_f, w, rho_jam, R, S, gamma, xi = params
    beta_bar = 1.0 - beta
    zero = torch.zeros_like(rho)
    inf = torch.tensor(float("inf"), dtype=rho.dtype, device=rho.device)

    # 1. On-ramp flow (eq. 4.2). NoControl: the metering term max{C, Q} is dropped.
    space = xi * torch.clamp(rho_jam - rho, min=0.0) * length / dt
    r = torch.minimum(torch.minimum(demand + queue / dt, space), R)
    r = torch.clamp(r, min=0.0)

    # 2. On-ramp queue (eq. 4.3).
    queue_new = torch.clamp(queue + (demand - r) * dt, min=0.0)

    # 3. Mainline flow f_i (eq. 4.4); free flow downstream of cell N drops the
    #    receiving term (eq. 4.5).
    rho_eff = rho + gamma * r * dt / length          # on-ramp-blended density
    send = beta_bar * v_f * rho_eff                  # mainline sending
    recv = w * torch.clamp(rho_jam - rho_eff, min=0.0)  # receiving capacity per cell
    if rho.shape[0] > 1:
        recv_down = torch.cat([recv[1:], inf.reshape(1)])
    else:
        recv_down = inf.reshape(1)
    # Off-ramp-capacity term (beta_bar/beta)*S restricts f_i so the beta fraction
    # matches S. Guards avoid 0/0 and 0*inf: beta<=0 -> inf (no restriction),
    # beta>=1 -> 0 (all flow exits via off-ramp).
    safe_beta = torch.where(beta > 0.0, beta, torch.ones_like(beta))
    offcap_general = beta_bar / safe_beta * S
    offcap = torch.where(
        beta <= 0.0,
        inf.expand_as(beta),
        torch.where(beta >= 1.0, zero, offcap_general),
    )
    f = torch.minimum(torch.minimum(send, recv_down), torch.minimum(offcap, q_max))
    f = torch.clamp(f, min=0.0)

    # 4. Off-ramp flow (eq. 4.6).
    safe_beta_bar = torch.where(beta_bar > 0.0, beta_bar, torch.ones_like(beta_bar))
    s_split = beta / safe_beta_bar * f
    s_full = torch.minimum(v_f * rho_eff, S)
    s = torch.where(beta < 1.0, s_split, s_full)

    # 5. Density (eq. 4.7). Upstream boundary admits min(inflow, cell-0 receiving)
    #    (standard CTM source; no upstream queue -- rejected demand is dropped).
    inflow_t = torch.as_tensor(inflow, dtype=rho.dtype, device=rho.device)
    boundary_inflow = torch.minimum(inflow_t, recv[0])
    if rho.shape[0] > 1:
        f_up = torch.cat([boundary_inflow.reshape(1), f[:-1]])
    else:
        f_up = boundary_inflow.reshape(1)
    rho_new = rho + dt / length * (f_up + r - f - s)

    # 6. Mean speed (eq. 4.8).
    safe_rho = torch.where(rho_new > 0.0, rho_new, torch.ones_like(rho_new))
    speed = torch.where(rho_new > 0.0, torch.minimum(v_f, (f + s) / safe_rho), v_f)

    return TorchStepResult(
        rho=rho_new,
        queue=queue_new,
        mainline_flow=f,
        off_ramp=s,
        on_ramp=r,
        speed=speed,
        boundary_inflow=boundary_inflow,
    )
