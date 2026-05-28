"""Canonical CTMSIM per-step update.

:func:`step` is a pure, vectorized function implementing the six update equations
of ``docs/ctm_module.md`` (Kurzhanskiy 2007, sec. 4.2.1). The state carried
between steps is the density vector ``rho`` [veh/mi] and the on-ramp queue vector
``queue`` [veh]. Inputs are trusted (validate via :class:`~.model.Scenario` /
:meth:`~.model.Freeway.validate` first); ``step`` itself does no validation so it
stays cheap to call in a loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FreewayArrays


@dataclass
class StepResult:
    """State and flows produced by one :func:`step` (all arrays shaped (n_cells,))."""

    rho: np.ndarray            # density at k+1 [veh/mi]
    queue: np.ndarray          # on-ramp queue at k+1 [veh]
    mainline_flow: np.ndarray  # f_i, cell i -> i+1 [veh/h]
    off_ramp: np.ndarray       # s_i [veh/h]
    on_ramp: np.ndarray        # r_i [veh/h]
    speed: np.ndarray          # V_i mean speed [mi/h]


def step(
    arr: FreewayArrays,
    dt: float,
    rho,
    queue,
    demand,
    beta,
    inflow: float,
) -> StepResult:
    """Advance one time step. Equation numbers refer to docs/ctm_module.md.

    Parameters
    ----------
    arr : FreewayArrays
        Per-cell parameters (from :meth:`~.model.Freeway.arrays`).
    dt : float
        Time step ΔT [h].
    rho, queue : array-like, shape (n_cells,)
        Density [veh/mi] and on-ramp queue [veh] at step k.
    demand, beta : array-like broadcastable to (n_cells,)
        On-ramp arrival demand d_i(k) [veh/h] and off-ramp split ratio beta_i(k).
    inflow : float
        Upstream boundary demand f_0(k) [veh/h].
    """
    length, q_max, v_f, w, rho_jam, R, S, gamma, xi = arr
    rho = np.asarray(rho, dtype=float)
    queue = np.asarray(queue, dtype=float)
    demand = np.broadcast_to(np.asarray(demand, dtype=float), rho.shape).astype(float)
    beta = np.broadcast_to(np.asarray(beta, dtype=float), rho.shape).astype(float)
    beta_bar = 1.0 - beta

    # 1. On-ramp flow (eq. 4.2). NoControl: the metering term max{C, Q} is dropped.
    space = xi * np.maximum(rho_jam - rho, 0.0) * length / dt
    r = np.minimum(np.minimum(demand + queue / dt, space), R)
    r = np.maximum(r, 0.0)

    # 2. On-ramp queue (eq. 4.3).
    queue_new = np.maximum(queue + (demand - r) * dt, 0.0)

    # 3. Mainline flow f_i (eq. 4.4); free flow downstream of cell N drops the
    #    receiving term (eq. 4.5).
    rho_eff = rho + gamma * r * dt / length          # on-ramp-blended density
    send = beta_bar * v_f * rho_eff                  # mainline sending
    recv = w * np.maximum(rho_jam - rho_eff, 0.0)    # receiving capacity per cell
    recv_down = np.empty_like(rho)
    if rho.size > 1:
        recv_down[:-1] = recv[1:]
    recv_down[-1] = np.inf
    # Off-ramp-capacity term (beta_bar/beta)*S restricts the mainline f_i so that
    # the beta fraction matches S. Edge cases avoid 0/0 and 0*inf:
    #   beta = 0 -> no off-ramp restriction (offcap = inf)
    #   beta = 1 -> mainline must be 0 (all flow exits via off-ramp)
    with np.errstate(divide="ignore", invalid="ignore"):
        offcap_general = beta_bar / np.where(beta > 0.0, beta, 1.0) * S
    offcap = np.where(
        beta <= 0.0,
        np.inf,
        np.where(beta >= 1.0, 0.0, offcap_general),
    )
    f = np.minimum(np.minimum(send, recv_down), np.minimum(offcap, q_max))
    f = np.maximum(f, 0.0)

    # 4. Off-ramp flow (eq. 4.6).
    s_split = beta / np.where(beta_bar > 0.0, beta_bar, 1.0) * f
    s_full = np.minimum(v_f * rho_eff, S)
    s = np.where(beta < 1.0, s_split, s_full)

    # 5. Density (eq. 4.7). Upstream boundary admits min(inflow, cell-0 receiving)
    #    (standard CTM source; no upstream queue yet -- see docs Open items).
    f_up = np.empty_like(rho)
    f_up[0] = min(float(inflow), float(recv[0]))
    if rho.size > 1:
        f_up[1:] = f[:-1]
    rho_new = rho + dt / length * (f_up + r - f - s)

    # 6. Mean speed (eq. 4.8).
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(rho_new > 0.0, np.minimum(v_f, (f + s) / rho_new), v_f)

    return StepResult(
        rho=rho_new,
        queue=queue_new,
        mainline_flow=f,
        off_ramp=s,
        on_ramp=r,
        speed=speed,
    )
