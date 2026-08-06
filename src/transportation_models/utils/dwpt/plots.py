"""Plotting helpers for the DWPT demand module.

``plot_P_y``                    -- power-vs-position profile of the corridor.
``plot_demand_heatmap``         -- E(time, position) space-time heatmap.
``plot_aggregate_timeseries``   -- corridor-aggregate load vs time.
``plot_peak_load_profile``      -- per-position load at the peak timestamp.
``plot_load_duration_curve``    -- sorted corridor-aggregate load vs duration.

All follow the repo's ``utils/ctm/plots.py`` convention: take the data plus
a keyword-only ``out_path``, render, save at 120 dpi, and close the figure.
The three load plots express power in megawatts (energy divided by ``dt_h``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .model import DemandResult


def plot_P_y(
    P_y: np.ndarray,
    position_m: np.ndarray,
    *,
    title: str = "DWPT power-vs-position profile",
    out_path: Path,
) -> None:
    """Plot the spatial power profile ``P_y`` (kW) against position (m)."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(position_m, P_y / 1e3, lw=1.2)
    ax.set_xlabel("position [m]")
    ax.set_ylabel("power $P_y$ [kW]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_demand_heatmap(
    result: DemandResult,
    *,
    dt_h: float,
    title: str = "DWPT demand $E$(time, position)",
    cmap: str = "magma",
    out_path: Path,
) -> None:
    """Render ``result.E`` as a space-time heatmap (position x, time y).

    ``dt_h`` is the timestep duration in hours (e.g. ``freeway.dt``); the
    y-axis then runs from 0 to ``n_t * dt_h`` in hours rather than in raw
    timestep indices.
    """
    E = result.E
    position_m = result.position_m
    n_t = E.shape[0]

    dx = position_m[1] - position_m[0]
    x_edges = np.concatenate([position_m - dx / 2, [position_m[-1] + dx / 2]])
    y_edges = np.linspace(0.0, n_t * dt_h, n_t + 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    mesh = ax.pcolormesh(x_edges, y_edges, E, cmap=cmap, shading="flat")
    ax.set_xlabel("position [m]")
    ax.set_ylabel("time [h]")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="energy [Wh]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_aggregate_timeseries(
    result: DemandResult,
    *,
    dt_h: float,
    title: str = "DWPT corridor-aggregate load",
    out_path: Path,
) -> None:
    """Plot the corridor-aggregate load (MW) against time (h).

    The aggregate is ``E.sum(axis=1) / dt_h`` (energy over all positions per
    timestep, as power). The peak timestamp is marked.
    """
    power_MW = result.E.sum(axis=1) / dt_h / 1e6
    t_h = np.arange(power_MW.size) * dt_h
    i_peak = int(np.argmax(power_MW))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t_h, power_MW, lw=1.2)
    ax.plot(t_h[i_peak], power_MW[i_peak], "o", color="C3", zorder=5,
            label=f"peak {power_MW[i_peak]:.3g} MW @ {t_h[i_peak]:.2f} h")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("corridor power [MW]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_peak_load_profile(
    result: DemandResult,
    *,
    dt_h: float,
    segment_m: float = 20.0,
    cell_edges_m: np.ndarray | None = None,
    title: str = "DWPT instantaneous load profile at peak",
    out_path: Path,
) -> None:
    """Plot aggregated load (MW) at the peak-aggregate timestamp vs position.

    The peak timestamp is ``argmax`` of the corridor-aggregate load; the
    per-position load there is ``E[t*, :] / dt_h``. Positions are aggregated
    (summed) into contiguous bins to suppress the pad-vs-gap ripple. Bins are
    the CTM cells when ``cell_edges_m`` (an ``(N+1,)`` array of cell boundaries
    in meters) is given, else uniform ``segment_m``-wide segments. Bin loads
    sum to the peak corridor power either way.
    """
    t_star = int(np.argmax(result.E.sum(axis=1)))
    power_MW = result.E[t_star] / dt_h / 1e6
    pos = result.position_m

    if cell_edges_m is not None:
        edges = np.asarray(cell_edges_m, dtype=float)
        unit = "cell"
    else:
        k0 = int(np.floor(pos.min() / segment_m))
        k1 = int(np.floor(pos.max() / segment_m))
        edges = np.arange(k0, k1 + 2) * segment_m
        unit = f"{segment_m:g} m segment"

    n_bins = edges.size - 1
    idx = np.clip(np.digitize(pos, edges) - 1, 0, n_bins - 1)
    bin_power_MW = np.bincount(idx, weights=power_MW, minlength=n_bins)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(edges[:-1], bin_power_MW, width=np.diff(edges), align="edge",
           edgecolor="white", lw=0.3)
    ax.set_xlabel("position [m]")
    ax.set_ylabel(f"power per {unit} [MW]")
    ax.set_title(f"{title} (t = {t_star * dt_h:.2f} h)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_load_duration_curve(
    result: DemandResult,
    *,
    dt_h: float,
    title: str = "DWPT corridor-aggregate load duration curve",
    out_path: Path,
) -> None:
    """Plot the corridor-aggregate load (MW) sorted descending vs duration (h).

    Point ``k`` reads: the corridor load is at least this large for ``k``
    timesteps, i.e. ``k * dt_h`` hours.
    """
    power_MW = np.sort(result.E.sum(axis=1) / dt_h / 1e6)[::-1]
    duration_h = np.arange(1, power_MW.size + 1) * dt_h

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(duration_h, power_MW, lw=1.2)
    ax.set_xlabel("duration [h]")
    ax.set_ylabel("corridor power [MW]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
