"""Plotting helpers for the DWPT demand module.

``plot_P_y``           -- power-vs-position profile of the corridor.
``plot_demand_heatmap`` -- E(timestep, position) space-time heatmap.

Both follow the repo's ``utils/ctm/plots.py`` convention: take the data plus
a keyword-only ``out_path``, render, save at 120 dpi, and close the figure.
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
    title: str = "DWPT demand $E$(timestep, position)",
    cmap: str = "magma",
    out_path: Path,
) -> None:
    """Render ``result.E`` as a space-time heatmap (position x, timestep y)."""
    E = result.E
    position_m = result.position_m
    n_t = E.shape[0]

    dx = position_m[1] - position_m[0]
    x_edges = np.concatenate([position_m - dx / 2, [position_m[-1] + dx / 2]])
    y_edges = np.arange(n_t + 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    mesh = ax.pcolormesh(x_edges, y_edges, E, cmap=cmap, shading="flat")
    ax.set_xlabel("position [m]")
    ax.set_ylabel("timestep")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="energy [Wh]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
