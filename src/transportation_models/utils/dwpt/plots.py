"""Plotting helpers for the DWPT demand module.

``plot_P_y``                    -- power-vs-position profile of the corridor.
``plot_demand_heatmap``         -- E(time, position) space-time heatmap.
``plot_aggregate_timeseries``   -- corridor-aggregate load vs time.
``plot_peak_load_profile``      -- per-position load at the peak timestamp.
``plot_absolute_peak_profile``  -- per-position peak load over all timestamps.
``plot_average_load_profile``   -- per-position load averaged over all timestamps.
``plot_load_factor_profile``    -- per-position mean/peak load ratio.
``plot_load_distribution_profile`` -- per-position load distribution over time.
``plot_load_duration_curve``    -- sorted corridor-aggregate load vs duration.

All follow the repo's ``utils/ctm/plots.py`` convention: take the data plus
a keyword-only ``out_path``, render, save at 120 dpi, and close the figure.
The corridor-aggregate plots express power in megawatts (energy divided by
``dt_h``). The per-position profiles divide that by the bin's length as well,
giving a linear power density in MW/mi so that bins of unequal length -- CTM
cells in particular -- are directly comparable; the load factor is a ratio and
so is dimensionless either way. The per-position profiles put position on a
mile axis; ``plot_P_y`` and ``plot_demand_heatmap`` stay in meters, being pad-
and grid-scale views.

The five per-position profiles share the same binning: contiguous bins that
are the CTM cells when ``cell_edges_m`` is given, else uniform
``segment_m``-wide segments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .model import DemandResult

_METERS_PER_MILE = 1609.344


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


def _bin_edges(
    position_m: np.ndarray,
    segment_m: float,
    cell_edges_m: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """Contiguous bin edges over ``position_m``, plus a label for one bin.

    Bins are the CTM cells when ``cell_edges_m`` (an ``(N+1,)`` array of cell
    boundaries in meters) is given, else uniform ``segment_m``-wide segments
    spanning the corridor.
    """
    if cell_edges_m is not None:
        return np.asarray(cell_edges_m, dtype=float), "cell"
    k0 = int(np.floor(position_m.min() / segment_m))
    k1 = int(np.floor(position_m.max() / segment_m))
    return np.arange(k0, k1 + 2) * segment_m, f"{segment_m:g} m segment"


def _binned_density(
    E: np.ndarray,
    position_m: np.ndarray,
    dt_h: float,
    edges: np.ndarray,
) -> np.ndarray:
    """Load density (MW/mi) per timestep and bin: ``(T, n_bins)`` from ``(T, m)``.

    Summing positions into bins suppresses the pad-vs-gap ripple; dividing by
    the bin's length then makes bins of unequal length comparable, so a long
    CTM cell no longer reads as hot merely for being long. Multiply a column
    back by its bin width in miles to recover the MW carried by that bin.

    ``edges`` stays in meters, matching ``position_m``; only the density's
    denominator and the plotted axis are in miles.

    ``position_m`` is ascending, so every bin is a contiguous column slice.
    """
    n_bins = edges.size - 1
    idx = np.clip(np.digitize(position_m, edges) - 1, 0, n_bins - 1)
    bounds = np.searchsorted(idx, np.arange(n_bins + 1))
    binned_MW = np.column_stack([
        E[:, lo:hi].sum(axis=1) for lo, hi in zip(bounds[:-1], bounds[1:])
    ]) / dt_h / 1e6
    return binned_MW / (np.diff(edges) / _METERS_PER_MILE)


def _plot_profile_bars(
    edges: np.ndarray,
    values: np.ndarray,
    *,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
    out_path: Path,
) -> None:
    """Render one per-bin profile as a bar chart against position (mi).

    ``edges`` is in meters, as everywhere else in this module; the conversion
    to miles happens here so every profile shares one axis convention.

    Bars are centered on their bin and drawn to one uniform width. Because the
    values are densities, a bar's height alone is comparable across bins and
    its width carries no information -- so bins of unequal length (CTM cells)
    get equal-width bars rather than length-scaled ones. The width is the
    narrowest bin, which keeps bars from overlapping and, when every bin is
    the same width, reproduces a contiguous tiled profile.
    """
    edges_mi = edges / _METERS_PER_MILE
    widths_mi = np.diff(edges_mi)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(edges_mi[:-1] + widths_mi / 2, values, width=widths_mi.min(),
           align="center", edgecolor="white", lw=0.3)
    ax.set_xlabel("position [mi]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3, axis="y")
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
    per-position load there is ``E[t*, :] / dt_h``, summed into contiguous bins
    and divided by each bin's length. Bins are the CTM cells when
    ``cell_edges_m`` (an ``(N+1,)`` array of cell boundaries in meters) is
    given, else uniform ``segment_m``-wide segments. Because the bars are
    densities, they integrate -- rather than sum -- to the peak corridor power.
    """
    t_star = int(np.argmax(result.E.sum(axis=1)))
    edges, unit = _bin_edges(result.position_m, segment_m, cell_edges_m)
    density = _binned_density(
        result.E[t_star][None, :], result.position_m, dt_h, edges
    )[0]

    _plot_profile_bars(
        edges, density,
        ylabel="power density [MW/mi]",
        title=f"{title} ({unit} bins, t = {t_star * dt_h:.2f} h)",
        out_path=out_path,
    )


def plot_absolute_peak_profile(
    result: DemandResult,
    *,
    dt_h: float,
    segment_m: float = 20.0,
    cell_edges_m: np.ndarray | None = None,
    title: str = "DWPT absolute peak load by position",
    out_path: Path,
) -> None:
    """Plot each bin's largest load density (MW/mi) over all timesteps.

    Unlike ``plot_peak_load_profile``, every bin is read at its own worst
    timestamp, so the bars are non-coincident: each is at least as tall as in
    the peak-instant profile, and together they exceed the peak corridor power
    unless the whole corridor peaks at once.
    """
    edges, unit = _bin_edges(result.position_m, segment_m, cell_edges_m)
    peak_density = _binned_density(
        result.E, result.position_m, dt_h, edges
    ).max(axis=0)

    _plot_profile_bars(
        edges, peak_density,
        ylabel="peak power density [MW/mi]",
        title=f"{title} ({unit} bins)",
        out_path=out_path,
    )


def plot_average_load_profile(
    result: DemandResult,
    *,
    dt_h: float,
    segment_m: float = 20.0,
    cell_edges_m: np.ndarray | None = None,
    title: str = "DWPT average load by position",
    out_path: Path,
) -> None:
    """Plot each bin's load density (MW/mi) averaged over all timesteps.

    The mean runs over every simulated timestep, so a bin's bar is its total
    delivered energy divided by the simulated duration and by its own length.
    """
    edges, unit = _bin_edges(result.position_m, segment_m, cell_edges_m)
    mean_density = _binned_density(
        result.E, result.position_m, dt_h, edges
    ).mean(axis=0)

    _plot_profile_bars(
        edges, mean_density,
        ylabel="mean power density [MW/mi]",
        title=f"{title} ({unit} bins)",
        out_path=out_path,
    )


def plot_load_factor_profile(
    result: DemandResult,
    *,
    dt_h: float,
    segment_m: float = 20.0,
    cell_edges_m: np.ndarray | None = None,
    title: str = "DWPT load factor by position",
    out_path: Path,
) -> None:
    """Plot each bin's load factor -- its mean load over its own peak load.

    A value near 1 means the bin's load is steady; a small value means the bin
    must be sized for a peak it rarely sees. Bins that never see load read 0.
    Bin length cancels in the ratio, so this profile is the one that does not
    change under the MW/mi normalization the other profiles use.
    """
    edges, unit = _bin_edges(result.position_m, segment_m, cell_edges_m)
    density = _binned_density(result.E, result.position_m, dt_h, edges)
    peak = density.max(axis=0)
    load_factor = np.divide(
        density.mean(axis=0), peak,
        out=np.zeros_like(peak), where=peak > 0,
    )

    _plot_profile_bars(
        edges, load_factor,
        ylabel="load factor [-]",
        title=f"{title} ({unit} bins)",
        ylim=(0.0, 1.0),
        out_path=out_path,
    )


def plot_load_distribution_profile(
    result: DemandResult,
    *,
    dt_h: float,
    segment_m: float = 20.0,
    cell_edges_m: np.ndarray | None = None,
    title: str = "DWPT load distribution by position",
    out_path: Path,
) -> None:
    """Box-plot each bin's load density (MW/mi) over all timesteps.

    One box per bin summarizes that bin's load across every simulated
    timestep: median, interquartile range, 1.5-IQR whiskers, and outliers
    beyond them. Boxes sit at their bin's center and share one uniform width,
    matching the bar profiles, so the x-axis reads the same across the set --
    the whisker top is that bin's absolute peak or near it, and the box shows
    how much of the day is spent well below it.

    A long corridor binned into narrow segments yields hundreds of boxes;
    widen the bins (``segment_m``, or ``cell_edges_m``) to keep them legible.
    """
    edges, unit = _bin_edges(result.position_m, segment_m, cell_edges_m)
    density = _binned_density(result.E, result.position_m, dt_h, edges)
    edges_mi = edges / _METERS_PER_MILE
    widths_mi = np.diff(edges_mi)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot(
        density,
        positions=edges_mi[:-1] + widths_mi / 2,
        widths=0.8 * widths_mi.min(),
        manage_ticks=False,  # keep the numeric position axis, not 1..N labels
        flierprops=dict(marker=".", ms=2, mec="none", mfc="C0", alpha=0.3),
    )
    ax.set_xlabel("position [mi]")
    ax.set_ylabel("power density [MW/mi]")
    ax.set_title(f"{title} ({unit} bins)")
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
