"""Reference DWC systems for testing and demos."""

from .model import CorridorSpec, PadSpec


def newbolt_small_scale() -> CorridorSpec:
    """Newbolt 2024a small-scale validation system (Table 1).

    Three Tx pads with the paper's dimensions, laid out as three
    ``(alpha + lambda_gap)`` tiles in a single CTM-equivalent cell.

    ``dx_grid`` is set to 0.0001 m (0.1 mm) — the coarsest grid where
    ``alpha=2 m``, ``delta=1.5 m``, and ``lambda_gap=0.1524 m`` all
    resolve to integer grid units (20 000, 15 000, and 1 524).
    """
    pad = PadSpec(
        alpha=2.0,
        delta=1.5,
        lambda_gap=0.1524,
        beta=150_000.0,
        beta_prime=150_000.0,
    )
    dx_grid = 0.0001
    cell_length = 3 * (pad.alpha + pad.lambda_gap)
    return CorridorSpec(
        cell_lengths_m=(cell_length,),
        pad=pad,
        dx_grid=dx_grid,
    )
