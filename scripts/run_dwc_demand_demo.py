"""Demo: tiny CTM run -> DWC demand -> plots.

Runs the four-cell CTM example (Kurzhanskiy diss. §3.4), maps it through the
adapted mCONV demand module at a coarse position grid, and writes two plots:
the corridor power-vs-position profile and the demand space-time heatmap.
Output defaults to ``scripts/output/dwc_demand_demo/`` (gitignored).

    python scripts/run_dwc_demand_demo.py [--out-dir <dir>] [--steps <n>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transportation_models.utils.ctm.engine import simulate
from transportation_models.utils.ctm.examples import (
    four_cell_freeway,
    four_cell_scenario,
)
from transportation_models.utils.dwc import plots
from transportation_models.utils.dwc.adapter import corridor_from_ctm, mainline_vht
from transportation_models.utils.dwc.demand import compute
from transportation_models.utils.dwc.model import PadSpec

# Coarse pad + grid so the dense T_R over four 1-mile cells stays small.
_PAD = PadSpec(
    alpha=20.0, delta=20.0, lambda_gap=10.0, beta=150_000.0, beta_prime=150_000.0
)
_DX_GRID = 10.0
_ETA_EV = 0.3

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO / "scripts" / "output" / "dwc_demand_demo"


def main(out_dir: Path, *, steps: int = 60) -> Path:
    """Run the demo and write both plots to ``out_dir``; returns ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = simulate(four_cell_freeway(), four_cell_scenario(steps=steps))
    corridor = corridor_from_ctm(result, _PAD, _DX_GRID)
    demand = compute(mainline_vht(result), corridor, _ETA_EV)

    plots.plot_P_y(
        corridor.build_P_y(), corridor.position_m,
        out_path=out_dir / "dwc_P_y.png",
    )
    plots.plot_demand_heatmap(
        demand, dt_h=result.freeway.dt, out_path=out_dir / "dwc_demand_heatmap.png",
    )
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    written = main(args.out_dir, steps=args.steps)
    print(f"Wrote dwc_P_y.png and dwc_demand_heatmap.png to {written}")
