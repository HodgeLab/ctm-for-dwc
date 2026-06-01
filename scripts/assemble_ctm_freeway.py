"""Step 6: assemble a CTM-ready freeway table from Step 1+2 + Step 4 outputs.

This script joins three artifacts and writes a single per-cell freeway CSV
that slots directly into :func:`utils.ctm.io.freeway_from_dataframe`:

  * ``cells.csv``                       (Step 1+2)  -- cell geometry +
                                                       mainline/ramp VDS ids.
  * ``station_metadata_calibrated.csv`` (Step 4)    -- per-VDS triangular
                                                       FD parameters
                                                       (per-lane).
  * ``ramp_metadata_calibrated.csv``    (Step 4)    -- per-VDS ramp
                                                       capacity (total
                                                       flow, optional).

The assembly:

  1. Joins each cell's ``vds_id`` against the mainline calibration to
     look up FD params.
  2. **Scales per-lane FD quantities (q_max, rho_jam, rho_crit) to per-cell
     totals by multiplying by the cell's lane count.** Free-flow speed
     and congestion-wave speed are intensive and pass through unchanged.
  3. Looks up on/off-ramp capacities by ``on_ramp_vds_id`` /
     ``off_ramp_vds_id``. Cells with no calibrated ramp capacity get NaN,
     which :func:`freeway_from_dataframe` translates into infinite ramp
     capacity (the "no measurement -> no constraint" fall-back, valid in
     the CTM).
  4. Prints a CFL Δt advisory: per-cell :math:`l_i / v_{f,i}`, the
     binding (most-restrictive) cell, and a suggested sub-5-s round Δt.
     **The script stops short of building a Freeway** -- the user picks
     ``dt`` from the advisory and feeds the CSV to
     ``freeway_from_dataframe`` themselves at sim time.

Run from the repo root::

    python scripts/assemble_ctm_freeway.py \\
        --cells scripts/output/ctm_corridor/I_210_W/cells.csv \\
        --calibrated data/pems/calibrated/station_metadata_calibrated.csv \\
        --ramp-calibrated data/pems/calibrated/ramp_metadata_calibrated.csv \\
        --out scripts/output/ctm_corridor/I_210_W/freeway.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from transportation_models.utils.ctm.assembly import (
    assemble_freeway_table,
    compute_cfl_advisory,
    format_cfl_advisory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cells", required=True, type=Path,
        help="Step 1+2 cells.csv (the user-edited / regenerated version).",
    )
    parser.add_argument(
        "--calibrated", required=True, type=Path,
        help=("Step 4 mainline calibration CSV "
              "(station_metadata_calibrated.csv)."),
    )
    parser.add_argument(
        "--ramp-calibrated", default=None, type=Path,
        help=("Step 4 ramp calibration CSV (ramp_metadata_calibrated.csv). "
              "Optional -- cells without a matched ramp VDS get infinite "
              "ramp capacity, which the CTM allows."),
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Destination freeway.csv path.",
    )
    args = parser.parse_args()

    cells_df = pd.read_csv(args.cells)
    calibrated_df = pd.read_csv(args.calibrated)
    ramp_calibrated_df = (
        pd.read_csv(args.ramp_calibrated) if args.ramp_calibrated else None
    )

    freeway_df = assemble_freeway_table(
        cells_df, calibrated_df, ramp_calibrated_df=ramp_calibrated_df,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    freeway_df.to_csv(args.out, index=False)
    print(f"freeway table -> {args.out}  ({len(freeway_df)} cells)",
          file=sys.stderr)

    advisory = compute_cfl_advisory(freeway_df)
    print()
    print(format_cfl_advisory(advisory))


if __name__ == "__main__":
    main()
