"""Step 5 helper: regenerate cell artifacts after manual edits to ``cells.csv``.

Step 1 (``scripts/build_ctm_from_osm.py``) writes:

* ``cells.csv``         -- the cell table.
* ``cells.geojson``     -- one LineString per cell.
* ``corridor_map.png``  -- lat/lon overlay of the mainline + ramps.
* ``cell_layout.png``   -- strip plot of cells along postmile.
* ``mainline.geojson`` + ``ramps.geojson`` + ``corridor.json``
                        -- round-trip sidecars (the corridor's geometry +
                        metadata) so this script can rebuild everything
                        without re-running the osmnx extraction.

Users hand-edit ``cells.csv`` during Step 5 (Cell Length Tuning): merging
short cells, splitting long ones, or nudging postmiles. After editing, run::

    python scripts/regenerate_cell_artifacts.py \\
        --dir scripts/output/ctm_corridor/I_210_W

to rebuild ``cells.geojson``, ``corridor_map.png``, and ``cell_layout.png``
from the edited table. The ``length_warning`` column is recomputed from
the edited lengths so users can immediately see whether their adjustments
brought every cell inside the CFL-friendly bounds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ctm_for_dwc.utils.ctm.cells import (
    cells_to_geodataframe,
    corridor_from_artifacts,
    flag_cell_length_warnings,
)
from ctm_for_dwc.utils.ctm.plots import (
    plot_cell_layout,
    plot_corridor_map,
)


def regenerate_cell_artifacts(
    in_dir: Path | str,
    *,
    out_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Rebuild cells.geojson + corridor_map.png + cell_layout.png from cells.csv.

    Reads:
      * ``<in_dir>/cells.csv``
      * ``<in_dir>/mainline.geojson`` + ``ramps.geojson`` + ``corridor.json``
        (the round-trip sidecars from ``corridor_to_artifacts``)

    Writes (defaulting to ``in_dir`` if ``out_dir`` is omitted):
      * ``cells.geojson``    -- cells geometry rebuilt from edited postmiles.
      * ``corridor_map.png`` -- redrawn from the cached corridor.
      * ``cell_layout.png``  -- strip plot from the edited cells table.

    The ``length_warning`` column on cells.csv is **recomputed** (so it stays
    consistent with the edited ``length`` values) and the updated table is
    written back to ``out_dir/cells.csv``.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir) if out_dir is not None else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    corridor = corridor_from_artifacts(in_dir)
    cells_df = pd.read_csv(in_dir / "cells.csv")
    cells_df = flag_cell_length_warnings(cells_df)

    cells_df.to_csv(out_dir / "cells.csv", index=False)
    cells_gdf = cells_to_geodataframe(cells_df, corridor)
    cells_gdf.to_file(out_dir / "cells.geojson", driver="GeoJSON")
    plot_corridor_map(corridor, out_dir / "corridor_map.png")
    plot_cell_layout(corridor, cells_df, out_dir / "cell_layout.png")

    return {
        "cells_csv": out_dir / "cells.csv",
        "cells_geojson": out_dir / "cells.geojson",
        "corridor_map_png": out_dir / "corridor_map.png",
        "cell_layout_png": out_dir / "cell_layout.png",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dir", required=True, type=Path,
        help="Step-1 output directory containing cells.csv + sidecars.",
    )
    parser.add_argument(
        "--out-dir", default=None, type=Path,
        help="Where to write the regenerated artifacts (default: --dir, in-place).",
    )
    args = parser.parse_args()
    written = regenerate_cell_artifacts(args.dir, out_dir=args.out_dir)
    print(f"Regenerated from {args.dir}/")
    for label, path in written.items():
        print(f"  {label}: {path.name}")


if __name__ == "__main__":
    main()
