"""Generate mock PeMS 5-minute timeseries CSVs for a handful of VDS stations.

Writes one ``{vds}.csv`` per station under ``mock_data/pems_timeseries/`` using
the same column schema the preprocessing pipeline consumes
(``timestamp, station, pct_observed, total_flow_[veh/5-min], avg_speed_[mph]``,
see ``PeMSDataProcessor.timeseries_cols``). Flow follows a typical weekday
diurnal shape with AM/PM peaks; speed drops as flow approaches capacity.
"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd

VDS_IDS = [400662, 400141, 402960]
DAY = "2025-01-07"  # a representative weekday (Tuesday)
FREEFLOW_MPH = 65.0
_REPO = Path(__file__).resolve().parents[1]
OUT_DIR = _REPO / "data" / "pems" / "mock_data"


def _diurnal_flow(hours: np.ndarray) -> np.ndarray:
    """Flow [veh/5-min] over the day: two Gaussian commute peaks + a base."""
    am_peak = np.exp(-0.5 * ((hours - 7.75) / 1.1) ** 2)
    pm_peak = np.exp(-0.5 * ((hours - 17.25) / 1.3) ** 2)
    base = 0.12 + 0.10 * np.sin((hours - 3) / 24 * 2 * np.pi)  # daytime > night
    shape = np.clip(base + 0.95 * am_peak + 1.0 * pm_peak, 0.03, None)
    return shape  # normalized 0..~1, scaled per-station below


def generate_station(vds: int, rng: np.random.Generator) -> pd.DataFrame:
    timestamps = pd.date_range(start=DAY, periods=288, freq="5min")
    hours = timestamps.hour + timestamps.minute / 60.0

    # Per-station peak capacity so the three detectors differ a little.
    peak_flow = rng.uniform(140, 175)  # veh/5-min at the busiest interval
    shape = _diurnal_flow(np.asarray(hours))
    flow = shape / shape.max() * peak_flow
    flow = np.maximum(0.0, flow + rng.normal(0, 3.0, len(flow)))

    # Speed: free-flow when uncongested, drops as flow nears the peak.
    saturation = flow / peak_flow
    speed = FREEFLOW_MPH - 28.0 * np.clip(saturation - 0.55, 0, None) / 0.45
    speed = np.clip(speed + rng.normal(0, 1.5, len(speed)), 18.0, FREEFLOW_MPH)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "station": vds,
            "pct_observed": 100.0,
            "total_flow_[veh/5-min]": np.round(flow, 1),
            "avg_speed_[mph]": np.round(speed, 1),
        }
    )


# %%
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for vds in VDS_IDS:
        # Seed off the VDS id so output is reproducible per station.
        rng = np.random.default_rng(vds)
        df = generate_station(vds, rng)
        out = OUT_DIR / f"{vds}.csv"
        df.to_csv(out, index=False)
        print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
