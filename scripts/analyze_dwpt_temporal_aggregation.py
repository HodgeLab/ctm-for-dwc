"""Evaluate the effect of temporal aggregation on the DWPT energy profile.

The DWPT demand ``E`` (Wh per CTM timestep) is a high-resolution load profile.
Reporting it at a coarser temporal resolution -- averaging power over 5-, 15-,
30-, 60-min windows, the way a distribution-grid study might -- conserves total
energy but *attenuates the peak power* a feeder must carry, and flattens the
load shape. This script quantifies that, on a CTM case study, for two purposes:

* **Grid sizing** -- how much the corridor-total (and worst per-mile-segment)
  peak power is understated at each aggregation window.
* **Modeling fidelity** -- how the load-duration shape departs from the
  native-``dt`` baseline (peak-timing shift, load-duration-curve divergence).

Baseline is the CTM's native ``dt``; windows aggregate up from there. Energy
conservation is reported as a sanity check (should be ~0).

Run from the repo root, e.g.:

    python scripts/analyze_dwpt_temporal_aggregation.py \\
        --ctm-result case_studies/I880N_10mi/histAve_ramps/result.npz

Pad geometry / ``eta_EV`` / ``dx_grid`` default to the study's values but are
all CLI-overridable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transportation_models.utils.ctm.results import SimulationResult
from transportation_models.utils.dwpt import aggregation as agg
from transportation_models.utils.dwpt.adapter import corridor_from_ctm, mainline_vht
from transportation_models.utils.dwpt.demand import compute
from transportation_models.utils.dwpt.model import PadSpec


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--ctm-result",
        type=Path,
        default=Path("case_studies/I880N_10mi/histAve_ramps/result.npz"),
        help="Path to the CTM simulation .npz (default: I880N_10mi/histAve_ramps)",
    )
    p.add_argument("--alpha", type=float, default=2.0, help="Tx pad length [m]")
    p.add_argument("--lambda-gap", type=float, default=0.5, help="Tx pad gap [m]")
    p.add_argument("--delta", type=float, default=1.5, help="Rx pad length [m]")
    p.add_argument("--beta", type=float, default=150000.0, help="Rx rated power [W]")
    p.add_argument(
        "--beta-prime", type=float, default=150000.0, help="Tx rated power [W]"
    )
    p.add_argument("--dx-grid", type=float, default=0.5, help="Spatial grid [m]")
    p.add_argument("--eta-ev", type=float, default=0.25, help="EV fraction [0,1]")
    p.add_argument(
        "--windows-min",
        type=str,
        default="1,5,15,30,60",
        help="Comma-separated aggregation windows in minutes (default 1,5,15,30,60)",
    )
    p.add_argument(
        "--segment-len-mi",
        type=float,
        default=1.0,
        help="Per-segment length for the local-peak metric [mi] (default 1.0)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <ctm-result-dir>/aggregation)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.ctm_result.exists():
        raise SystemExit(f"--ctm-result not found: {args.ctm_result}")

    result = SimulationResult.from_npz(args.ctm_result.resolve())
    dt_h = result.freeway.dt
    dt_s = dt_h * 3600.0

    out_dir = (
        args.out_dir if args.out_dir is not None
        else args.ctm_result.parent / "aggregation"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build the native-resolution DWPT profile -------------------------
    pad = PadSpec(
        alpha=args.alpha, delta=args.delta, lambda_gap=args.lambda_gap,
        beta=args.beta, beta_prime=args.beta_prime,
    )
    corridor = corridor_from_ctm(result, pad, args.dx_grid)
    demand = compute(mainline_vht(result), corridor, args.eta_ev)
    E = demand.E  # (T, m) Wh

    # Corridor-total energy-per-step, and native power series + its weights.
    e_total = agg.energy_per_step(E)  # (T,) Wh
    native_power, native_counts = agg.window_power(e_total, 1, dt_h)
    native_peak, native_idx = agg.peak(native_power)
    native_peak_time_h = agg.bin_center_times_h(native_counts, dt_h)[native_idx]
    total_energy_wh = float(e_total.sum())

    # Per-mile-segment energy-per-step series (worst-local-peak metric).
    segments = agg.mile_segment_columns(
        corridor.position_m, corridor.corridor_length_m, segment_len_mi=args.segment_len_mi
    )
    segment_e_step = {c: E[:, cols].sum(axis=1) for c, cols in segments.items()}
    segment_native_peak = {
        c: agg.window_power(e, 1, dt_h)[0].max() for c, e in segment_e_step.items()
    }

    # --- Sweep the aggregation windows ------------------------------------
    windows_min = [float(w) for w in args.windows_min.split(",")]
    native_min = dt_s / 60.0
    # Reference row (native dt) first, then each requested window.
    ladder = [("native", native_min, 1)] + [
        (f"{w:g}min", w, max(1, round(w * 60.0 / dt_s))) for w in windows_min
    ]

    rows = []
    window_power_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, w_min, steps in ladder:
        power, counts = agg.window_power(e_total, steps, dt_h)
        window_power_series[label] = (power, counts)
        pk, idx = agg.peak(power)
        pk_time = agg.bin_center_times_h(counts, dt_h)[idx]

        # Worst per-mile-segment peak attenuation at this window.
        segment_atten = {
            c: agg.pct_attenuation(
                segment_native_peak[c], agg.window_power(e, steps, dt_h)[0].max()
            )
            for c, e in segment_e_step.items()
        }
        worst_segment = max(segment_atten, key=segment_atten.get) if segment_atten else -1
        worst_atten = segment_atten[worst_segment] if segment_atten else 0.0

        ldc = agg.ldc_divergence(
            native_power, native_counts, power, counts
        )
        binned_energy = float((power * counts * dt_h).sum())

        rows.append(
            {
                "window": label,
                "window_min": w_min,
                "peak_power_kW": pk / 1e3,
                "peak_pct_atten": agg.pct_attenuation(native_peak, pk),
                "peak_time_h": pk_time,
                "peak_shift_min": (pk_time - native_peak_time_h) * 60.0,
                "worst_segment_idx": worst_segment,
                "worst_segment_pct_atten": worst_atten,
                "ldc_divergence": ldc,
                "total_energy_kWh": binned_energy / 1e3,
                "energy_rel_err": (binned_energy - total_energy_wh)
                / total_energy_wh,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "temporal_aggregation_summary.csv"
    df.to_csv(csv_path, index=False)

    _plot_load_profiles(window_power_series, dt_h, out_dir / "load_profiles.png")
    _plot_peak_attenuation(df, out_dir / "peak_attenuation.png")
    _plot_load_duration_curves(
        window_power_series, out_dir / "load_duration_curves.png"
    )

    _print_summary(args, result, corridor, df, total_energy_wh, len(segments), out_dir)


def _plot_load_profiles(series, dt_h, out_path):
    """Overlay corridor-total power vs time-of-day at each resolution."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, (power, counts) in series.items():
        t = agg.bin_center_times_h(counts, dt_h)
        lw = 0.7 if label == "native" else 1.4
        ax.step(t, power / 1e3, where="mid", lw=lw, label=label)
    ax.set_xlabel("time of day [h]")
    ax.set_ylabel("corridor DWPT power [kW]")
    ax.set_title("Corridor-total DWPT load vs temporal resolution")
    ax.grid(alpha=0.3)
    ax.legend(title="window", ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_peak_attenuation(df, out_path):
    """Peak understatement (%) vs window: corridor-total and worst mile segment."""
    agg_rows = df[df["window"] != "native"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        agg_rows["window_min"], agg_rows["peak_pct_atten"],
        "o-", label="corridor total",
    )
    ax.plot(
        agg_rows["window_min"], agg_rows["worst_segment_pct_atten"],
        "s--", label="worst mile segment",
    )
    ax.set_xlabel("aggregation window [min]")
    ax.set_ylabel("peak power understatement [%]")
    ax.set_title("DWPT peak attenuation vs temporal aggregation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_load_duration_curves(series, out_path):
    """Overlay load-duration curves at each resolution."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, (power, counts) in series.items():
        xs, ldc = agg.load_duration_curve(power, counts)
        lw = 0.9 if label == "native" else 1.4
        ax.plot(xs * 100.0, ldc / 1e3, lw=lw, label=label)
    ax.set_xlabel("% of horizon power is exceeded")
    ax.set_ylabel("corridor DWPT power [kW]")
    ax.set_title("DWPT load-duration curve vs temporal resolution")
    ax.grid(alpha=0.3)
    ax.legend(title="window", ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _print_summary(args, result, corridor, df, total_energy_wh, n_segments, out_dir):
    dt_s = result.freeway.dt * 3600.0
    native = df[df["window"] == "native"].iloc[0]
    lines = [
        "=== DWPT temporal-aggregation analysis ===",
        f"  CTM result        : {args.ctm_result}",
        f"  native dt         : {dt_s:g} s   |  horizon {result.n_steps * result.freeway.dt:g} h",
        f"  corridor          : {corridor.corridor_length_m / 1609.344:.2f} mi, "
        f"{n_segments} segment(s) @ {args.segment_len_mi:g} mi",
        f"  eta_EV            : {args.eta_ev}   |  total energy {total_energy_wh / 1e3:.1f} kWh",
        f"  native peak power : {native['peak_power_kW']:.1f} kW "
        f"@ t={native['peak_time_h']:.2f} h",
        "",
        "  window   peak[kW]  atten%  worstSegment%  peakShift[min]  ldcDiv  energyErr",
    ]
    for _, r in df[df["window"] != "native"].iterrows():
        lines.append(
            f"  {r['window']:>6}  {r['peak_power_kW']:8.1f}  {r['peak_pct_atten']:6.1f}"
            f"  {r['worst_segment_pct_atten']:10.1f}  {r['peak_shift_min']:13.1f}"
            f"  {r['ldc_divergence']:6.3f}  {r['energy_rel_err']:+.2e}"
        )
    summary = "\n".join(lines) + "\n"
    (out_dir / "temporal_aggregation_summary.txt").write_text(summary)
    print()
    print(summary)
    print(f"Wrote {out_dir}/")
    for name in (
        "temporal_aggregation_summary.csv",
        "temporal_aggregation_summary.txt",
        "load_profiles.png",
        "peak_attenuation.png",
        "load_duration_curves.png",
    ):
        print(f"  {name}")


if __name__ == "__main__":
    main()
