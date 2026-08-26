"""Evaluate the effect of temporal aggregation on the DWPT energy profile.

The DWPT demand ``E`` (Wh per CTM timestep) is a high-resolution load profile.
Reporting it at a coarser temporal resolution -- averaging power over 10-s to
15-min windows, the way a distribution-grid study might -- conserves total
energy but *attenuates the peak power* a feeder must carry, and flattens the
load shape. This script quantifies that, on a CTM case study, for two purposes:

* **Grid sizing** -- how much the corridor-total peak power, and the worst
  local peak at each segment length, is understated at each aggregation
  window.
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
        "--windows-s",
        type=str,
        default="10,30,60,300,600,900",
        help="Comma-separated aggregation windows in seconds "
        "(default 10,30,60,300,600,900)",
    )
    p.add_argument(
        "--segment-lens-mi",
        type=str,
        default="0.25,0.5,1,2.5,5,10,20",
        help="Comma-separated segment lengths for the local-peak metric [mi] "
        "(default 0.25,0.5,1,2.5,5,10,20)",
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

    # Per-segment energy-per-step series at each requested segment length
    # (worst-local-peak metric). Lengths longer than the corridor are dropped --
    # they would just restate the corridor total.
    corridor_mi = corridor.corridor_length_m / 1609.344
    requested_lens = [float(x) for x in args.segment_lens_mi.split(",")]
    segment_lens_mi = [L for L in requested_lens if L <= corridor_mi]
    for L in requested_lens:
        if L > corridor_mi:
            print(f"note: skipping {L:g} mi segments (corridor is {corridor_mi:.2f} mi)")

    segment_e_step: dict[float, dict[int, np.ndarray]] = {}
    segment_native_peak: dict[float, dict[int, float]] = {}
    for L in segment_lens_mi:
        cols = agg.mile_segment_columns(
            corridor.position_m, corridor.corridor_length_m, segment_len_mi=L
        )
        segment_e_step[L] = {c: E[:, idx].sum(axis=1) for c, idx in cols.items()}
        segment_native_peak[L] = {
            c: agg.window_power(e, 1, dt_h)[0].max()
            for c, e in segment_e_step[L].items()
        }

    # --- Sweep the aggregation windows ------------------------------------
    requested_windows_s = [float(w) for w in args.windows_s.split(",")]
    windows_s = [w for w in requested_windows_s if w >= dt_s]
    for w in requested_windows_s:
        if w < dt_s:
            print(f"note: skipping {w:g} s window (native dt is {dt_s:g} s)")
    # Reference row (native dt) first, then each requested window. A window is
    # realized as a whole number of native steps, so its actual duration
    # (``window_actual_s``) can differ from the requested one.
    ladder = [("native", dt_s, 1)] + [
        (_window_label(w), w, round(w / dt_s)) for w in windows_s
    ]

    rows = []
    window_power_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, w_s, steps in ladder:
        power, counts = agg.window_power(e_total, steps, dt_h)
        window_power_series[label] = (power, counts)
        pk, idx = agg.peak(power)
        pk_time = agg.bin_center_times_h(counts, dt_h)[idx]

        ldc = agg.ldc_divergence(
            native_power, native_counts, power, counts
        )
        binned_energy = float((power * counts * dt_h).sum())

        row = {
            "window": label,
            "window_s": w_s,
            "window_actual_s": steps * dt_s,
            "peak_power_kW": pk / 1e3,
            "peak_pct_atten": agg.pct_attenuation(native_peak, pk),
            "peak_time_h": pk_time,
            "peak_shift_min": (pk_time - native_peak_time_h) * 60.0,
            "ldc_divergence": ldc,
            "total_energy_kWh": binned_energy / 1e3,
            "energy_rel_err": (binned_energy - total_energy_wh) / total_energy_wh,
        }

        # Worst local peak attenuation at each segment length.
        for L in segment_lens_mi:
            segment_atten = {
                c: agg.pct_attenuation(
                    segment_native_peak[L][c],
                    agg.window_power(e, steps, dt_h)[0].max(),
                )
                for c, e in segment_e_step[L].items()
            }
            worst_segment = max(segment_atten, key=segment_atten.get)
            row[f"worst_seg_{L:g}mi_idx"] = worst_segment
            row[f"worst_seg_{L:g}mi_pct_atten"] = segment_atten[worst_segment]

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "temporal_aggregation_summary.csv"
    df.to_csv(csv_path, index=False)

    _plot_load_profiles(window_power_series, dt_h, out_dir / "load_profiles.png")
    _plot_peak_attenuation(df, segment_lens_mi, out_dir / "peak_attenuation.png")
    _plot_load_duration_curves(
        window_power_series, out_dir / "load_duration_curves.png"
    )

    _print_summary(args, result, corridor, df, total_energy_wh, segment_lens_mi, out_dir)


def _window_label(window_s: float) -> str:
    """Human-readable window label: sub-minute in seconds, otherwise minutes."""
    return f"{window_s:g}s" if window_s < 60.0 else f"{window_s / 60.0:g}min"


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


def _plot_peak_attenuation(df, segment_lens_mi, out_path):
    """Peak understatement (%) vs window: corridor total and each segment length.

    Segment length is an ordered magnitude, so the segment series share one hue
    ramp (short segments light, long segments dark); the corridor total is a
    separate black reference line.
    """
    agg_rows = df[df["window"] != "native"]
    x = agg_rows["window_actual_s"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    shades = plt.cm.Blues(np.linspace(0.4, 0.95, len(segment_lens_mi)))
    for L, color in zip(segment_lens_mi, shades):
        ax.plot(
            x, agg_rows[f"worst_seg_{L:g}mi_pct_atten"],
            "s--", color=color, lw=1.5, ms=4, label=f"worst {L:g} mi segment",
        )
    ax.plot(x, agg_rows["peak_pct_atten"], "o-", color="k", lw=2,
            ms=5, label="corridor total")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(agg_rows["window"])
    ax.minorticks_off()  # log minor ticks would be unlabeled
    ax.set_xlabel("aggregation window")
    ax.set_ylabel("peak power understatement [%]")
    ax.set_title("DWPT peak attenuation vs temporal aggregation")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
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


def _print_summary(args, result, corridor, df, total_energy_wh, segment_lens_mi, out_dir):
    dt_s = result.freeway.dt * 3600.0
    native = df[df["window"] == "native"].iloc[0]
    seg_header = "".join(f"{L:g}mi%".rjust(8) for L in segment_lens_mi)
    lines = [
        "=== DWPT temporal-aggregation analysis ===",
        f"  CTM result        : {args.ctm_result}",
        f"  native dt         : {dt_s:g} s   |  horizon {result.n_steps * result.freeway.dt:g} h",
        f"  corridor          : {corridor.corridor_length_m / 1609.344:.2f} mi, "
        f"segment lengths {', '.join(f'{L:g}' for L in segment_lens_mi)} mi",
        f"  eta_EV            : {args.eta_ev}   |  total energy {total_energy_wh / 1e3:.1f} kWh",
        f"  native peak power : {native['peak_power_kW']:.1f} kW "
        f"@ t={native['peak_time_h']:.2f} h",
        "",
        "  worst local peak understatement (%) by segment length:",
        f"  window   peak[kW]  atten%{seg_header}  peakShift[min]  ldcDiv  energyErr",
    ]
    for _, r in df[df["window"] != "native"].iterrows():
        seg_cells = "".join(
            f"{r[f'worst_seg_{L:g}mi_pct_atten']:8.1f}" for L in segment_lens_mi
        )
        lines.append(
            f"  {r['window']:>6}  {r['peak_power_kW']:8.1f}  {r['peak_pct_atten']:6.1f}"
            f"{seg_cells}  {r['peak_shift_min']:13.1f}"
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
