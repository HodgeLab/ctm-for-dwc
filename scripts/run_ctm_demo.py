"""Standalone CTM demo: Kurzhanskiy diss. Ch. 3, Example 1 (p. 46-47).

Two identical cells (v_f=60, w=20, rho_crit=100, rho_jam=400, q_max=6000) with
upstream inflow r_0 = 4800 veh/h and an on-ramp r_2 = 1200 veh/h at the
downstream cell. Cell 2 is the only bottleneck (f_2 = q_max).

Expected steady states (from Kurzhanskiy 2007):
    start="empty"  ->  uncongested equilibrium rho^u   = (80, 100)
    start="jam"    ->  most congested equilibrium rho^c = (160, 160)

Run from the repo root:
    python scripts/run_ctm_demo.py
"""

from __future__ import annotations

from transportation_models.utils.ctm import compute_metrics, examples, simulate


def report(start: str, steps: int = 2400) -> None:
    fwy = examples.example_1_freeway()
    scn = examples.example_1_scenario(steps=steps, start=start)
    res = simulate(fwy, scn)
    metrics = compute_metrics(res)
    expected = {"empty": (80.0, 100.0), "jam": (160.0, 160.0)}[start]

    print(f"\n=== Example 1: start='{start}', dt={fwy.dt * 3600:.0f}s, "
          f"{steps} steps ({steps * fwy.dt:.1f} h) ===")
    print(f"  expected equilibrium rho = ({expected[0]:.0f}, {expected[1]:.0f})  veh/mi\n")
    print(f"  {'t [h]':>6}   {'rho_0':>7}  {'rho_1':>7}    {'f_0':>7}  {'f_1':>7}  veh/h")
    print(f"  {'-' * 6}   {'-' * 7}  {'-' * 7}    {'-' * 7}  {'-' * 7}")
    snap = [k for k in (0, 30, 60, 120, 300, 600, 1200, steps) if k <= steps]
    for k in snap:
        t_h = k * fwy.dt
        r0, r1 = res.density[:, k]
        if k < steps:
            f0, f1 = res.mainline_flow[:, k]
            print(f"  {t_h:6.2f}   {r0:7.3f}  {r1:7.3f}    {f0:7.1f}  {f1:7.1f}")
        else:
            print(f"  {t_h:6.2f}   {r0:7.3f}  {r1:7.3f}    {'(final)':>15}")
    err0 = abs(res.density[0, -1] - expected[0])
    err1 = abs(res.density[1, -1] - expected[1])
    print(f"\n  |rho_final - rho_expected| = ({err0:.4g}, {err1:.4g}) veh/mi")

    # Final-step performance metrics (eqs. 4.9-4.17).
    print(f"\n  Final-step metrics (eqs. 4.9-4.17):")
    print(f"    travel_time      = {metrics.travel_time[-1] * 60:8.4f}  min")
    print(f"    VHT              = {metrics.vht[-1]:8.4f}  veh*h")
    print(f"    VMT              = {metrics.vmt[-1]:8.4f}  veh*mi")
    print(f"    delay            = {metrics.delay[-1]:8.4f}  veh*h")
    print(f"    productivity_loss= {metrics.productivity_loss[-1]:8.4f}  mi*h")


if __name__ == "__main__":
    report("empty")
    report("jam")
