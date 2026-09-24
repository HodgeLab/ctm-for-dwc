# Spec: DWC Macro-vs-Micro Validation Harness (P7)

> **SUPERSEDED.** The code this document specifies — `microsim/` and
> `scripts/run_dwc_validation.py` — was removed when the repository was
> refocused on the CTM-corridor-simulation → DWC-charging-demand workstream.
> The macro-vs-micro validation is not part of that workstream. This spec is
> kept for reference only; recover the implementation with
> `git show pre-cleanup:<path>`.

> Status: **DRAFT — awaiting review.** This is the Phase-1 specification for P7
> of [dwc_demand_implementation_plan.md](dwc_demand_implementation_plan.md).
> The plan/spec for the DWC demand module itself remains
> [dwc_demand_module.md](dwc_demand_module.md); this document specifies only
> the validation harness and the microscopic reference it requires.

## Objective

Build a **two-stage accuracy-vs-efficiency study** that grounds this repo's
macroscopic (CTM + adapted mCONV) DWC-demand pipeline against a from-scratch
implementation of the Newbolt microscopic simulation.

**Source lineage (two papers, same authors).** The plan's "Newbolt 2024b" is
*DWC-Integrated Microscopic Traffic Flow for Distribution Grid Voltage
Stability Analysis* (NAPS 2024) — the two-lane microsim, but with a crude binary
`±v` car-following rule (its eq 2). Its expanded successor, *A Novel Approach to
Dual-Lane Microscopic Simulations…* (IEEE TTE 12(1), 2026), replaces that with a
**modified Gipps** longitudinal model (eq 1–5). This harness uses the **2026
modified Gipps** car-following (the rigorous version, approved for single-lane)
and takes the **initial/desired velocity distribution U(45, 90 mph)** from the
2024 NAPS paper. Equation numbers below refer to the 2026 paper unless noted. The study is run across **multiple real corridors of
increasing length** (first: the prepared `scripts/output/1mi_case_study`,
I-880 N).

The deliverable is a **research result** (numbers + figures quantifying the
trade-off), and the implementation that produces it is **test-backed**.

### The two stages

- **Stage 1 — traffic fidelity.** Run the microsim and the CTM on the same
  corridor; score *each* one's density and flow against **PeMS** ground truth
  using the **existing `validation.csv` schema** (per-cell `density_rmse`,
  `density_mape`, `flow_rmse`, `flow_mape` vs. the assigned VDS detectors);
  add **wall-clock compute cost** per method. The CTM side already exists
  (`compare_against_historical`); the new work is producing the **micro side
  in the same format** — which requires aggregating microscopic trajectories
  into macroscopic density/flow at the fixed VDS locations.
- **Stage 2 — demand fidelity.** Compare **original mCONV** (charging-load
  convolution integrated over micro trajectories, Newbolt eq 30–32) against
  the **adapted mCONV** (`E = η_EV · VHT · P_y`, the repo's `compute`). Pin the
  **uniform-density limit as a regression test** (corridor-aggregate energy
  relative error `< 1e-6`); **quantify the divergence** in realistic
  (non-uniform) regimes. Do *not* run the original mCONV on CTM-resampled
  trajectories — the two stages isolate concerns by construction (Stage 1 =
  traffic physics vs. PeMS; Stage 2 = the mCONV adaptation).

### Microsim scope

> **Revised (v2).** The original single-lane scope could not match the CTM's
> multi-lane capacity in congestion (a literal single lane caps near ~2000
> veh/h vs. I-880's 4-lane `q_max ≈ 6155`), which defeats the goal of scoring
> both models against *multi-lane* PeMS data over congested periods. Scope is
> revised to a real **N-lane** model. See "Assumptions surfaced in review" and
> the updated Resolved Decisions.

In:
- **Modified Gipps longitudinal car-following** (Newbolt eq 1–5), with **paper
  parameter values** (`a`, `b`, `s_o`, lengths from Newbolt Tables II/IV;
  L1 Tesla Model 3: `a = 4.79`, `b = 2.50 m/s²`, `s_o = 25 m`), CLI-overridable.
- **N-lane with lane-changing** — `N` = the corridor's PeMS lane count
  (4 for I-880), using Newbolt's incentive + random lane-change model
  (eq 6–14, Table I, Alg 3) generalized from 2 to `N` lanes. Required to
  reproduce multi-lane flow and congestion for a fair comparison to PeMS.
- **DWC in all lanes; EVs have no charging-lane preference.** Tx pads span
  every lane, so the macro (lane-agnostic `E = η_EV · VHT · P_y`) stays
  comparable in Stage 2. Consequently EVs and non-EVs share the **same**
  lane-change logic — Newbolt's EV charging-lane-seeking incentive (eq 13) is
  **not** reproduced (documented deviation; it presumes a single charging lane).
- **Charging-load convolution only** (eq 30–32): per-vehicle `P_y` integrated
  over time. **No** SOC, discharge, HVAC, lighting, or priority cutoff.
- **Vehicle seeding from the raw PeMS timeseries** (the same source the CTM
  inflow derives from, for a fair comparison), via a **Poisson arrival process**
  with rate from the measured PeMS flow (not Newbolt's TTF MC — see decision 8).
  Each vehicle's Gipps cap `v_MAX` (= initial speed) is drawn from
  `U(v_f − 15, v_f + 15) mph` (uniform per 2024 NAPS, re-centered on `v_f`).
- **Micro→macro aggregator** producing per-cell density (veh/mi) and flow
  (veh/h), **summed/averaged across lanes**, on the 5-min PeMS grid.

Out of scope, **architecture left open**:
- EV discharge / HVAC / lighting / SOC battery stack + priority-charging
  cutoff (eq 15–29, 33–39).
- Lane-specific DWC (single charging lane + Newbolt EV-seeking, eq 13) and a
  matching lane-specific macro mCONV — the faithful-Newbolt alternative to
  "DWC all lanes", deferred to keep Stage 2 macro-comparable now.
- Per-class fleet mix (Newbolt's 10 vehicle models, Tables II–III); a single
  homogeneous class is used, consistent with the DWC module's v0.

## Inputs (per corridor)

A prepared case-study directory under `scripts/output/<name>/`, as already
materialized for `1mi_case_study` (I-880 N):

| Artifact | Used for |
|---|---|
| `corridor.json` | corridor `ref`/`direction`, Caltrans postmiles |
| `cells.csv` | per-cell `length`, `lanes`, `vds_id`, `vds_lanes`, `vds_pm`, ramp flags/ids |
| `freeway.csv` | per-cell FD params (`q_max`, `v_f`, `w`, `rho_jam`, `rho_crit`) |
| `sim_no_ramps/result.npz` | the CTM `SimulationResult` (density/flow/speed, `dt`, boundary inflow) |
| `sim_no_ramps/boundary_inflow.csv` | upstream admitted inflow (veh/h) for seeding |
| `sim_no_ramps/validation.csv` | the CTM-vs-PeMS reference (Stage-1 macro half, already computed) |
| PeMS per-VDS timeseries (`<vds_id>.csv`) | ground truth (external; see Boundaries) |

The CTM ran I-880 N at `dt = 20 s` over a 24 h horizon (`T = 4320`), 2 cells,
4 lanes, with VDS `400662` / `400141`.

The **PeMS per-VDS timeseries is a runtime dependency**: the Stage-1
micro-vs-PeMS scoring needs `<vds_id>.csv` present at run time, but it is *not*
required to build or run the unit tests (layers 1–3 use synthetic fixtures). The
driver fails loudly with the missing path if it is absent.

## Tech Stack

- Python ≥ 3.10, NumPy core (matching the DWC module). `pandas` for the
  validation-CSV / timeseries I/O. `matplotlib` (Agg in tests) for figures.
  No new heavy dependencies.
- Reuses `ctm_for_dwc.ctm` (`SimulationResult`,
  `compare_against_historical`) and `ctm_for_dwc.dwc`
  (`build_P_y`, `compute`, `CorridorSpec`, `PadSpec`).

## Commands

```
Test (this work):  pytest tests/test_microsim_*.py tests/test_dwc_validation*.py
Test (full suite): pytest
Lint:              (none configured in repo; match existing style)
Run the study:     python scripts/run_dwc_validation.py --case-dir scripts/output/1mi_case_study
Slow/opt-in tests: pytest -m slow      # macro-vs-micro equivalence harness
```

## Project Structure

Library code under `src/` (importable); the study driver is a repo-root
entrypoint (run, don't import) — matching the repo's `src/ctm_for_dwc/` vs `scripts/`
convention.

```
src/ctm_for_dwc/microsim/
  __init__.py        # public API
  model.py           # MicrosimSpec / Vehicles / MicrosimResult / GuardStats
  gipps.py           # modified Gipps longitudinal update (eq 1–5)
  lanechange.py      # N-lane incentive + random lane change (eq 6–14, Alg 3)
  seeding.py         # PeMS-flow → Poisson arrivals + random entry lane
  simulate.py        # N-lane MS main loop (lane-change + longitudinal + guard)
  aggregate.py       # trajectories → per-cell (density, flow), Edie, summed across lanes
  charging.py        # original mCONV over trajectories (eq 30–32), reuses build_P_y
  examples.py        # synthetic corridors; uniform-density fixture

scripts/run_dwc_validation.py   # case-dir → micro run → Stage 1 + Stage 2 → figures/tables
tests/test_microsim_*.py         # gipps, seeding, aggregate, charging — hand-calc + property
tests/test_dwc_validation.py    # Stage-2 macro-vs-micro equivalence (marked slow)
```

## Code Style

Match the DWC/CTM modules: NumPy-only pure functions where possible, frozen
dataclasses with `__post_init__` validation, NumPy-style docstrings, grid/SI
units fixed at one validated boundary. Example (the Gipps safe-velocity step):

```python
def safe_velocity(v: float, gap: float, *, b: float, dt: float, s0: float) -> float:
    """Gipps collision-free velocity (Newbolt eq 5).

    Parameters
    ----------
    v : float
        Subject vehicle velocity [m/s] this step.
    gap : float
        Bumper-to-bumper gap to the leader [m] (Newbolt eq 1).
    b, dt, s0 : float
        Deceleration [m/s^2], step [s], desired safe spacing [m].

    Returns
    -------
    float
        Velocity [m/s] that maintains the safe distance.
    """
    return -b * dt + math.sqrt(b**2 * dt**2 + v**2 + 2 * b * (gap - s0))
```

Units convention: microsim internals in **SI (m, s, m/s)** (Gipps is naturally
metric); the aggregator converts to the validation schema's **veh/mi, veh/h,
mi/h** at the seam, mirroring how the DWC adapter converts miles→m once.

## Testing Strategy

`pytest` (config in `pyproject.toml`, `testpaths=["tests"]`). Mirror the DWC
module's four-layer strategy:

1. **Per-function hand calculations** — small inputs, round numbers:
   - Gipps: a two-vehicle step where the leader is far (accelerate, clamped to
     `v_MAX`) and one where it is close (decelerate to `v_SAFE`), checked by hand.
   - Seeding: a constant inflow → expected vehicle count and (seeded) headways
     to a known tolerance.
   - Aggregator: vehicles at known uniform density/velocity in one cell →
     Edie density = `N/L`, flow = `N·v/L`, exact.
   - Charging: one vehicle at constant velocity over the corridor → energy
     equals `∫ P_y dt` along its path, hand-checked against `build_P_y`.
2. **Invariants / properties** — no collisions/overlaps (`gap ≥ 0`); velocities
   in `[v_MIN, v_MAX]`; aggregated density/flow non-negative; energy `≥ 0`.
3. **Cross-method (uniform limit)** — the central Stage-2 identity: seed
   vehicles at constant density+velocity; corridor-aggregate micro energy vs.
   adapted-mCONV energy agree to **relative error `< 1e-6`**. (Plan layer 4.)
4. **End-to-end smoke** — the study driver runs on `1mi_case_study` and emits
   the Stage-1 table (micro `validation.csv`-schema rows + compute time) and the
   Stage-2 divergence number, under a slow/opt-in marker.

Coverage expectation: every pure function in `gipps`/`seeding`/`aggregate`/
`charging` has ≥1 hand-calc and ≥1 property test green before composition; the
slow harness is opt-in (not on the default `pytest` path's fast subset).

## Boundaries

- **Always:** run the relevant `pytest` before commit; keep microsim internals
  in SI and convert once at the aggregator seam; reuse
  `compare_against_historical` rather than re-implementing the PeMS comparison;
  flag missing data paths instead of assuming the external drive is mounted.
- **Ask first:** adding any dependency (e.g., a traffic-sim library — the intent
  is from-scratch); changing the `validation.csv` schema; introducing the
  dual-lane or battery-stack code paths now (deferred); editing the shared
  `dwc_demand_module.md` source-of-truth.
- **Never:** commit data or secrets; remove or weaken the `< 1e-6` uniform-limit
  test to make a run pass; "clean" NaN rows out of PeMS 5-min grids; vendor or
  copy Newbolt source (we have only the paper).

## Success Criteria

1. N-lane modified-Gipps microsim (with lane-changing) implemented, seeded from
   the raw PeMS timeseries, with all per-function hand-calc + property tests
   green (incl. no-overlap within each lane and lane-change validity).
2. Micro→macro aggregator emits per-cell density/flow on the PeMS 5-min grid
   that flows through `compare_against_historical` to produce a micro
   `validation.csv` in the **same schema** as the CTM's.
3. **Stage 1** produces, per corridor: micro-vs-PeMS accuracy (validation
   schema) alongside the existing CTM-vs-PeMS numbers, plus **wall-clock compute
   cost** for each method.
4. **Stage 2** uniform-density limit passes as a test (`rel err < 1e-6`); a
   realistic-regime **divergence number** is reported.
5. The study runs across **≥2 corridor lengths** (starting with
   `1mi_case_study`) and the results are presented comparably across lengths.
6. README/module map updated to mention the `microsim` package and the
   `run_dwc_validation.py` entrypoint.

## Resolved Decisions

1. **Micro→macro aggregation — Edie per-cell box.** Use Edie's generalized
   definitions over each cell's full space-time region (density = total
   time-in-cell / area; flow = total distance-in-cell / area). `validation.csv`
   is per-cell and the CTM produces per-cell density, so the Edie per-cell box
   is the like-for-like quantity. Point-detector emulation at `vds_pm` is *not*
   pursued this round.
2. **On-ramp inflow — mainline-only.** The microsim models upstream mainline
   inflow only, mirroring the `sim_no_ramps/` CTM artifact the study consumes
   and keeping Stage 1 like-for-like. The 1-mile on-ramp (`on_ramp_vds_id
   402960`) is not injected. (Consistent with the DWC ramp-handling assumption.)
3. **Stage-2 realistic regime — actual corridor now, sweep later.** Report the
   divergence on the real CTM/micro density field of the case-study corridor
   this round. The synthetic within-cell gradient sweep (`dwc_demand_module.md`
   validation item 2) is a documented future/stretch item, not built now.

### Adopted defaults (in effect unless flagged before the Plan phase)

4. **Microsim time step `dt = 1 s`** per Newbolt. Independent of the CTM's
   `20 s`; the aggregator reconciles both to the PeMS 5-min grid.
5. **Single homogeneous vehicle class.** One representative LDV (e.g. Newbolt
   Table I/II L1, Tesla Model 3: length 4.69 m) for the fixed Gipps params
   (`a`, `b`, `s_o`, length) + the corridor's existing `PadSpec`. Each vehicle's
   **desired/free-flow speed `v_MAX` (also its initial velocity) is sampled from
   `U(v_f − 15, v_f + 15) mph`** — uniform per the 2024 NAPS paper, but
   **re-centered on the corridor free-flow speed `v_f`** (mean of cell `v_f`)
   rather than the paper's literal `U(45,90)`. Rationale: the literal band's mean
   (67.5 mph) makes the micro outrun a corridor with `v_f ≈ 59 mph` and biases
   Stage 1; centering on `v_f` keeps free-flow micro speed consistent with the
   CTM/PeMS while retaining uniform sampling + ±15 mph heterogeneity.
   (Revises the earlier "not tuned to `v_f`" stance — a deliberate, transparent
   coupling.) Acceleration/deceleration constants `a`, `b` and desired safe
   spacing `s_o` taken from the 2026 paper's parameter tables. (Per-class fleet
   mix remains deferred.)

### v2 decisions (after multi-lane review)

6. **N-lane with lane-changing.** `N` = the corridor's PeMS lane count;
   Newbolt's incentive + random lane-change (eq 6–14, Table I, Alg 3)
   generalized to `N` lanes. Replaces the single-lane scope, which could not
   reproduce multi-lane capacity/congestion vs. PeMS.
7. **DWC in all lanes; EVs no lane preference.** Keeps the lane-agnostic macro
   comparable in Stage 2; EVs and non-EVs share lane-change logic (Newbolt's
   EV-seeking eq 13 dropped — see Deviations). Lane-specific DWC + a
   lane-specific macro is the deferred faithful-Newbolt alternative.
8. **Seed from raw PeMS timeseries via a Poisson process.** Both macro and
   micro inputs derive from the same raw PeMS source (mainline boundary VDS
   flow). Arrivals are drawn as a per-step Poisson process with rate from the
   measured 5-min PeMS flow (resampled to the micro `dt`). Newbolt's TTF
   Monte-Carlo (ref [31], Alg 1) is **not** reproduced: that method exists to
   *estimate* arrivals from junction left/right turning counts, which we don't
   need given direct mainline flow counts — Poisson is the standard, simpler
   arrival model for a measured rate. EV classification is independent
   `Bernoulli(eta)` per arrival (Newbolt Alg 2's density sampling).
9. **Paper parameter values + CLI overrides.** `a`/`b`/`s_o`/lengths and pad
   specs default to Newbolt's tables (fixing the earlier invented `a=1.5`,
   `b=2.0`), overridable per run for sensitivity.

### Assumptions surfaced in review (were unflagged in v1)

A review of the v1 micro module found decisions made without sign-off; their v2
disposition:

| Assumption (v1) | Disposition (v2) |
|---|---|
| `a=1.5`, `b=2.0`, pad defaults — invented, not Newbolt's | **Fixed** — use paper tables (decision 9). |
| Boundary admission gate (`min_x−length>s_o`, drop if blocked) — invented | **Replaced** by lane-based entry (random lane; Newbolt bumps conflicts across lanes). |
| Independent per-step Poisson arrivals | **Kept** — Poisson is the standard arrival model given direct flow counts (decision 8); Newbolt's TTF MC not needed. |
| Admitted vs. offered inflow (seeded from CTM admitted) | **Replaced** — seed from raw PeMS flow (decision 8). |
| `v_min = 0` (Gipps floor) | **Open** — keep `0` (vehicles can stop in congestion) unless the paper pins a 20 mph floor; revisit with ref [31]. |
| EV assignment `Bernoulli(eta)` (stochastic count) | **Kept**, documented; multi-seed CIs average it out. |
| Charging integral is a Riemann sum (start-of-interval `P_y`) | **Kept** — exact in the grid-aligned fixture, converges as `dt→0`; documented. |
| `v_f` band center = mean of per-cell `v_f` | **Kept**, documented. |

### Assumptions surfaced in review (v2, the N-lane rewrite)

A review of the v2 code found these decisions made without explicit sign-off.
Grouped by kind; **awaiting disposition**.

**Lane-change rule (`lanechange.py`) — generalizing a 2-lane published rule:**

| # | Assumption | Disposition |
|---|---|---|
| V1 | **N-lane generalization is mine.** Newbolt eq 13/14 are strictly 2-lane (`y∈{1,2}`). I generalized to "blocked in current lane → move to a feasible adjacent lane," which is *not* literally in the paper. | **Keep** (paper is 2-lane); documented as a designed generalization. |
| V2 | **Backward gap convention differs.** Newbolt eq 7 `s_B = x_i + l_i − (x_{j-1} + l_{j-1})` is a back-to-back distance; I used a **bumper gap** `x − x_B − length` (one length stricter). | **Change → match eq 7** (back-to-back distance) for fidelity; the bumper-gap version was an unprompted stricter choice. |
| V3 | **Lane choice among feasible adjacents.** I used largest-forward-gap. | **Change → lowest feasible lane index** (deterministic first pass, per user). |
| V4 | **Lane changes were simultaneous (vectorized), not sequential** as Newbolt's per-vehicle Alg 1 loop. | **Change → sequential per-vehicle loop** (Newbolt Alg 1; 2024 Figs 3-4). Faithful *and* gives an honest micro compute cost for the Stage-1 efficiency comparison; the vectorized version understated it. |

**Entry / seeding (`simulate.py`, `seeding.py`):**

| # | Assumption | Disposition |
|---|---|---|
| V5 | **Boundary admission "most-room" bump + drop-if-never-admitted.** A due vehicle takes its entry lane, else the lane whose nearest active vehicle is farthest downstream; if no lane has room it waits and (rarely) may never enter. My rule, not the paper's. | **Keep**; documented (sequential loop + 4 lanes makes blocking rare). |
| V6 | **Uniform-random entry lane.** Newbolt 2026 studies *preprocessing lane-determination ratios* (e.g. 40/60). I split lanes uniformly. | **Keep** uniform (user); expose a ratio later. |
| V7 | **Per-lane inflow split is random.** PeMS flow is a station total (sum across lanes); I seed the total then assign lanes uniformly, rather than using measured per-lane flows. | **Keep** (user); documented (per-lane PeMS could refine). |

**Parameters / wiring (`model.py`, driver):**

| # | Assumption | Disposition |
|---|---|---|
| V8 | **`lane_change_prob` default 0.5** (Newbolt `w_o` base case). | **Keep** 0.5 (user); CLI-overridable. |
| V9 | **Boundary VDS = cell-0 `vds_id`** as the inflow source. | **Confirmed** correct (user). |
| V10 | **`n_lanes = max(cells.lanes)`** — one corridor-wide lane count (can't vary per cell). | **Keep** (fine for uniform I-880). Per-cell lane counts (lane drop/add + merge logic) deferred to **future work**. |
| V11 | **Pad defaults kept non-Newbolt** (`α=3.5/λ=0.5/δ=1.0`, 150 kW) because Newbolt's 25 mm gap doesn't resolve on `dx_grid=0.5`. | **Keep** (user); Stage 2 compares the *same* pad both ways, so values don't bias it. Newbolt's 350 kW / 3 m / 25 mm would need a finer `dx_grid`. |
| V12 | **Lane-change RNG seed = `spec.seed + 1`** to decorrelate from the seeding stream. | **Keep** (user); documented. |
| V13 | **`v_min = 0`** carried from v1; ref [31] has no Gipps model / floor. | **Keep** `0` (user). |

## Deviations from the published model

Faithful reproduction is the goal, but some of the published model is not
usable as-is for this study; each divergence is documented:

- **EV charging-lane-seeking dropped (eq 13).** Because DWC spans all lanes
  here (decision 7, to keep the macro comparable), EVs have no reason to prefer
  a lane, so EVs and non-EVs share the same lane-change logic. Newbolt's
  EV-specific incentive (eq 13), which presumes a single charging lane, is not
  reproduced. The faithful alternative (single charging lane + lane-specific
  macro) is deferred.
- **Displacement-cap collision guard (micro loop).** Newbolt's
  simplified Gipps safe-velocity (2026 eq 5) uses the *subject's* speed and a
  hard binary accelerate/decelerate switch at `s_o`, and is **not
  collision-free** at practical time steps: a fast follower closing on a slow
  leader overshoots and passes through it (bumper overlaps of a few meters
  observed at `dt = 0.25–1 s` under ordinary `U(v_f ± 15)` heterogeneity;
  clean only near `dt = 0.1 s`, ~10× compute, and not even monotone in `dt`).
  To keep the sim physical at `dt = 1 s` we **diverge from the paper**: the
  velocity formula is kept verbatim, then each vehicle's per-step displacement
  is capped at the gap to its leader (`v ≤ gap/dt`), so positions can never
  overlap. The guard binds only in the rare overshoot. Its footprint is
  reported in `MicrosimResult.guard_stats` — activation count and rate, max
  speed reduction, and max acceleration delta (preferred − actual) — so the
  divergence is **auditable, never silent**. If activations are frequent on a
  real corridor, that is a signal to revisit `dt` or the safe-velocity law.

---

## Implementation Plan (Phase 2)

> Status: **DRAFT — awaiting review.** Reviewable build strategy derived from the
> spec above. Phase 3 (discrete tasks) follows after sign-off.

### Components and dependencies

```
model.py ─┬─→ gipps.py ──────┐
          ├─→ seeding.py ─────┼─→ simulate.py ─┬─→ aggregate.py ─┐
          └─→ examples.py ────┘                └─→ charging.py ──┤
                                                                 └─→ scripts/run_dwc_validation.py
                       (reuse) ctm.compare_against_historical ───┘
                       (reuse) dwc.build_P_y / dwc.compute  ───┘
```

| Component | Role | Depends on | New/Reuse |
|---|---|---|---|
| `model.py` | `MicrosimSpec` (Gipps params, `dt`, velocity range, RNG seed, `PadSpec`, cell geometry), `Vehicle` state, `MicrosimResult` (trajectories) | — | New |
| `gipps.py` | longitudinal update: gap (eq 1), accel/decel branch (eq 2), `v_a`/`v_b`/`v_SAFE` (eq 3–5), clamp | `model` | New |
| `seeding.py` | `boundary_inflow.csv` → vehicle entry times (Monte-Carlo) + initial velocity ~ `U(45,90 mph)` (→ per-vehicle `v_MAX`) | `model` | New |
| `simulate.py` | single-lane main loop (Alg 1–2), vectorized across active vehicles → `MicrosimResult` | `gipps`, `seeding`, `model` | New |
| `aggregate.py` | trajectories → per-cell (density, flow) via Edie, on the 5-min grid | `model` | New |
| `charging.py` | original mCONV (eq 30–32): index `P_y` at each vehicle position, integrate over time → micro demand | `dwc.build_P_y`, `model` | New + reuse |
| `examples.py` | synthetic corridors; deterministic uniform-density fixture | `model` | New |
| `run_dwc_validation.py` | case-dir → micro run → Stage 1 + Stage 2 → tables/figures | all + `ctm`, `dwc` | New |

### Implementation order (critical path + parallelism)

1. **`model.py`** — foundation; nothing else compiles without the dataclasses.
2. **`gipps.py`** and **`seeding.py`** — independent of each other (both need only
   `model`); buildable in parallel. `examples.py` can start here too.
3. **`simulate.py`** — composes 1–2 into the loop. Critical path.
4. **`aggregate.py`** and **`charging.py`** — both consume `simulate` output but
   are independent of each other; parallelizable.
5. **`run_dwc_validation.py`** — last; wires everything to the case dir and the
   reused `compare_against_historical` / `compute`.

Sequential critical path: `model → gipps → simulate → {aggregate | charging} →
driver`. Parallel opportunities: (gipps ∥ seeding ∥ examples), then (aggregate ∥
charging). TDD per function throughout (test-first, per the DWC module's habit).

### Stage wiring at the seams

- **Stage 1 (micro side):** `aggregate.py` emits per-cell density `(n_cells,
  n_5min+1)` and flow `(n_cells, n_5min)` directly on the 5-min grid, wrapped in
  a `SimulationResult`-compatible object (the `SimpleNamespace` shape
  `validate_ctm_corridor.py::_load_sim_arrays` already feeds
  `compare_against_historical`, with `dt = 5 min` so `steps_per_5min = 1`). The
  comparator is **reused unchanged**; the CTM-side `validation.csv` already
  exists. Add wall-clock timing around each engine.
- **Stage 2:** `charging.py` produces the micro demand profile; the adapted side
  is the existing `dwc.compute`/`from_ctm`. Compare corridor-aggregate energy.

### Risks and mitigations

1. **Stochasticity vs. reproducible tests.** The sim is random (seeding,
   `U(45,90)`). → `MicrosimSpec` carries an explicit RNG seed; the uniform-limit
   equivalence test bypasses randomness entirely (deterministic uniform spacing +
   constant velocity).
2. **Uniform-density limit is the subtle one.** The `< 1e-6` identity only holds
   in steady-state interior, excluding the approach/exit transient that `M_CTM`
   zeros (center-reference). → Construct the fixture to match the `M_CTM`
   convention exactly (constant ρ, constant v, interior only); this is the test
   that pins Stage 2, so it gets dedicated care and a hand-derived expected value.
3. **Micro free-flow speed alignment.** The literal `U(45,90)` (mean 67.5 mph)
   would make the micro outrun I-880's `v_f ≈ 58–60 mph` and bias Stage 1. →
   Mitigated by sampling `v_MAX ~ U(v_f − 15, v_f + 15)`, centering the mean on
   the corridor `v_f`. Residual realized-speed differences (from congestion
   dynamics, not the desired-speed mean) are legitimately part of the Stage-1
   fidelity measurement; document the speed-distribution choice beside the numbers.
4. **Compute-time comparison is apples-to-oranges.** Pure-Python per-vehicle vs.
   the CTM's vectorized engine. **Superseded by v2 decision V4:** the micro is
   deliberately a *sequential per-vehicle loop* (Newbolt Alg 1), **not**
   vectorized — vectorizing would understate the micro's true cost and bias the
   Stage-1 efficiency comparison. The micro is honestly the slow one (~O(active²)
   per step); that gap is the result the study reports. Report wall-clock for
   both with the structural caveat.
5. **Edie ↔ comparator array contract.** Shapes/units must match what
   `compare_against_historical` expects (density veh/mi, flow veh/h, 5-min grid,
   one row per cell). → Pin with the uniform fixture (Edie density `= N/L`, flow
   `= N·v/L`) before wiring the driver.
6. **Multi-corridor generality.** Only `1mi_case_study` exists now. → Driver is
   parametrized by `--case-dir`; longer corridors are new input dirs, no code
   change. Keep corridor geometry read from `cells.csv`/`freeway.csv`.

### Verification checkpoints (each ends green)

| CP | After | Verify |
|---|---|---|
| CP1 | `model` + `gipps` | hand-calc: far-leader→accelerate (clamped to `v_MAX`); near-leader→`v_SAFE` decelerate |
| CP2 | `seeding` | constant inflow → expected vehicle count + headways; velocity draws in `[45,90] mph` |
| CP3 | `simulate` | invariants on a synthetic corridor: `gap ≥ 0` (no overlap), `v ∈ [v_MIN, v_MAX]` |
| CP4 | `aggregate` | Edie exact on uniform fixture (`ρ=N/L`, `q=N·v/L`); micro `validation.csv` produced via reused comparator |
| CP5 | `charging` | single-vehicle constant-v hand-calc (`∫P_y dt`); **uniform-density `< 1e-6`** equivalence vs `dwc.compute` |
| CP6 | `run_dwc_validation.py` | end-to-end on `1mi_case_study`: Stage-1 table (micro vs PeMS + compute) + Stage-2 divergence; marked `slow` |

CP5's `< 1e-6` is the headline regression test; CP1–CP4 are fast unit layers; CP6
is the opt-in end-to-end harness.

---

## Tasks (Phase 3)

> Status: **DRAFT — awaiting review.** Dependency-ordered. Order: T1 → (T2 ∥ T3 ∥
> T4) → T5 → (T6 ∥ T7) → T8 → T9. Each task is test-first.

- [ ] **T1 — `model.py`: data model.**
  - Acceptance: frozen dataclasses `MicrosimSpec` (Gipps params `a`, `b`, `s_o`,
    vehicle `length`, `dt`, `v_f`-centered speed-band half-width = 15 mph, RNG
    `seed`, `PadSpec`, per-cell `length`/`v_f`), `Vehicle` (position, velocity,
    `v_max`, `is_ev`, entry step), `MicrosimResult` (trajectory arrays +
    timestep/cell axes). `__post_init__` validates positivity and units (SI
    internally), mirroring `dwc/model.py`.
  - Verify: `pytest tests/test_microsim_model.py` — field values, validation
    rejects non-positive params, derived properties.
  - Files: `microsim/__init__.py`, `microsim/model.py`,
    `tests/test_microsim_model.py`.

- [ ] **T2 — `gipps.py`: longitudinal update (CP1).**
  - Acceptance: pure functions `gap` (eq 1), `safe_velocity` (eq 5),
    `next_velocity` (eq 2–4: accel branch clamped to `v_max`, decel branch to
    `v_SAFE`, floored at `v_MIN`). SI units.
  - Verify: `pytest tests/test_microsim_gipps.py` — hand calc: far leader →
    accelerate, clamped to `v_max`; near leader → decelerate to `v_SAFE`;
    property: output in `[v_MIN, v_max]`.
  - Files: `microsim/gipps.py`, `tests/test_microsim_gipps.py`.

- [ ] **T3 — `seeding.py`: inflow → vehicles (CP2).**
  - Acceptance: `seed_vehicles(boundary_inflow, spec, rng)` converts the inflow
    series to per-vehicle entry steps (Monte-Carlo, ref [31] spirit) and draws
    each `v_max` (= initial v) from `U(v_f − 15, v_f + 15) mph`. Deterministic
    under a fixed seed.
  - Verify: `pytest tests/test_microsim_seeding.py` — constant inflow → expected
    count + headways within tolerance; all `v_max ∈ [v_f−15, v_f+15] mph`; same
    seed → identical output.
  - Files: `microsim/seeding.py`, `tests/test_microsim_seeding.py`.

- [ ] **T4 — `examples.py`: synthetic corridors + uniform fixture.**
  - Acceptance: tiny synthetic corridor builder and a **deterministic
    uniform-density fixture** (constant ρ, constant v, uniform spacing, interior
    only — matching the `M_CTM` center-reference convention) for the CP5
    equivalence test.
  - Verify: `pytest tests/test_microsim_examples.py` — fixture has the asserted
    constant density/velocity and vehicle count.
  - Files: `microsim/examples.py`, `tests/test_microsim_examples.py`.

- [ ] **T5 — `simulate.py`: single-lane main loop (CP3).**
  - Acceptance: `simulate(spec, vehicles) -> MicrosimResult`; vectorized across
    active vehicles (NumPy arrays, not objects), Alg 1–2 single-lane; vehicles
    enter at their step, leave past corridor end.
  - Verify: `pytest tests/test_microsim_simulate.py` — invariants on a synthetic
    corridor: `gap ≥ 0` (no overlap) every step; `v ∈ [v_MIN, v_max]`; smoke run
    completes.
  - Files: `microsim/simulate.py`, `tests/test_microsim_simulate.py`.

- [ ] **T6 — `aggregate.py`: Edie + comparator wiring (CP4).**
  - Acceptance: `aggregate_to_cells(result, cells, *, grid="5min")` returns
    per-cell density `(n_cells, n_5min+1)` [veh/mi] and flow `(n_cells, n_5min)`
    [veh/h] via Edie's box; plus a `SimulationResult`-compatible wrapper so the
    **reused** `compare_against_historical` runs unchanged.
  - Verify: `pytest tests/test_microsim_aggregate.py` — Edie exact on the uniform
    fixture (`ρ = N/L`, `q = N·v/L`); wrapper produces a micro `validation.csv`
    in the CTM schema.
  - Files: `microsim/aggregate.py`, `tests/test_microsim_aggregate.py`.

- [ ] **T7 — `charging.py`: original mCONV + uniform-limit equivalence (CP5).**
  - Acceptance: `micro_demand(result, pad, eta_EV)` indexes `dwc.build_P_y` at
    each vehicle's position per step and integrates (eq 30–32) → micro demand
    profile + corridor-aggregate energy. No SOC/discharge.
  - Verify: `pytest tests/test_microsim_charging.py` — single-vehicle constant-v
    hand calc (`∫ P_y dt`). `pytest -m slow tests/test_dwc_validation.py` —
    **uniform-density limit: corridor-aggregate micro energy vs `dwc.compute`
    relative error `< 1e-6`** (the headline regression test).
  - Files: `microsim/charging.py`, `tests/test_microsim_charging.py`,
    `tests/test_dwc_validation.py`.

- [ ] **T8 — `run_dwc_validation.py`: study driver (CP6).**
  - Acceptance: `--case-dir` → seed + simulate micro → Stage 1 (Edie aggregate +
    `compare_against_historical` for micro; read existing CTM `validation.csv`;
    wall-clock time both engines) → Stage 2 (`micro_demand` vs `dwc.compute`
    divergence on the actual corridor field) → writes a Stage-1 comparison table
    and Stage-2 divergence + figures. Fails loudly if PeMS timeseries absent.
    Register the `slow` marker in `pyproject.toml` if not present.
  - Verify: `python scripts/run_dwc_validation.py --case-dir
    scripts/output/1mi_case_study` emits the table + figures;
    `pytest -m slow tests/test_dwc_validation.py` end-to-end smoke passes.
  - Files: `scripts/run_dwc_validation.py`, `tests/test_dwc_validation.py`,
    `pyproject.toml`.

- [ ] **T9 — Wire-up + docs.**
  - Acceptance: `microsim/__init__.py` exports the public API; `README.md`
    module map mentions `microsim` + the driver; `dwc_demand_implementation_plan.md`
    P7 progress-log entry added (date, what shipped, what was learned).
  - Verify: `pytest` (full suite green); `python -c "import
    ctm_for_dwc.microsim"` succeeds; README/plan reviewed.
  - Files: `microsim/__init__.py`, `README.md`,
    `docs/dwc_demand_implementation_plan.md`.
```
