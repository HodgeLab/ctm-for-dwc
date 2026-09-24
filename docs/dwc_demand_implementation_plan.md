# DWC Demand Module — Implementation Plan

Implementation plan for a Dynamic Wireless Charging (DWC) demand
module that adapts the *mCONV* method of
[Newbolt 2024a](https://ieeexplore.ieee.org/document/10750800) to consume
macroscopic outputs from this repo's CTM engine. The spec
[dwc_demand_module.md](dwc_demand_module.md) is the source of truth for
equations, notation, and assumptions; this doc is the build plan and the
running progress log.

## v0 status & retrospective

**v0 is complete** (commits `2600440`..`b6eaa7c` on `add_mconv_module`): the
pipeline runs end-to-end — CTM `SimulationResult` → `from_ctm` / `compute` →
`DemandResult` → plots — at 78 passing tests. Phasing and as-built deltas are
in the **Progress log** below; the sections between here and there are the
*original* plan. The spec remains the source of truth for equations; the
**code** is the source of truth for signatures.

### What we learned (and why it mattered)

- **The power normalization was under-specified — and we'd have got it
  wrong.** The spec first asserted both "γ = min(β, β′)" and "γ is the paper's
  ρ", but Algorithm 3 makes ρ a *fitting* factor, `min(β,β′)/max(P_A)`.
  Without the `/max(P_A)` the peak overshoots and every power magnitude is
  off. Fix: reproduce Newbolt's Algorithms 1–3 verbatim in the spec and pin
  the `peak = γ` identity with a test. *Lesson: reproduce the source, don't
  paraphrase it.*

- **The dense `T_R` "memory wall" was an artifact, not a law.** We first
  framed resolution-vs-compute as a fundamental `O(n²)` wall. It isn't:
  `P_A = T_R·S_T` *is* a convolution, computable in `O(n)` with no matrix —
  the dense path was just one *representation*. Fix: `build_P_y` convolves
  (`method="auto"`); `build_T_R` stays only for exact reproduction. *Lesson:
  separate the algorithm from the data structure before declaring a
  bottleneck.*

- **The off-by-one (`m = n + δ_grid − 1`) was real and load-bearing.** It
  threatened every downstream array shape. Flagged at P0, pinned in P1,
  resolved in P3 by the **center-reference** `M_CTM` convention (chosen from
  three options, evaluated in a notebook): mapping traversal positions by the
  pad *center* keeps a 1:1 correspondence with the `n` corridor positions, so
  `n_i = positions_per_cell` exactly (no divide-by-zero) and entry/exit are
  symmetric. *Lesson: surface index conventions early; settle them with a
  worked example, not prose.*

- **The CTM's `VHT` bakes in the on-ramp queue; DWC must exclude it.**
  `metrics.vht_per_cell = (ρL + queue)·dt`; mainline-only is `ρL·dt`. We
  compute that from the `SimulationResult` (deviating from the planned
  `Metrics` seam, which can't separate the queue). On a real metered-on-ramp
  run, including the queue overcounts demand by **21.9 %**. *Lesson: verify an
  upstream quantity against its definition, not its name.*

- **Grid-integer signatures, one conversion.** The pure functions take grid
  units; the single meter→grid conversion (and its validation) lives in
  `CorridorSpec`, and real CTM cell lengths (miles→m) are *snapped* to the
  grid at the seam (error ≤ `dx_grid`/2). *Lesson: one validated boundary
  beats a conversion duplicated per function.*

- **Process.** Spec-as-source-of-truth caught the normalization error before
  any code; test-first with exact integer hand-calcs made the convolution
  refactor safe (behavioral tests passed unchanged); and checking the *actual*
  paper and *actual* CTM code — not assumptions — corrected the plan
  repeatedly.

## Scope

- **In:** spatial submodule (build $\mathbf{P_y}$ from corridor geometry +
  pad specs), cell→position mapping (build $\mathbf{M_{CTM}}$), and demand
  assembly ($\mathbf{E} = \eta_{EV}\,(\mathbf{VHT}^T \mathbf{M_{CTM}}) *
  \mathbf{P_y}$). Single-lane-equivalent corridor, single vehicle class,
  ramps excluded — per the spec's v0 assumptions.
- **Out (for now):** per-class fleet mix, lane-specific DWC
  infrastructure, velocity-dependent $\mathbf{P_y}$, multi-vehicle Tx-pad
  saturation. Listed in the spec; not designed for here.
- **Couples to CTM via one seam:** reads $\mathbf{VHT}$ (shape `(N, T)`)
  from `ctm.metrics.Metrics.vht_per_cell`, plus per-cell `length` from
  `Freeway`. No other coupling.

## Module layout

A self-contained subpackage alongside `ctm/`:

```
src/ctm_for_dwc/dwc/
  __init__.py     # public API
  spatial.py      # build_S_T, build_S_R, build_T_R, build_P_y
  mapping.py      # build_M_CTM (cell→position partition-of-unity)
  demand.py       # compute_demand (VHT × M_CTM × P_y × eta_EV)
  model.py        # PadSpec, CorridorSpec, DemandResult dataclasses
  examples.py     # Newbolt small-scale system; tiny synthetic corridors
  plots.py        # P_y vs position; E heatmap vs (time, position)
scripts/run_dwc_demand_demo.py    # CTM run → demand → plot
tests/test_dwc_*.py
```

## Core data model

- **`PadSpec`** — Tx params (`alpha`, `lambda_gap`, `beta_prime`) and Rx
  params (`delta`, `beta`). Units: meters and watts. Validated on
  construction (positive lengths; `dx_grid` divides `alpha`, `delta`,
  `lambda_gap` cleanly).
- **`CorridorSpec`** — ordered list of per-cell lengths (meters; converted
  from the CTM `Freeway`'s miles via a single adapter), plus a `PadSpec`
  and the position grid spacing `dx_grid`. Computes derived quantities
  (corridor length in position units `n`, traversal positions `m`,
  positions per cell `n_i`).
- **`DemandResult`** — `E` array `(T, m)` in Wh; carries axis labels
  (position-meter for columns, timestep for rows) and a `.to_dataframe()`
  helper.

**Validation up front:** position grid resolves all pad dimensions
(`alpha`, `delta`, `lambda_gap` are integer multiples of `dx_grid`); cell
boundaries align to position-grid edges (no fractional positions per
cell — split case is deferred per the spec); `eta_EV ∈ [0, 1]`;
`beta, beta_prime > 0`. Bad configs fail loudly.

## Functions map 1:1 to the spec

> **As-built note.** Signatures below are the *original* plan. The shipped
> functions take **grid-unit integers** (the meter→grid conversion lives once
> in `CorridorSpec`): `build_P_y(S_T, S_R, gamma, *, method="auto")`
> (convolution, not `T_R @ S_T`), `build_M_CTM(positions_per_cell,
> delta_grid)`, and orchestrators `CorridorSpec.build_P_y()` /
> `compute(VHT, corridor, eta_EV)` / `from_ctm(result, pad_spec, dx_grid,
> eta_EV)`. See the progress log for the why.

Pure functions, NumPy-only:

1. **`build_S_T(corridor_length_m, alpha, lambda_gap, beta_prime, dx_grid)
   -> ndarray (n,)`** — repeats the `[β' × α, 0 × λ]` tile across the
   corridor. (Spec §"Spatial Dependence", Algorithm 1.)
2. **`build_S_R(delta, beta, dx_grid) -> ndarray (δ_grid,)`** — constant-
   magnitude Rx kernel. (Spec §"Spatial Dependence", Algorithm 2.)
3. **`build_T_R(S_R, n) -> ndarray (m, n)`** — Toeplitz of `S_R` over an
   `(n + δ_grid - 1) × n` band, mimicking Rx traversal. (Spec eq. for
   $\mathbf{T_R}$.)
4. **`build_P_y(S_T, S_R, gamma, *, method="auto") -> ndarray (m,)`** —
   implements Newbolt Algorithm 3: form the raw area profile
   `P_A = S_R * S_T` (a convolution — same quantity as `T_R @ S_T`), then
   return `gamma * P_A / P_A.max()` so the peak equals `gamma = min(beta,
   beta_prime)`. The `/ P_A.max()` normalization **is** the spec's
   $\gamma = \min(\beta,\beta')/\max(\mathbf{T_R}\mathbf{S_T})$ — do not drop
   it, or the peak overshoots by a factor of `P_A.max()`. `method` is
   `auto` (convolve for small kernels, fft for large) / `convolve` / `fft` /
   `toeplitz` (the dense reference). (Spec eq. for $\mathbf{P_y}$;
   Algorithm 3; Resolution-vs-compute.)
5. **`build_M_CTM(cell_lengths_m, n_corridor, dx_grid, m_traversal,
   approach_positions) -> ndarray (N, m)`** — row-partition-of-unity:
   `M[i, j] = 1/n_i` for `j` in cell `i`, else 0. Approach/exit positions
   (the first/last `δ_grid - 1` traversal positions) get zero columns.
   (Spec eq. for $\mathbf{M_{CTM}}$.)
6. **`compute_demand(VHT, M_CTM, P_y, eta_EV) -> ndarray (T, m)`** —
   `eta_EV * (VHT.T @ M_CTM) * P_y` with broadcast. (Spec eq. for
   $\mathbf{E}$.)

Two-line orchestrator `CorridorSpec.build()` composes 1→4 to a `P_y`, and
a top-level `compute(VHT, corridor) -> DemandResult` composes 5–6.

## Verification strategy (four independent layers)

1. **Per-function hand calculations** — small inputs with round numbers,
   every function tested against a value computed by hand:
   - `build_S_T` with 2 tiles → exact bit pattern.
   - `build_T_R` with `δ_grid=3, n=4` → check banded-Toeplitz entries
     directly.
   - `build_P_y` with `α=2, δ=2, λ=2` (toy units) → triangular peaks whose
     height equals `gamma` exactly (the `/ P_A.max()` normalization pins the
     peak) with valleys of known fraction between.
   - `build_M_CTM` with 2 cells of 4 and 6 positions → row sums to 1,
     entries `0.25` and `0.1667`.
   - `compute_demand` with constant `VHT` and constant `P_y` on 1 cell →
     `E = eta_EV · VHT · P_y / n_i` at every position.
2. **Invariants / properties** —
   - Row sums of `M_CTM` equal 1 over corridor cells (off-corridor rows = 0).
   - `P_y ≤ gamma` everywhere (no overshoot of the rated capacity).
   - `E ≥ 0` everywhere; `E = 0` where `VHT = 0`.
   - **Energy conservation:** `E.sum(axis=1)[k] == eta_EV ·
     sum_i(VHT[i,k] · mean_j∈cell_i(P_y[j]))`. This is the spec's central
     identity and pins the row-normalization choice.
   - Linearity in `eta_EV` and `VHT` (scaling either scales `E`
     proportionally).
3. **Cross-method check (Newbolt small-scale)** — reproduce the
   3-Tx-pad system from Newbolt et al. (their Table 1: `α=2 m`,
   `δ=1.5 m`, `λ=0.1524 m`, `β=β'=150 kW`). Assert the `P_y` profile
   matches the qualitative shape of their Fig. 4 (three peaks at full
   overlap; valleys between; outermost positions zero). Peak height
   = 150 kW exactly; valley height a known fraction of peak from the
   convolution geometry.
4. **Macroscopic-vs-microscopic equivalence (the spec's validation plan)** —
   a synthetic test where vehicles are seeded uniformly in a single cell
   at known density and velocity, with the adapted module run against the
   cell's `VHT`. Compare the corridor-aggregate energy to a microscopic
   computation (`N_vehicles` independent traversals through the
   conventional mCONV pipeline). Should agree within numerical tolerance
   in the uniform-density limit, which is the regime in which the
   adaptation is exact. **Lives in `tests/test_dwc_validation.py` but
   marked `slow` / opt-in.**

## Phasing (each phase ends green)

| Phase | Deliverable | Verify |
|---|---|---|
| **P0** | `PadSpec` / `CorridorSpec` + validation + Newbolt example builder | config-validation tests |
| **P1** | `build_S_T`, `build_S_R`, `build_T_R` pure functions | per-function hand-calc tests (layer 1) |
| **P2** | `build_P_y` | hand-calc tests + Newbolt Fig 4 qualitative match (layer 3) |
| **P3** | `build_M_CTM` | row-sum and off-corridor invariants (layers 1+2) |
| **P4** | `compute_demand` + `DemandResult` | energy conservation identity (layer 2) |
| **P5** | CTM↔DWC adapter (`from_ctm(freeway, metrics, pad_spec)`) | end-to-end smoke test on a tiny CTM run |
| **P6** | demo script + plotting | runnable sandbox |
| **P7 (opt-in)** | macroscopic-vs-microscopic validation harness | layer 4 |

**Status:** P0–P7 shipped (see progress log). P7 (macroscopic-vs-microscopic
validation harness) is specified in [dwc_validation_spec.md](dwc_validation_spec.md)
and implemented under `microsim/` + `scripts/run_dwc_validation.py`.

## TDD workflow per function

For each function in P1–P4, **the test comes first**:

1. Write the hand-calc test in the corresponding `tests/test_dwc_*.py`.
   Test name encodes the property (e.g.,
   `test_build_M_CTM_rows_sum_to_one`,
   `test_compute_demand_conserves_energy`).
2. Run the test — it should fail with `ImportError` or `NotImplementedError`.
3. Implement the function. Run the test until green.
4. Add the next property test (invariant or edge case) for the same
   function, repeat 2–3.
5. Only move to the next function when both hand-calc and at least one
   property test are green.

## Defaults chosen (override if needed)

- **NumPy core; scipy for the fft path.** `build_P_y` forms `P_A` by
  convolution and never materializes `T_R`: `np.convolve` for small kernels,
  `scipy.signal.fftconvolve` for large ones (`method="auto"` picks). This
  keeps memory `O(n)` at any corridor length (see Resolution-vs-compute);
  `scipy` is imported lazily, only on the fft path.
- **Position grid `dx_grid` is user-specified** (no GCD default — Newbolt's
  `λ = 0.1524 m` shares no clean GCD with `α`/`δ`). `CorridorSpec` validates
  that all pad dims and cell lengths are integer multiples of `dx_grid` within
  a `1e-6` ratio tolerance; the CTM adapter snaps real cell lengths to the grid.
- **Units convention:** all lengths in meters, all powers in watts inside
  the DWC module. The CTM adapter converts the freeway's miles → m at
  the seam.
- **Output:** `DemandResult.E` is in **Wh** (per spec); the `to_dataframe`
  helper provides a long-format `(timestep, position_m, energy_wh)` table
  for joining with other analyses.

## Resolution vs. compute

The spatial submodule discretizes the corridor onto a grid of spacing
`dx_grid`, so the number of positions is `n = corridor_length / dx_grid`.
The one quantity that blew up — a **dense** `T_R` — is `O(n²)`, and it was
the real bottleneck. But it is an **avoidable artifact**: `P_A = T_R @ S_T`
is exactly the discrete convolution `S_R * S_T`, so `build_P_y` forms it
directly (`np.convolve` / `scipy.signal.fftconvolve`) in `O(n)` memory and
never materializes the matrix. `build_T_R` is kept only to reproduce the
original mCONV exactly (`method="toeplitz"`).

What remains scales **linearly** in `n`, with small constants: `P_y` and
`S_T` are `O(n)` vectors (~100 KB even at 4 mi / 0.5 m); `E = (T, m)` is
`O(T·n)` — the demand profile itself, aggregable to per-pad / per-feeder
demand if `m` is large. There is no quadratic wall once the dense `T_R` is
gone.

**Benchmark** (`scripts/benchmark_dwc_convolution.py`; Rx pad δ = 1.5 m,
kernel = round(δ/dx); time to form `P_A`):

| corridor | resolution | n | kernel | np.convolve | fftconvolve | dense `T_R` |
|---|---|---|---|---|---|---|
| 1 mi | 0.5 m | 3.2 K | 3 | 0.00 ms | 0.06 ms | 7.0 ms |
| 50 mi | 0.5 m | 161 K | 3 | 0.08 ms | 2.3 ms | OOM (207 GB) |
| 50 mi | 0.1 m | 805 K | 15 | 5.4 ms | 12 ms | OOM (5.2 TB) |
| 10 mi | 0.01 m | 1.6 M | 150 | 34 ms | 29 ms | OOM (21 TB) |
| 1 mi | 0.001 m | 1.6 M | 1500 | 446 ms | 30 ms | OOM (21 TB) |
| 12 m bench | 0.1 mm | 120 K | 15000 | 409 ms | 1.9 ms | OOM (130 GB) |

Reading: the dense `T_R` is OOM beyond a 1-mile coarse corridor;
`np.convolve` wins for small kernels (coarse grids), `fftconvolve` wins for
large kernels (fine grids — Newbolt fidelity: 1.9 ms vs 409 ms). The
crossover sits near a kernel of ~150–256 grid units, which is why
`build_P_y(method="auto")` switches to `fft` above `len(S_R) > 256`.

Consequences for the build:
- `build_P_y` defaults to `method="auto"` (convolve for coarse grids, fft
  for fine), with `convolve` / `fft` / `toeplitz` selectable for comparison
  and exact reproduction.
- `dx_grid` stays a per-study choice — fidelity to small `λ` drives it down;
  output size (`E`) and the snap-to-grid rounding set practical bounds — but
  it is no longer gated by a quadratic `T_R`.
- For very large `(T, m)` outputs, aggregate `E` / `P_y` to per-pad demand
  (a future output-format option, out of scope for the convolution change).

## Open items — final status

Every item flagged during the build (top-level + per-phase in the progress
log) and its disposition:

| Item (where raised) | Status |
|---|---|
| Dense `T_R` materialization (plan) | **Resolved** — `build_P_y` convolves (`O(n)`); dense kept only for `method="toeplitz"`. |
| `dx_grid` default / `λ` not grid-clean (plan, P0) | **Decided** — `dx_grid` user-specified (no GCD); CTM adapter snaps cell lengths to the grid (≤ `dx_grid`/2 error). |
| Approach/exit position handling (plan) | **Resolved** — center-reference `M_CTM` zeros off-corridor columns (P3). Reporting demand *during* the transient → v1. |
| CTM `VHT` includes ramp queue (plan) | **Resolved** — mainline VHT (`ρL·dt`) from `SimulationResult` (P5), confirmed on a real queuing run (~21.9 % effect). |
| Float drift in tile-length composition (P0) | **Mitigated** — snap-to-grid bounds per-cell error; `1e-6` validation tolerance. |
| `m_traversal` off-by-one (P0, P1) | **Resolved** — pinned in P1, center-reference `M_CTM` in P3. |
| Spec to name the `M_CTM` convention (P3) | **Resolved** — spec touch-up documents center-reference. |
| `compute_demand` input validation (P4) | **Resolved** — `eta_EV ∈ [0,1]` + shape checks at the `compute()` boundary (P5). |
| `build_P_y` divide-by-zero guard (P2) | **Decided** — none; all-zero `S_T` is impossible (`CorridorSpec` rejects empty cells). |
| Four-cell example never queues (P5) | **Resolved** — `metered_on_ramp_scenario` + 3 end-to-end tests (follow-up). |

## Future work (v1+)

Deferred from the spec's v0 assumptions and surfaced during the build:

- **P7 — macroscopic-vs-microscopic validation harness** — **done** (2026-06-15;
  see progress log and [dwc_validation_spec.md](dwc_validation_spec.md)).
  Remaining refinements surfaced during the build:
  - **Multi-seed confidence intervals** — the microsim is stochastic; the
    Stage-2 divergence on a single seed swings with the horizon. Average over
    seeds and report CIs (cf. Newbolt's 30 runs) before drawing study
    conclusions.
  - **Lane-changing + battery stack** — dual-lane (eq 6-14) and the
    discharge/HVAC/SOC model (eq 15-39) were deferred; the microsim
    architecture leaves both open.
- **Heterogeneous corridors** — a `Cell`-level dataclass for DWC-on/off
  stretches and per-cell pad variation (P0); the **split-cell** case
  (fractional positions per cell), which `CorridorSpec` currently rejects.
- **Fleet & infrastructure realism** (spec) — per-class fleet mix
  (class-indexed `VHT`/`P_y`), lane-specific DWC (lane-by-lane `VHT`/`P_y`),
  velocity-dependent `P_y`, multi-vehicle Tx-pad saturation.
- **Output scaling** — aggregate `E`/`P_y` to per-pad / per-feeder demand so
  fine-grid, long-corridor, long-horizon `(T, m)` outputs stay tractable (the
  linear residual in Resolution-vs-compute).
- **Approach/exit transient** — report demand as the EV enters/leaves the
  corridor (currently zeroed in `M_CTM`).

## Progress log

*(Update as phases land. Entry per phase: date, what shipped, what we
learned that wasn't in the plan.)*

### P0 — 2026-06-10 — data model + Newbolt example builder

**Shipped.**
[`src/ctm_for_dwc/dwc/`](../src/ctm_for_dwc/dwc/):
`__init__.py`, `model.py` (`PadSpec`, `CorridorSpec`), `examples.py`
(`newbolt_small_scale`).
[`tests/test_dwc_model.py`](../tests/test_dwc_model.py): 22 tests
covering field values, `gamma = min(beta, beta_prime)`, non-positive
rejection across all `PadSpec` fields, grid-alignment validation for
both pad dims and cell lengths, derived integer-grid properties
(`n_corridor`, `alpha_grid`, `delta_grid`, `lambda_grid`,
`m_traversal`, `positions_per_cell`), and Newbolt Table-1 values via
the example builder.

**Implementation assumptions made (not pre-specified by the plan).**
- Used frozen `@dataclass` with `__post_init__` validation instead of
  CTM's explicit `.validate()` method. Configs are immutable; bad
  configs fail at construction without a second call.
- `dx_grid` is **user-specified**, not auto-derived. No GCD default,
  because Newbolt's `lambda_gap = 0.1524 m` doesn't share a clean GCD
  with `alpha`/`delta`. The validator enforces integer-multiple
  alignment within `1e-6` float tolerance on the ratio.
- Newbolt example uses `dx_grid = 0.0001 m` — the coarsest grid that
  resolves `lambda_gap` to an integer (`1 524` units). A coarser grid
  (1 mm) leaves `lambda_gap / dx = 152.4` and fails validation.
- Newbolt example corridor = 3 tiles of `(alpha + lambda_gap) = 2.1524
  m`, total `6.4572 m`, single cell. Tile convention is from the spec
  (`S_T` repeats `[alpha-on, lambda-off]`), so corridor ends in a gap.

**Open items raised during build (logged for later phases).**
1. **Float-precision drift in tile-length composition.** `3 * (2.0 +
   0.1524) / 0.0001` round-trips to `64572` cleanly under Python's
   float64, but for a long corridor (hundreds of tiles) the accumulated
   error in `corridor_length_m` could approach the `1e-6` tolerance.
   Watch at P5 (CTM adapter) when real cell lengths flow in from
   miles→meter conversion. May need to raise the tolerance or accept
   small rounding in tile counts.
2. **No `Cell`-level dataclass.** `CorridorSpec.cell_lengths_m` is just
   a tuple of floats — no per-cell metadata (e.g., DWC-on vs DWC-off
   stretches, or per-cell pad variations). If the corridor ever has
   heterogeneous Tx layouts (one of the deferred items in the spec),
   we'll need a `Cell` struct. Not blocking v0.
3. **`m_traversal = n + delta_grid - 1` convention.** This treats the
   first traversal position as "Rx pad right edge at corridor start"
   and the last as "Rx pad left edge at corridor end" — matching
   Newbolt's Fig. 2 illustration. Worth verifying against the actual
   `T_R` indexing in P1, since an off-by-one here propagates into
   every downstream array shape.

**Verify.** `pytest tests/test_dwc_model.py` → 22 passed in 0.02s.
Full suite (excluding pre-existing failures in
`tests/test_model_prototyping.py`, which is unrelated GRU work): 400
passed.

### P1 — 2026-06-10 — spatial pure functions

**Shipped.**
[`src/ctm_for_dwc/dwc/spatial.py`](../src/ctm_for_dwc/dwc/spatial.py):
`build_S_T`, `build_S_R`, `build_T_R` (Newbolt 2024a Algorithms 1-2).
[`tests/test_dwc_spatial.py`](../tests/test_dwc_spatial.py): 11
hand-calc + property tests (layer 1) — `S_T` tile bit-pattern,
truncation, and value set; `S_R` constant magnitude and length; `T_R`
banded-Toeplitz entries by hand, `(n + δ_grid - 1) × n` shape,
off-band zeros, and the `δ_grid = 1` diagonal case.

**Implementation assumptions made (not pre-specified by the plan).**
- **Grid-integer signatures** instead of the plan's meter+`dx_grid`
  signatures (user-approved). The functions take grid-unit integers
  (`n`, `alpha_grid`, `lambda_grid`, `delta_grid`); the meter→grid
  conversion stays solely in `CorridorSpec` (P0), which already
  validates it. This makes all three consistent with the plan's own
  grid-based `build_T_R(S_R, n)` and avoids duplicating the conversion.
  P2's `CorridorSpec.build()` composes them via the validated `*_grid`
  properties.
- `T_R` is built dense via a per-column loop (NumPy-only, clarity over
  speed). The `fftconvolve` / stride-trick path remains deferred to P5
  per the plan's "T_R materialization" open item.

**Open items raised during build.**
1. The off-by-one convention (`m = n + δ_grid - 1`, no trailing zero
   row) is now pinned in code and tested. P3's `build_M_CTM` must agree:
   the approach/exit positions it zeroes are the first/last partial-
   overlap rows of this `T_R`. Verify the column alignment there.

**Verify.** `pytest tests/test_dwc_spatial.py` → 11 passed in 0.07s;
`tests/test_dwc_model.py tests/test_dwc_spatial.py` → 33 passed.
Purely additive (no existing module imports `spatial` yet), so no
regression surface elsewhere.

### P2 — 2026-06-10 — P_y power profile

**Shipped.**
[`build_P_y`](../src/ctm_for_dwc/dwc/spatial.py) in
`spatial.py` — implements Newbolt 2024a Algorithm 3:
`gamma * (T_R @ S_T) / (T_R @ S_T).max()`, so the peak equals
`gamma = min(beta, beta_prime)`.
[`tests/test_dwc_spatial.py`](../tests/test_dwc_spatial.py): 6 new
tests — exact two-pad hand-calc (`[5,10,5,0,5,10,5]`), peak-equals-gamma
(150 kW), `P_y <= gamma` and `P_y >= 0` invariants (layer 2), linearity
in `gamma`, and the Fig-4 qualitative shape (layer 3).

**Implementation assumptions made (not pre-specified by the plan).**
- **Layer-3 uses a coarse 3-pad analog, not the literal Table-1 dims.**
  At the Newbolt example's `dx_grid = 0.0001 m`, `T_R` is
  ~79571 × 64572 dense float64 ≈ 41 GB — the dense-`T_R` ceiling the
  plan flags for P5. The shape test therefore uses a geometry-preserving
  coarse system (`alpha_grid=3 > delta_grid=2`, `lambda_grid=1`, 3 pads)
  and asserts the qualitative Fig-4 properties (one hump per pad,
  inter-pad valleys, tapered ends) plus the exact peak height. A
  faithful full-resolution check waits on the P5 `fftconvolve` path.

**Open items raised during build.**
1. `build_P_y` divides by `P_A.max()` with no guard for an all-zero
   `S_T` (`max == 0`). That config is impossible for a real corridor
   (always has pads) and `CorridorSpec` rejects empty cells, so no guard
   was added per "no error handling for impossible scenarios."

**Verify.** `pytest tests/test_dwc_spatial.py` → 17 passed in 0.07s;
`tests/test_dwc_model.py tests/test_dwc_spatial.py` → 39 passed.

### P3 — 2026-06-10 — M_CTM cell->position map

**Shipped.**
[`build_M_CTM`](../src/ctm_for_dwc/dwc/mapping.py) in
`mapping.py` — the `(N, m)` row-partition-of-unity.
[`tests/test_dwc_mapping.py`](../tests/test_dwc_mapping.py): 8 tests —
hand calcs (symmetric `δ_grid=3`, even-`δ_grid` floor, `δ_grid=1` plain
partition) plus invariants (shape, rows sum to 1, `n_i` entries each
`1/n_i`, exactly `δ_grid-1` zeroed boundary columns, one cell per mapped
column).

**Decision: center-reference convention (Option C), chosen by the user**
after evaluating three options in
[`scripts/dwc_m_ctm_options.ipynb`](../src/ctm_for_dwc/scripts/dwc_m_ctm_options.ipynb).
Traversal position `j` maps to corridor position `j - off`, where
`off = (δ_grid - 1)//2` is the floored Rx-pad center; off-corridor centers
get zero columns. This resolves the long-standing off-by-one (P0 open
item, spec edit #2): the map is 1:1 over the `n` corridor positions, so
`n_i = positions_per_cell[i]` exactly (no division by zero) and the
boundary treatment is symmetric. Signature is grid-integer
(`positions_per_cell`, `delta_grid`), consistent with P1.

**Implementation assumptions made (not pre-specified by the plan).**
- **Even `δ_grid` floors the offset.** Newbolt's `δ_grid = 15000` is even,
  so `off = (δ_grid-1)//2` zeroes one more column at the exit than the
  approach (a single-position asymmetry, negligible for `n ≫ δ_grid`).
  Pinned by `test_build_M_CTM_hand_calc_even_delta_floors_offset`.

**Open items raised during build.**
1. The spec's `M_CTM` paragraph still says the boundary indexing is "fixed
   in the implementation" (deferred). Now that Option C is chosen, the
   spec could name the center-reference convention. Left for a spec touch-up
   so the build stays moving; not blocking.

**Verify.** `pytest tests/test_dwc_mapping.py` → 8 passed in 0.07s;
`tests/test_dwc_model.py tests/test_dwc_spatial.py tests/test_dwc_mapping.py`
→ 47 passed.

### P4 — 2026-06-10 — compute_demand + DemandResult

**Shipped.**
[`compute_demand`](../src/ctm_for_dwc/dwc/demand.py) in
`demand.py` — `eta_EV * (VHT.T @ M_CTM) * P_y`, shape `(T, m)` in Wh.
[`DemandResult`](../src/ctm_for_dwc/dwc/model.py) in
`model.py` — frozen container holding `E` (Wh), `position_m`, `timesteps`,
with shape validation and a `to_dataframe()` long-format helper; exported
from the package `__init__`.
[`tests/test_dwc_demand.py`](../tests/test_dwc_demand.py): 7 tests —
single-cell hand calc, shape, the energy-conservation identity (checked
against a per-cell mean power computed directly from `P_y`), linearity in
`eta_EV` and `VHT`, non-negativity, and zero-where-`VHT`-zero.
4 `DemandResult` tests added to `tests/test_dwc_model.py`.

**Scope decision (user): container only; orchestrator deferred to P5.**
P4 delivers `compute_demand` + the `DemandResult` container. The
`compute(VHT, corridor)` orchestrator, `CorridorSpec.build_P_y()`, and the
center-reference `position_m` derivation move to P5, where they pair with
the CTM adapter (`from_ctm`).

**Implementation assumptions made (not pre-specified by the plan).**
- `compute_demand` is a pure array function with no `eta_EV ∈ [0,1]` or
  shape validation; that guarding lives at the `compute()` orchestrator
  boundary (P5). Units assume `VHT` in vehicle-hours so `E` is in Wh.
- `to_dataframe` imports pandas lazily so importing `PadSpec`/`CorridorSpec`
  does not pull in pandas.

**Verify.** `pytest tests/test_dwc_demand.py` → 7 passed; full DWC suite
(`test_dwc_model.py test_dwc_spatial.py test_dwc_mapping.py
test_dwc_demand.py`) → 58 passed.

### P5 — 2026-06-10 — CTM adapter + compute orchestrator

**Shipped.**
- [`CorridorSpec.build_P_y()`](../src/ctm_for_dwc/dwc/model.py)
  — composes P1-P2 over the corridor's grid properties.
- [`compute(VHT, corridor, eta_EV)`](../src/ctm_for_dwc/dwc/demand.py)
  — orchestrates `build_M_CTM` + `build_P_y` + `compute_demand` into a
  labeled `DemandResult` (center-reference `position_m`, timestep-index
  rows; validates `eta_EV ∈ [0,1]` and VHT cell count).
- [`adapter.py`](../src/ctm_for_dwc/dwc/adapter.py) (new
  module): `mainline_vht(result)` and `from_ctm(result, pad_spec, dx_grid,
  eta_EV)` — the single CTM seam.
- `compute` and `from_ctm` exported from the package `__init__`.
[`tests/test_dwc_compute.py`](../tests/test_dwc_compute.py) (6) and
[`tests/test_dwc_adapter.py`](../tests/test_dwc_adapter.py) (4), incl. an
end-to-end smoke test on the four-cell CTM run (Kurzhanskiy §3.4).

**Decisions (user, after CTM-code review).**
1. **Mainline VHT from `SimulationResult`** (`density[:,1:] * L * dt`),
   queue-free by construction. Deviates from the plan's "read
   `Metrics.vht_per_cell`" seam — that field includes the on-ramp queue —
   so the adapter takes the `SimulationResult` and recomputes (one-line
   duplication of the `rho·L·dt` formula).
2. **Snap cell lengths to the grid** in `from_ctm` (round `L_m/dx_grid`),
   error ≤ `dx_grid`/2 per cell, keeping `CorridorSpec` strict.
3. **Dense `T_R`, `fftconvolve` deferred.** P5 verifies on a tiny CTM run
   at coarse `dx_grid`; mile-scale corridors at fine `dx_grid` remain
   infeasible until the convolution path is added (resolution-vs-compute
   section).

**Implementation notes (not pre-specified by the plan).**
- New `adapter.py` module (not in the original layout list) isolates the
  CTM import; the DWC package has no runtime CTM dependency (`from_ctm`
  duck-types the passed result; CTM types are `TYPE_CHECKING`-only).
- **The four-cell example never spills back into an on-ramp queue**, so
  `mainline_vht`'s queue exclusion is proven on a controlled synthetic
  result (`test_mainline_vht_excludes_queue_contribution`); the real-run
  test checks `mainline = metrics_VHT − queue·dt`.

**Verify.** `pytest tests/test_dwc_compute.py tests/test_dwc_adapter.py`
→ 10 passed; full DWC suite (6 files) → 68 passed.

### P6 — 2026-06-10 — demo script + plotting

**Shipped.**
- [`plots.py`](../src/ctm_for_dwc/dwc/plots.py):
  `plot_P_y` (power-vs-position) and `plot_demand_heatmap`
  (E space-time heatmap), following the `ctm/plots.py` convention
  (data + keyword-only `out_path`, save at 120 dpi, close).
- [`scripts/run_dwc_demand_demo.py`](../scripts/run_dwc_demand_demo.py):
  four-cell CTM run -> `compute` -> both plots; run via
  `python scripts/run_dwc_demand_demo.py`. Lives in the repo-root `scripts/`
  (the CTM module's entrypoint convention), **not** in the package — only
  importable library code belongs under `src/`.
[`tests/test_dwc_plots.py`](../tests/test_dwc_plots.py): 4 tests
(`position_m` property, `corridor_from_ctm` snapping, two plot smoke
tests under the Agg backend). The demo entrypoint is not unit-tested
(its pieces are), matching the CTM convention for runnable scripts.

**Small refactors folded in (DRY).**
- Added `CorridorSpec.position_m` property (center-reference); `compute`
  now uses it instead of recomputing the offset inline.
- Extracted `corridor_from_ctm(result, pad, dx_grid)` from `from_ctm` so
  the demo (and callers) can get the snapped corridor without duplicating
  the miles->grid logic. Exported from `__init__`.

**Visual check.** Ran the demo for real: `P_y` oscillates 150 kW (peak =
gamma) to 75 kW (gap valleys, overlap 2->1 at `delta_grid=2`); the heatmap
shows the pad striping, per-cell VHT bands, and the empty-start fill-in
transient (downstream cells dark until vehicles arrive). Both sensible.

**Verify.** `pytest tests/test_dwc_plots.py` → 4 passed;
`python scripts/run_dwc_demand_demo.py` writes two PNGs; full DWC suite
(7 files) → 72 passed.

### Follow-up — 2026-06-10 — queuing confirmation

The P5 caveat — that the four-cell example never spills back into a queue,
so the queue exclusion was only proven synthetically — is now closed. A new
CTM example, `metered_on_ramp_scenario` (capacity-limited on-ramp, demand
1000 > R=600 veh/h), drives a real on-ramp queue. Three tests in
[`tests/test_dwc_adapter.py`](../tests/test_dwc_adapter.py) confirm
end-to-end that DWC demand excludes queue VHT: the scenario builds queue;
`mainline_vht` is strictly below the CTM's (queue-inclusive) VHT; and, by
linearity of `compute`, demand splits as mainline + queue-only with
`from_ctm` keeping only the mainline part. On a 40-step run the excluded
queue contribution is ~277 kWh — **21.9 % of the queue-inclusive total** —
so the exclusion is materially significant, not cosmetic.

**Verify.** `pytest tests/test_dwc_adapter.py` → 7 passed; full DWC suite
→ 75 passed.

### Follow-up — 2026-06-11 — build_P_y via convolution

`build_P_y` no longer materializes the dense `T_R`. It forms
`P_A = S_R * S_T` directly, signature now
`build_P_y(S_T, S_R, gamma, *, method="auto")`:

- `method="auto"` (default) picks `np.convolve` for small kernels (coarse
  grids) and `scipy.signal.fftconvolve` for large ones (`len(S_R) > 256`,
  the benchmarked crossover); `convolve` / `fft` / `toeplitz` are selectable.
- `build_T_R` is retained as the reference / `method="toeplitz"` path and
  documented as off the hot path.
- `E` is **unchanged** — only how `P_y` is computed differs (identical values
  to float round-off), so everything downstream is untouched.

[`scripts/benchmark_dwc_convolution.py`](../scripts/benchmark_dwc_convolution.py)
sweeps corridor length × resolution (physical units) across all three methods;
results are tabled in the Resolution-vs-compute section. Headline: the dense
`T_R` is OOM beyond ~1 coarse mile; convolution keeps memory `O(n)` and runs
50 mi / 0.1 m in ~5–12 ms.

New tests in [`tests/test_dwc_spatial.py`](../tests/test_dwc_spatial.py):
all three methods agree, a 200k-position corridor runs with no dense matrix,
and an unknown `method` is rejected. The "Resolution vs. compute" section and
"Defaults chosen" / Functions-map / `T_R` open item were corrected — the
earlier "no representation trick escapes this" framing overstated a bottleneck
that convolution removes.

**Verify.** `pytest tests/test_dwc_spatial.py` → passed;
`python scripts/benchmark_dwc_convolution.py` prints the sweep;
full DWC suite → 78 passed.

### P7 — 2026-06-15 — macroscopic-vs-microscopic validation harness

**Shipped.** A from-scratch single-lane **Newbolt microsimulation** and a
two-stage accuracy-vs-efficiency study, specified in
[dwc_validation_spec.md](dwc_validation_spec.md) (full spec-driven workflow:
interview → spec → plan → tasks → implement).
[`src/ctm_for_dwc/microsim/`](../src/ctm_for_dwc/microsim/):
`model.py` (`MicrosimSpec`/`Vehicles`/`MicrosimResult`/`GuardStats`), `gipps.py`
(modified-Gipps eq 1-5), `seeding.py` (Poisson inflow → vehicles, `v_max ~
U(v_f±15 mph)`), `simulate.py` (single-lane loop), `aggregate.py` (Edie
per-cell density/flow → reused `compare_against_historical`), `charging.py`
(original mCONV eq 30-32), `examples.py` (uniform-limit fixture).
[`scripts/run_dwc_validation.py`](../scripts/run_dwc_validation.py): the
study driver. 47 tests across `tests/test_microsim_*.py` +
`tests/test_dwc_validation.py`.

**Decisions (user, via interview + spec gates).**
- **Two stages, isolated by construction:** Stage 1 scores micro *and* CTM
  against PeMS (`validation.csv` schema, RMSE/MAPE) + wall-clock; Stage 2
  compares original mCONV (micro trajectories) vs adapted mCONV (CTM VHT). No
  original-mCONV-on-CTM-resampled path.
- **Scope:** single-lane equivalent, charging-load only (no SOC/discharge/HVAC),
  modified Gipps from the 2026 TTE paper; dual-lane + battery stack deferred,
  architecture left open. `v_max` ~ `U(v_f−15, v_f+15) mph` (uniform per 2024
  NAPS, re-centered on corridor `v_f` to avoid biasing Stage 1).
- **Edie per-cell aggregation** (matches the CTM's per-cell quantity), mainline
  inflow only, Stage-2 divergence reported on the actual corridor.

**What we learned (not in the plan).**
- **Newbolt's simplified Gipps is not collision-free at practical `dt`.** Its
  safe-velocity (eq 5) uses the *subject's* speed + a binary switch, so a fast
  follower overshoots a slow leader (overlaps at `dt = 0.25–1 s`; clean only
  near `0.1 s`, ~10× compute, non-monotone). Resolved with a **documented
  displacement-cap guard** (cap displacement at the gap; velocity formula kept
  verbatim) whose activation stats live in `MicrosimResult.guard_stats`. See
  the spec's "Deviations from the published model".
- **The uniform-density equivalence is exact by construction** (CP5,
  `rel err < 1e-6`, per-position to `1e-9`): mapping each EV's in-corridor
  position to the *interior* `P_y` index (center-reference) means the micro
  charges exactly where `M_CTM` is non-zero — the approach/exit transient is
  excluded automatically (resolves the plan's long-standing Risk-2 concern).
- **On the real I-880 corridor the Stage-2 divergence swings with the horizon**
  (−1.5% at 1 h, +8.5% at 3 h, −30% at 8 h on a single seed) — the genuine
  macro-vs-micro traffic-physics signal varying with the day's inflow profile,
  plus single-seed sampling noise. **Multi-seed CIs** (cf. Newbolt's 30 runs)
  are the natural next refinement.

**Verify.** `pytest tests/test_microsim_*.py tests/test_dwc_validation.py` →
passes; `pytest -m slow` runs the end-to-end driver smoke; full suite green
(excluding the pre-existing unrelated `test_model_prototyping.py` GRU
failures). `python scripts/run_dwc_validation.py --case-dir
scripts/output/1mi_case_study` writes a validation summary.

### P7 v2 — 2026-06-15 — N-lane (lane-changing) rewrite

**Why.** The single-lane v1 could not reproduce the CTM's multi-lane capacity
(a literal lane caps ~2000 veh/h vs. I-880's 4-lane `q_max ≈ 6155`), so it
could not be scored fairly against multi-lane PeMS over congested periods. A
review also surfaced v1 assumptions made without sign-off (invented `a`/`b`,
boundary-admission rule, admitted-vs-offered inflow); all are recorded and
dispositioned in [dwc_validation_spec.md](dwc_validation_spec.md).

**Shipped (v1 committed first at `5329ca1` as a restore point).**
- `lanechange.py` (new): Newbolt's incentive + random lane change (eq 6–14,
  Alg 3) **generalized to N lanes**, evaluated per subject; the EV
  charging-lane-seeking term (eq 13) is dropped because DWC spans all lanes (so
  EVs/non-EVs share the rule). Backward gap matches eq 7 (back-to-back);
  selection among feasible lanes is lowest-index (v2 audit V2/V3).
- `simulate.py`: rewritten to a **sequential per-vehicle loop** (Newbolt Alg 1):
  admit → per-vehicle {lane change → Gipps velocity → advance} → retire. The
  loop is deliberately not vectorized so the micro's compute cost is honest for
  the Stage-1 efficiency comparison (v2 audit V4).
- `seeding.py`: Poisson arrivals from **raw PeMS flow** (same source as the CTM
  inflow; driver reads the upstream boundary VDS) + random entry lane.
- `model.py`: `lane`/`n_lanes` on results, `entry_lane` on vehicles,
  **Newbolt paper defaults** (`a=4.79`, `b=2.50`, …) + `lane_change_prob`.
- `aggregate.py`/`charging.py`: confirmed **lane-agnostic** (Edie sums across
  lanes → PeMS-style station totals; charging sums EVs by position) — no logic
  change needed.
- Driver: lane count from `cells.csv`, raw-PeMS seeding (CTM-inflow fallback
  when PeMS absent), paper Gipps params with CLI overrides.

**Decisions (user).** Faithful Poisson (not Newbolt's TTF MC — that estimates
arrivals from junction turning, unnecessary with direct flow counts); DWC all
lanes; paper params + CLI overrides.

**What we learned.** Multi-lane makes the model markedly healthier: the
displacement-guard activation rate on the 1-mile run dropped from ~1.95%
(single-lane) to ~0.06% (4-lane) — lane-changing relieves the congestion that
caused the overshoots. The uniform-limit `< 1e-6` equivalence is unaffected
(lane-agnostic charging/aggregation). A v2 code audit (same as v1) surfaced 13
unflagged assumptions; all are dispositioned in the spec (V2 backward-gap →
match eq 7; V3 lane choice → lowest index; V4 → sequential loop; the rest
kept/confirmed with the user).

**Verify.** `pytest tests/test_microsim_*.py tests/test_dwc_validation.py
-m "slow or not slow"` → 68 passed (incl. end-to-end). Multi-seed CIs and
lane-specific DWC (per-cell lane counts + merge logic) remain future work.
