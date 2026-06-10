# DWPT Demand Module — Implementation Plan

Implementation plan for a Dynamic Wireless Power Transfer (DWPT) demand
module that adapts the *mCONV* method of
[Newbolt 2024a](https://ieeexplore.ieee.org/document/10750800) to consume
macroscopic outputs from this repo's CTM engine. The spec
[dwpt_demand_module.md](dwpt_demand_module.md) is the source of truth for
equations, notation, and assumptions; this doc is the build plan and the
running progress log.

## Scope

- **In:** spatial submodule (build $\mathbf{P_y}$ from corridor geometry +
  pad specs), cell→position mapping (build $\mathbf{M_{CTM}}$), and demand
  assembly ($\mathbf{E} = \eta_{EV}\,(\mathbf{VHT}^T \mathbf{M_{CTM}}) *
  \mathbf{P_y}$). Single-lane-equivalent corridor, single vehicle class,
  ramps excluded — per the spec's v0 assumptions.
- **Out (for now):** per-class fleet mix, lane-specific DWPT
  infrastructure, velocity-dependent $\mathbf{P_y}$, multi-vehicle Tx-pad
  saturation. Listed in the spec; not designed for here.
- **Couples to CTM via one seam:** reads $\mathbf{VHT}$ (shape `(N, T)`)
  from `utils.ctm.metrics.Metrics.vht_per_cell`, plus per-cell `length` from
  `Freeway`. No other coupling.

## Module layout

A self-contained subpackage alongside `utils/ctm/`:

```
src/transportation_models/utils/dwpt/
  __init__.py     # public API
  spatial.py      # build_S_T, build_S_R, build_T_R, build_P_y
  mapping.py      # build_M_CTM (cell→position partition-of-unity)
  demand.py       # compute_demand (VHT × M_CTM × P_y × eta_EV)
  model.py        # PadSpec, CorridorSpec, DemandResult dataclasses
  examples.py     # Newbolt small-scale system; tiny synthetic corridors
  plots.py        # P_y vs position; E heatmap vs (time, position)
scripts/run_dwpt_demand_demo.py    # CTM run → demand → plot
tests/test_dwpt_*.py
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

Pure functions, NumPy-only:

1. **`build_S_T(corridor_length_m, alpha, lambda_gap, beta_prime, dx_grid)
   -> ndarray (n,)`** — repeats the `[β' × α, 0 × λ]` tile across the
   corridor. (Spec §"Spatial Dependence", Algorithm 1.)
2. **`build_S_R(delta, beta, dx_grid) -> ndarray (δ_grid,)`** — constant-
   magnitude Rx kernel. (Spec §"Spatial Dependence", Algorithm 2.)
3. **`build_T_R(S_R, n) -> ndarray (m, n)`** — Toeplitz of `S_R` over an
   `(n + δ_grid - 1) × n` band, mimicking Rx traversal. (Spec eq. for
   $\mathbf{T_R}$.)
4. **`build_P_y(T_R, S_T, gamma) -> ndarray (m,)`** — implements Newbolt
   Algorithm 3: form the raw area profile `P_A = T_R @ S_T`, then return
   `gamma * P_A / P_A.max()` so the peak equals `gamma = min(beta,
   beta_prime)`. The `/ P_A.max()` normalization **is** the spec's
   $\gamma = \min(\beta,\beta')/\max(\mathbf{T_R}\mathbf{S_T})$ — do not drop
   it, or the peak overshoots by a factor of `P_A.max()`. (Spec eq. for
   $\mathbf{P_y}$; Algorithm 3.)
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
   adaptation is exact. **Lives in `tests/test_dwpt_validation.py` but
   marked `slow` / opt-in.**

## Phasing (each phase ends green)

| Phase | Deliverable | Verify |
|---|---|---|
| **P0** | `PadSpec` / `CorridorSpec` + validation + Newbolt example builder | config-validation tests |
| **P1** | `build_S_T`, `build_S_R`, `build_T_R` pure functions | per-function hand-calc tests (layer 1) |
| **P2** | `build_P_y` | hand-calc tests + Newbolt Fig 4 qualitative match (layer 3) |
| **P3** | `build_M_CTM` | row-sum and off-corridor invariants (layers 1+2) |
| **P4** | `compute_demand` + `DemandResult` | energy conservation identity (layer 2) |
| **P5** | CTM↔DWPT adapter (`from_ctm(freeway, metrics, pad_spec)`) | end-to-end smoke test on a tiny CTM run |
| **P6** | demo script + plotting | runnable sandbox |
| **P7 (opt-in)** | macroscopic-vs-microscopic validation harness | layer 4 |

## TDD workflow per function

For each function in P1–P4, **the test comes first**:

1. Write the hand-calc test in the corresponding `tests/test_dwpt_*.py`.
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

- **NumPy-only core** (no PyTorch / no sparse libs in v0). `T_R` is
  built dense; for the scale of corridors we model (≤ 10 mi, `dx_grid` ~
  0.1 m → `n` ~ 160 000 cells of `S_T`), this stays well under 1 GB if
  we use `float32` and avoid materializing `T_R` for the multiplication
  (compute `T_R @ S_T` via stride tricks or `scipy.signal.fftconvolve`).
  **Revisit if memory bites.**
- **Position grid `dx_grid` defaults to `gcd(alpha, delta, lambda_gap)`**
  rounded to the nearest mm — guarantees integer-position pad dimensions.
- **Units convention:** all lengths in meters, all powers in watts inside
  the DWPT module. The CTM adapter converts the freeway's miles → m at
  the seam.
- **Output:** `DemandResult.E` is in **Wh** (per spec); the `to_dataframe`
  helper provides a long-format `(timestep, position_m, energy_wh)` table
  for joining with other analyses.

## Open items flagged during build

- **`T_R` materialization.** For corridors > ~2 mi the dense `(m, n)`
  Toeplitz is wasteful. Plan: keep the dense path for tests (clarity)
  but expose an `fft_convolve=True` flag on `build_P_y` for production
  runs. Decide based on profiling at P5.
- **`dx_grid` default.** Newbolt's `λ = 0.1524 m` is not a nice round
  number. If we round to mm, the tile won't tessellate cleanly. May
  need to upsample to 0.1 mm or accept a tiny rounding error in the
  corridor length. Decide at P0.
- **Approach/exit position handling.** `T_R` has `m > n` rows for the
  vehicle entering/leaving the corridor. We zero those columns in
  `M_CTM`. If a future use case wants to report demand "as the EV
  enters the corridor" the approach rows would need their own
  treatment. Out of scope for v0.
- **CTM `VHT` includes ramp-queue contribution.** Per the spec's
  ramp-handling assumption, this should be excluded from the mainline
  demand. The CTM adapter needs to subtract ramp-queue VHT or pull
  mainline-only VHT directly from `Metrics`. Confirm which is cleaner
  at P5 (may require a small `metrics.py` addition rather than
  doing it on the DWPT side).

## Progress log

*(Update as phases land. Entry per phase: date, what shipped, what we
learned that wasn't in the plan.)*

### P0 — 2026-06-10 — data model + Newbolt example builder

**Shipped.**
[`src/transportation_models/utils/dwpt/`](../src/transportation_models/utils/dwpt/):
`__init__.py`, `model.py` (`PadSpec`, `CorridorSpec`), `examples.py`
(`newbolt_small_scale`).
[`tests/test_dwpt_model.py`](../tests/test_dwpt_model.py): 22 tests
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
   a tuple of floats — no per-cell metadata (e.g., DWPT-on vs DWPT-off
   stretches, or per-cell pad variations). If the corridor ever has
   heterogeneous Tx layouts (one of the deferred items in the spec),
   we'll need a `Cell` struct. Not blocking v0.
3. **`m_traversal = n + delta_grid - 1` convention.** This treats the
   first traversal position as "Rx pad right edge at corridor start"
   and the last as "Rx pad left edge at corridor end" — matching
   Newbolt's Fig. 2 illustration. Worth verifying against the actual
   `T_R` indexing in P1, since an off-by-one here propagates into
   every downstream array shape.

**Verify.** `pytest tests/test_dwpt_model.py` → 22 passed in 0.02s.
Full suite (excluding pre-existing failures in
`tests/test_model_prototyping.py`, which is unrelated GRU work): 400
passed.

### P1 — 2026-06-10 — spatial pure functions

**Shipped.**
[`src/transportation_models/utils/dwpt/spatial.py`](../src/transportation_models/utils/dwpt/spatial.py):
`build_S_T`, `build_S_R`, `build_T_R` (Newbolt 2024a Algorithms 1-2).
[`tests/test_dwpt_spatial.py`](../tests/test_dwpt_spatial.py): 11
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

**Verify.** `pytest tests/test_dwpt_spatial.py` → 11 passed in 0.07s;
`tests/test_dwpt_model.py tests/test_dwpt_spatial.py` → 33 passed.
Purely additive (no existing module imports `spatial` yet), so no
regression surface elsewhere.
