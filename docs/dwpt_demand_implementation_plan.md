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

## Resolution vs. compute (fundamental tradeoff)

The spatial submodule discretizes the corridor onto a grid of spacing
`dx_grid`, so the number of positions is `n = corridor_length / dx_grid`.
**Every** spatial quantity scales with `n`: `S_T`, `S_R`, and `P_y` are
`O(n)`; `M_CTM` and `E` are `O(N·m)` and `O(T·m)` with `m ≈ n`; a dense
`T_R` is `O(n·δ_grid)` nonzeros.

The pad geometry forces the grid fine. Newbolt's `λ = 0.1524 m` only
resolves to integer grid units at `dx_grid = 0.0001 m` (0.1 mm), and at
that resolution `n` explodes with corridor length:

| Corridor | `dx_grid` | `n` | dense `T_R` (float64) |
|---|---|---|---|
| 6.5 m bench | 0.1 mm | 64,572 | ~41 GB |
| 1 mile | 0.1 mm | ~16 million | infeasible |
| 10 miles | 0.1 mm | ~160 million | infeasible (even 1-D `P_y` ≈ 1.3 GB) |
| 10 miles | 10 cm | ~160,000 | small — **but does not resolve `λ`** |

**This is the core bottleneck, and a sparse/`fftconvolve` `T_R` does not
solve it.** Replacing the dense Toeplitz with a convolution removes the
`O(n·δ_grid)` materialization, but `n` itself — and therefore `P_y`,
`M_CTM`, and `E` — still grows linearly with `corridor_length / dx_grid`.
There is no representation trick that escapes this. We are balancing
**model resolution against compute/memory**: fidelity to the physical pad
geometry (small `λ`) drives `dx_grid` down, while usable corridor length
and runtime drive it up.

Consequences for the build:
- v0 keeps the dense path for clarity/correctness and validates on small,
  coarse systems. The P2 layer-3 test uses a coarse analog for exactly
  this reason — the literal Table-1 dimensions are not testable densely.
- Before modeling real corridors (P5+) we must (a) move to a convolution
  path (`fftconvolve`, `O(n log n)`, no dense `T_R`) **and/or** (b)
  coarsen `dx_grid` and accept a bounded rounding error in pad dimensions
  (snap `λ` to the grid), trading geometric fidelity for tractability.
- `dx_grid` is therefore a per-study decision, not a global default:
  bench validation wants 0.1 mm; a 10-mile grid-impact study may tolerate
  cm-scale spacing with quantified error.

## Open items flagged during build

- **`T_R` materialization.** For corridors > ~2 mi the dense `(m, n)`
  Toeplitz is wasteful. Plan: keep the dense path for tests (clarity)
  but expose an `fft_convolve=True` flag on `build_P_y` for production
  runs. Decide based on profiling at P5. Note this only addresses the
  `T_R` term — the deeper `n`-scaling limit is the resolution-vs-compute
  tradeoff above.
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

### P2 — 2026-06-10 — P_y power profile

**Shipped.**
[`build_P_y`](../src/transportation_models/utils/dwpt/spatial.py) in
`spatial.py` — implements Newbolt 2024a Algorithm 3:
`gamma * (T_R @ S_T) / (T_R @ S_T).max()`, so the peak equals
`gamma = min(beta, beta_prime)`.
[`tests/test_dwpt_spatial.py`](../tests/test_dwpt_spatial.py): 6 new
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

**Verify.** `pytest tests/test_dwpt_spatial.py` → 17 passed in 0.07s;
`tests/test_dwpt_model.py tests/test_dwpt_spatial.py` → 39 passed.

### P3 — 2026-06-10 — M_CTM cell->position map

**Shipped.**
[`build_M_CTM`](../src/transportation_models/utils/dwpt/mapping.py) in
`mapping.py` — the `(N, m)` row-partition-of-unity.
[`tests/test_dwpt_mapping.py`](../tests/test_dwpt_mapping.py): 8 tests —
hand calcs (symmetric `δ_grid=3`, even-`δ_grid` floor, `δ_grid=1` plain
partition) plus invariants (shape, rows sum to 1, `n_i` entries each
`1/n_i`, exactly `δ_grid-1` zeroed boundary columns, one cell per mapped
column).

**Decision: center-reference convention (Option C), chosen by the user**
after evaluating three options in
[`scripts/dwpt_m_ctm_options.ipynb`](../src/transportation_models/scripts/dwpt_m_ctm_options.ipynb).
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

**Verify.** `pytest tests/test_dwpt_mapping.py` → 8 passed in 0.07s;
`tests/test_dwpt_model.py tests/test_dwpt_spatial.py tests/test_dwpt_mapping.py`
→ 47 passed.

### P4 — 2026-06-10 — compute_demand + DemandResult

**Shipped.**
[`compute_demand`](../src/transportation_models/utils/dwpt/demand.py) in
`demand.py` — `eta_EV * (VHT.T @ M_CTM) * P_y`, shape `(T, m)` in Wh.
[`DemandResult`](../src/transportation_models/utils/dwpt/model.py) in
`model.py` — frozen container holding `E` (Wh), `position_m`, `timesteps`,
with shape validation and a `to_dataframe()` long-format helper; exported
from the package `__init__`.
[`tests/test_dwpt_demand.py`](../tests/test_dwpt_demand.py): 7 tests —
single-cell hand calc, shape, the energy-conservation identity (checked
against a per-cell mean power computed directly from `P_y`), linearity in
`eta_EV` and `VHT`, non-negativity, and zero-where-`VHT`-zero.
4 `DemandResult` tests added to `tests/test_dwpt_model.py`.

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

**Verify.** `pytest tests/test_dwpt_demand.py` → 7 passed; full DWPT suite
(`test_dwpt_model.py test_dwpt_spatial.py test_dwpt_mapping.py
test_dwpt_demand.py`) → 58 passed.

### P5 — 2026-06-10 — CTM adapter + compute orchestrator

**Shipped.**
- [`CorridorSpec.build_P_y()`](../src/transportation_models/utils/dwpt/model.py)
  — composes P1-P2 over the corridor's grid properties.
- [`compute(VHT, corridor, eta_EV)`](../src/transportation_models/utils/dwpt/demand.py)
  — orchestrates `build_M_CTM` + `build_P_y` + `compute_demand` into a
  labeled `DemandResult` (center-reference `position_m`, timestep-index
  rows; validates `eta_EV ∈ [0,1]` and VHT cell count).
- [`adapter.py`](../src/transportation_models/utils/dwpt/adapter.py) (new
  module): `mainline_vht(result)` and `from_ctm(result, pad_spec, dx_grid,
  eta_EV)` — the single CTM seam.
- `compute` and `from_ctm` exported from the package `__init__`.
[`tests/test_dwpt_compute.py`](../tests/test_dwpt_compute.py) (6) and
[`tests/test_dwpt_adapter.py`](../tests/test_dwpt_adapter.py) (4), incl. an
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
  CTM import; the DWPT package has no runtime CTM dependency (`from_ctm`
  duck-types the passed result; CTM types are `TYPE_CHECKING`-only).
- **The four-cell example never spills back into an on-ramp queue**, so
  `mainline_vht`'s queue exclusion is proven on a controlled synthetic
  result (`test_mainline_vht_excludes_queue_contribution`); the real-run
  test checks `mainline = metrics_VHT − queue·dt`.

**Verify.** `pytest tests/test_dwpt_compute.py tests/test_dwpt_adapter.py`
→ 10 passed; full DWPT suite (6 files) → 68 passed.

### P6 — 2026-06-10 — demo script + plotting

**Shipped.**
- [`plots.py`](../src/transportation_models/utils/dwpt/plots.py):
  `plot_P_y` (power-vs-position) and `plot_demand_heatmap`
  (E space-time heatmap), following the `ctm/plots.py` convention
  (data + keyword-only `out_path`, save at 120 dpi, close).
- [`scripts/run_dwpt_demand_demo.py`](../scripts/run_dwpt_demand_demo.py):
  four-cell CTM run -> `compute` -> both plots; run via
  `python scripts/run_dwpt_demand_demo.py`. Lives in the repo-root `scripts/`
  (the CTM module's entrypoint convention), **not** in the package — only
  importable library code belongs under `src/`.
[`tests/test_dwpt_plots.py`](../tests/test_dwpt_plots.py): 4 tests
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

**Verify.** `pytest tests/test_dwpt_plots.py` → 4 passed;
`python scripts/run_dwpt_demand_demo.py` writes two PNGs; full DWPT suite
(7 files) → 72 passed.

### Follow-up — 2026-06-10 — queuing confirmation

The P5 caveat — that the four-cell example never spills back into a queue,
so the queue exclusion was only proven synthetically — is now closed. A new
CTM example, `metered_on_ramp_scenario` (capacity-limited on-ramp, demand
1000 > R=600 veh/h), drives a real on-ramp queue. Three tests in
[`tests/test_dwpt_adapter.py`](../tests/test_dwpt_adapter.py) confirm
end-to-end that DWPT demand excludes queue VHT: the scenario builds queue;
`mainline_vht` is strictly below the CTM's (queue-inclusive) VHT; and, by
linearity of `compute`, demand splits as mainline + queue-only with
`from_ctm` keeping only the mainline part. On a 40-step run the excluded
queue contribution is ~277 kWh — **21.9 % of the queue-inclusive total** —
so the exclusion is materially significant, not cosmetic.

**Verify.** `pytest tests/test_dwpt_adapter.py` → 7 passed; full DWPT suite
→ 75 passed.
