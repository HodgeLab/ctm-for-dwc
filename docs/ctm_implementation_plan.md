# CTM Module — Implementation Plan

Implementation plan for a Cell-Transmission-Model engine in this library, based on
the **canonical CTMSIM update** specified in [ctm_module.md](ctm_module.md)
(Kurzhanskiy 2007, §4.2.1). The spec doc is the source of truth for the equations;
this doc is the build plan.

## Scope

- **In:** full canonical CTMSIM update — a single freeway as a linear chain of
  cells, one on- and one off-ramp per cell, with the on-ramp blending/allocation
  factors $\gamma_i,\xi_i$, ramp capacities $R_i,S_i$, on-ramp queues, and a
  metering hook. State per cell is $(\rho_i, q_i)$.
- **Out (for now):** Aurora-style networks (merges/diverges, multiple freeways),
  signal control, arterials. Noted as future extensions; not designed for yet.
- **Clean slate:** the CTM module defines its **own** freeway/cell representation.
  It does not depend on the old `utils/representation.py` `Freeway` graph.

## Module layout

A self-contained subpackage under the library (`src/ctm_for_dwc/` = importable):

```
src/ctm_for_dwc/ctm/
  __init__.py     # public API
  model.py        # Cell, Freeway, Scenario dataclasses + validation
  engine.py       # step() and simulate() — the six canonical equations
  metrics.py      # VHT / VMT / delay / productivity loss (eqs 4.9–4.17)
  controllers.py  # ramp-metering interface + NoControl default
  results.py      # SimulationResult container (+ to_dataframe / plotting)
  examples.py     # tiny hand-built systems (dissertation Example 1, …)
  io.py           # adapter: build a Freeway from Step-2 calibrated metadata
scripts/run_ctm_demo.py   # standalone: build a small freeway → run → plot
tests/test_ctm_*.py
```

## Core data model

- **`Cell`** — geometry `length`; FD params `q_max, v_f, w, rho_jam, rho_crit`
  (rho_crit defaults to `q_max/v_f`); ramp params `on_ramp_capacity (R)`,
  `off_ramp_capacity (S)`, `gamma`, `xi`. Validated on construction.
- **`Freeway`** — `dt` ($\Delta T$) + ordered `list[Cell]`. Compiles cells to NumPy
  arrays for the update loop; the dataclass is the inspectable/serializable surface.
- **`Scenario`** — initial `rho0`, `q0`; time-varying `demand[i,k]` (on-ramp
  arrivals), `beta[i,k]` (off-ramp split), `inflow[k]` (upstream boundary). Constant
  shorthands allowed.
- **`SimulationResult`** — arrays `(n_cells, n_steps)` for
  `density, mainline_flow, off_ramp_flow, on_ramp_flow, queue, speed`, plus
  `.to_dataframe()` (long format) and `.metrics()`.

**Validation up front:** FD consistency
$q_{max}=v_f\rho_{crit}=w(\rho_{jam}-\rho_{crit})$ within tolerance; CFL
$v_f\Delta T\le l$ (spec Step-5 rule); $\beta,\gamma,\xi\in[0,1]$. Bad configs fail
loudly, not silently.

## Engine maps 1:1 to the spec

`step(...)` is a **pure function** (no hidden state → unit-testable) executing the
six steps in order, each a vectorized NumPy expression, with docstrings citing the
equation tags already in the spec:

1. on-ramp flow `r` — eq 4.2 · 2. queue `q` — eq 4.3 · 3. mainline flow `f` (+
boundary `f_N`) — eqs 4.4/4.5 · 4. off-ramp `s` — eq 4.6 · 5. density `ρ` — eq 4.7
· 6. speed `V` — eq 4.8.

`simulate(...)` loops `step` over the horizon and packs a `SimulationResult`.

## Verification strategy (four independent layers)

1. **Per-equation hand calculations** — a 2–3 cell system with round numbers, one
   step computed by hand for *each FD regime* (free/congested in & out), asserting
   the engine matches each of the six equations. Pins the equations themselves.
2. **Invariants / properties** — mass conservation every step (vehicles in = out +
   Δstored); $0\le\rho_i\le\rho_{jam}$, $0\le f\le q_{max}$, $q_i\ge0$; monotonicity
   of the update map. Over randomized configs (optionally `hypothesis`).
3. **Golden tests vs. the dissertation** — reproduce **Example 1** (Ch. 3, p. 46):
   two identical cells, $v_f=60, w=20, \rho_{crit}=100, \rho_{jam}=400$, demand
   $r_0=4800, r_2=1200$. The paper states the equilibria: from empty →
   $\rho^u=(80,100)$; from jam → $\rho^{con}=(160,160)$. Assert convergence to both.
   Targets come from the paper, not our code → genuinely independent. (Examples 2/3
   give two more.)
4. **Analytical wave speeds** — a density pulse propagates downstream at $v_f$ (one
   cell/step when $v_f\Delta T=l$); a jam propagates upstream at $w$.

*Integration check (not in CI):* run with Step-2 FD params + measured demand/splits
and compare density/flow contours to PeMS — the dissertation's own calibration
workflow (Ch. 4.3); belongs to a workstream, not the engine's test suite.

## Standalone + pluggable

- **Standalone:** `scripts/run_ctm_demo.py` builds a tiny freeway in code (the
  Example-1 system), runs a day, prints/plots density & speed contours — a runnable
  sandbox.
- **Pluggable:** the engine's only contract is *config + scenario in → arrays out*,
  NumPy-only, no W&B / no old-graph coupling. Workstreams attach at the edges:
  - `io.py` builds a `Freeway` from Step-2 calibrated metadata (the one seam to the
    existing pipeline);
  - the controller interface feeds the metering term in eq 4.2 (ships `NoControl`,
    room for ALINEA later);
  - `SimulationResult.to_dataframe()` joins cleanly to PeMS or feeds the
    imputation / ramp-estimation work (Step 6).

## Phasing (each phase ends green)

| Phase | Deliverable | Verify |
|---|---|---|
| **P0** | `Cell`/`Freeway`/`Scenario` + validation + Example-1 builder | config-validation tests |
| **P1** | pure `step()` for the six equations | per-equation hand-calc tests (layer 1) |
| **P2** | `simulate()` + `SimulationResult` | conservation/invariants (layer 2) |
| **P3** | golden equilibria | Example 1 convergence (layer 3) + wave speeds (layer 4) |
| **P4** | metrics (VHT/VMT/delay/PL) | vs. hand calcs |
| **P5** | demo script + `io.py` adapter | runnable sandbox |
| **P6 (later)** | controller interface + ALINEA | — |

## Defaults chosen (override if needed)

- **Vectorized NumPy** core (not per-cell Python loops); `step` stays pure/array-based.
- **Output = arrays + `to_dataframe()`** (no xarray dependency).
- **On-ramp default $\gamma=0$** (drops blending; the simple-freeway case), even
  though CTMSIM's own default is 1 — fully overridable.
- **Demo system = dissertation Example 1**, doubling as the golden test.

## Open items flagged during build

- **Upstream boundary inflow.** P0–P1 admits `f_0 = min(inflow_demand, cell-1
  receiving)` (standard CTM source, keeps $\rho\le\rho_{jam}$); no upstream queue
  yet. The spec calls `f_0` "exogenous." Revisit whether to add a boundary queue
  (cf. dissertation's $n_0$) in P2.
