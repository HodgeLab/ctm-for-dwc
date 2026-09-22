# Ramp Flow Estimation

> **PARTIALLY SUPERSEDED.** This document describes several estimators, not all
> of which still exist. The active workflow is: train a GRU
> (`train_ramp_flow_gru.py`) → build a ramp scenario from it
> (`build_ctm_ramp_scenario.py`) → build the optimization bundle
> (`build_ctm_diffsim_inputs.py`, renamed from `build_ctm_qp_inputs.py`) →
> optimize (`optimize_ramp_flows_diffsim.py`).
>
> Removed, and described here for reference only: the **Julia inverse-CTM QP**
> (`julia/ramp_qp/`, `scripts/verify_ramp_qp.py`) and the **Zhang estimator**
> (`zhang.py`, `zhang_features.py`, and the `*_zhang_*` scripts). Recover with
> `git show pre-cleanup:<path>`.

A fundamental issue in traffic flow modeling is the existence of missing ramp flows, which are critical boundary conditions for freeway traffic simulation.
The problem is described in detail in [ctm_module](ctm_module.md); here, we describe the different methods for estimating missing ramp flows. Most methods are specific to Type 1 estimation on stretches which are configuration type (c), except for the model-based optimization method.

## Table of Contents

- [Module layout](#module-layout)
- [Stretch identification](#stretch-identification-pems-native)
- [Kan estimator](#kan-estimator)
- [GRU estimator](#gru-estimator)
- [Zhang estimator](#zhang-estimator-transfer-learning-gru--dda--mt)
- [Model-based optimization](#model-based-optimization)

## Module layout

The data processing and estimator validation machinery are method-agnostic; everything method-specific receives its own file.

| file | responsibility |
|---|---|
| `features.py` | corridor windows → feature matrix `Z` + measured `(r, s)` targets; the IO assembler returns `(TrainingData, stretch_ctx)` — the corpus plus a per-stretch bounds-context table (`c_w`, `r_demand`, `s_qmax`). `require_capacity=False` keeps stretches without calibrated capacity (GRU-only corpora) |
| `bounds.py` | Kan Appendix A bounds, determined-first rule, α↔flow, out-of-band detection |
| `kan.py` | RF/GBM α-regressor behind the shared interface: derives α targets internally (`alpha_targets`), expands the ctx table (`row_context`), reconstructs via `bounds` |
| `gru.py` | direct joint `(r, s)` GRU regressor + `Normalizer`; checkpoint save/load |
| `zhang_features.py` | Zhang 2024 corpus: mainline preprocessing (fill rules) + per-stretch samples (ρ = q/v, per-step hour/minute) |
| `zhang.py` | Zhang 2024 estimator: GRU backbone + Deep Domain Adaptation + Model Transfer, staged interface |
| `validation.py` | leave-`k`-out CV, stretch-level train/val/test split, NRMSE/R²/BIAS, Scenario 1–4 aggregation |
| `manifest.py` | split manifest (corpus params + split + digest over corpus **and** ctx) so separate runs provably share one corpus |


## Stretch identification (PeMS-native)

Ramp-configuration type and the mainline↔ramp topology are determined **directly
from PeMS station postmiles**, not from the CTM cell grid (which snaps ramps to
cell edges and filters detectors through cell assignment). `scripts/build_pems_stretches.py`
reads `data/pems/station_metadata.csv` and, per `(Fwy, Dir)`, forms a *unit
stretch* between each pair of consecutive mainline (`ML`) detectors, ordered by
`Abs_PM`, with the on-ramp (`OR`), off-ramp (`FR`), and connector (`FF`) stations
in between. It writes a reviewable `stretches.csv` (one per direction). A stretch
is type (c) when it holds ≥1 on-ramp and ≥1 off-ramp.

`scripts/audit_ramp_determinability.py` then scans the full multi-year timeseries
for sliding windows (default 3 consecutive samples, Kan's `t, t-1, t-2`) in which a
stretch is *determinable* — both `ML` detectors observed and ≤1 ramp unobserved
(conservation recovers one). Type-(c) stretches with enough determinable windows
form the **training corpus** for the estimator below.

## Kan estimator

This is a faithful implementation of [Kan et al. 2021](https://doi.org/10.1109/TITS.2020.2989365). The core is implemented in `utils.ramp_flow_estimation.kan.py`, with an entry point in `scripts/estimate_ramp_flows_kan.py`.

### Formulation (Kan Appendix A)

At each 5-min step, with upstream/downstream mainline flows `q_up`, `q_down` and
weaving-area capacity `C_w`, the ramp flows are bounded by conservation and capacity:

$$r_{min} = \max(q^{down} - q^{up},\,0), \quad s_{min} = \max(q^{up} - q^{down},\,0)$$
$$R_{max} = \min(C_w - q^{up},\, r_{cap},\, r_{demand}), \quad S_{max} = \min(C_w - q^{down},\, s_{cap},\, s_{qmax})$$
$$r_{max} = \min(q^{down} - q^{up} + S_{max},\, R_{max}), \quad s_{max} = \min(q^{up} - q^{down} + R_{max},\, S_{max})$$

Kan predicts the **determined-first** ramp — the one with a positive conservation
floor — via a coefficient $\alpha \in [0, 1]$ blended over its band, and recovers
the pair by conservation (Kan A.18–A.19):

$$q^{down} \ge q^{up}:\quad \hat r = \alpha\, r_{max} + (1-\alpha)\, r_{min},\quad \hat s = \max(\hat r + q^{up} - q^{down},\,0)$$
$$q^{up} > q^{down}:\quad \hat s = \alpha\, s_{max} + (1-\alpha)\, s_{min},\quad \hat r = \max(\hat s + q^{down} - q^{up},\,0)$$

**Bound decisions (documented approximations):**

* `C_w` = the nearest-upstream mainline detector's (`up_ml_id`) calibrated **per-lane**
  `capacity` (veh/hr/lane, ~1500–2000) × its number of `Lanes`, from
  `station_metadata_calibrated.csv` — i.e. total mainline capacity in veh/hr.
* `r_cap, r_demand, s_cap, s_qmax` all default to `+∞`, so `R_max`/`S_max` default to
  the weaving cap `C_w − q`. This keeps A.10/A.11 exact while yielding a band that is
  computable for **target (unmeasured) ramps** — which have no data-derived capacity —
  since `y_min` comes from conservation and `y_max` from `C_w` and mainline flow alone.
* `r_demand` / `s_qmax` are set from a ramp's **historical peak observed flow** when it
  has history (tightens the band for measured ramps; targets stay at `C_w − q`).
  "Peak demand" is approximated by peak observed *flow* (metered demand is unobservable).
* `r_cap` / `s_cap` stay `+∞`; an **HCM geometry-based ramp capacity** (from ramp lanes)
  is a noted follow-up.
* A.11 as *printed* uses `C_w − q^{up}` for `S_max`; per A.9 and conservation this is an
  apparent typo, and we use `C_w − q^{down}`.
* In congestion `C_w − q ≤ 0` yields a degenerate band; those windows have no α target
  and are excluded from Kan *fitting* (the corpus itself keeps them — they carry valid
  flow targets for direct estimators).

### Estimator + features

* **Units**: PeMS `total_flow` is veh/5-min; the assembler scales flows to **veh/hr**
  on load so that `q_up`, `q_down`, `r`, `s`, and the bounds match the veh/hr `C_w`.
* **Features `Z`** (Kan Table III): upstream and downstream mainline `total_flow`,
  `avg_speed`, `avg_occupancy` at `t, t-1, t-2`, plus three window-level time
  features at the prediction step `t`: day-of-week (0–6), hour-of-day (0–23),
  and the 5-minute slot within the hour (0–11). The latter two decompose Kan's
  time-of-day index `k ∈ [0, 287]` (`k = 12·hour + slot`); day-of-week adds
  weekly structure. They are appended as the last 3 columns of `Z`.
* **Target**: $\alpha = (y - y_{min}) / (y_{max} - y_{min})$ of the determined-first ramp,
  derived *inside the estimator* from the corpus and its bounds context (the corpus
  itself is method-agnostic; see the module layout below). `q_up`/`q_down` are recovered
  from the feature matrix's last-lag block; `C_w`/`r_demand`/`s_qmax` arrive as the
  per-stretch bounds context, expanded row-aligned via `kan.row_context`.
* **Model**: random forest (gradient boosting optional), per Kan.

### Validation

Leave-`k`-stretches-out cross-validation (a stretch is Kan's "ramp pair"): remove all
of `k` held-out stretches' windows, train on the rest, reconstruct `r̂, ŝ` for the
held-out windows, and score against known `r, s`. `k` is configurable (default 1 =
Kan Scenario 1); higher `k` reproduces Kan's Section V.C Scenarios 2–4, aggregating
over held-out combinations (enumerated when tractable, else sampled) as mean±std.
Metrics: **NRMSE** and **R²** (plus BIAS for Kan parity) on reconstructed flows.

**Target windows**: by default the corpus uses only **fully-measured** windows (both ramps'
detectors report at `t`), giving clean directly-measured flow targets. A
`require_both_measured=False` flag additionally admits `conservation_resolvable` windows
(one ramp measured, the other recovered by conservation) — whose target is then partly a
function of the mainline inputs, so it is opt-in.

Run via `scripts/estimate_ramp_flows_kan.py` over `data/pems/stretches/*.csv`.

**Out of scope for this build**: the temporal/conservation dispatch, the CTM
`demand`/`beta` CSV export, and predicting real never-measured target stretches.

## GRU estimator
We have observed in the raw PeMS data that flow is generally not conserved at a ramp gores, i.e., 
$$q_{up} + r_{true} \ \ != \ q_{down} + s_{true}$$

This poses a challenge to Kan's method, which encodes the assumption that flow is conserved through the $r_{min}$ and $s_{min}$ calculations (A.3 and A.6, respectively).

It may be possible to develop a more accurate estimator for $r$ and $s$, which captures this aspect of the data, by learning the ramp flows directly.

Using the same feature vector as Kan, we train a gated recurrent unit (GRU) to estimate the ramp flows directly (`gru.py`):

* **Inputs**: the same feature windows as Kan — the traffic block reshaped to a
  `(window_size, 6)` sequence of [up/down × flow/speed/occ] steps, with the three
  window-level time features (day-of-week, hour, 5-min slot at `t`) broadcast onto
  every timestep, giving a `(window_size, 9)` sequence; the estimator accepts whatever
  window the corpus carries (default 3, matching Kan for a controlled comparison).
  numpy→tensor conversion happens once at the estimator boundary; everything inside
  is torch.
* **Normalization** (`gru.Normalizer`, fitted on the train split): features are scaled
  per physical **variable** — one mean/std each for flow, speed, occupancy, shared
  across the up/downstream stations and all lags; each time channel gets its own
  independent mean/std. Targets are configurable
  (`target_transform`): `"zscore"` (center+scale) or `"log1p"` (log1p then
  center+scale, weighting relative rather than absolute errors).
* **Output**: joint prediction — GRU → 2-unit linear head in normalized-target space;
  the inverse transform clamps flows at zero. No conservation floor or capacity ceiling
  is imposed, since the bounds are derived from the conservation assumption under test.
* **Targets**: directly-measured $(r, s)$ from fully-measured windows. The
  method-agnostic corpus keeps *all* such windows — including degenerate-band and
  conservation-violating (α-clipped) ones, which are precisely the windows motivating
  this estimator; `kan.alpha_targets` recomputes those diagnostics wherever needed.
* **Training**: Adam/MSE, early stopping on validation loss with best-weight restore.
* **Corpora**: the benchmark corpus requires calibrated upstream capacity
  (`require_capacity=True`) so Kan can run on identical rows; a second, GRU-only
  configuration (`--include-uncalibrated`) additionally trains on stretches without
  calibrated capacity for best-achievable accuracy.

### Evaluation protocol (GRU vs. Kan)

A fixed stretch-level train/val/test split (`validation.train_val_test_split`): one
stretch's windows to validation (early stopping/tuning), one to test, the rest to
train — seeded-random selection, overridable by stretch ID. The corpus build parameters,
resolved split, and a digest over the corpus + bounds context are recorded in a
**split manifest** (`manifest.py`), so the two separately-launched entrypoints provably
see identical data:

* `scripts/train_ramp_flow_gru.py` — GRU-only training driver. Logs the full config and
  per-epoch train/val losses to Weights & Biases (project `ramp-flow-estimation`;
  `WANDB_MODE=offline` supported for HPC), writes the manifest, and checkpoints the
  model (checkpoint + manifest saved as W&B artifacts). The test stretch is never
  touched here. Use a distinct `--out-dir` per configuration.
* `scripts/compare_ramp_flow_estimators.py` — slurm-friendly CLI. Rebuilds the corpus
  from the manifest (hard failure if the digest no longer matches), re-derives the
  identical split, trains Kan RF on the train rows (degenerate-band rows are excluded
  from its fit internally), loads the GRU checkpoint (or `--retrain-gru`), and reports
  NRMSE/R²/BIAS on the test stretch — overall and sliced by in-band / α-clipped /
  degenerate-band windows, so any gap is attributable to the estimator and localized to
  the conservation-violating regime. Requires a `require_capacity` manifest.

A clean negative result (GRU no better than Kan) is a valid outcome; the protocol's job
is attribution, not advocacy.

## Zhang estimator (transfer learning: GRU + DDA + MT)

A faithful implementation of [Zhang et al. 2024](https://doi.org/10.1109/TITS.2023.3315693)
(`zhang_features.py` + `zhang.py`), which frames the type-(c) problem as
*transfer learning*: a **Source Stretch** with measured ramps trains an
estimator, adapted to a **Target Stretch** whose ramps are unmeasured. Unlike
the pooled-corpus estimators above, the pipeline is pairwise: one explicit
source stretch, one explicit target stretch, three stages evaluated in order
so each component's contribution is attributable.

1. **GRU backbone** (paper Fig. 3) — per step, the mainline state
   `[q_up, q_down, v_up, v_down, ρ_up, ρ_down]` plus that step's time
   features `[hour, minute, day-of-week (0–6), is-holiday (0/1)]`, over
   `H = 5` history steps. Per-step Linear embedding (10→64) → GRU (hidden
   64) → FC (64→256) → hidden state `H_t` → separate FC heads for `r` and
   `s`. Trained on the source with Adam (lr 1e-4), MSE, batch 64, ≤100
   epochs; early stopping on a seeded held-out fraction of **source days**
   (default 20%, drawn from all days). No conservation or capacity structure.
2. **Deep Domain Adaptation** (DDA) — reduces the marginal distribution
   difference using only the target's *unlabeled* mainline windows: a fresh
   BatchNorm(256) is inserted before `H_t` (the paper adds BN only at this
   stage — Fig. 2 vs Fig. 3), the embedding+GRU are frozen, and FC/BN/heads
   fine-tune on `source MSE + λ·MMD(H_src, H_tgt)` with `λ = 0.01` for a
   **fixed** number of epochs (no target labels ⇒ no early-stop signal).
3. **Model Transfer** (MT, paper Eq. 5) — per-ramp amplitude scalars
   `h = mean(y / ŷ)` over one *Traffic-Survey* day's samples with nonzero
   estimates; predictions are multiplied by `(h_r, h_s)`. In evaluation the
   survey is simulated from one full day of the target's real ramp flows,
   repeated over 5 seeded random days (metrics mean ± std, each repetition
   scored on the target minus its survey day, Kan/Zhang's `T = n_t − n_p`).
   In real deployment this requires a field count — the known limitation.

**Zhang-specific preprocessing** (`zhang_features.preprocess_mainline`).
Applied at runtime to **mainline series only** — the on-disk 5-minute grid is
never modified, and ramp targets are never filled:

1. mainline samples with `pct_observed` below a threshold (CLI
   `--pct-observed-threshold`, default 50) are treated as missing;
2. missing runs of 1–2 steps are filled by persistence (a run at the record
   start falls through to rule 3);
3. longer runs are filled by the detector's historical mean for that
   (day-of-week, time-of-day) slot; never-observed slots stay NaN and the
   window is unusable;
4. windows whose mainline data are *entirely* filled are dropped.

### Decision log vs. the paper

The paper's description is the default; every departure is deliberate and
recorded here so the implementation is auditable.

**Deviations from the paper's specification:**

* **Default deployment pools sources and omits DDA.** The paper's pipeline is
  single-source backbone → Deep Domain Adaptation → Model Transfer. Our
  `train_ramp_flow_zhang.py` **defaults** to training the backbone on a *pool*
  of source stretches (all `--stretch-ids` except `--target-stretch`) and
  **skipping** DDA, because the 2×2 pooling experiment below found
  pooled-no-DDA beats the faithful single-source + DDA on target NRMSE mean
  *and* variance — pooling appears to buy what DDA was meant to. The knobs are
  orthogonal: `--mode {pooled,single}` selects the source set and `--dda`
  toggles the adaptation stage, so `--mode single --dda` reproduces the
  paper-faithful pipeline exactly. Model Transfer runs in every configuration.
  When DDA is off there is no adaptation-layer distance to report, so the
  per-stage output is backbone → +MT (the `mt` stage) instead of
  backbone → +DDA → +DDA+MT.
* **Preprocessing is replaced wholesale.** The paper fills large gaps with
  the same-weekday historical average, small gaps by linear interpolation,
  and deletes+re-interpolates 3σ outliers. We use the four rules above
  instead: a PeMS-specific `pct_observed` quality gate (the paper has no
  equivalent), **persistence** rather than linear interpolation for short
  gaps, **no outlier removal**, and the all-filled-window drop. Fill rules
  apply to mainline only — the paper preprocesses the whole dataset, ramp
  series included, whereas our targets are always directly-measured values.
* **Density is derived, not measured**: `ρ = q/v` (PeMS carries occupancy,
  not the density the paper's feature vector specifies).
* **No early stopping in the DDA stage.** The paper says early stopping is
  used "in training and fine-tuning" but never names the fine-tuning signal,
  and no target labels exist to supply one — the DDA stage runs a fixed
  `--dda-epochs` instead. (Backbone early stopping is kept; see below.)
* **BatchNorm is inserted at the DDA stage.** Fig. 3 (backbone) has no BN;
  Fig. 2 (transfer framework) shows BN before the adaptation layer. We follow
  the text ("the BN Layer is *added* before the Adaptation layer… in
  adaptation"): the backbone trains without BN, and a fresh BN(256) is
  inserted when `adapt` begins.
* **Evaluation is one target per run.** The paper's groups average the
  two directions of each stretch pair (and its no-transfer baseline is a
  single-source model); we score one explicit `--target-stretch` per
  invocation — against the pooled source by default, or a single
  `--source-stretch` under `--mode single` — with per-stage metrics.
  Looping over configurations is a deferred open item.
* **All days, with day-type feature channels.** The paper restricts to
  workday data; we instead extend the per-step time features with
  day-of-week (integer 0–6) and an is-holiday flag
  (`USFederalHolidayCalendar`, observed dates) and train on **all days by
  default** — more data, and weekend/holiday ramp behavior stays learnable.
  `--weekdays-only` opts back into the paper-faithful workday-only corpus
  (results predating this change were produced in that mode). Simulated
  Traffic-Survey days remain **weekday-only** regardless (a real survey runs
  on a weekday, and a weekend-calibrated MT scalar would mis-scale weekday
  predictions); the source-validation day holdout is unrestricted so it
  mirrors the training distribution.

**Where the paper is silent, choices made:**

* **MMD kernel** (Eq. 2 never states φ): Gaussian RBF, per-batch
  median-heuristic bandwidth; the loss is the RKHS *norm* (square root of the
  biased squared-MMD estimator), matching Eq. 2 as printed.
* **Feature/target scaling** (unstated): per-channel z-score fitted on the
  source training split only; predictions clamped ≥ 0 on the inverse.
* **Embedding width** (unstated): 64. The FC after the GRU maps 64 → 256 so
  that the adaptation layer matches the stated "BN hidden dimension = 256"
  (the text's word "reduce" is treated as loose wording).
* **Backbone early-stopping signal** (unstated): source-val MSE on a seeded
  held-out fraction of **source days** (default 20%, `--source-val-frac`).
* **Survey-day selection** (unstated): seeded random draw among target
  **weekdays** carrying ≥ 50% of the busiest such day's window count (so a
  sparsely-observed day can't stand in for a full survey day).
* **Per-pair MMD distance** (`ZhangEstimator.pair_mmd`): the paper reports
  the MMD between each source/target pair (Tables II/IV/VI) as
  transfer-difficulty context. Per Eq. 2 and §III-D this is defined over the
  **Adaptation Layer** — the hidden state `H_t` after the GRU module — so it
  requires a trained backbone and is computed with the same Gaussian-RBF
  median-heuristic estimator as the DDA loss (the kernel itself remains our
  choice), on a seeded subsample of ≤ 2048 windows per stretch (the kernel
  matrix is O(n²)). The driver reports it twice: after backbone training
  (the paper-analogous pre-adaptation distance) and after DDA (which should
  shrink it). Values are comparable across our runs but not to the paper's
  tables (their kernel/scaling are unknown).

**Matched to the paper** (for the record): `H = 5`, per-step
`[hour, minute]` time features (ours adds day-of-week/holiday; see the
deviations), Adam lr 1e-4 / MSE / batch 64 / 100 epochs, GRU hidden 64,
`λ_mmd = 0.01`, frozen embedding+GRU during DDA, MT over the
nonzero-estimate set with 5 survey days reported as mean ± std, and
NRMSE/R² pooled over both ramps (Eqs. 6–7).

Run via `scripts/train_ramp_flow_zhang.py --stretch-ids <ids…> --target-stretch <id>`
for the pooled-no-DDA default, or `--mode single --dda --source-stretch <id>`
for the paper-faithful pipeline (same W&B/out-dir conventions as the GRU
driver; writes `pair_manifest.json` with build params + per-source array
digests, `result.json` with the source↔target MMD distance and per-stage
target metrics, and the trained checkpoint — post-DDA when `--dda` is set,
else the pooled backbone).
A single-source `--dda` sweep's results aggregate offline via `scripts/aggregate_zhang_pairs.py
--results-dir <sweep dir>`: `pairs.csv` plus per-metric (NRMSE/R²/NBIAS)
plots against the **post-DDA** pair MMD — the DDA stage as a scatter, the
DDA+MT stage as mean ± std error bars across survey days, with x-binned
mean ± std bands. Passing `--stretches`/`--timeseries-dir` additionally
computes each pair's **flow-amplitude ratio** per ramp type — the mean of
log(source/target) over the rank-matched top-500 ramp flows (positive =
source peaks hotter; an MT-aligned difficulty axis orthogonal to MMD) — and
emits the matching per-ramp `{metric}_{on,off}_vs_flowratio.png` plots. The
samples are rebuilt with each pair's `pair_manifest.json` build parameters,
so the ratios always match the corpus that produced that pair's results; no
re-training is involved.

### Pooled-vs-DDA 2×2 experiment

Does pooling training data across stretches already buy what DDA buys? The
experiment trains the full 2×2 — {single-source, pooled} × {no-DDA, +DDA} —
on the Zhang backbone/corpus and scores every arm on the same held-out
target stretch (all target windows, full flow metrics + adaptation-layer
pair MMD; **no MT** — the question lives in the mainline-features-only
regime). The pool is an explicit stretch-ID list minus the target, so it
includes the single source and pooled-vs-single is a pure data-addition
comparison; the no-DDA/+DDA arms of each pool share one backbone (evaluated
before and after `adapt`), and both backbones start from the same model
seed. Early stopping uses the same seeded day-level holdout, drawn over each
arm's own training stretches.

* `scripts/compare_zhang_pooling.py` — one (target, single-source)
  combination per run: builds each stretch's samples once (byte-identical
  across arms, digests recorded), trains the four arms, logs per-epoch
  curves under arm prefixes and final target metrics over `arm_idx` to W&B,
  writes the 2×2 table to `result.json`.
* `scripts/aggregate_zhang_pooling.py` — offline evaluation over a sweep
  directory of `result.json` files (no W&B): `combos.csv` (one row per
  combination × arm: NRMSE/R²/NBIAS + MMD), `summary.csv` (per-arm mean/std
  across combinations), and per-metric scatter+trend plots of accuracy vs
  pair MMD, one series per arm. The x-axis is the **single/no-DDA arm's**
  pair MMD — the distance as seen by a source-trained backbone before
  adaptation (each arm's own MMD is also in the CSV).

Reading the result: if `pooled` tracks the DDA arms flat across MMD, pooling
subsumes DDA; if the no-DDA arms degrade with MMD while the DDA arms stay
flat, the transfer machinery earns its keep.

This experiment is what motivated the pooled-no-DDA **default** in
`train_ramp_flow_zhang.py` (see the first deviation above): pooled-no-DDA came
out ahead of single-source + DDA on target NRMSE mean and variance, so the
deployment recipe pools sources and omits adaptation while keeping Model
Transfer.

**Open items (deferred by design)**: looping over many source/target
configurations; source-selection rules (e.g. min-MMD); MT ratio-robustness
variants (mean-of-ratios inflates when ŷ is small — kept faithful to Eq. 5
for now); congestion-regime slicing; multi-seed error bars for the 2×2
experiment if the arm gaps look seed-sized.

## Model-based optimization
This is a method inspired by [Muralidharan and Horowitz 2009](https://doi.org/10.3141/2099-07), though the implementation is distinct.

Implemented across three pieces: `scripts/build_ctm_qp_inputs.py` (Python prep),
`julia/ramp_qp/solve_ramp_qp.jl` (the solver), and `scripts/verify_ramp_qp.py`
(round-trip verification). Unlike the other estimators, this method estimates the ramp flows in *cells*; every ramp in the freeway is estimated, without any historical data.

**Idea.** Rather than reconstruct each ramp series on its own, fit the *whole
corridor* at once: find the ramp flows that make the CTM's mainline density
trajectory match observed PeMS density, with the CTM update equations enforced
as constraints. Observed ramp data is deliberately **excluded** from the fit,
so the held-out ramp measurements become an honest test set for "can mainline
density + CTM dynamics alone reconstruct ramp flow?"

**Formulation (a convex QP).** Solved with JuMP + Gurobi:

* **Decision variables** — on-ramp *admitted flow* $r_i(k)$ and off-ramp *flow*
  $s_i(k)$, both $\ge 0$, piecewise-constant over 5-min blocks (PeMS cadence).
* **State variables** — density $\rho_i(k)$ and mainline flow $f_i(k)$, pinned by
  the CTM update as equality/inequality constraints (simultaneous transcription).
* **Objective** — $\sum (\rho_i(k_m) - \hat\rho_i(m))^2$ over the **direct /
  tiebreak** cells (genuine observations) at the 5-min sample grid, with
  $\hat\rho = \text{total\_flow}\cdot 12 / \text{avg\_speed}$.
* **Key linearizations** that keep it a *convex* QP:
  * The off-ramp **split** $\beta$ is not a variable — using it would make
    $\beta\rho$, $\beta f$, $1/\beta$ nonconvex. We solve for the off-ramp
    **flow** $s_i$ instead and recover $\beta_i = s_i/(f_i+s_i)$ afterward.
  * **$\gamma = 1$** (the engine/CTMSIM default). Under the $s$-form $\gamma$
    enters only linearly ($\rho_i + \gamma r_i\,\Delta T/l_i$), so matching the
    forward simulator costs nothing in convexity.
  * **No on-ramp queue** ($q\equiv 0$): demand is unidentifiable from mainline
    density (only the *admitted* flow $r_i$ appears in conservation), and the
    held-out target is observed ramp *flow*. So $r_i$ is the variable and is
    output as `--demand`; this also round-trips the on-ramp exactly.
  * The CTM `min` operators are **relaxed to $\le$** (Ziliaskopoulos-style), so
    no binaries. The relaxation is exact only where the bound is active; the
    round-trip check below measures this.
* **Boundary/initial conditions fixed from PeMS** — $\rho_i(0)$ from
  `initial_state_from_vds` and upstream inflow $f_0(k)$ from `inflow_from_vds`
  enter as parameters, not variables, so the QP is fed exactly what the forward
  simulator is fed (keeping the round-trip honest).
* **No regularization** beyond $r,s\ge 0$ — pure least squares. The inverse is
  generally **non-unique**, so the recovered flows are one of many
  density-equivalent solutions. (Deferred: a small temporal-smoothness /
  minimal-norm penalty to select a physical minimizer; and a big-M **MIQP**
  escalation to make the `min` exact where the relaxation is loose.)

**Pipeline.**

```
# 1. Prep the QP bundle (reuses initial_state_from_vds / inflow_from_vds /
#    the Step-9 observed-density extraction):
python scripts/build_ctm_qp_inputs.py \
    --freeway scripts/output/ctm_corridor/<case>/freeway.csv \
    --cells   scripts/output/ctm_corridor/<case>/cells.csv \
    --timeseries-dir data/pems/csv_files \
    --start "2022-04-12 06:00" --end "2022-04-12 09:00"
# -> writes freeway.csv, rho0.csv, inflow.csv, observed_density.csv, meta.json
#    into <case>/qp_inputs/. dt defaults to the largest CFL-stable 5-min divisor.

# 2. Solve (Julia). Default optimizer is Gurobi; set RAMP_QP_OPTIMIZER=clarabel
#    for a license-free convex-QP solver (used by the test suite):
julia --project=julia/ramp_qp julia/ramp_qp/solve_ramp_qp.jl <case>/qp_inputs
# -> demand.csv, beta.csv (the --demand / --beta inputs), plus *_fitted.csv and
#    solve_report.json.

# 3. Verify the round-trip + relaxation tightness:
python scripts/verify_ramp_qp.py --bundle <case>/qp_inputs
```

**Round-trip = the tightness diagnostic.** Because the `min` constraints are
relaxed, the QP's fitted trajectory only equals a real CTM run where those
relaxations are tight. `verify_ramp_qp.py` feeds the solver's `demand`/`beta`
back through the **exact** forward simulator (with `gamma=1`, matching the QP)
and reports (a) the density gap between the forward run and the QP's fitted
density, and (b) a per-cell/step map of how much slack sits on each mainline
flow's binding bound. A large gap with many loose operators means the recovered
ramp flows, while density-matching, are **not** a physical CTM solution — the
expected symptom of the unregularized, relaxed inverse, and the signal to add
regularization or escalate those operators to a big-M MIQP.

**Status.** Prep, solver, and verification are implemented and unit-tested
(`tests/test_build_ctm_qp_inputs.py`, `tests/test_verify_ramp_qp.py`); the Julia
solver is exercised manually (Gurobi license / cross-language, so it is not in
CI). A synthetic identifiability check (known freeway $\to$ forward-sim density
$\to$ QP) confirms the QP reproduces the observed density to ~$10^{-8}$.

**Finding — density transfers, flow does not, and the gap grows with length.**
On the I-880 N case studies (`case_studies/I880N_{1,5,10}mi`,
2023-06-01, 24 h, dt 5 s) the QP matches observed density essentially exactly
(objective $\approx 0$). Run through the **exact** forward simulator and scored
with `validate_ctm_corridor.py` against historical PeMS (vs the
historical-average ramp fill), the result is corridor-length-dependent:

| case | metric | Level 3 | histAve |
|---|---|---|---|
| 1mi (2 cells)  | density RMSE / flow RMSE | **19.9** / 638 | 24.6 / **342** |
| 1mi            | flow MAPE / GEH<5        | 12.3% / 62%    | 11.8% / 60%    |
| 5mi (14 cells) | density MAPE / flow MAPE | 47% / 50%      | **27% / 17%**  |
| 5mi            | flow RMSE / GEH<5        | 955 / 24%      | **443 / 44%**  |

On the short corridor Level 3 is competitive — it even beats historical average
on **density** (the quantity it optimizes) — but on the 5mi it is clearly worse,
especially on **flow** and the aggregate/GEH measures. The split is diagnostic:
the objective (density) stays competitive while flow degrades and worsens with
length. Two coupled causes:

1. **Relaxation looseness (dominant).** The QP fits density under the *relaxed*
   CTM (`min` $\to$ `$\le$`), but `simulate`/`validate` use the *exact* `min`.
   The optimizer exploits the slack — choosing flows below their true `min` — so
   its trajectory is not a valid CTM run. When the exact operators "snap back,"
   density diverges (`verify_ramp_qp` round-trip: ~190 RMS veh/mi on the 5mi,
   100 % of mainline operators loose). This is independent of the validator, so
   it is the relaxation, not a wiring bug.
2. **Flow is never in the objective** — only density. Historical average feeds
   *physically real* ramp flows that the exact simulator handles sanely, so its
   density *and* flow validate well.

Critically, regularization-for-uniqueness alone would **not** fix this: a unique
loose solution is still loose. Making the optimum a genuine exact-CTM trajectory
is what guarantees transfer. Two principled routes (under evaluation):

* **Exact MIQP** — big-M binaries make the `min` exact, keeping the JuMP+Gurobi
  constraint formulation; the solution round-trips by construction. Cost:
  tens of thousands of binaries (tractability on 5mi/10mi is the open question).
* **Differentiable forward-simulation** — optimize the ramp inputs by running
  the *exact* forward CTM each iteration and descending the density error;
  structurally immune to the transfer failure because it never leaves the exact
  dynamics. (This is the "option B" set aside during design.)

Baseline `validate_ctm_corridor` stats for the current relaxed-QP output are
saved under each case study's `qp_ramps/validation/` for comparison against
whichever fix lands next.