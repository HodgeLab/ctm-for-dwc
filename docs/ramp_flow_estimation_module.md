# Ramp Flow Estimation
A fundamental issue in traffic flow modeling is the existence of missing ramp flows, which are critical boundary conditions for freeway traffic simulation.
For example, in the Cell Transmission Model (CTM), on- and off-ramp flows are used to determine the on-ramp demand and off-ramp split ratio at each cell.
The existence of missing ramp flow data may result in a transportation network that is not fully observable, which is a severe challenge to overcome when modeling traffic flow in the network.

The estimation of missing ramp flow data is commonly categorized into one of two types:

1. Historical data is 100% lacking for the ramp of interest
2. Some historical data exists

For type 2, it may be possible to exploit temporal dependencies in the historical data to fill in, or impute, the missing values. This is commonly achieved through either a statistical method (e.g., ARIMA) or a machine learning-based method (e.g., recurrent neural network).
Type 1 is a more challenging problem, which has been addressed in [Muralidharan and Horowitz 2009](https://doi.org/10.3141/2099-07), [Kan et al. 2021](https://doi.org/10.1109/TITS.2020.2989365), and [Zhang et al. 2024](https://doi.org/10.1109/TITS.2023.3315693).

Furthermore, Type 1 may be further classified on the basis of ramp configuration.
Consider the three options presented in Figure 1 from Zhang et al. 2024.

![ramp_configs](ramp_configurations.png)

When we say we are concerned with estimating missing ramp flows, we are interested in determining the quantities $r$ and $s$, and we assume that the quantities $q^{up}$ and $q^{down}$ are known.
For stretches of type (a) or (b), we use conservation of flow to determine the missing ramp flow, setting $s$ or $r$ to zero depending on the type.

$$ q^{up} + r = q^{down} + s $$

For stretches of type (c), it is impossible to uniquely determine the pair of missing ramp flows from mainline flow alone.
This is the problem that Kan et al. 2021 and Zhang et al. 2024 address.

## Implementation Plan
We require a standalone ramp flow estimation module which is capable of generating CSV files for CTM cell demand and split ratios that are compatible with the CTM simulation engine.
Based on the classifications of missing ramp flow estimation previously discussed, the module is implemented as follows:

```python
if n_historical_points > threshold:
    temporal_fill()
elif (cell_type == TYPE_A) or (cell_type == TYPE_B):
    conservation_fill()
else:
    pair_fill()
```

For each vehicle detector station (VDS) associated with a ramp, we will determine whether the number of historical points in the available timeseries for the VDS(`n_historical_points`) exceeds some user-defined `threshold`.
If so, we will employ a fill strategy which exploits the temporal dependency of the historical data.
If not, we will evaluate the cell with which the VDS is associated and determine the configuration type as previously defined.
If the configuration type permits missing ramp flow estimation by conservation of flow, then that method will be used.
Finally, if the ramp flow data cannot be described by the prior methods, we will implement a fill strategy consistent with the approaches described by Kan et al. and Zhang et al.

Note that we make the problem specific to missing VDS estimation here. This is because a given cell may have multiple VDSs assigned to its on-ramp and/or off-ramp, and it is the *total* demand from these VDSs with which we are concerned, per the CTM Module's [Step 5](ctm_module.md#step-5-manual-validation-of-cell-mappings).

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

## Kan estimator (`utils/ramp_flow_estimation`)

The `pair_fill()` strategy for type-(c) stretches is a faithful implementation of
[Kan et al. 2021](https://doi.org/10.1109/TITS.2020.2989365). The module is named
generally (`ramp_flow_estimation`) so alternative methods (e.g. Zhang 2024) can be
added later; the method-agnostic pieces (bounds, feature extraction, validation)
are separated from the Kan-specific estimator.

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

### Module layout

The corpus and validation machinery are method-agnostic; everything Kan-specific
lives in `kan.py`/`bounds.py`. Estimators share the interface
`fit(X, r, s, ctx=None)` / `predict_flows(X, ctx=None)`, where `ctx` is an opaque
dict of row-aligned context arrays (Kan's bounds context; the GRU ignores it).

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
   `[q_up, q_down, v_up, v_down, ρ_up, ρ_down]` plus that step's Time Feature
   `[hour, minute]`, over `H = 5` history steps. Per-step Linear embedding
   (8→64) → GRU (hidden 64) → FC (64→256) → hidden state `H_t` → separate FC
   heads for `r` and `s`. Trained on the source with Adam (lr 1e-4), MSE,
   batch 64, ≤100 epochs; early stopping on a seeded held-out fraction of
   **source days** (default 20%). No conservation or capacity structure.
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
* **Evaluation is one direction per run.** The paper's groups average the
  two directions of each stretch pair (and its no-transfer baseline is a
  single-source model); we run one explicit `--source-stretch` →
  `--target-stretch` direction per invocation, with per-stage metrics.
  Looping over configurations is a deferred open item.

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
* **Survey-day selection** (unstated): seeded random draw among target days
  carrying ≥ 50% of the busiest day's window count (so a sparsely-observed
  day can't stand in for a full survey day).
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

**Matched to the paper** (for the record): workday-only data (weekend
windows dropped by default; `--include-weekends` opts out; public holidays
are *not* excluded — a refinement over-and-above the weekday filter),
`H = 5`, per-step `[hour, minute]` time features, Adam lr 1e-4 / MSE /
batch 64 / 100 epochs, GRU hidden 64, `λ_mmd = 0.01`, frozen embedding+GRU
during DDA, MT over the nonzero-estimate set with 5 survey days reported as
mean ± std, and NRMSE/R² pooled over both ramps (Eqs. 6–7).

Run via `scripts/train_ramp_flow_zhang.py --source-stretch <id> --target-stretch <id>`
(same W&B/out-dir conventions as the GRU driver; writes `pair_manifest.json`
with build params + array digests, `result.json` with the source↔target MMD
distance and per-stage target metrics, and the post-DDA checkpoint).

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

**Open items (deferred by design)**: looping over many source/target
configurations; source-selection rules (e.g. min-MMD); MT ratio-robustness
variants (mean-of-ratios inflates when ŷ is small — kept faithful to Eq. 5
for now); congestion-regime slicing; multi-seed error bars for the 2×2
experiment if the arm gaps look seed-sized.
