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
* In congestion `C_w − q ≤ 0` yields a degenerate band; those windows are flagged and dropped.

### Estimator + features

* **Features `Z`** (Kan Table III): upstream and downstream mainline `total_flow`,
  `avg_speed`, `avg_occupancy` at `t, t-1, t-2`.
* **Target**: $\alpha = (y - y_{min}) / (y_{max} - y_{min})$ of the determined-first ramp.
* **Model**: random forest (gradient boosting optional), per Kan.

### Validation

Leave-`k`-stretches-out cross-validation (a stretch is Kan's "ramp pair"): remove all
of `k` held-out stretches' windows, train on the rest, reconstruct `r̂, ŝ` for the
held-out windows, and score against known `r, s`. `k` is configurable (default 1 =
Kan Scenario 1); higher `k` reproduces Kan's Section V.C Scenarios 2–4, aggregating
over held-out combinations (enumerated when tractable, else sampled) as mean±std.
Metrics: **NRMSE** and **R²** (plus BIAS for Kan parity) on reconstructed flows.

**Target windows**: by default training uses only **fully-measured** windows (both ramps'
detectors report at `t`), giving clean directly-measured `α` targets. A
`require_both_measured=False` flag additionally admits `conservation_resolvable` windows
(one ramp measured, the other recovered by conservation) — whose target is then partly a
function of the mainline inputs, so it is opt-in.

### Module layout

| file | responsibility |
|---|---|
| `bounds.py` | Kan Appendix A bounds, determined-first rule, α↔flow, degenerate-band handling (method-agnostic) |
| `features.py` | corridor windows → feature matrix `Z` + `α` targets (method-agnostic; reuses the audit's window logic) |
| `kan.py` | RF/GBM α-regressor: `fit`, `predict`, reconstruct via `bounds` |
| `validation.py` | leave-`k`-out CV, NRMSE/R²/BIAS, Scenario 1–4 aggregation (method-agnostic) |

Run via `scripts/evaluate_ramp_flow_estimation.py` over `data/pems/stretches/*.csv`.

**Out of scope for this build**: the temporal/conservation dispatch, the CTM
`demand`/`beta` CSV export, and predicting real never-measured target stretches.