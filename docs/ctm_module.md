# Cell Transmission Model (CTM) Module
The cell transmission model (CTM) is a first-order macroscopic transportation model introduced by 
[Daganzo](https://www.sciencedirect.com/science/article/pii/0191261594900027?via%3Dihub) and adapted by [Munoz](https://ieeexplore.ieee.org/document/1383703). Later, [Kurzhanskiy](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2007/EECS-2007-148.html) introduced the link-node cell transmission model (LN-CTM). In any case, the model discretizes a freeway into a set of n cells, each characterized by a fundamental diagram, and iteratively evolves the density of traffic within each cell according to a set of [state update equations](#ctm-state-update-equations). The [Caltrans Performance Measurement System (PeMS) database](https://dot.ca.gov/programs/traffic-operations/mpr/pems-source) is a frequently used data source for CTM case studies, thanks to its densely-instrumented network. 

Application of the CTM to a given freeway corridor involves the following steps: 

1) Identification of vehicle detector stations (VDSs) associated with the corridor and collection of data from each VDS
2) Calibration of fundamental diagram parameters for each available VDS
3) Mapping of roadway to CTM cells
4) Mapping of VDSs to CTM cells
5) Tuning of CTM cell lengths
6) Ramp flow estimation

## Step 1: VDS Identification and Download
`transportation_models.utils.data_downloading` includes two classes, `PeMSDownloader` and `PeMSExtractor` for interfacing with the PeMS database and downloading VDS data.

## Step 2: Fundamental Diagram Calibration
`transportation_models.utils.data_processing` includes a class, `PeMSDataProcessor` with methods for calibrating fundamental diagram parameters.
Key parameters are:
1) Capacity $q_{max,i}$
2) Free-flow speed $v_{f,i}$
3) Congestion-wave speed $w_i$
4) Jam density $\rho_{jam,i}$
5) Critical density $\rho_{crit,i}$

## Step 3: Roadway --> Cell Mapping
Implemented in `transportation_models.utils.ctm.osm` (corridor extraction),
`transportation_models.utils.ctm.cells` (cell layout), and
`transportation_models.utils.ctm.caltrans` (optional Caltrans postmile
annotation). End-to-end demo: `scripts/build_ctm_from_osm.py`.

The mapping is a three-stage pipeline that takes an OpenStreetMap network for a
region and emits a freeway-schema cell table ready for
`utils.ctm.io.freeway_from_dataframe`. Each stage is intentionally small so it
can be tested and swapped in isolation.

**Stage 1: OSM → `Corridor`.** The caller fetches a roadway graph via any
osmnx download method (`graph_from_bbox`, `graph_from_place`,
`graph_from_polygon`, `load_graphml`, …) and runs `ox.simplify_graph` on it.
`corridor_from_graph(graph, ref="I 880", direction="N")` then:

1. Filters edges to `highway=motorway` matching the requested `ref` (matches a
   bare string, list-valued OSM tags, or `;`-joined values).
2. Drops mainline edges whose bearing is more than `bearing_tolerance` degrees
   off the requested direction, which picks one carriageway out of a
   bidirectional bbox.
3. Takes the largest weakly-connected component of what remains.
4. Walks the result upstream → downstream, picking the dominant carriageway at
   every fork via a `(lanes, length, bearing-closeness)` tuple score. This
   tolerates OSM's habit of tagging parallel HOV / express / auxiliary lanes
   with the same `ref` as the trunk (e.g., I-880 in Hayward, I-210 in Pasadena
   where the bbox truncates an HOV stub a few meters from the trunk).
5. Identifies ramps as `highway=motorway_link` edges with exactly one endpoint
   on the mainline path (source on mainline ⇒ off-ramp; sink on mainline ⇒
   on-ramp).
6. Optionally crops to `pm_range=(lo, hi)` in cumulative-mile coordinates.

The output `Corridor` holds an ordered list of `MainlineSegment`s
(geometry, lane count, `pm_start`/`pm_end` in cumulative miles from upstream)
and `RampJunction`s (kind, postmile, gore-point geometry, ramp name).

If the user doesn't know the right OSM `ref` for a region, `summarize_refs(g)`
inventories the refs present, ranked by total mainline length, with the
most-common OSM `name` tag attached (e.g. `Nimitz Freeway` for `I 880`).
`scripts/list_osm_refs.py` is the CLI wrapper.

**Stage 2 (optional): Caltrans postmile annotation.** When
`corridor_from_graph(..., postmiles=path_or_gdf)` is given a Caltrans SHN
postmile dataset (point geometries with `Route` and `PM` columns), it
attaches a `caltrans_postmiles: dict[node_id → PM]` to the corridor. For each
mainline node it finds the nearest matching-route marker and adjusts that
marker's PM by the signed distance along the mainline from marker to node,
using the route's direction convention (PMs increase N/E along the route, so
westbound and southbound corridors get monotonically *decreasing* PMs along
travel — matching what CTMSIM / the dissertation report). The downloader
`download_caltrans_postmiles(routes=[880])` pulls a route-filtered subset
from the Caltrans FeatureServer (≈1 MB per route, vs ~144 MB statewide) and
caches it under `data/caltrans/` (gitignored).

**Stage 3: cell layout.** `cells_from_corridor(corridor)` collects breakpoints
from the union of on-ramp gores, off-ramp gores, mainline lane-count changes,
the corridor endpoints, and any `extra_break_postmiles` the caller passes in
(Step 4 uses the last to enforce "one VDS per cell"). Adjacent breakpoints
become cells, and each cell records `length`, `lanes`, `on_ramp`, `off_ramp`,
the ramp names if present, and — when Caltrans PMs were attached in stage 2 —
`caltrans_pm_start`/`caltrans_pm_end`. The resulting DataFrame slots directly
into `freeway_from_dataframe` once Step 2 supplies the FD parameters.

The CTMSIM/dissertation convention that "on-ramps live at the *start* of a
cell, off-ramps at the *end*" is encoded exactly here: a cell's `on_ramp`
flag is True iff its `pm_start` sits on an on-ramp gore, and `off_ramp` iff
its `pm_end` sits on an off-ramp gore. A ramp at the corridor's upstream or
downstream end therefore belongs to the first or last cell; co-located
on + off ramps split cleanly across adjacent cells (off → upstream cell,
on → downstream cell).

Because the on-ramp lives at the cell's upstream edge, on-ramp vehicles
have a full ΔT to traverse the cell at free-flow speed (the CFL condition
$v_{f,i}\Delta T \le l_i$ guarantees they don't overshoot). They therefore
*do* contribute to the cell's sending function in the same step, which
fixes the on-ramp blending factor at $\gamma_i = 1$. `Cell.gamma` defaults
to 1.0 accordingly (matching CTMSIM canonical); the dissertation's
Examples 1 and §3.4 — which are stated in the Gomes & Horowitz / four-mode
form with $\gamma_i = 0$ — pin γ explicitly in
`utils.ctm.examples`. The on-ramp allocation factor $\xi_i$ is orthogonal
to ramp position (it controls how much of the cell's empty space the
on-ramp can fill) and defaults to 1.0, matching both conventions.

**Validation.** The Hayward I-880 N case end-to-end: `python
scripts/build_ctm_from_osm.py --bbox -122.08,37.57,-122.04,37.62 --ref "I 880"
--direction N --postmiles auto` extracts 8 mainline segments and 7 cells over
3.89 mi with Caltrans PM 10.9 → 14.7 (NB direction → PMs increase along
travel). The I-210 W golden test against the Kurzhanskiy 2007 dissertation
fixture (`ctmsim_configs/w060412.mat`) confirms the corridor's Caltrans PMs
span the dissertation's PM 24-39 range when anchored at Vernon Ave and the
SR-134 split.

## Step 4: VDS --> Cell Mapping
Currently implemented in `transportation_models.utils.representation` through a class called `Freeway`.
Needs to be re-done.

## Step 5: Cell Length Tuning
Not currently implemented.
A requirement of the CTM is that each cell length is at least as large as the distance covered by vehicles moving at free-flow speed through that cell, i.e., $v_{f,i} ΔT ≤ l_i$. 

## Step 6: Ramp Flow Estimation
Not currently implemented.


## CTM State Update Equations
The LN-CTM is a Godunov discretization of the LWR conservation law with a triangular fundamental diagram. The state of each cell $i$ at step $k$ is its **density** $\rho_i(k)$ (veh/mi) together with its on-ramp queue $q_i(k)$ (veh); flows are veh/h and speeds mi/h. The canonical update below is the **CTMSIM computational model** ([Kurzhanskiy](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2007/EECS-2007-148.html), §4.2.1) — the same model implemented in Aurora. The compact **four-mode switching form** (FF/FC/CC/CF) that [Muralidharan & Horowitz](https://journals.sagepub.com/doi/10.3141/2099-07) reduce it to for a freeway with one on- and one off-ramp per cell is given afterward as an equivalent expansion.

### Notation (units per Kurzhanskiy Table 4.1)

| Symbol | Meaning | Unit |
|---|---|---|
| $l_i$ | cell length | mi |
| $\Delta T$ | time step | h |
| $q_{max,i}$ | mainline capacity | veh/h |
| $R_i,\ S_i$ | on-ramp / off-ramp capacity | veh/h |
| $v_{f,i}$ | free-flow speed | mi/h |
| $w_i$ | congestion-wave speed | mi/h |
| $\rho_{jam,i}$ | jam density | veh/mi |
| $\rho_{crit,i}$ | critical density | veh/mi |
| $\rho_i(k)$ | density of cell $i$ — *state* | veh/mi |
| $q_i(k)$ | on-ramp queue at cell $i$ — *state* | veh |
| $\beta_i(k)$ | off-ramp split ratio | – |
| $\gamma_i$ | on-ramp flow blending factor | – |
| $\xi_i$ | on-ramp flow allocation factor | – |
| $f_i(k)$ | mainline flow, cell $i\to i{+}1$ | veh/h |
| $s_i(k),\ r_i(k)$ | off-ramp / on-ramp flow at cell $i$ | veh/h |
| $d_i(k)$ | on-ramp demand at cell $i$ | veh/h |
| $V_i(k)$ | mean speed in cell $i$ | mi/h |

The five FD parameters $q_{max,i},\,v_{f,i},\,w_i,\,\rho_{jam,i},\,\rho_{crit,i}$ are the ones calibrated in Step 2. The triangle has only three degrees of freedom, so they satisfy

$$q_{max,i} = v_{f,i}\,\rho_{crit,i} = w_i\,(\rho_{jam,i} - \rho_{crit,i})$$

*Source:* Kurzhanskiy eq. (3.1), which writes it as $F_i = \bar\beta_i v_{f,i}\rho_{crit,i} = w_i(\rho_{jam,i}-\rho_{crit,i})$; the plain triangle here (no split factor $\bar\beta_i$) is the one CTMSIM's FD editor parameterizes (§4.2.2).

### CTMSIM update (canonical)

Each step advances $(\rho_i, q_i)$ for every cell in the order below. Write $\bar\beta_i = 1-\beta_i$.

**1. On-ramp flow** *(Kurzhanskiy eq. 4.2)* — the on-ramp admits the lesser of its waiting demand, the empty space it can reach, and its capacity:

$$r_i(k{+}1) = \min\!\left\{\ \underbrace{d_i(k{+}1) + \tfrac{q_i(k)}{\Delta T}}_{\text{demand}+\text{queue}},\ \ \underbrace{\xi_i\,(\rho_{jam,i}-\rho_i(k))\,\tfrac{l_i}{\Delta T}}_{\text{reachable space}},\ \ R_i,\ \ \underbrace{\max\{C_i,\,\mathcal{Q}_i\}}_{\text{metering}}\ \right\}$$

where $C_i,\mathcal{Q}_i$ are the mainline- and queue-controller set points (the last term is present **only under ramp metering**).

**2. On-ramp queue** *(Kurzhanskiy eq. 4.3)* — carries unmet demand forward:

$$q_i(k{+}1) = \max\!\big\{\, q_i(k) + \big(d_i(k{+}1) - r_i(k{+}1)\big)\,\Delta T,\ \ 0 \,\big\}$$

**3. Mainline flow** $i\to i{+}1$ *(Kurzhanskiy eq. 4.4)* — min of sending, downstream supply, off-ramp capacity, and capacity:

$$f_i(k{+}1) = \min\!\left\{\ \bar\beta_i\,v_{f,i}\Big(\rho_i(k) + \gamma_i\,r_i(k{+}1)\tfrac{\Delta T}{l_i}\Big),\ \ w_{i+1}\Big(\rho_{jam,i+1} - \rho_{i+1}(k) - \gamma_{i+1}\,r_{i+1}(k{+}1)\tfrac{\Delta T}{l_{i+1}}\Big),\ \ \tfrac{\bar\beta_i}{\beta_i}\,S_i,\ \ q_{max,i}\ \right\}$$

At the downstream boundary *(Kurzhanskiy eq. 4.5)* the supply term drops (free flow downstream of cell $N$), and $f_0$ is the exogenous upstream inflow:

$$f_N(k{+}1) = \min\!\left\{\ \bar\beta_N\,v_{f,N}\Big(\rho_N(k) + \gamma_N\,r_N(k{+}1)\tfrac{\Delta T}{l_N}\Big),\ \ \tfrac{\bar\beta_N}{\beta_N}\,S_N,\ \ q_{max,N}\ \right\}$$

**4. Off-ramp flow** *(Kurzhanskiy eq. 4.6)* — the split is taken off the total leaving the cell, so it is metered in lockstep with $f_i$ (FIFO):

$$s_i(k{+}1) = \begin{cases} \dfrac{\beta_i}{\bar\beta_i}\,f_i(k{+}1), & \beta_i < 1 \\[8pt] \min\!\Big\{\, v_{f,i}\big(\rho_i(k) + \gamma_i\,r_i(k{+}1)\tfrac{\Delta T}{l_i}\big),\ S_i \,\Big\}, & \beta_i = 1 \end{cases}$$

**5. Density** *(Kurzhanskiy eq. 4.7)* — the conservation law; the $\tfrac{\Delta T}{l_i}$ factor converts a net flow in veh/h to a density change in veh/mi:

$$\rho_i(k{+}1) = \rho_i(k) + \frac{\Delta T}{l_i}\big(\, f_{i-1}(k{+}1) + r_i(k{+}1) - f_i(k{+}1) - s_i(k{+}1) \,\big)$$

**6. Mean speed** *(Kurzhanskiy eq. 4.8)* — derived, for travel-time / VHT outputs:

$$V_i(k{+}1) = \min\!\left\{\, v_{f,i},\ \ \frac{f_i(k{+}1) + s_i(k{+}1)}{\rho_i(k{+}1)} \,\right\}$$

**Simplifications** (recover the plain freeway model by parameter choice):

- $\gamma_i = 0$ — drops the on-ramp blending term, so the sending function uses $\rho_i$ alone (the assumption in Gomes & Horowitz and in the four-mode form below). General $\gamma_i\in[0,1]$ places the on-ramp within the cell.
- $\xi_i = 1$ — on-ramp vehicles may use the cell's entire empty space.
- $S_i \to \infty$ — drops the $\tfrac{\bar\beta_i}{\beta_i}S_i$ term, so off-ramps never restrict the mainline (Muralidharan & Horowitz assume this).
- No metering — drops the $\max\{C_i,\mathcal{Q}_i\}$ term, so $r_i$ is limited only by demand+queue, reachable space, and ramp capacity.

### Equivalent four-mode form (Muralidharan & Horowitz)

In the simplified case ($\gamma_i=0$, uncapacitated off-ramps), define **capacity-limited speeds** and **total node demand** $c_i$ *(all from M&H, p. 59)*:

$$\bar v_i = \min\!\Big(v_{f,i},\ \tfrac{q_{max,i}}{\rho_i}\Big), \quad \bar w_i = \min\!\Big(w_i,\ \tfrac{q_{max,i}}{\rho_{jam,i}-\rho_i}\Big), \quad c_i = \bar\beta_i\,\rho_i\bar v_i + d_i \quad[\text{veh/h}]$$

so the sending is $\rho_i\bar v_i$ and the receiving (supply) is $(\rho_{jam,i}-\rho_i)\bar w_i$. With the downstream-acceptance ratio (a shorthand for the supply/demand fraction in M&H's flow expressions, p. 59–60)

$$\eta_i = \frac{\min\!\big((\rho_{jam,i+1}-\rho_{i+1})\bar w_{i+1},\ c_i\big)}{c_i}\ \in (0,1]\qquad(\eta_i=1\text{ in free flow}),$$

cell $i$'s outflow is $\rho_i\bar v_i$ in free flow and $\eta_i\rho_i\bar v_i$ when the downstream link is congested. Expanding the in/out switches gives four closed-form density updates *(M&H, p. 59 — labeled FF/FC/CC/CF there, not equation-numbered)* (first letter = inflow node $i{-}1$, second = outflow node $i$):

| Mode | $\rho_i(k{+}1) = \rho_i(k) + \tfrac{\Delta T}{l_i}\big(\,\text{inflow} - \text{outflow}\,\big)$ |
|---|---|
| **FF** | $c_{i-1} \;-\; \rho_i\bar v_i$ |
| **FC** | $c_{i-1} \;-\; \dfrac{(\rho_{jam,i+1}-\rho_{i+1})\bar w_{i+1}}{c_i}\,\rho_i\bar v_i$ |
| **CC** | $(\rho_{jam,i}-\rho_i)\bar w_i \;-\; \dfrac{(\rho_{jam,i+1}-\rho_{i+1})\bar w_{i+1}}{c_i}\,\rho_i\bar v_i$ |
| **CF** | $(\rho_{jam,i}-\rho_i)\bar w_i \;-\; \rho_i\bar v_i$ |

These reproduce the canonical update for $\beta_i=0$ or whenever flow is below capacity. They differ only in where the capacity cap sits: M&H apply $q_{max,i}$ to the cell's *total* throughput (via $\bar v_i$), whereas CTMSIM (step 3) caps the *mainline* flow $f_i$.

### Cell-length rule (Step 5)

The Step-5 requirement $v_{f,i}\,\Delta T \le l_i$ is the Courant (CFL) condition for this scheme: a vehicle at free-flow speed cannot cross more than one cell per step. The dissertation states it as $\Delta T < \min_i l_i / v_{f,i}$ *(Kurzhanskiy eq. 4.1)*.

> Notation note: the four-mode form comes from Muralidharan & Horowitz, who — like the dissertation's Ch. 3 — work in *normalized* units: the state is the vehicle count per cell (their Table 1 lists jam density as veh/section, i.e. per cell, not per mile), with cell length and time step folded into the speeds, so no $\tfrac{\Delta T}{l_i}$ factor appears in their conservation equation. Converting that count $n_i$ to the physical density used here is $\rho_i = n_i/l_i$ — standard CTM unit bookkeeping, not a formula written in the paper.