# Step 7 Level 3: inverse-CTM ramp-flow imputation as a convex QP (JuMP + Gurobi).
#
# Reads the CSV/JSON bundle written by `scripts/build_ctm_qp_inputs.py`, builds
# the simultaneous-transcription QP described in docs/ctm_module.md (Level 3),
# solves it with Gurobi, and writes `demand.csv` / `beta.csv` (the `--demand` /
# `--beta` inputs) plus `solve_report.json`.
#
# Formulation (gamma = 1, off-ramp split eliminated via the off-ramp *flow*
# s_i(k); on-ramp *admitted flow* r_i(k), no queue):
#   * decision variables : r_i(k), s_i(k)  (piecewise-constant over 5-min blocks)
#   * state variables     : rho_i(k), f_i(k), and the upstream boundary f0(k)
#   * objective           : sum of (rho - rho_obs)^2 at direct/tiebreak cells
#                           on the 5-min sample grid
#   * the CTM `min` operators are RELAXED to `<=` (convex QP). Round-trip
#     through the forward simulator (`scripts/verify_ramp_qp.py`) is the
#     relaxation-tightness diagnostic.
#
# Usage:
#   julia --project=julia/ramp_qp julia/ramp_qp/solve_ramp_qp.jl <bundle_dir> [out_dir]

using CSV
using DataFrames
using JSON
using JuMP

# Optimizer is pluggable so the convex QP can be validated without a Gurobi
# license: set RAMP_QP_OPTIMIZER=clarabel (a free, pure-Julia QP solver) for
# license-free testing. Default is Gurobi, the production solver. The `using`
# happens at top level (not inside a function) so the solver's methods are in
# the right world age by the time JuMP instantiates the model.
const _OPT_NAME = lowercase(get(ENV, "RAMP_QP_OPTIMIZER", "gurobi"))
if _OPT_NAME == "gurobi"
    using Gurobi
    const OPTIMIZER = Gurobi.Optimizer
elseif _OPT_NAME == "clarabel"
    using Clarabel
    const OPTIMIZER = Clarabel.Optimizer
else
    error("unknown RAMP_QP_OPTIMIZER=$_OPT_NAME (expected gurobi|clarabel)")
end

function main(bundle_dir::String, out_dir::String)
    meta = JSON.parsefile(joinpath(bundle_dir, "meta.json"))
    freeway = CSV.read(joinpath(bundle_dir, "freeway.csv"), DataFrame)
    rho0_df = CSV.read(joinpath(bundle_dir, "rho0.csv"), DataFrame)
    inflow_df = CSV.read(joinpath(bundle_dir, "inflow.csv"), DataFrame)
    obs_df = CSV.read(joinpath(bundle_dir, "observed_density.csv"), DataFrame)

    N = Int(meta["n_cells"])
    T = Int(meta["T"])
    sp5 = Int(meta["steps_per_5min"])
    B = Int(meta["n_5min"])
    dt = Float64(meta["dt_h"])
    gamma = Float64(meta["gamma"])
    xi = Float64(meta["xi"])

    # Per-cell FD params (1-based Julia rows == cell order).
    l = Float64.(freeway.length)
    qmax = Float64.(freeway.q_max)
    vf = Float64.(freeway.v_f)
    w = Float64.(freeway.w)
    rho_jam = Float64.(freeway.rho_jam)
    rho0 = Float64.(rho0_df.rho0)            # length N, cell order
    inflow = Float64.(inflow_df.inflow)      # length T, step order (k = 0..T-1)

    # Ramp structure (meta cell indices are 0-based -> +1 for Julia).
    on_cells = Int.(meta["on_ramp_cells"]) .+ 1
    off_cells = Int.(meta["off_ramp_cells"]) .+ 1
    on_set = Set(on_cells)
    off_set = Set(off_cells)
    Rcap = Dict{Int,Union{Nothing,Float64}}()
    for (k, v) in meta["on_ramp_capacity"]
        Rcap[parse(Int, k) + 1] = v === nothing ? nothing : Float64(v)
    end
    Scap = Dict{Int,Union{Nothing,Float64}}()
    for (k, v) in meta["off_ramp_capacity"]
        Scap[parse(Int, k) + 1] = v === nothing ? nothing : Float64(v)
    end

    # Observed (cell, m, rho_obs), cell 0-based -> +1.
    obs = [(Int(r.cell) + 1, Int(r.m_5min), Float64(r.rho_obs)) for r in eachrow(obs_df)]

    model = Model(OPTIMIZER)

    @variable(model, 0 <= rho[i = 1:N, k = 0:T] <= rho_jam[i])
    @variable(model, f[i = 1:N, k = 0:(T - 1)] >= 0)
    @variable(model, f0[k = 0:(T - 1)] >= 0)            # upstream boundary inflow
    @variable(model, r_blk[i = on_cells, b = 0:(B - 1)] >= 0)
    @variable(model, s_blk[i = off_cells, b = 0:(B - 1)] >= 0)

    # Ramp flow at sim step k: block-constant; 0 when the cell has no ramp.
    ron(i, k) = i in on_set ? r_blk[i, div(k, sp5)] : 0
    soff(i, k) = i in off_set ? s_blk[i, div(k, sp5)] : 0
    # gamma=1 on-ramp-blended density used by sending + receiving.
    rho_eff(i, k) = rho[i, k] + gamma * ron(i, k) * dt / l[i]

    # Initial state fixed from PeMS.
    for i in 1:N
        @constraint(model, rho[i, 0] == rho0[i])
    end

    for k in 0:(T - 1)
        for i in 1:N
            # Mainline sending (total leaving <= capacity-limited demand) + q_max.
            @constraint(model, f[i, k] + soff(i, k) <= vf[i] * rho_eff(i, k))
            @constraint(model, f[i, k] <= qmax[i])
            # Downstream receiving (interior cells only; last cell free-flow out).
            if i < N
                @constraint(model, f[i, k] <= w[i + 1] * (rho_jam[i + 1] - rho_eff(i + 1, k)))
            end
            # On-ramp: reachable space + (optional) capacity.
            if i in on_set
                @constraint(model, ron(i, k) <= xi * (rho_jam[i] - rho[i, k]) * l[i] / dt)
                if Rcap[i] !== nothing
                    @constraint(model, ron(i, k) <= Rcap[i])
                end
            end
            # Off-ramp: (optional) capacity.
            if i in off_set && Scap[i] !== nothing
                @constraint(model, soff(i, k) <= Scap[i])
            end
            # Conservation. Upstream inflow is f0 for cell 1, else the cell above.
            fin = i == 1 ? f0[k] : f[i - 1, k]
            @constraint(model,
                rho[i, k + 1] == rho[i, k] + dt / l[i] * (fin + ron(i, k) - f[i, k] - soff(i, k)))
        end
        # Upstream boundary: min(exogenous inflow, cell-1 receiving), relaxed.
        @constraint(model, f0[k] <= inflow[k + 1])
        @constraint(model, f0[k] <= w[1] * (rho_jam[1] - rho_eff(1, k)))
    end

    # Objective: squared density error at direct/tiebreak cells, 5-min grid.
    @objective(model, Min,
        sum((rho[c, m * sp5] - ro)^2 for (c, m, ro) in obs))

    optimize!(model)
    status = string(termination_status(model))
    obj = objective_value(model)

    # --- extract outputs (wide T x N, 0-based cell-index column labels) ---
    demand = zeros(T, N)
    beta = zeros(T, N)
    for k in 0:(T - 1)
        for i in 1:N
            if i in on_set
                demand[k + 1, i] = value(r_blk[i, div(k, sp5)])
            end
            if i in off_set
                sval = value(s_blk[i, div(k, sp5)])
                fval = value(f[i, k])
                beta[k + 1, i] = (fval + sval > 0) ? sval / (fval + sval) : 0.0
            end
        end
    end

    colnames = string.(0:(N - 1))
    CSV.write(joinpath(out_dir, "demand.csv"), DataFrame(demand, colnames))
    CSV.write(joinpath(out_dir, "beta.csv"), DataFrame(beta, colnames))

    # Fitted state/flow trajectories, so verify_ramp_qp.py can measure the
    # forward-sim round-trip gap and per-operator relaxation tightness.
    rho_fit = [value(rho[i, k]) for k in 0:T, i in 1:N]          # (T+1) x N
    f_fit = [value(f[i, k]) for k in 0:(T - 1), i in 1:N]        # T x N
    s_fit = [i in off_set ? value(s_blk[i, div(k, sp5)]) : 0.0 for k in 0:(T - 1), i in 1:N]
    CSV.write(joinpath(out_dir, "rho_fitted.csv"), DataFrame(rho_fit, colnames))
    CSV.write(joinpath(out_dir, "mainline_flow_fitted.csv"), DataFrame(f_fit, colnames))
    CSV.write(joinpath(out_dir, "off_ramp_flow_fitted.csv"), DataFrame(s_fit, colnames))

    # Per-cell density RMSE at observed points (sanity / report only).
    per_cell = Dict{String,Float64}()
    counts = Dict{Int,Int}()
    sse = Dict{Int,Float64}()
    for (c, m, ro) in obs
        e = value(rho[c, m * sp5]) - ro
        sse[c] = get(sse, c, 0.0) + e^2
        counts[c] = get(counts, c, 0) + 1
    end
    for (c, n) in counts
        per_cell[string(c - 1)] = sqrt(sse[c] / n)   # back to 0-based label
    end

    report = Dict(
        "status" => status,
        "objective" => obj,
        "n_observed" => length(obs),
        "density_rmse_per_cell" => per_cell,
    )
    open(joinpath(out_dir, "solve_report.json"), "w") do io
        JSON.print(io, report, 2)
    end

    println("status=$status  objective=$obj  -> $(out_dir)/{demand,beta}.csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
    bundle_dir = length(ARGS) >= 1 ? ARGS[1] : error("usage: solve_ramp_qp.jl <bundle_dir> [out_dir]")
    out_dir = length(ARGS) >= 2 ? ARGS[2] : bundle_dir
    main(bundle_dir, out_dir)
end
