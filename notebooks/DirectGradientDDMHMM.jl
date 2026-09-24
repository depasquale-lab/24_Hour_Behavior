using Pkg
Pkg.activate("notebooks")

using Random
using Distributions
using Dates
using Statistics
using Printf
using DriftDiffusionModels
using HiddenMarkovModels
using DensityInterface
using CSV
using DataFrames
using BSON: @save

const Optim = DriftDiffusionModels.Optim
const RAT = "1062"
const K = 4
const N_INITS = 30
const MAX_ITER = 2500              # L-BFGS iterations for direct approach
const TIED = Symbol[]         # [] = full, [:v] = tied drift, etc.

Random.seed!(69)

# Load one rat's data
data_file = joinpath("data", "processed_rat_data.csv.gz")
rat_df = CSV.read(data_file, DataFrame)
rat_df = rat_df[rat_df.daily .== "24 hr", :]
replace!(rat_df[!, :choose_right], 0 => -1)
side_mapping = Dict("right" => 1, "left" => -1)
DataFrames.transform!(
    rat_df,
    :correct_side => ByRow(cs -> get(side_mapping, cs, missing)) => :correct_side_numeric,
)

sub = rat_df[rat_df.name .== RAT, :]
@assert !isempty(sub) "rat $RAT not in data"

dates = [Date(split(dt)[1]) for dt in sub.trial_datetime]
unique_dates = sort(unique(dates))

sessions = Vector{Vector{DDMResult}}()
for date in unique_dates
    idx = findall(dates .== date)
    isempty(idx) && continue
    push!(
        sessions,
        [
            DDMResult(rt, ch, st) for (rt, ch, st) in
            zip(sub.rt[idx], sub.choose_right[idx], sub.correct_side_numeric[idx])
        ],
    )
end
const OBS_SEQ = reduce(vcat, sessions)
const SEQ_ENDS = cumsum([length(s) for s in sessions])

@info "Rat $RAT: $(length(OBS_SEQ)) trials across $(length(sessions)) sessions, K=$K"

# Parameter layout (all unconstrained ℝⁿ; transforms below)
#
#   θ = [ init_logits (K)        # softmax → init distribution
#       | trans_logits (K*K)     # softmax per row → transition matrix
#       | ddm_y (4*K) ]          # per-state (log_B, log_v, logit_a₀, log_τ)
#
# Softmax on init/trans is technically overparameterized (one gauge
# dimension per simplex) but L-BFGS handles it fine, and it avoids
# having to choose which logit to pin.
const N_INIT_LOGITS = K
const N_TRANS_LOGITS = K * K
const N_DDM_PARAMS = 4 * K
const N_PARAMS = N_INIT_LOGITS + N_TRANS_LOGITS + N_DDM_PARAMS

function softmax1(x)
    m = maximum(x)
    e = exp.(x .- m)
    return e ./ sum(e)
end

sigmoid(y) = 1 / (1 + exp(-y))

"Return (init::Vector, trans::Matrix, ddm_y::Vector) from a flat θ."
function unpack(θ)
    init_logits = @view θ[1:N_INIT_LOGITS]
    trans_logits_flat = @view θ[(N_INIT_LOGITS + 1):(N_INIT_LOGITS + N_TRANS_LOGITS)]
    ddm_y = @view θ[(N_INIT_LOGITS + N_TRANS_LOGITS + 1):end]

    init = softmax1(init_logits)
    trans = Matrix{eltype(θ)}(undef, K, K)
    for k in 1:K
        trans[k, :] .= softmax1(@view trans_logits_flat[((k - 1) * K + 1):(k * K)])
    end
    return init, trans, ddm_y
end

"Convert y-space ddm params at state k → (B, v, a₀, τ). Applies tying if configured."
function ddm_params_for_state(k, ddm_y, tied_idx_map)
    B_y, v_y, a_y, τ_y = (
        ddm_y[tied_idx_map[k, 1]],
        ddm_y[tied_idx_map[k, 2]],
        ddm_y[tied_idx_map[k, 3]],
        ddm_y[tied_idx_map[k, 4]],
    )
    return exp(B_y), exp(v_y), sigmoid(a_y), exp(τ_y)
end

"Build index map accounting for tied params (same logic as fit!)"
function build_tied_idx_map(tied::Vector{Symbol})
    DDM_PARAMS = (:B, :v, :a₀, :τ)
    free = Symbol[p for p in DDM_PARAMS if !(p in tied)]
    nfree = length(free)
    idx = Matrix{Int}(undef, K, length(DDM_PARAMS))
    for (pi, p) in enumerate(DDM_PARAMS)
        if p in tied
            pos = K * nfree + findfirst(==(p), tied)
            for k in 1:K
                ;
                idx[k, pi] = pos;
            end
        else
            pos = findfirst(==(p), free)
            for k in 1:K
                ;
                idx[k, pi] = (k - 1) * nfree + pos;
            end
        end
    end
    return idx, K * nfree + length(tied)
end

const TIED_IDX_MAP, N_DDM_TIED = build_tied_idx_map(TIED)

# Override total param count if tying reduces it
const N_PARAMS_EFF = N_INIT_LOGITS + N_TRANS_LOGITS + N_DDM_TIED

# Parametric DDM emission so ForwardDiff Duals flow through.
struct DDMEmission{T<:Real}
    B::T
    v::T
    a₀::T
    τ::T
end

DensityInterface.DensityKind(::DDMEmission) = HasDensity()

function DensityInterface.logdensityof(d::DDMEmission, x::DDMResult)
    return DriftDiffusionModels.logdensityof(d.B, d.v, d.a₀, d.τ, x.rt, x.choice, x.s)
end

# Minimal AbstractHMM wrapper
struct DDMHMM{T<:Real} <: HiddenMarkovModels.AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dists::Vector{DDMEmission{T}}
end

Base.length(h::DDMHMM) = length(h.init)
HiddenMarkovModels.initialization(h::DDMHMM) = h.init
HiddenMarkovModels.transition_matrix(h::DDMHMM) = h.trans
HiddenMarkovModels.obs_distributions(h::DDMHMM) = h.dists

function neg_logL(θ)
    init, trans, ddm_y = unpack(θ)
    dists = [DDMEmission(ddm_params_for_state(k, ddm_y, TIED_IDX_MAP)...) for k in 1:K]
    hmm = DDMHMM(init, trans, dists)
    return -HiddenMarkovModels.logdensityof(hmm, OBS_SEQ; seq_ends=SEQ_ENDS)
end

# Random initialization in θ-space.
# Widths chosen to actually probe the landscape (vs. cluster around prior mean):
#   init logits   σ = 1.0
#   trans logits  σ = 0.7, diagonal bias jittered Uniform(0.5, 3.0) per-init
#                 (so some starts are sticky, some less so)
#   DDM y-space:  log B σ=0.8 around log(2)         → B ∈ ~[0.9, 4.4]
#                 log v σ=1.0 around 0              → v ∈ ~[0.37, 2.7]
#                 logit a₀ σ=0.8 around 0           → bias ∈ ~[0.18, 0.82]
#                 log τ  σ=0.7 around log(0.1)      → τ ∈ ~[0.025, 0.4]
function random_init_θ()
    θ = zeros(N_PARAMS_EFF)
    θ[1:N_INIT_LOGITS] .= randn(N_INIT_LOGITS) .* 1.0
    diag_bias = 0.5 + 2.5 * rand()
    for k in 1:K
        base = randn(K) .* 0.7
        base[k] += diag_bias
        θ[(N_INIT_LOGITS + (k - 1) * K + 1):(N_INIT_LOGITS + k * K)] .= base
    end
    off = N_INIT_LOGITS + N_TRANS_LOGITS
    DDM_PARAMS = (:B, :v, :a₀, :τ)
    for k in 1:K, (pi, p) in enumerate(DDM_PARAMS)
        pos = off + TIED_IDX_MAP[k, pi]
        if p == :B
            θ[pos] = log(2.0) + randn() * 0.8
        elseif p == :v
            θ[pos] = 0.0 + randn() * 1.0
        elseif p == :a₀
            θ[pos] = 0.0 + randn() * 0.8
        elseif p == :τ
            θ[pos] = log(0.1) + randn() * 0.7
        end
    end
    return θ
end

# Run
@info "=== Direct gradient descent ($N_INITS inits, threads=$(Threads.nthreads())) ==="

# Pre-generate inits so RNG order is deterministic regardless of thread scheduling.
θ0s = [random_init_θ() for _ in 1:N_INITS]

# Each slot: (init_id, logL, θ_opt, seconds, iterations)
direct_results = Vector{Any}(nothing, N_INITS)
t_wall_start = time()

Threads.@threads for init_id in 1:N_INITS
    θ0 = θ0s[init_id]
    t0 = time()
    result = Optim.optimize(
        neg_logL,
        θ0,
        Optim.LBFGS(linesearch=Optim.LineSearches.BackTracking()),
        Optim.Options(iterations=MAX_ITER, show_trace=false);
        autodiff=:forward,
    )
    dt = time() - t0
    logL = -Optim.minimum(result)
    θ_opt = Optim.minimizer(result)
    direct_results[init_id] = (init_id, logL, θ_opt, dt, result.iterations)
    @info @sprintf(
        "  init %d: logL = %.2f  (%.1fs, %d iters)", init_id, logL, dt, result.iterations
    )
end

t_wall = time() - t_wall_start
t_direct_total = sum(r[4] for r in direct_results)  # summed CPU time across threads
best_direct = maximum(r[2] for r in direct_results)

# Report
tied_tag = isempty(TIED) ? "full" : "tied-" * join(string.(TIED), "_")

@info """

============================================================
  Rat $RAT  •  K=$K  •  config=$tied_tag  •  $(length(OBS_SEQ)) trials
============================================================
  Direct gradient: best logL = $(round(best_direct; digits=2))
                   wall clock $(round(t_wall; digits=1))s   (with $(Threads.nthreads()) threads)
                   CPU total  $(round(t_direct_total; digits=1))s
                   per-init   $(round(t_direct_total / N_INITS; digits=1))s avg
============================================================
"""

# Save artifacts
out_dir = joinpath("results", "ddm_hmm_constrained", "direct_gradient_$(RAT)")
isdir(out_dir) || mkpath(out_dir)

best_i = argmax(r[2] for r in direct_results)
best_θ = direct_results[best_i][3]
@save joinpath(out_dir, "best_direct.bson") best_θ best_direct t_direct_total TIED K RAT

CSV.write(
    joinpath(out_dir, "per_init_results.csv"),
    DataFrame(;
        init_id=[r[1] for r in direct_results],
        logL=[r[2] for r in direct_results],
        seconds=[r[4] for r in direct_results],
        iterations=[r[5] for r in direct_results],
    ),
)
