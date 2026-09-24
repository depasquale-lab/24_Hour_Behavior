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

# Mirrors FitConstrainedDDMHMMs.jl so BIC numbers are comparable across approaches.
const K_STATES = 4
const N_INITS = 30
const MAX_ITER = 3000
const TIED_CONFIGS = [Symbol[], [:τ], [:a₀], [:τ, :a₀], [:v], [:B]]

Random.seed!(69)

# Source of rat_idx (in priority order):
#   1. $RAT_LIST + $SGE_TASK_ID  (array job over a non-contiguous subset:
#                                 RAT_LIST=5:12:14:15:18 with -t 1-5 picks
#                                 rats 5, 12, 14, 15, 18 in turn).
#                                 Use ':' as the separator — qsub -v parses
#                                 ',' as a delimiter between vars.
#   2. $SGE_TASK_ID  (UGE/SGE array job, task id is the rat idx directly)
#   3. $RAT_IDX      (manual override, 1-based)
#   4. first CLI arg (e.g. `julia FitOneRatGradient.jl 3`)
function _resolve_rat_idx()
    sge_id = get(ENV, "SGE_TASK_ID", "")
    sge_set = !(isempty(sge_id) || sge_id == "undefined")

    list_env = get(ENV, "RAT_LIST", "")
    if !isempty(list_env) && sge_set
        # Accept ':' or ',' so the var works whether passed via -v (needs ':')
        # or pre-exported with qsub -V (either works).
        indices = parse.(Int, split(list_env, r"[:,]"))
        slot = parse(Int, sge_id)
        @assert 1 <= slot <= length(indices) "SGE_TASK_ID=$slot out of range 1:$(length(indices)) for RAT_LIST=$list_env"
        return indices[slot]
    end

    if sge_set
        return parse(Int, sge_id)
    end

    rat_idx_env = get(ENV, "RAT_IDX", "")
    if !(isempty(rat_idx_env) || rat_idx_env == "undefined")
        return parse(Int, rat_idx_env)
    end

    if !isempty(ARGS)
        return parse(Int, ARGS[1])
    end
    error(
        "FitOneRatGradient.jl: no rat index provided. Set SGE_TASK_ID (optionally with RAT_LIST) or RAT_IDX, or pass an integer CLI arg.",
    )
end

# Load + preprocess data
data_file = joinpath("data", "processed_rat_data.csv.gz")
rat_df = CSV.read(data_file, DataFrame)
rat_df = rat_df[rat_df.daily .== "24 hr", :]
replace!(rat_df[!, :choose_right], 0 => -1)
side_mapping = Dict("right" => 1, "left" => -1)
DataFrames.transform!(
    rat_df,
    :correct_side => ByRow(cs -> get(side_mapping, cs, missing)) => :correct_side_numeric,
)

const rat_names = String.(unique(rat_df[!, "name"]))

const RAT_IDX = _resolve_rat_idx()
@assert 1 <= RAT_IDX <= length(rat_names) "RAT_IDX=$RAT_IDX is out of range 1:$(length(rat_names))"
const THIS_RAT = rat_names[RAT_IDX]

function load_obs_for_rat(rat::String)
    sub = rat_df[rat_df.name .== rat, :]
    @assert !isempty(sub) "rat $rat not in data"
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
    obs_seq = reduce(vcat, sessions)
    seq_ends = cumsum([length(s) for s in sessions])
    return obs_seq, seq_ends
end

const OBS_SEQ, SEQ_ENDS = load_obs_for_rat(THIS_RAT)

@info "== Task $(RAT_IDX): rat=$THIS_RAT  ($(length(OBS_SEQ)) trials, $(length(SEQ_ENDS)) sessions, threads=$(Threads.nthreads())) =="

# Reparameterization helpers (lifted from DirectGradientDDMHMM.jl)
softmax1(x) = (m = maximum(x); e = exp.(x .- m); e ./ sum(e))
sigmoid(y) = 1 / (1 + exp(-y))

const DDM_PARAMS = (:B, :v, :a₀, :τ)

function build_tied_idx_map(tied::AbstractVector{Symbol}, K::Int)
    free = Symbol[p for p in DDM_PARAMS if !(p in tied)]
    nfree = length(free)
    idx = Matrix{Int}(undef, K, length(DDM_PARAMS))
    for (pi, p) in enumerate(DDM_PARAMS)
        if p in tied
            pos = K * nfree + findfirst(==(p), tied)
            for k in 1:K
                idx[k, pi] = pos
            end
        else
            pos = findfirst(==(p), free)
            for k in 1:K
                idx[k, pi] = (k - 1) * nfree + pos
            end
        end
    end
    return idx, K * nfree + length(tied)
end

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

struct DDMHMM{T<:Real} <: HiddenMarkovModels.AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dists::Vector{DDMEmission{T}}
end
Base.length(h::DDMHMM) = length(h.init)
HiddenMarkovModels.initialization(h::DDMHMM) = h.init
HiddenMarkovModels.transition_matrix(h::DDMHMM) = h.trans
HiddenMarkovModels.obs_distributions(h::DDMHMM) = h.dists

function fit_gradient(
    obs_seq,
    seq_ends,
    tied::AbstractVector{Symbol};
    K::Int=K_STATES,
    n_inits::Int=N_INITS,
    max_iter::Int=MAX_ITER,
)
    n_init_logits = K
    n_trans_logits = K * K
    tied_idx_map, n_ddm_tied = build_tied_idx_map(tied, K)
    n_params_eff = n_init_logits + n_trans_logits + n_ddm_tied

    function unpack(θ)
        init_logits = @view θ[1:n_init_logits]
        trans_logits_flat = @view θ[(n_init_logits + 1):(n_init_logits + n_trans_logits)]
        ddm_y = @view θ[(n_init_logits + n_trans_logits + 1):end]
        init = softmax1(init_logits)
        trans = Matrix{eltype(θ)}(undef, K, K)
        for k in 1:K
            trans[k, :] .= softmax1(@view trans_logits_flat[((k - 1) * K + 1):(k * K)])
        end
        return init, trans, ddm_y
    end

    function ddm_params_for_state(k, ddm_y)
        B_y = ddm_y[tied_idx_map[k, 1]]
        v_y = ddm_y[tied_idx_map[k, 2]]
        a_y = ddm_y[tied_idx_map[k, 3]]
        τ_y = ddm_y[tied_idx_map[k, 4]]
        return exp(B_y), exp(v_y), sigmoid(a_y), exp(τ_y)
    end

    function neg_logL(θ)
        init, trans, ddm_y = unpack(θ)
        dists = [DDMEmission(ddm_params_for_state(k, ddm_y)...) for k in 1:K]
        hmm = DDMHMM(init, trans, dists)
        return -HiddenMarkovModels.logdensityof(hmm, obs_seq; seq_ends=seq_ends)
    end

    function random_init_θ()
        θ = zeros(n_params_eff)
        θ[1:n_init_logits] .= randn(n_init_logits) .* 1.0
        diag_bias = 0.5 + 2.5 * rand()
        for k in 1:K
            base = randn(K) .* 0.7
            base[k] += diag_bias
            θ[(n_init_logits + (k - 1) * K + 1):(n_init_logits + k * K)] .= base
        end
        off = n_init_logits + n_trans_logits
        for k in 1:K, (pi, p) in enumerate(DDM_PARAMS)
            pos = off + tied_idx_map[k, pi]
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

    θ0s = [random_init_θ() for _ in 1:n_inits]
    results = Vector{Any}(nothing, n_inits)
    t_wall_start = time()

    # Inits are run sequentially. HMMs.jl parallelizes the forward-backward
    # across sequences (one thread per session), so giving each L-BFGS run
    # the full thread pool is faster than nesting Threads.@threads here —
    # the outer @threads would consume the pool and starve the inner one.
    for init_id in 1:n_inits
        θ0 = θ0s[init_id]
        t0 = time()
        try
            result = Optim.optimize(
                neg_logL,
                θ0,
                Optim.LBFGS(; linesearch=Optim.LineSearches.BackTracking()),
                Optim.Options(; iterations=max_iter, show_trace=false);
                autodiff=:forward,
            )
            dt = time() - t0
            logL = -Optim.minimum(result)
            θ_opt = Optim.minimizer(result)
            results[init_id] = (init_id, logL, θ_opt, dt, result.iterations)
            @info @sprintf(
                "  init %d: logL = %.2f  (%.1fs, %d iters)",
                init_id,
                logL,
                dt,
                result.iterations
            )
        catch e
            @warn "L-BFGS failed" init_id exception = (e, catch_backtrace())
        end
    end

    t_wall = time() - t_wall_start

    valid = [r for r in results if r !== nothing]
    @assert !isempty(valid) "all $(n_inits) inits failed"
    best_idx = argmax([r[2] for r in valid])
    best = valid[best_idx]
    best_θ, best_logL = best[3], best[2]

    init_best, trans_best, ddm_y_best = unpack(best_θ)
    dists_best = [DDMEmission(ddm_params_for_state(k, ddm_y_best)...) for k in 1:K]

    return (
        best_θ=best_θ,
        best_logL=best_logL,
        init=init_best,
        trans=trans_best,
        dists=dists_best,
        per_init=results,
        t_wall=t_wall,
        n_params_eff=n_params_eff,
    )
end

# Effective free param count (ignores softmax gauge dimensions). Matches FitConstrainedDDMHMMs.jl.
function count_free_params(K::Int, tied::AbstractVector{Symbol})
    return (K - 1) + K * (K - 1) + 4K - length(tied) * (K - 1)
end
bic(logL::Real, k::Int, N::Int) = k * log(N) - 2 * logL

# Drive the fits across tied configs for this rat
results_dir = joinpath("results", "ddm_hmm_gradient")
isdir(results_dir) || mkpath(results_dir)
per_task_dir = joinpath(results_dir, "per_task_summaries")
isdir(per_task_dir) || mkpath(per_task_dir)
per_init_dir = joinpath(results_dir, "per_init_summaries")
isdir(per_init_dir) || mkpath(per_init_dir)

summary_rows = DataFrame(;
    rat=String[],
    K=Int[],
    tied=String[],
    n_trials=Int[],
    n_params=Int[],
    logL=Float64[],
    bic=Float64[],
)

const N_TRIALS = length(OBS_SEQ)

for tied in TIED_CONFIGS
    tied_tag = isempty(tied) ? "full" : join(string.(tied), "_")
    @info "Gradient-fit rat=$THIS_RAT K=$K_STATES tied=$tied_tag with $N_INITS inits..."

    fit = fit_gradient(OBS_SEQ, SEQ_ENDS, tied)

    k_eff = count_free_params(K_STATES, tied)
    bic_val = bic(fit.best_logL, k_eff, N_TRIALS)

    bson_path = joinpath(results_dir, "$(THIS_RAT)_K$(K_STATES)_tied-$(tied_tag).bson")
    best_θ = fit.best_θ
    best_logL = fit.best_logL
    hmm_init = fit.init
    hmm_trans = fit.trans
    hmm_dists = fit.dists
    @save bson_path best_θ best_logL hmm_init hmm_trans hmm_dists tied K_STATES rat =
        THIS_RAT n_trials = N_TRIALS n_params = k_eff bic = bic_val

    CSV.write(
        joinpath(per_init_dir, "$(THIS_RAT)_K$(K_STATES)_tied-$(tied_tag).csv"),
        DataFrame(;
            init_id=[r === nothing ? missing : r[1] for r in fit.per_init],
            logL=[r === nothing ? missing : r[2] for r in fit.per_init],
            seconds=[r === nothing ? missing : r[4] for r in fit.per_init],
            iterations=[r === nothing ? missing : r[5] for r in fit.per_init],
        ),
    )

    push!(
        summary_rows,
        (
            rat=THIS_RAT,
            K=K_STATES,
            tied=tied_tag,
            n_trials=N_TRIALS,
            n_params=k_eff,
            logL=fit.best_logL,
            bic=bic_val,
        ),
    )

    @info "  logL=$(round(fit.best_logL; digits=2))  k=$k_eff  BIC=$(round(bic_val; digits=2))  →  $bson_path"
end

CSV.write(joinpath(per_task_dir, "$(THIS_RAT)_K$(K_STATES).csv"), summary_rows)
@info "Wrote per-task summary for $THIS_RAT"
