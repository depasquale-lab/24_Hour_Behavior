using Pkg
Pkg.activate("notebooks")

using Random
using Distributions
using DriftDiffusionModels
using HiddenMarkovModels
using Dates
using CSV
using DataFrames
using Statistics
using Printf
using BSON: @save

const Optim = DriftDiffusionModels.Optim
const StatsAPI = DriftDiffusionModels.StatsAPI

# Config

const K_STATES = 4
const N_FOLDS = 5
const FOLD_IDX = 1                  # which fold to hold out (1..N_FOLDS)
const N_INITS = 5                  # random initializations per fit
const MAX_ITER = 100
const TIED_CONFIGS = [Symbol[], [:v]]  # "full" and "tied-v"

Random.seed!(67)

# Data loading

const DATA_FILE = joinpath("data", "processed_rat_data.csv.gz")

rat_df = CSV.read(DATA_FILE, DataFrame)
rat_df = rat_df[rat_df.daily .== "24 hr", :]
replace!(rat_df[!, :choose_right], 0 => -1)
side_mapping = Dict("right" => 1, "left" => -1)
DataFrames.transform!(
    rat_df,
    :correct_side => ByRow(cs -> get(side_mapping, cs, missing)) => :correct_side_numeric,
)

"Return results-by-date (one `Vector{DDMResult}` per session)."
function sessions_for_rat(rat::AbstractString)
    sub = rat_df[rat_df.name .== rat, :]
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
    return sessions
end

"5-fold split by session. fold = 1 holds out the first 1/5 of sessions."
function train_test_split(
    sessions::Vector{Vector{DDMResult}}; n_folds::Int=N_FOLDS, fold::Int=FOLD_IDX
)
    n = length(sessions)
    fold_size = div(n, n_folds)
    test_start = (fold - 1) * fold_size + 1
    test_end = fold == n_folds ? n : fold * fold_size
    test_idx = test_start:test_end
    train_idx = setdiff(1:n, test_idx)

    train_sessions = sessions[train_idx]
    test_sessions = sessions[test_idx]

    train_data = reduce(vcat, train_sessions)
    test_data = reduce(vcat, test_sessions)
    train_seq_ends = cumsum([length(s) for s in train_sessions])
    test_seq_ends = cumsum([length(s) for s in test_sessions])

    return (
        train_data,
        train_seq_ends,
        test_data,
        test_seq_ends,
        length(train_idx),
        length(test_idx),
    )
end

# TiedPriorHMM, copied from FitConstrainedDDMHMMs.jl (full = empty tied set).

const DDM_PARAMS = (:B, :v, :a₀, :τ)
const DDM_BOUNDS = Dict(
    :B => (0.001, 50.0), :v => (0.0, 10.0), :a₀ => (0.0, 1.0), :τ => (1e-3, 5.0)
)

mutable struct TiedPriorHMM{T<:Real} <: HiddenMarkovModels.AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dists::Vector{DriftDiffusionModel}
    α_trans::Matrix{T}
    α_init::Vector{T}
    tied::Vector{Symbol}
end

function TiedPriorHMM(prior::PriorHMM, tied::AbstractVector{Symbol})
    @assert all(p in DDM_PARAMS for p in tied) "tied must be subset of $DDM_PARAMS"
    return TiedPriorHMM(
        copy(prior.init),
        copy(prior.trans),
        deepcopy(prior.dists),
        copy(prior.α_trans),
        copy(prior.α_init),
        collect(tied),
    )
end

Base.length(hmm::TiedPriorHMM) = length(hmm.init)
HiddenMarkovModels.initialization(hmm::TiedPriorHMM) = hmm.init
HiddenMarkovModels.transition_matrix(hmm::TiedPriorHMM) = hmm.trans
HiddenMarkovModels.obs_distributions(hmm::TiedPriorHMM) = hmm.dists

function StatsAPI.fit!(
    hmm::TiedPriorHMM,
    fb::HiddenMarkovModels.ForwardBackwardStorage,
    obs_seq::AbstractVector;
    seq_ends,
)
    K = length(hmm)

    init_counts = hmm.α_init .- 1
    trans_counts = hmm.α_trans .- 1
    for k in eachindex(seq_ends)
        t1, t2 = HiddenMarkovModels.seq_limits(seq_ends, k)
        init_counts .+= fb.γ[:, t1]
        trans_counts .+= sum(fb.ξ[t1:t2])
    end
    hmm.init .= init_counts ./ sum(init_counts)
    hmm.trans .= trans_counts ./ sum(trans_counts; dims=2)

    tied = hmm.tied
    free = Symbol[p for p in DDM_PARAMS if !(p in tied)]

    # Unconstrained y-space (log for B, v, τ; logit for a₀): plain LBFGS, ~5-10× faster than Fminbox.
    logit(p) = log(p / (1 - p))
    sigmoid(y) = 1 / (1 + exp(-y))
    to_y(p::Symbol, x) = p == :a₀ ? logit(clamp(x, 1e-6, 1 - 1e-6)) : log(max(x, 1e-9))

    y0 = Float64[]
    for k in 1:K, p in free
        push!(y0, to_y(p, getfield(hmm.dists[k], p)))
    end
    state_weights = [sum(@view fb.γ[k, :]) for k in 1:K]
    total_weight = sum(state_weights)
    for p in tied
        vals = [getfield(hmm.dists[k], p) for k in 1:K]
        push!(y0, to_y(p, sum(state_weights .* vals) / total_weight))
    end

    nfree = length(free)
    param_idx = Matrix{Int}(undef, K, length(DDM_PARAMS))
    for (pi, p) in enumerate(DDM_PARAMS)
        if p in tied
            pos = K * nfree + findfirst(==(p), tied)
            for k in 1:K
                ;
                param_idx[k, pi] = pos;
            end
        else
            pos = findfirst(==(p), free)
            for k in 1:K
                ;
                param_idx[k, pi] = (k - 1) * nfree + pos;
            end
        end
    end

    function neg_log_likelihood(y)
        total = zero(eltype(y))
        for k in 1:K
            B = exp(y[param_idx[k, 1]])
            v = exp(y[param_idx[k, 2]])
            a₀ = 1 / (1 + exp(-y[param_idx[k, 3]]))
            τ = exp(y[param_idx[k, 4]])
            γk = @view fb.γ[k, :]
            for i in eachindex(obs_seq)
                total +=
                    γk[i] * DriftDiffusionModels.logdensityof(
                        B, v, a₀, τ, obs_seq[i].rt, obs_seq[i].choice, obs_seq[i].s
                    )
            end
        end
        return -total
    end

    result = Optim.optimize(
        neg_log_likelihood,
        y0,
        Optim.LBFGS(; linesearch=Optim.LineSearches.BackTracking());
        autodiff=:forward,
    )
    y_opt = Optim.minimizer(result)

    for k in 1:K
        B = clamp(exp(y_opt[param_idx[k, 1]]), DDM_BOUNDS[:B][1], DDM_BOUNDS[:B][2])
        v = clamp(exp(y_opt[param_idx[k, 2]]), DDM_BOUNDS[:v][1], DDM_BOUNDS[:v][2])
        a₀ = clamp(sigmoid(y_opt[param_idx[k, 3]]), DDM_BOUNDS[:a₀][1], DDM_BOUNDS[:a₀][2])
        τ = clamp(exp(y_opt[param_idx[k, 4]]), DDM_BOUNDS[:τ][1], DDM_BOUNDS[:τ][2])
        setfield!(hmm.dists[k], :B, B)
        setfield!(hmm.dists[k], :v, v)
        setfield!(hmm.dists[k], :a₀, a₀)
        setfield!(hmm.dists[k], :τ, τ)
    end

    @assert HiddenMarkovModels.valid_hmm(hmm)
    return nothing
end

# Fitting / evaluation

function generate_init(n_states::Int)
    init_init = rand(Dirichlet(fill(1.0, n_states)))
    trans_init = zeros(Float64, n_states, n_states)
    for i in 1:n_states
        dv = ones(n_states);
        dv[i] = 10.0
        trans_init[i, :] .= rand(Dirichlet(dv))
    end
    log_drift = rand(Normal(0.0, 0.5), n_states)
    log_boundary = rand(Normal(log(2), 0.5), n_states)
    bias = rand(Beta(10, 10), n_states)
    emissions = [
        DriftDiffusionModel(;
            B=exp(log_boundary[i]), v=exp(log_drift[i]), a₀=bias[i], τ=0.1
        ) for i in 1:n_states
    ]
    return PriorHMM(init_init, trans_init, emissions, 1, 1)
end

function fit_one(
    tied::AbstractVector{Symbol},
    train_data,
    train_seq_ends;
    n_states::Int=K_STATES,
    n_inits::Int=N_INITS,
    max_iter::Int=MAX_ITER,
)
    # Generate priors up-front (avoid RNG contention inside threads).
    priors = [generate_init(n_states) for _ in 1:n_inits]

    # Each entry: (hmm, final_ll, logL_evolution) or nothing on failure.
    results = Vector{Any}(nothing, n_inits)

    Threads.@threads for init_id in 1:n_inits
        tied_prior = TiedPriorHMM(priors[init_id], tied)
        try
            hmm_est, logL_evolution = HiddenMarkovModels.baum_welch(
                tied_prior,
                train_data;
                seq_ends=train_seq_ends,
                atol=1e-3,
                max_iterations=max_iter,
                loglikelihood_increasing=false,
            )
            results[init_id] = (hmm_est, last(logL_evolution), logL_evolution)
        catch e
            @warn "Baum–Welch failed" tied init_id exception=(e, catch_backtrace())
        end
    end

    best_hmm = nothing
    best_ll = -Inf
    best_evolution = Float64[]
    for r in results
        r === nothing && continue
        hmm_est, final_ll, evol = r
        if final_ll > best_ll
            best_ll = final_ll
            best_hmm = hmm_est
            best_evolution = evol
        end
    end
    return best_hmm, best_ll, best_evolution
end

"Held-out log-likelihood under a trained HMM."
function test_loglike(hmm, test_data, test_seq_ends)
    return HiddenMarkovModels.logdensityof(hmm, test_data; seq_ends=test_seq_ends)
end

# Rat selection: $SGE_TASK_ID, then $RAT_IDX, then ARGS[1]; none → all rats.

const ALL_RATS = String.(unique(rat_df[!, "name"]))

function _resolve_rat_list()
    for k in ("SGE_TASK_ID", "RAT_IDX")
        v = get(ENV, k, "")
        (isempty(v) || v == "undefined") && continue
        i = parse(Int, v)
        return [ALL_RATS[i]]
    end
    if !isempty(ARGS)
        return [ALL_RATS[parse(Int, ARGS[1])]]
    end
    return ALL_RATS
end

const RAT_LIST = _resolve_rat_list()
@info "CV will process $(length(RAT_LIST)) rat(s): $(RAT_LIST)  (threads=$(Threads.nthreads()))"

# Main loop
out_dir = joinpath("results", "ddm_hmm_constrained", "cv_v_tied")
isdir(out_dir) || mkpath(out_dir)
per_task_dir = joinpath(out_dir, "per_task_summaries")
isdir(per_task_dir) || mkpath(per_task_dir)

summary = DataFrame(;
    rat=String[],
    config=String[],
    n_train_sessions=Int[],
    n_test_sessions=Int[],
    n_train_trials=Int[],
    n_test_trials=Int[],
    train_logL=Float64[],
    test_logL=Float64[],
    test_logL_per_trial=Float64[],
)

for rat in RAT_LIST
    @info "=== CV rat $rat (fold $FOLD_IDX / $N_FOLDS) ==="
    sessions = sessions_for_rat(rat)
    train_data, train_ends, test_data, test_ends, n_tr_sess, n_te_sess = train_test_split(
        sessions
    )

    @info "  $(length(train_data)) train trials ($n_tr_sess sessions) | $(length(test_data)) test trials ($n_te_sess sessions)"

    for tied in TIED_CONFIGS
        tag = isempty(tied) ? "full" : "tied-" * join(string.(tied), "_")
        @info "  fitting $tag..."
        hmm_trained, train_ll, evolution = fit_one(tied, train_data, train_ends)

        test_ll = test_loglike(hmm_trained, test_data, test_ends)
        per_trial = test_ll / length(test_data)

        @info "    train_LL=$(round(train_ll; digits=2))  test_LL=$(round(test_ll; digits=2))  per-trial=$(round(per_trial; digits=4))"

        outfile = joinpath(out_dir, "$(rat)_K$(K_STATES)_fold$(FOLD_IDX)_$(tag).bson")
        @save outfile hmm_trained tied train_ll test_ll evolution rat K_STATES FOLD_IDX

        push!(
            summary,
            (
                rat=rat,
                config=tag,
                n_train_sessions=n_tr_sess,
                n_test_sessions=n_te_sess,
                n_train_trials=length(train_data),
                n_test_trials=length(test_data),
                train_logL=train_ll,
                test_logL=test_ll,
                test_logL_per_trial=per_trial,
            ),
        )
    end
end

# Per-task summary (one file per rat, so array jobs don't clobber each other).
if length(RAT_LIST) == 1
    CSV.write(
        joinpath(per_task_dir, "$(RAT_LIST[1])_K$(K_STATES)_fold$(FOLD_IDX).csv"), summary
    )
    @info "Wrote per-task CV summary for $(RAT_LIST[1])"
else
    # Legacy serial path — write the merged file directly.
    summary_path = joinpath(out_dir, "cv_summary_fold$(FOLD_IDX).csv")
    CSV.write(summary_path, summary)
    @info "Wrote CV summary to $summary_path"

    wide = unstack(summary, :rat, :config, :test_logL_per_trial)
    rename!(wide, "tied-v" => :test_per_trial_tiedv, "full" => :test_per_trial_full)
    wide.Δ_test_per_trial = wide.test_per_trial_tiedv .- wide.test_per_trial_full
    wide.tiedv_wins_cv = wide.Δ_test_per_trial .> 0
    CSV.write(joinpath(out_dir, "cv_comparison_fold$(FOLD_IDX).csv"), wide)
    @info "CV comparison (per-trial held-out logL, tied-v − full; positive ⇒ tied-v generalizes better):"
    show(wide; allrows=true, allcols=true);
    println()
end
