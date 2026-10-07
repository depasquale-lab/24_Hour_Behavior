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
using BSON: @save

# Reuse DriftDiffusionModels' deps rather than adding them to the environment.
const Optim = DriftDiffusionModels.Optim
const StatsAPI = DriftDiffusionModels.StatsAPI

struct ConstrainedDDMHMMFit
    hmm::Any
    tied::Vector{Symbol}
    logL::Float64
    logL_evolution::Vector{Float64}
    n_trials::Int
    n_free_params::Int
    bic::Float64
end

const N_ITERS = 10
const K_STATES = 4

# Parameters tied (shared across states); the rest vary per state.
#   Symbol[]   full model
#   [:τ], [:a₀], [:τ, :a₀]   nuisance params animal-level
#   [:v], [:B]               does drift / caution need to vary?
const TIED_CONFIGS = [Symbol[], [:τ], [:a₀], [:τ, :a₀], [:v], [:B]]

Random.seed!(67)

data_file = joinpath("data", "processed_rat_data.csv.gz")
rat_df = CSV.read(data_file, DataFrame)
rat_df = rat_df[rat_df.daily .== "24 hr", :]

replace!(rat_df[!, :choose_right], 0 => -1)
side_mapping = Dict("right" => 1, "left" => -1)
DataFrames.transform!(
    rat_df,
    :correct_side => ByRow(cs -> get(side_mapping, cs, missing)) => :correct_side_numeric,
)

rat_names = String.(unique(rat_df[!, "name"]))

function data_for_ddmhmm(rat_idx::Int)
    rat = rat_names[rat_idx]
    rat_of_interest = rat_df[rat_df.name .== rat, :]
    dates = [Date(split(dt)[1]) for dt in rat_of_interest.trial_datetime]
    unique_dates = sort(unique(dates))

    results_by_date = Vector{Vector{DDMResult}}()
    for date in unique_dates
        day_indices = findall(dates .== date)
        isempty(day_indices) && continue
        day_rts = rat_of_interest.rt[day_indices]
        day_outcomes = rat_of_interest.choose_right[day_indices]
        day_stim_side = rat_of_interest.correct_side_numeric[day_indices]
        day_results = [
            DDMResult(rt, choice, stim) for
            (rt, choice, stim) in zip(day_rts, day_outcomes, day_stim_side)
        ]
        push!(results_by_date, day_results)
    end

    seq_ends = cumsum([length(seq) for seq in results_by_date])
    all_results = reduce(vcat, results_by_date)
    return all_results, seq_ends
end

function generate_ddmhmm_initialization(n_states::Int)
    init_init = rand(Dirichlet(fill(1.0, n_states)))

    trans_init = zeros(Float64, n_states, n_states)
    for i in 1:n_states
        dirichlet_vector = ones(n_states)
        dirichlet_vector[i] = 10.0
        trans_init[i, :] .= rand(Dirichlet(dirichlet_vector))
    end

    log_drift = rand(Normal(0.0, 0.5), n_states)
    log_boundary = rand(Normal(log(2), 0.5), n_states)
    bias = rand(Beta(10, 10), n_states)
    non_decision_time = 0.1

    emissions_init = [
        DriftDiffusionModel(;
            B=exp(log_boundary[i]), v=exp(log_drift[i]), a₀=bias[i], τ=non_decision_time
        ) for i in 1:n_states
    ]

    return PriorHMM(init_init, trans_init, emissions_init, 1, 1)
end

# TiedPriorHMM: PriorHMM whose emission M-step also fits the tied parameters.
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

    #Init & transition M-step (identical to PriorHMM)
    init_counts = hmm.α_init .- 1
    trans_counts = hmm.α_trans .- 1
    for k in eachindex(seq_ends)
        t1, t2 = HiddenMarkovModels.seq_limits(seq_ends, k)
        init_counts .+= fb.γ[:, t1]
        trans_counts .+= sum(fb.ξ[t1:t2])
    end
    hmm.init .= init_counts ./ sum(init_counts)
    hmm.trans .= trans_counts ./ sum(trans_counts; dims=2)

    # Emission M-step with tying
    tied = hmm.tied
    free = Symbol[p for p in DDM_PARAMS if !(p in tied)]

    # Unconstrained y-space (log for B, v, τ; logit for a₀): plain LBFGS, ~5-10× faster than Fminbox.
    logit(p) = log(p / (1 - p))
    sigmoid(y) = 1 / (1 + exp(-y))
    to_y(p::Symbol, x) = p == :a₀ ? logit(clamp(x, 1e-6, 1 - 1e-6)) : log(max(x, 1e-9))

    # Parameter layout (flat vector in y-space):
    #   per-state free params for state 1, then state 2, ..., then tied params.
    y0 = Float64[]
    for k in 1:K
        for p in free
            push!(y0, to_y(p, getfield(hmm.dists[k], p)))
        end
    end

    # Initialize tied params at γ-weighted average of current per-state values.
    state_weights = [sum(@view fb.γ[k, :]) for k in 1:K]
    total_weight = sum(state_weights)
    for p in tied
        vals = [getfield(hmm.dists[k], p) for k in 1:K]
        push!(y0, to_y(p, sum(state_weights .* vals) / total_weight))
    end

    nfree = length(free)

    # Flat-vector index per (state, parameter), precomputed.
    param_idx = Matrix{Int}(undef, K, length(DDM_PARAMS))
    for (pi, p) in enumerate(DDM_PARAMS)
        if p in tied
            pos = K * nfree + findfirst(==(p), tied)
            for k in 1:K
                param_idx[k, pi] = pos
            end
        else
            pos = findfirst(==(p), free)
            for k in 1:K
                param_idx[k, pi] = (k - 1) * nfree + pos
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

# BIC & fitting driver
"""
    count_free_params(K, tied)

Number of free parameters in the HMM-DDM with `K` states and the given `tied`
set of DDM parameters shared across states.

- initial distribution:        K - 1
- transition matrix:            K * (K - 1)
- emissions:                    4K - |tied| * (K - 1)
"""
function count_free_params(K::Int, tied::AbstractVector{Symbol})
    return (K - 1) + K * (K - 1) + 4K - length(tied) * (K - 1)
end

bic(logL::Real, k::Int, N::Int) = k * log(N) - 2 * logL

function fit_constrained_ddmhmm_for_rat(
    rat_idx::Int,
    tied::AbstractVector{Symbol};
    n_states::Int=K_STATES,
    n_inits::Int=N_ITERS,
    atol::Float64=1e-3,
    max_iter::Int=100,
    loglikelihood_increasing::Bool=false,
)
    obs_seq, seq_ends = data_for_ddmhmm(rat_idx)
    N = length(obs_seq)

    # Generate all priors up-front so RNG order is independent of thread scheduling.
    priors = [generate_ddmhmm_initialization(n_states) for _ in 1:n_inits]
    results = Vector{Any}(nothing, n_inits)

    Threads.@threads for init_id in 1:n_inits
        tied_prior = TiedPriorHMM(priors[init_id], tied)
        try
            hmm_est, logL_evolution = HiddenMarkovModels.baum_welch(
                tied_prior,
                obs_seq;
                seq_ends=seq_ends,
                atol=atol,
                max_iterations=max_iter,
                loglikelihood_increasing=loglikelihood_increasing,
            )
            results[init_id] = (hmm_est, last(logL_evolution), logL_evolution)
        catch e
            @warn "Baum–Welch failed" rat_idx tied init_id exception = (
                e, catch_backtrace()
            )
        end
    end

    best_hmm = nothing
    best_logL = -Inf
    best_logL_evolution = Float64[]
    for r in results
        r === nothing && continue
        hmm_est, final_ll, evol = r
        if final_ll > best_logL
            best_logL = final_ll
            best_hmm = hmm_est
            best_logL_evolution = evol
        end
    end

    k = count_free_params(n_states, tied)
    return ConstrainedDDMHMMFit(
        best_hmm, collect(tied), best_logL, best_logL_evolution, N, k, bic(best_logL, k, N)
    )
end

# Run fits across rats and tied configurations
if abspath(PROGRAM_FILE) == @__FILE__
    results_dir = joinpath("results", "ddm_hmm_constrained")
    isdir(results_dir) || mkpath(results_dir)

    summary_rows = DataFrame(;
        rat=String[],
        K=Int[],
        tied=String[],
        n_trials=Int[],
        n_params=Int[],
        logL=Float64[],
        bic=Float64[],
    )

    for (rat_idx, rat) in enumerate(rat_names)
        for tied in TIED_CONFIGS
            tied_tag = isempty(tied) ? "full" : join(string.(tied), "_")
            @info "Fitting rat=$rat K=$K_STATES tied=$tied_tag with $N_ITERS inits..."

            fit = fit_constrained_ddmhmm_for_rat(rat_idx, tied)

            filename = joinpath(results_dir, "$(rat)_K$(K_STATES)_tied-$(tied_tag).bson")
            @save filename fit rat K_STATES tied

            push!(
                summary_rows,
                (
                    rat=rat,
                    K=K_STATES,
                    tied=tied_tag,
                    n_trials=fit.n_trials,
                    n_params=fit.n_free_params,
                    logL=fit.logL,
                    bic=fit.bic,
                ),
            )

            @info "  logL=$(round(fit.logL; digits=2))  k=$(fit.n_free_params)  BIC=$(round(fit.bic; digits=2))  →  $filename"
        end
    end

    summary_path = joinpath(results_dir, "bic_summary.csv")
    CSV.write(summary_path, summary_rows)
    @info "Wrote BIC summary to $summary_path"
end
