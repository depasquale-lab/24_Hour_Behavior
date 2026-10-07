#=
State-number sweep for the session-based ("daily") animals: K = 1, 2, ... fits
with full-data logL, AIC and BIC. Fitting setup matches FitDDMHMMs.jl /
FitConstrainedDDMHMMs.jl (Baum-Welch, N_INITS restarts, best wins).

Rat selection: $SGE_TASK_ID, then $RAT_IDX, then ARGS[1]; none -> every daily rat.
=#

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

# Config

const K_LIST = let v = get(ENV, "K_LIST", "")
    isempty(v) ? collect(1:6) : [parse(Int, s) for s in split(v, ',')]
end
const N_INITS = parse(Int, get(ENV, "N_INITS", "10"))
const MAX_ITER = parse(Int, get(ENV, "MAX_ITER", "100"))

Random.seed!(67)  # this seed is bussin fr fr on god

# Data loading (session-based / "daily" animals only)

const DATA_FILE = joinpath("data", "processed_rat_data.csv.gz")

rat_df = CSV.read(DATA_FILE, DataFrame)
rat_df = rat_df[rat_df.daily .== "daily", :]   # session-based training group
replace!(rat_df[!, :choose_right], 0 => -1)
side_mapping = Dict("right" => 1, "left" => -1)
DataFrames.transform!(
    rat_df,
    :correct_side => ByRow(cs -> get(side_mapping, cs, missing)) => :correct_side_numeric,
)

const DAILY_RATS = String.(unique(rat_df[!, "name"]))

"Return (obs_seq, seq_ends) for one rat, with one sequence per session/day."
function data_for_ddmhmm(rat::AbstractString)
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

    seq_ends = cumsum([length(s) for s in sessions])
    return reduce(vcat, sessions), seq_ends, length(sessions)
end

# Model setup

"Random DDM-HMM initialization with `n_states` states (same scheme as FitDDMHMMs.jl)."
function generate_ddmhmm_initialization(n_states::Int)
    init_init = rand(Dirichlet(fill(1.0, n_states)))

    trans_init = zeros(Float64, n_states, n_states)
    for i in 1:n_states
        dirichlet_vector = ones(n_states)
        dirichlet_vector[i] = 10.0    # bias towards self-transition
        trans_init[i, :] .= rand(Dirichlet(dirichlet_vector))
    end

    log_drift = rand(Normal(0.0, 0.5), n_states)
    log_boundary = rand(Normal(log(2), 0.5), n_states)
    bias = rand(Beta(10, 10), n_states)
    non_decision_time = 0.1

    emissions_init = [
        DriftDiffusionModel(
            exp(log_drift[i]), exp(log_boundary[i]), bias[i], non_decision_time
        ) for i in 1:n_states
    ]

    return PriorHMM(init_init, trans_init, emissions_init, 1, 1)
end

"""
    count_free_params(K)

Free parameters in the unconstrained (no tying) DDM-HMM:
initial distribution K - 1, transitions K * (K - 1), emissions 4K.
Matches `count_free_params(K, Symbol[])` in FitConstrainedDDMHMMs.jl.
"""
count_free_params(K::Int) = (K - 1) + K * (K - 1) + 4K

aic(logL::Real, k::Int) = 2 * (k - logL)
bic(logL::Real, k::Int, N::Int) = k * log(N) - 2 * logL

"Baum-Welch from `n_inits` random restarts; returns the best restart."
function fit_best(obs_seq, seq_ends, n_states::Int; n_inits::Int=N_INITS, max_iter::Int=MAX_ITER)
    # Generate priors up-front to avoid RNG contention inside threads.
    priors = [generate_ddmhmm_initialization(n_states) for _ in 1:n_inits]
    results = Vector{Any}(nothing, n_inits)

    Threads.@threads for init_id in 1:n_inits
        try
            hmm_est, logL_evolution = HiddenMarkovModels.baum_welch(
                priors[init_id],
                obs_seq;
                seq_ends=seq_ends,
                atol=1e-3,
                max_iterations=max_iter,
                loglikelihood_increasing=false,
            )
            results[init_id] = (hmm_est, last(logL_evolution), logL_evolution)
        catch e
            @warn "Baum-Welch failed" n_states init_id exception = (e, catch_backtrace())
        end
    end

    best_hmm, best_ll, best_evolution, n_ok = nothing, -Inf, Float64[], 0
    for r in results
        r === nothing && continue
        n_ok += 1
        hmm_est, final_ll, evol = r
        if final_ll > best_ll
            best_hmm, best_ll, best_evolution = hmm_est, final_ll, evol
        end
    end
    return best_hmm, best_ll, best_evolution, n_ok
end

# Rat selection

function _resolve_rat_list()
    for k in ("SGE_TASK_ID", "RAT_IDX")
        v = get(ENV, k, "")
        (isempty(v) || v == "undefined") && continue
        i = parse(Int, v)
        @assert 1 <= i <= length(DAILY_RATS) "index $i out of range 1:$(length(DAILY_RATS))"
        return [DAILY_RATS[i]]
    end
    if !isempty(ARGS)
        return [DAILY_RATS[parse(Int, ARGS[1])]]
    end
    return DAILY_RATS
end

const RAT_LIST = _resolve_rat_list()
@info "State sweep over K = $K_LIST for $(length(RAT_LIST)) rat(s): $(RAT_LIST)  (threads=$(Threads.nthreads()))"

# Main loop

out_dir = joinpath("results", "ddm_hmm_state_sweep")
per_task_dir = joinpath(out_dir, "per_task_summaries")
mkpath(per_task_dir)

summary = DataFrame(;
    rat=String[],
    K=Int[],
    n_trials=Int[],
    n_sessions=Int[],
    n_params=Int[],
    n_successful_inits=Int[],
    logL=Float64[],
    logL_per_trial=Float64[],
    aic=Float64[],
    bic=Float64[],
)

for rat in RAT_LIST
    obs_seq, seq_ends, n_sessions = data_for_ddmhmm(rat)
    N = length(obs_seq)
    @info "=== rat $rat: $N trials, $n_sessions sessions ==="

    for K in K_LIST
        @info "  fitting K = $K ($N_INITS inits)..."
        hmm, logL, evolution, n_ok = fit_best(obs_seq, seq_ends, K)

        if hmm === nothing
            @warn "  all inits failed for rat $rat, K = $K -- skipping"
            continue
        end

        k = count_free_params(K)
        @save joinpath(out_dir, "$(rat)_K$(K)_daily_sweep.bson") hmm logL evolution rat K N

        push!(
            summary,
            (
                rat=rat,
                K=K,
                n_trials=N,
                n_sessions=n_sessions,
                n_params=k,
                n_successful_inits=n_ok,
                logL=logL,
                logL_per_trial=logL / N,
                aic=aic(logL, k),
                bic=bic(logL, k, N),
            ),
        )

        @info "    logL=$(round(logL; digits=2))  per-trial=$(round(logL / N; digits=4))  k=$k  BIC=$(round(bic(logL, k, N); digits=2))"
    end
end

# Per-task summary (one file per rat, so array tasks don't clobber each other).
if length(RAT_LIST) == 1
    CSV.write(joinpath(per_task_dir, "$(RAT_LIST[1])_state_sweep.csv"), summary)
    @info "Wrote per-task sweep summary for $(RAT_LIST[1])"
else
    CSV.write(joinpath(out_dir, "state_sweep_summary.csv"), summary)
    @info "Wrote sweep summary to $(joinpath(out_dir, "state_sweep_summary.csv"))"
end

show(summary; allrows=true, allcols=true)
println()
