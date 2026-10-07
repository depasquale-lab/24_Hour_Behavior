#=
Session-blocked 5-fold CV over K = 1..5 for the session-based ("daily") animals;
N_INITS restarts per fit, best by training logL, scored on held-out sessions.
Folds, restarts and Baum-Welch settings match CrossValidateVTied.jl / FitDDMHMMs.jl.
Unlike the raw-logL sweep (StateSweepDaily.jl), held-out logL can select a K.

One task = one (rat, fold): $SGE_TASK_ID (rat-major index), or $RAT_IDX/$FOLD_IDX,
or ARGS; none -> every rat and fold, serially.

Env knobs: GROUP=daily|24hr, SHUFFLE=0|1, RAT_SET=fitted|all, SKIP_DONE=1|0,
K_LIST=1,2,3,4,5, N_INITS, MAX_ITER, N_FOLDS
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
    isempty(v) ? collect(1:5) : [parse(Int, s) for s in split(v, ',')]
end
const N_FOLDS = parse(Int, get(ENV, "N_FOLDS", "5"))
const N_INITS = parse(Int, get(ENV, "N_INITS", "10"))
const MAX_ITER = parse(Int, get(ENV, "MAX_ITER", "100"))

# GROUP: "daily" = session-based, "24hr" = 24 hr animals (sessions = calendar days).
const GROUP = get(ENV, "GROUP", "daily")
GROUP in ("daily", "24hr") || error("GROUP must be \"daily\" or \"24hr\", got \"$GROUP\"")

# RAT_SET: "fitted" = daily rats with K4 fits in results/ddm_hmm; "all" = every
# animal (24hr always). SHUFFLE=1 permutes trial order within training sessions
# (a mixture of DDMs); held-out sessions keep their real order.
const SHUFFLE = get(ENV, "SHUFFLE", "0") == "1"

const RAT_SET = GROUP == "24hr" ? "all" : get(ENV, "RAT_SET", "fitted")
const ALREADY_FIT = ["Daenerys", "Dobby", "Dory", "Regina", "Rhubarb"]

Random.seed!(67)  # this seed is bussin fr fr on god

# Data loading

const DATA_FILE = joinpath("data", "processed_rat_data.csv.gz")

rat_df = CSV.read(DATA_FILE, DataFrame)
rat_df = rat_df[rat_df.daily .== (GROUP == "24hr" ? "24 hr" : "daily"), :]
replace!(rat_df[!, :choose_right], 0 => -1)
side_mapping = Dict("right" => 1, "left" => -1)
DataFrames.transform!(
    rat_df,
    :correct_side => ByRow(cs -> get(side_mapping, cs, missing)) => :correct_side_numeric,
)

n_sessions(rat) = length(unique(split(dt)[1] for dt in rat_df.trial_datetime[rat_df.name .== rat]))

# Rats with fewer sessions than folds (608840: 2 sessions) can't be split.
const DAILY_RATS = filter(r -> n_sessions(r) >= N_FOLDS, String.(unique(rat_df[!, "name"])))
const RATS = if RAT_SET == "all"
    DAILY_RATS
elseif RAT_SET == "fitted"
    filter(in(Set(ALREADY_FIT)), DAILY_RATS)
else
    error("RAT_SET must be \"fitted\" or \"all\", got \"$RAT_SET\"")
end
@assert !isempty(RATS) "no rats selected (RAT_SET=$RAT_SET)"

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

"""
    train_test_split(sessions; n_folds, fold)

Session-blocked k-fold split: fold `f` holds out the f-th contiguous block of
sessions (chronological). Identical to the split in CrossValidateVTied.jl.
"""
function train_test_split(
    sessions::Vector{Vector{DDMResult}}; n_folds::Int=N_FOLDS, fold::Int=1
)
    n = length(sessions)
    fold_size = div(n, n_folds)
    fold_size < 1 && error("rat has only $n sessions; cannot run $n_folds-fold CV")

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

Free parameters in the unconstrained DDM-HMM: initial distribution K - 1,
transitions K * (K - 1), emissions 4K. Matches FitConstrainedDDMHMMs.jl.
"""
count_free_params(K::Int) = (K - 1) + K * (K - 1) + 4K

"""
    fit_best(train_data, train_seq_ends, n_states)

Baum-Welch from `n_inits` random restarts on the TRAINING sessions only.
The winning restart is picked by training logL — the held-out set is never
touched during fitting.
"""
function fit_best(
    train_data, train_seq_ends, n_states::Int; n_inits::Int=N_INITS, max_iter::Int=MAX_ITER
)
    # Generate priors up-front to avoid RNG contention inside threads.
    priors = [generate_ddmhmm_initialization(n_states) for _ in 1:n_inits]
    results = Vector{Any}(nothing, n_inits)

    Threads.@threads for init_id in 1:n_inits
        try
            hmm_est, logL_evolution = HiddenMarkovModels.baum_welch(
                priors[init_id],
                train_data;
                seq_ends=train_seq_ends,
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

"Held-out log-likelihood under a trained HMM."
function test_loglike(hmm, test_data, test_seq_ends)
    return HiddenMarkovModels.logdensityof(hmm, test_data; seq_ends=test_seq_ends)
end

# Task selection: one task = one (rat, fold) pair, rat-major.

const JOB_GRID = [(r, f) for r in RATS for f in 1:N_FOLDS]

function _resolve_jobs()
    rat_env = get(ENV, "RAT_IDX", "")
    fold_env = get(ENV, "FOLD_IDX", "")
    if !isempty(rat_env) && rat_env != "undefined"
        r = parse(Int, rat_env)
        @assert 1 <= r <= length(RATS) "RAT_IDX=$r out of range 1:$(length(RATS))"
        if !isempty(fold_env) && fold_env != "undefined"
            f = parse(Int, fold_env)
            @assert 1 <= f <= N_FOLDS "FOLD_IDX=$f out of range 1:$N_FOLDS"
            return [(RATS[r], f)]
        end
        return [(RATS[r], f) for f in 1:N_FOLDS]
    end

    task_env = get(ENV, "SGE_TASK_ID", "")
    if !isempty(task_env) && task_env != "undefined"
        t = parse(Int, task_env)
        @assert 1 <= t <= length(JOB_GRID) "SGE_TASK_ID=$t out of range 1:$(length(JOB_GRID))"
        return [JOB_GRID[t]]
    end

    if length(ARGS) >= 2
        return [(RATS[parse(Int, ARGS[1])], parse(Int, ARGS[2]))]
    elseif length(ARGS) == 1
        return [JOB_GRID[parse(Int, ARGS[1])]]
    end
    return JOB_GRID
end

const JOBS = _resolve_jobs()
@info "CV state sweep ($GROUP): K = $K_LIST, $N_FOLDS folds, $N_INITS inits | rats=$RATS" n_jobs = length(
    JOBS
) threads = Threads.nthreads()

# Main loop

out_dir = joinpath(
    "results", "ddm_hmm_state_sweep", (GROUP == "24hr" ? "cv_24hr" : "cv") * (SHUFFLE ? "_shuffled" : "")
)
per_task_dir = joinpath(out_dir, "per_task_summaries")
mkpath(per_task_dir)

summary = DataFrame(;
    rat=String[],
    K=Int[],
    fold=Int[],
    n_params=Int[],
    n_train_sessions=Int[],
    n_test_sessions=Int[],
    n_train_trials=Int[],
    n_test_trials=Int[],
    n_successful_inits=Int[],
    train_logL=Float64[],
    train_logL_per_trial=Float64[],
    test_logL=Float64[],
    test_logL_per_trial=Float64[],
)

const SKIP_DONE = get(ENV, "SKIP_DONE", "1") == "1"

for (rat, fold) in JOBS
    if SKIP_DONE && isfile(joinpath(per_task_dir, "$(rat)_fold$(fold)_cv.csv"))
        @info "rat $rat fold $fold already done -- skipping (SKIP_DONE=0 to refit)"
        continue
    end
    sessions = sessions_for_rat(rat)
    train_data, train_ends, test_data, test_ends, n_tr_sess, n_te_sess = train_test_split(
        sessions; fold=fold
    )
    if SHUFFLE
        rng = MersenneTwister(hash((rat, fold)))
        starts = [1; train_ends[1:(end - 1)] .+ 1]
        for (a, b) in zip(starts, train_ends)
            shuffle!(rng, view(train_data, a:b))
        end
    end
    @info "=== rat $rat, fold $fold/$N_FOLDS: $(length(train_data)) train trials ($n_tr_sess sessions) | $(length(test_data)) test trials ($n_te_sess sessions) ==="

    for K in K_LIST
        @info "  fitting K = $K ($N_INITS inits)..."
        hmm, train_ll, evolution, n_ok = fit_best(train_data, train_ends, K)

        if hmm === nothing
            @warn "  all inits failed for rat $rat, K = $K, fold $fold -- skipping"
            continue
        end

        test_ll = test_loglike(hmm, test_data, test_ends)
        per_trial = test_ll / length(test_data)

        @save joinpath(out_dir, "$(rat)_K$(K)_fold$(fold)_$(GROUP)_cv.bson") hmm train_ll test_ll evolution rat K fold

        push!(
            summary,
            (
                rat=rat,
                K=K,
                fold=fold,
                n_params=count_free_params(K),
                n_train_sessions=n_tr_sess,
                n_test_sessions=n_te_sess,
                n_train_trials=length(train_data),
                n_test_trials=length(test_data),
                n_successful_inits=n_ok,
                train_logL=train_ll,
                train_logL_per_trial=train_ll / length(train_data),
                test_logL=test_ll,
                test_logL_per_trial=per_trial,
            ),
        )

        @info "    train_LL=$(round(train_ll; digits=2))  test_LL=$(round(test_ll; digits=2))  test per-trial=$(round(per_trial; digits=4))"
    end
end

# Per-task summary (one file per (rat, fold), so array tasks don't clobber each other).
if isempty(summary)
    @info "nothing fit in this task"
elseif length(JOBS) == 1
    rat, fold = JOBS[1]
    CSV.write(joinpath(per_task_dir, "$(rat)_fold$(fold)_cv.csv"), summary)
    @info "Wrote per-task CV summary for $rat fold $fold"
else
    CSV.write(joinpath(out_dir, "cv_state_sweep_summary.csv"), summary)
    @info "Wrote CV summary to $(joinpath(out_dir, "cv_state_sweep_summary.csv"))"
end

show(summary; allrows=true, allcols=true)
println()
