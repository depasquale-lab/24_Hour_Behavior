#=
Exact (maximum marginal likelihood) refit of the per-rat eDDM, replacing the
variational fit in eDDM.jl. Hyperparameters (m, σ0) by fit_mlddm_exact
(Gauss–Hermite quadrature, L-BFGS, several starts, quadrature order raised until
error < 1e-3 nats/trial); per-trial posteriors from trial_posteriors.

Outputs (results/eddm_exact/, same columns as the VI outputs so eDDM_posthoc.jl
reads either; no ELBO history):
  eddm_trial_posteriors_by_rat.csv.gz   μ_u* = posterior mean, logσ_u* = log posterior SD
  eddm_hyperparams_by_rat.csv           m_u*, logσ0_u* = log σ0
  eddm_fit_summary_by_rat.csv           loglik, BIC, convergence, quadrature error

Task: $SGE_TASK_ID or ARGS[1] = rat index; none -> every rat. `merge` combines
the per-rat files.

`cv`: the session-blocked 5-fold split of CrossValidateStatesDaily.jl, so
held-out logL lines up with the DDM-HMM CV. Each fold is refit on its training
sessions, warm-started from the full-data optimum (one start, fixed quadrature
order). $SGE_TASK_ID indexes the (rat x fold) grid, rat-major. Writes
OUT_DIR/cv/<rat>_fold<f>_cv.csv; `cv-merge` stitches them.

Env knobs: GROUP (24hr | daily), N_STARTS (4), Q (8), OUT_DIR, N_FOLDS (5),
MAX_TRIALS (0 = all; smoke tests only)
=#

using Pkg
Pkg.activate("notebooks")

using Random
using DriftDiffusionModels
using CSV
using DataFrames
using Statistics
using Serialization
using Dates

const N_STARTS = parse(Int, get(ENV, "N_STARTS", "4"))
const Q = parse(Int, get(ENV, "Q", "8"))
const MAX_TRIALS = parse(Int, get(ENV, "MAX_TRIALS", "0"))
const N_FOLDS = parse(Int, get(ENV, "N_FOLDS", "5"))
const GROUP = get(ENV, "GROUP", "24hr")
GROUP in ("daily", "24hr") || error("GROUP must be \"daily\" or \"24hr\", got \"$GROUP\"")
const OUT_DIR = get(
    ENV, "OUT_DIR", joinpath("results", GROUP == "24hr" ? "eddm_exact" : "eddm_exact_daily")
)
const PER_RAT_DIR = joinpath(OUT_DIR, "per_rat")
const CV_DIR = joinpath(OUT_DIR, "cv")

const PARAMS = ["B", "τ", "v", "a0"]

rat_df = CSV.read(joinpath("data", "processed_rat_data.csv.gz"), DataFrame)
rat_df = rat_df[rat_df.daily .== (GROUP == "24hr" ? "24 hr" : "daily"), :]
replace!(rat_df[!, :choose_right], 0 => -1)
mapping = Dict("right" => 1, "left" => -1)
transform!(
    rat_df, :correct_side => ByRow(cs -> get(mapping, cs, missing)) => :correct_side_numeric
)

session_dates(sub) = [Date(split(dt)[1]) for dt in sub.trial_datetime]

# Rats with fewer sessions than folds (608840: 2 sessions) are dropped, as in
# CrossValidateStatesDaily.jl.
const RATS = filter(
    r -> length(unique(session_dates(rat_df[rat_df.name .== r, :]))) >= N_FOLDS,
    String.(unique(rat_df.name)),
)

rat_data(sub) = [
    DDMResult(rt, ch, st) for
    (rt, ch, st) in zip(sub.rt, sub.choose_right, sub.correct_side_numeric)
]

to_natural(u) = (exp(u[1]), exp(u[2]), exp(u[3]), logistic(u[4]))

function fit_rat(rat_idx::Int)
    rat = RATS[rat_idx]
    sub = rat_df[rat_df.name .== rat, :]
    data = rat_data(sub)
    @info "== $rat: $(length(data)) trials (threads=$(Threads.nthreads())) =="

    fit = fit_mlddm_exact(
        data;
        q=Q,
        qτ=2Q,
        n_starts=N_STARTS,
        rng=MersenneTwister(67 + rat_idx),
        max_trials=MAX_TRIALS,
    )
    show(stdout, fit)

    μ, sd = trial_posteriors(data, fit.m, fit.σ0; q=fit.q, qτ=fit.qτ)

    trial_df = DataFrame(;
        rat_name=fill(rat, length(data)),
        trial_idx=1:length(data),
        rt=Float64.(sub.rt),
        choice=Int.(sub.choose_right),
        stim=Int.(sub.correct_side_numeric),
    )
    for (d, p) in enumerate(PARAMS)
        trial_df[!, "μ_u$p"] = μ[d, :]
    end
    for (d, p) in enumerate(PARAMS)
        trial_df[!, "logσ_u$p"] = log.(sd[d, :])
    end
    nat = [to_natural(μ[:, t]) for t in axes(μ, 2)]
    for (d, p) in enumerate(PARAMS)
        trial_df[!, "$(p)_mean"] = getindex.(nat, d)
    end

    g = to_natural(fit.m)
    hyper_df = DataFrame(
        :rat_name => [rat],
        [Symbol("m_u$p") => [fit.m[d]] for (d, p) in enumerate(PARAMS)]...,
        [Symbol("logσ0_u$p") => [log(fit.σ0[d])] for (d, p) in enumerate(PARAMS)]...,
        [Symbol("$(p)_group_mean") => [g[d]] for (d, p) in enumerate(PARAMS)]...,
    )

    summary_df = DataFrame(;
        rat_name=[rat],
        n_trials=[fit.n_total],
        n_used=[fit.n_used],
        loglik=[fit.loglik],
        bic=[fit.bic],
        converged=[fit.converged],
        n_converged=[fit.n_converged],
        n_starts=[fit.n_starts],
        q=[fit.q],
        qτ=[fit.qτ],
        quad_err=[fit.quad_err],
        n_zero=[fit.n_zero],
        boundary=[join(PARAMS[fit.boundary], ";")],
        seconds=[fit.seconds],
    )

    CSV.write(joinpath(PER_RAT_DIR, "$(rat)_trial_posteriors.csv.gz"), trial_df; compress=true)
    CSV.write(joinpath(PER_RAT_DIR, "$(rat)_hyperparams.csv"), hyper_df)
    CSV.write(joinpath(PER_RAT_DIR, "$(rat)_fit_summary.csv"), summary_df)
    serialize(joinpath(PER_RAT_DIR, "$(rat)_fit.jls"), fit)
    @info "$rat done in $(round(fit.seconds / 60; digits=1)) min"
end

function merge_rats()
    read_all(suffix) = reduce(
        vcat,
        [
            CSV.read(joinpath(PER_RAT_DIR, "$(r)_$suffix"), DataFrame; types=Dict(:rat_name => String)) for
            r in RATS if isfile(joinpath(PER_RAT_DIR, "$(r)_$suffix"))
        ],
    )
    summary = read_all("fit_summary.csv")
    missing_rats = setdiff(RATS, summary.rat_name)
    isempty(missing_rats) || @warn "missing rats: $(join(missing_rats, ", "))"

    CSV.write(joinpath(OUT_DIR, "eddm_fit_summary_by_rat.csv"), summary)
    CSV.write(joinpath(OUT_DIR, "eddm_hyperparams_by_rat.csv"), read_all("hyperparams.csv"))
    CSV.write(
        joinpath(OUT_DIR, "eddm_trial_posteriors_by_rat.csv.gz"),
        read_all("trial_posteriors.csv.gz");
        compress=true,
    )
    println(summary)
end

"""
    session_folds(sub)

Trial indices of the train and test rows for each fold: fold `f` holds out the
f-th contiguous block of sessions, exactly as `train_test_split` in
CrossValidateStatesDaily.jl.
"""
function session_folds(sub)
    dates = session_dates(sub)
    udates = sort(unique(dates))
    n = length(udates)
    fold_size = div(n, N_FOLDS)
    map(1:N_FOLDS) do f
        test_end = f == N_FOLDS ? n : f * fold_size
        test_dates = Set(udates[((f - 1) * fold_size + 1):test_end])
        # Rows in session order, matching reduce(vcat, sessions) in the CV script.
        order = sortperm(dates; alg=MergeSort)
        test = filter(i -> dates[i] in test_dates, order)
        train = filter(i -> !(dates[i] in test_dates), order)
        (train=train, test=test, n_train_sessions=n - length(test_dates),
         n_test_sessions=length(test_dates))
    end
end

const CV_GRID = [(r, f) for r in eachindex(RATS) for f in 1:N_FOLDS]

function cv_task(task::Int)
    rat_idx, fold = CV_GRID[task]
    rat = RATS[rat_idx]
    out = joinpath(CV_DIR, "$(rat)_fold$(fold)_cv.csv")
    isfile(out) && return @info "$rat fold $fold already done"

    full_path = joinpath(PER_RAT_DIR, "$(rat)_fit.jls")
    isfile(full_path) || error("no full-data fit for $rat at $full_path; run the fit step first")
    full = deserialize(full_path)

    sub = rat_df[rat_df.name .== rat, :]
    data = rat_data(sub)
    split = session_folds(sub)[fold]
    train, test = data[split.train], data[split.test]
    @info "== $rat fold $fold/$N_FOLDS: $(length(train)) train | $(length(test)) test trials =="

    fit = fit_mlddm_exact(
        train;
        q=full.q,
        qτ=full.qτ,
        n_starts=1,
        init=(full.m, full.σ0),
        escalate=false,
        rng=MersenneTwister(67 + task),
    )
    test_ll = marginal_loglik(test, fit.m, log.(fit.σ0); q=fit.q, qτ=fit.qτ)

    CSV.write(
        out,
        DataFrame(;
            rat=[rat],
            fold=[fold],
            n_params=[8],
            n_train_sessions=[split.n_train_sessions],
            n_test_sessions=[split.n_test_sessions],
            n_train_trials=[length(train)],
            n_test_trials=[length(test)],
            train_logL=[fit.loglik],
            train_logL_per_trial=[fit.loglik / length(train)],
            test_logL=[test_ll],
            test_logL_per_trial=[test_ll / length(test)],
            converged=[fit.converged],
            q=[fit.q],
            qτ=[fit.qτ],
            seconds=[fit.seconds],
        ),
    )
    @info "$rat fold $fold: test logL/trial = $(round(test_ll / length(test); digits=4))"
end

function cv_merge()
    files = filter(f -> endswith(f, "_cv.csv"), readdir(CV_DIR; join=true))
    df = sort!(reduce(vcat, [CSV.read(f, DataFrame; types=Dict(:rat => String)) for f in files]),
               [:rat, :fold])
    CSV.write(joinpath(OUT_DIR, "eddm_cv_summary.csv"), df)
    @info "merged $(nrow(df)) of $(length(CV_GRID)) (rat, fold) rows"
    println(df)
end

task_ids(all_ids) =
    let v = get(ENV, "SGE_TASK_ID", "")
        if !isempty(v) && v != "undefined"
            [parse(Int, v)]
        elseif length(ARGS) >= 2
            [parse(Int, ARGS[2])]
        else
            all_ids
        end
    end

if abspath(PROGRAM_FILE) == @__FILE__
    mkpath(PER_RAT_DIR)
    mkpath(CV_DIR)
    mode = get(ARGS, 1, "")
    if mode == "merge"
        merge_rats()
    elseif mode == "cv"
        foreach(cv_task, task_ids(eachindex(CV_GRID)))
    elseif mode == "cv-merge"
        cv_merge()
    else
        v = get(ENV, "SGE_TASK_ID", "")
        tasks = if !isempty(v) && v != "undefined"
            [parse(Int, v)]
        elseif !isempty(ARGS)
            [parse(Int, ARGS[1])]
        else
            eachindex(RATS)
        end
        foreach(fit_rat, tasks)
    end
end
