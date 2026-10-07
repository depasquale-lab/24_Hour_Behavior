#=
Within-session vs. between-session control. Each session is split in time:
first TRAIN_FRAC of trials train, the rest are held out.

  ddm              one DDM per rat (sessions pooled)
  session_ddm      one DDM per session, all 4 parameters free
  session_ddm_tau  one DDM per session, τ fixed at the rat-level value
  hmm              K-state DDM-HMM on the training parts; held-out end scored
                   conditional on its session's start

If the HMM beats the per-session DDMs, its gain is within-session, not just
"sessions differ". Sessions shorter than MIN_TRIALS are dropped.

Task: $SGE_TASK_ID or ARGS[1] = rat index; none -> every rat. `merge` stitches
the per-rat CSVs. FULL_FIT=1 (daily default) also fits the K-state DDM-HMM on
all of the rat's data, to OUT_DIR/full_fits/.

Env knobs: GROUP (daily | 24hr), K (4), N_INITS (10), MAX_ITER (100),
TRAIN_FRAC (0.8), MIN_TRIALS (50), FULL_FIT (1 for daily, 0 for 24hr)
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

const Optim = DriftDiffusionModels.Optim
const ForwardDiff = DriftDiffusionModels.ForwardDiff

const GROUP = get(ENV, "GROUP", "daily")
GROUP in ("daily", "24hr") || error("GROUP must be \"daily\" or \"24hr\", got \"$GROUP\"")
const K = parse(Int, get(ENV, "K", "4"))
const N_INITS = parse(Int, get(ENV, "N_INITS", "10"))
const MAX_ITER = parse(Int, get(ENV, "MAX_ITER", "100"))
const TRAIN_FRAC = parse(Float64, get(ENV, "TRAIN_FRAC", "0.8"))
const MIN_TRIALS = parse(Int, get(ENV, "MIN_TRIALS", "50"))
const FULL_FIT = get(ENV, "FULL_FIT", GROUP == "daily" ? "1" : "0") == "1"
const N_FOLDS = 5  # only used to drop rats the CV scripts drop

const OUT_DIR = joinpath("results", "session_cohort", "split_$(GROUP)")
const FULL_DIR = joinpath("results", "session_cohort", "full_fits")

Random.seed!(67)

rat_df = CSV.read(joinpath("data", "processed_rat_data.csv.gz"), DataFrame)
rat_df = rat_df[rat_df.daily .== (GROUP == "24hr" ? "24 hr" : "daily"), :]
replace!(rat_df[!, :choose_right], 0 => -1)
side_mapping = Dict("right" => 1, "left" => -1)
DataFrames.transform!(
    rat_df,
    :correct_side => ByRow(cs -> get(side_mapping, cs, missing)) => :correct_side_numeric,
)

"Results-by-date (one `Vector{DDMResult}` per session), chronological."
function sessions_for_rat(rat::AbstractString)
    sub = rat_df[rat_df.name .== rat, :]
    dates = [Date(split(dt)[1]) for dt in sub.trial_datetime]
    map(sort(unique(dates))) do d
        idx = findall(dates .== d)
        [
            DDMResult(rt, ch, st) for
            (rt, ch, st) in zip(sub.rt[idx], sub.choose_right[idx], sub.correct_side_numeric[idx])
        ]
    end
end

# Same rat set as CrossValidateStatesDaily.jl with RAT_SET=all.
const RATS = filter(
    r -> length(sessions_for_rat(r)) >= N_FOLDS, String.(unique(rat_df[!, "name"]))
)

# Models

"Random DDM-HMM initialization (same scheme as FitDDMHMMs.jl)."
function generate_ddmhmm_initialization(n_states::Int)
    init_init = rand(Dirichlet(fill(1.0, n_states)))
    trans_init = zeros(Float64, n_states, n_states)
    for i in 1:n_states
        α = ones(n_states)
        α[i] = 10.0
        trans_init[i, :] .= rand(Dirichlet(α))
    end
    log_drift = rand(Normal(0.0, 0.5), n_states)
    log_boundary = rand(Normal(log(2), 0.5), n_states)
    bias = rand(Beta(10, 10), n_states)
    emissions = [
        DriftDiffusionModel(exp(log_drift[i]), exp(log_boundary[i]), bias[i], 0.1) for
        i in 1:n_states
    ]
    return PriorHMM(init_init, trans_init, emissions, 1, 1)
end

"Best-of-`N_INITS` Baum-Welch fit, picked by training logL."
function fit_best(data, seq_ends, n_states::Int)
    priors = [generate_ddmhmm_initialization(n_states) for _ in 1:N_INITS]
    results = Vector{Any}(nothing, N_INITS)
    Threads.@threads for i in 1:N_INITS
        try
            hmm, evol = HiddenMarkovModels.baum_welch(
                priors[i],
                data;
                seq_ends=seq_ends,
                atol=1e-3,
                max_iterations=MAX_ITER,
                loglikelihood_increasing=false,
            )
            results[i] = (hmm, last(evol))
        catch e
            @warn "Baum-Welch failed" n_states i exception = (e, catch_backtrace())
        end
    end
    ok = filter(!isnothing, results)
    isempty(ok) && error("all $N_INITS inits failed")
    best = argmax(r -> r[2], ok)
    return best[1], best[2], length(ok)
end

loglik(m::DriftDiffusionModel, x) = sum(logdensityof(m, o) for o in x)

"Single DDM by MLE, started from `init` (τ clamped below the fastest RT)."
function fit_ddm(x::Vector{DDMResult}, init::DriftDiffusionModel)
    τ0 = min(init.τ, 0.9 * minimum(o.rt for o in x))
    m = DriftDiffusionModel(init.B, max(init.v, 0.05), clamp(init.a₀, 0.05, 0.95), max(τ0, 2e-3))
    return fit!(m, x)
end

"Single DDM with τ fixed; fits (B, v, a₀) by bounded L-BFGS."
function fit_ddm_fixed_tau(x::Vector{DDMResult}, init::DriftDiffusionModel, τ::Float64)
    nll(p) = -sum(logdensityof(p[1], p[2], p[3], τ, o.rt, o.choice, o.s) for o in x)
    g!(g, p) = ForwardDiff.gradient!(g, nll, p)
    p0 = [init.B, clamp(init.v, 0.05, 9.0), clamp(init.a₀, 0.05, 0.95)]
    res = Optim.optimize(
        nll,
        g!,
        [0.001, 0.0, 0.0],
        [50.0, 10.0, 1.0],
        p0,
        Optim.Fminbox(Optim.LBFGS(; linesearch=Optim.LineSearches.BackTracking())),
    )
    p = Optim.minimizer(res)
    return DriftDiffusionModel(p[1], p[2], p[3], τ)
end

# Per-rat task

function split_task(rat_idx::Int)
    rat = RATS[rat_idx]
    out = joinpath(OUT_DIR, "$(rat)_split.csv")
    sessions = filter(s -> length(s) >= MIN_TRIALS, sessions_for_rat(rat))
    n_train = [floor(Int, TRAIN_FRAC * length(s)) for s in sessions]
    trains = [s[1:n] for (s, n) in zip(sessions, n_train)]
    tests = [s[(n + 1):end] for (s, n) in zip(sessions, n_train)]
    @info "== $rat ($GROUP): $(length(sessions)) sessions, $(sum(n_train)) train | $(sum(length, tests)) test trials =="

    if !isfile(out)
        # Global DDM
        pooled = reduce(vcat, trains)
        ddm = fit_ddm(pooled, DriftDiffusionModel(2.0, 1.0, 0.5, 0.1))
        @info "  rat-level DDM: B=$(round(ddm.B; digits=3)) v=$(round(ddm.v; digits=3)) a0=$(round(ddm.a₀; digits=3)) τ=$(round(ddm.τ; digits=3))"

        # Per-session DDMs (threaded over sessions)
        n_s = length(sessions)
        ll_sess, ll_sess_tau = zeros(n_s), zeros(n_s)
        Threads.@threads for j in 1:n_s
            m_free = try
                fit_ddm(trains[j], ddm)
            catch e
                @warn "session DDM failed; using rat-level DDM" rat j exception = e
                ddm
            end
            m_tau = try
                fit_ddm_fixed_tau(trains[j], ddm, ddm.τ)
            catch e
                @warn "fixed-τ session DDM failed; using rat-level DDM" rat j exception = e
                ddm
            end
            ll_sess[j] = loglik(m_free, tests[j])
            ll_sess_tau[j] = loglik(m_tau, tests[j])
        end

        # DDM-HMM on the training parts, held-out ends scored given their session start
        hmm, train_ll, n_ok = fit_best(pooled, cumsum(length.(trains)), K)
        ll_hmm = [logdensityof(hmm, s) - logdensityof(hmm, tr) for (s, tr) in zip(sessions, trains)]
        @save joinpath(OUT_DIR, "$(rat)_K$(K)_split.bson") hmm train_ll ddm

        df = DataFrame(;
            rat=rat,
            session=1:n_s,
            n_trials=length.(sessions),
            n_train=n_train,
            n_test=length.(tests),
            ddm=[loglik(ddm, t) for t in tests],
            session_ddm=ll_sess,
            session_ddm_tau=ll_sess_tau,
            hmm=ll_hmm,
        )
        CSV.write(out, df)
        per = n -> round(sum(df[!, n]) / sum(df.n_test); digits=4)
        @info "  held-out logL/trial: ddm=$(per(:ddm)) session_ddm=$(per(:session_ddm)) session_ddm_tau=$(per(:session_ddm_tau)) hmm=$(per(:hmm))"
    else
        @info "  split already done"
    end

    if FULL_FIT
        full_out = joinpath(FULL_DIR, "$(rat)_K$(K)_$(GROUP).bson")
        if !isfile(full_out)
            all_sess = sessions_for_rat(rat)
            hmm, logL, _ = fit_best(reduce(vcat, all_sess), cumsum(length.(all_sess)), K)
            @save full_out hmm logL rat
            @info "  full-data K=$K fit: logL=$(round(logL; digits=1))"
        end
    end
end

function merge_split()
    files = filter(f -> endswith(f, "_split.csv"), readdir(OUT_DIR; join=true))
    df = reduce(vcat, [CSV.read(f, DataFrame; types=Dict(:rat => String)) for f in files])
    CSV.write(joinpath("results", "session_cohort", "split_$(GROUP)_sessions.csv"), df)
    s = combine(
        groupby(df, :rat),
        :n_test => sum => :n_test,
        [:ddm, :n_test] => ((a, n) -> sum(a) / sum(n)) => :ddm,
        [:session_ddm, :n_test] => ((a, n) -> sum(a) / sum(n)) => :session_ddm,
        [:session_ddm_tau, :n_test] => ((a, n) -> sum(a) / sum(n)) => :session_ddm_tau,
        [:hmm, :n_test] => ((a, n) -> sum(a) / sum(n)) => :hmm,
    )
    CSV.write(joinpath("results", "session_cohort", "split_$(GROUP)_summary.csv"), s)
    @info "merged $(nrow(s)) of $(length(RATS)) rats"
    println(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    mkpath(OUT_DIR)
    mkpath(FULL_DIR)
    if get(ARGS, 1, "") == "merge"
        merge_split()
    else
        v = get(ENV, "SGE_TASK_ID", "")
        tasks = if !isempty(v) && v != "undefined"
            [parse(Int, v)]
        elseif !isempty(ARGS)
            [parse(Int, ARGS[1])]
        else
            eachindex(RATS)
        end
        foreach(split_task, tasks)
    end
end
