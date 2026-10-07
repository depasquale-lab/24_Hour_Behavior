#=
Time-of-day DDM baseline for the 24 hr animals: a single-state DDM whose bound,
drift and starting point are Fourier functions of clock time (τ constant unless
VARY_TAU=1). H = 0 is the plain DDM; each H warm-starts from H - 1.

Scored on the session-blocked 5-fold split of CrossValidateStatesDaily.jl
(GROUP=24hr); each H is also fit on all data for BIC.

Task: $SGE_TASK_ID or ARGS[1] = rat index; none -> every rat. `merge`
concatenates the per-rat CSVs.

Env knobs: H_LIST=0,1,2,3, N_FOLDS, VARY_TAU=0|1
=#

using Pkg
Pkg.activate("notebooks")

using DriftDiffusionModels
using Dates
using CSV
using DataFrames
using Statistics
using BSON: @save

const H_LIST = let v = get(ENV, "H_LIST", "")
    isempty(v) ? collect(0:3) : [parse(Int, s) for s in split(v, ',')]
end
const N_FOLDS = parse(Int, get(ENV, "N_FOLDS", "5"))
const VARY = (true, true, true, get(ENV, "VARY_TAU", "0") == "1")   # (B, v, a₀, τ)

const OUT_DIR = joinpath("results", VARY[4] ? "tod_ddm_vary_tau" : "tod_ddm")
const PER_RAT_DIR = joinpath(OUT_DIR, "per_rat")

if !isempty(ARGS) && ARGS[1] == "merge"
    files = filter(endswith("_tod.csv"), readdir(PER_RAT_DIR; join=true))
    df = reduce(vcat, CSV.read.(files, DataFrame))
    CSV.write(joinpath(OUT_DIR, "tod_ddm_summary.csv"), df)
    @info "merged $(length(files)) rats -> $(joinpath(OUT_DIR, "tod_ddm_summary.csv"))"
    exit()
end

# Data loading (24 hr animals)

const DATA_FILE = joinpath("data", "processed_rat_data.csv.gz")

rat_df = CSV.read(DATA_FILE, DataFrame)
rat_df = rat_df[rat_df.daily .== "24 hr", :]
replace!(rat_df[!, :choose_right], 0 => -1)
side_mapping = Dict("right" => 1, "left" => -1)
DataFrames.transform!(
    rat_df,
    :correct_side => ByRow(cs -> get(side_mapping, cs, missing)) => :correct_side_numeric,
)

const RATS = String.(unique(rat_df[!, "name"]))

"Fractional clock hour. A handful of rows carry a date only; they map to 00:00."
function clock_hour(dt::AbstractString)
    parts = split(dt)
    length(parts) < 2 && return 0.0
    h, m, s = parse.(Float64, split(parts[2], ':'))
    return h + m / 60 + s / 3600
end

"""
    sessions_for_rat(rat)

One `(rt, choice, s, hour)` table per calendar date, in the same order as
`sessions_for_rat` in CrossValidateStatesDaily.jl.
"""
function sessions_for_rat(rat::AbstractString)
    sub = rat_df[rat_df.name .== rat, :]
    dates = [Date(split(dt)[1]) for dt in sub.trial_datetime]
    hours = clock_hour.(String.(sub.trial_datetime))

    sessions = Vector{Vector{NTuple{4,Float64}}}()
    for date in sort(unique(dates))
        idx = findall(dates .== date)
        isempty(idx) && continue
        push!(
            sessions,
            [
                (rt, ch, st, h) for (rt, ch, st, h) in
                zip(sub.rt[idx], sub.choose_right[idx], sub.correct_side_numeric[idx], hours[idx])
            ],
        )
    end
    return sessions
end

"Session-blocked split; identical indexing to CrossValidateStatesDaily.jl."
function train_test_split(sessions; n_folds::Int=N_FOLDS, fold::Int=1)
    n = length(sessions)
    fold_size = div(n, n_folds)
    fold_size < 1 && error("rat has only $n sessions; cannot run $n_folds-fold CV")
    test_start = (fold - 1) * fold_size + 1
    test_end = fold == n_folds ? n : fold * fold_size
    test_idx = test_start:test_end
    train_idx = setdiff(1:n, test_idx)
    return reduce(vcat, sessions[train_idx]), reduce(vcat, sessions[test_idx]),
           length(train_idx), length(test_idx)
end

to_obs(trials, H) = [RegDDMResult(t[1], Int(t[2]), Int(t[3]), fourier_basis(t[4]; n_harmonics=H))
                     for t in trials]

total_ll(m, obs) = sum(o -> logdensityof(m, o), obs)

"""
    fit_sweep(train, test)

Fit H = H_LIST in order, warm-starting each from the previous solution. Returns
one row per H (test columns are NaN when `test` is empty).
"""
function fit_sweep(train, test)
    d0 = DriftDiffusionModel(B=2.0, v=1.0, a₀=0.5, τ=0.1)
    fit!(d0, [DDMResult(t[1], Int(t[2]), Int(t[3])) for t in train])

    rows, models = NamedTuple[], Dict{Int,RegressionDDM}()
    prev = nothing
    for H in H_LIST
        P = 1 + 2H
        m = RegressionDDM(d0, P; vary=VARY)
        if prev !== nothing
            Pp = size(prev.β, 2)
            m.β[:, 1:min(P, Pp)] .= prev.β[:, 1:min(P, Pp)]
        end
        obs_tr = to_obs(train, H)
        fit!(m, obs_tr)
        tr = total_ll(m, obs_tr)
        te = isempty(test) ? NaN : total_ll(m, to_obs(test, H))
        push!(rows, (H=H, n_params=n_free_params(m), train_logL=tr, test_logL=te))
        models[H] = m
        prev = m
    end
    return rows, models
end

function _resolve_rats()
    task_env = get(ENV, "SGE_TASK_ID", "")
    if !isempty(task_env) && task_env != "undefined"
        return [RATS[parse(Int, task_env)]]
    elseif length(ARGS) >= 1
        return [RATS[parse(Int, ARGS[1])]]
    end
    return RATS
end

mkpath(PER_RAT_DIR)
const JOBS = _resolve_rats()
@info "Time-of-day DDM: H = $H_LIST, $N_FOLDS folds, vary (B,v,a₀,τ) = $VARY" rats = JOBS threads = Threads.nthreads()

for rat in JOBS
    sessions = sessions_for_rat(rat)

    # Folds 1..N_FOLDS for held-out logL, fold 0 = all data for BIC.
    tasks = 0:N_FOLDS
    out = Vector{Any}(undef, length(tasks))
    Threads.@threads for i in eachindex(tasks)
        fold = tasks[i]
        if fold == 0
            train, test = reduce(vcat, sessions), NTuple{4,Float64}[]
            n_tr, n_te = length(sessions), 0
        else
            train, test, n_tr, n_te = train_test_split(sessions; fold=fold)
        end
        rows, models = fit_sweep(train, test)
        out[i] = (fold, rows, models, n_tr, n_te, length(train), length(test))
    end

    summary = DataFrame()
    fits = Dict{Int,Any}()
    for (fold, rows, models, n_tr, n_te, nt_tr, nt_te) in out
        fits[fold] = Dict(H => (β=m.β, free=m.free) for (H, m) in models)
        for r in rows
            push!(summary, (
                rat=rat, H=r.H, fold=fold, n_params=r.n_params,
                n_train_sessions=n_tr, n_test_sessions=n_te,
                n_train_trials=nt_tr, n_test_trials=nt_te,
                train_logL=r.train_logL, train_logL_per_trial=r.train_logL / nt_tr,
                test_logL=r.test_logL, test_logL_per_trial=r.test_logL / max(nt_te, 1),
                bic=fold == 0 ? r.n_params * log(nt_tr) - 2r.train_logL : NaN,
            ); promote=true)
        end
    end
    sort!(summary, [:fold, :H])

    CSV.write(joinpath(PER_RAT_DIR, "$(rat)_tod.csv"), summary)
    @save joinpath(PER_RAT_DIR, "$(rat)_tod_fits.bson") fits rat
    @info "rat $rat done"
    show(summary; allrows=true, allcols=true)
    println()
end
