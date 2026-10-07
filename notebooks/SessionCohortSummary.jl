#=
Session vs. 24 hr cohort: one table of held-out fits plus within-session switching.

Reads (both cohorts, same session-blocked folds):
  DDM-HMM K = 1..5     results/ddm_hmm_state_sweep/cv{,_24hr}/per_task_summaries/
  shuffled K = 4       results/ddm_hmm_state_sweep/cv{,_24hr}_shuffled/per_task_summaries/
  multilevel DDM       results/eddm_exact{_daily,}/cv/
  within-session split results/session_cohort/split_{daily,24hr}/   (SessionSplitDDM.jl)

Only (rat, fold) pairs present for every model are compared; missing models are
skipped with a warning.

Writes results/session_cohort/:
  heldout_by_rat.csv     per rat: held-out logL per trial for every model
  heldout_vs_K.csv       per rat × K: Δ held-out logL per trial vs K = 1
  heldout_stats.csv      paired contrasts per cohort (Wilcoxon signed-rank)
  switching_by_rat.csv   Viterbi-decoded held-out sessions: switches, dwell, states used
  dwell_lengths.csv      every within-session dwell (run length, censored flag)
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Dates
using Statistics
using HypothesisTests
using DriftDiffusionModels
using HiddenMarkovModels
using BSON: @load

const OUT = joinpath("results", "session_cohort")
mkpath(OUT)
const COHORTS = ["daily", "24hr"]
const K_MAIN = 4
const N_FOLDS = 5
const MIN_OCC = 0.10  # a state "is used" in a session if it holds >= 10% of trials

cv_dir(c) = joinpath("results", "ddm_hmm_state_sweep", c == "24hr" ? "cv_24hr" : "cv")
mlddm_dir(c) = joinpath("results", c == "24hr" ? "eddm_exact" : "eddm_exact_daily", "cv")

function read_dir(dir; suffix=".csv")
    isdir(dir) || return DataFrame()
    files = filter(f -> endswith(f, suffix), readdir(dir; join=true))
    isempty(files) && return DataFrame()
    reduce(vcat, [CSV.read(f, DataFrame; types=Dict(:rat => String)) for f in files]; cols=:union)
end

# Held-out fits, one row per (cohort, rat, fold)

function fold_table(c)
    hmm = read_dir(joinpath(cv_dir(c), "per_task_summaries"))
    isempty(hmm) && error("no DDM-HMM CV results for $c")
    w = unstack(hmm, [:rat, :fold, :n_test_trials], :K, :test_logL; renamecols=k -> "hmm_K$k")
    rename!(w, "hmm_K1" => "ddm")
    for (name, df) in [
        "mixture" => read_dir(joinpath(cv_dir(c) * "_shuffled", "per_task_summaries")),
        "mlddm" => read_dir(mlddm_dir(c)),
    ]
        if isempty(df)
            @warn "$c: no $name results yet"
            w[!, name] = missings(Float64, nrow(w))
        else
            w = leftjoin(w, select(df, :rat, :fold, :test_logL => name); on=[:rat, :fold])
        end
    end
    w.cohort .= c
    return w
end

folds = reduce(vcat, fold_table.(COHORTS); cols=:union)
models = ["ddm", "mixture", "mlddm", ["hmm_K$k" for k in 2:5]...]

# Per-rat pooled logL per trial, over folds where every available model is present.
by_rat = combine(groupby(folds, [:cohort, :rat])) do sub
    have = [m for m in models if !all(ismissing, sub[!, m])]
    keep = completecases(sub, have)
    n = sum(sub.n_test_trials[keep])
    row = (n_folds=sum(keep), n_test_trials=n)
    merge(row, NamedTuple(Symbol(m) => (m in have ? sum(sub[keep, m]) / n : missing) for m in models))
end
sort!(by_rat, [:cohort, :rat])
CSV.write(joinpath(OUT, "heldout_by_rat.csv"), by_rat)

vs_K = DataFrame(; cohort=String[], rat=String[], K=Int[], Δ=Union{Missing,Float64}[])
for r in eachrow(by_rat), k in 1:5
    push!(vs_K, (r.cohort, r.rat, k, k == 1 ? 0.0 : r[Symbol("hmm_K$k")] - r.ddm))
end
CSV.write(joinpath(OUT, "heldout_vs_K.csv"), vs_K)

# Within-session split (SessionSplitDDM.jl)

split_df = reduce(
    vcat,
    [
        let s = read_dir(joinpath(OUT, "split_$c"); suffix="_split.csv")
            isempty(s) ? DataFrame() : (s.cohort .= c; s)
        end for c in COHORTS
    ];
    cols=:union,
)
split_by_rat = if isempty(split_df)
    @warn "no within-session split results yet"
    DataFrame(; cohort=String[], rat=String[])
else
    combine(
        groupby(split_df, [:cohort, :rat]),
        :n_test => sum => :split_n_test,
        [:ddm, :n_test] => ((a, n) -> sum(a) / sum(n)) => :split_ddm,
        # stronger of the two per-session baselines, per rat
        [:session_ddm, :session_ddm_tau, :n_test] =>
            ((a, b, n) -> max(sum(a), sum(b)) / sum(n)) => :split_session_ddm,
        [:hmm, :n_test] => ((a, n) -> sum(a) / sum(n)) => :split_hmm,
    )
end
CSV.write(joinpath(OUT, "split_by_rat.csv"), split_by_rat)

# Paired contrasts

contrasts = [
    ("DDM-HMM vs DDM", by_rat, :hmm_K4, :ddm),
    ("DDM-HMM vs multilevel DDM", by_rat, :hmm_K4, :mlddm),
    ("mixture vs DDM", by_rat, :mixture, :ddm),
    ("DDM-HMM vs mixture (dynamics)", by_rat, :hmm_K4, :mixture),
    ("DDM-HMM vs per-session DDM (split)", split_by_rat, :split_hmm, :split_session_ddm),
    ("per-session DDM vs DDM (split)", split_by_rat, :split_session_ddm, :split_ddm),
]
stats = DataFrame(;
    contrast=String[], cohort=String[], n_rats=Int[], n_positive=Int[],
    median_gain=Float64[], min_gain=Float64[], wilcoxon_p=Float64[],
)
for (label, df, a, b) in contrasts, c in COHORTS
    (hasproperty(df, a) && hasproperty(df, b)) || continue
    sub = dropmissing(df[df.cohort .== c, [a, b]])
    nrow(sub) == 0 && continue
    d = sub[!, a] .- sub[!, b]
    p = length(d) >= 2 ? pvalue(SignedRankTest(Float64.(d))) : NaN
    push!(stats, (label, c, length(d), count(>(0), d), median(d), minimum(d), p))
end
# Cohort difference in the headline gain (Mann-Whitney)
for (label, a, b) in [("DDM-HMM vs DDM", :hmm_K4, :ddm), ("DDM-HMM vs multilevel DDM", :hmm_K4, :mlddm)]
    g = [collect(skipmissing(by_rat[by_rat.cohort .== c, a] .- by_rat[by_rat.cohort .== c, b])) for c in COHORTS]
    all(length.(g) .>= 2) || continue
    push!(stats, (label * ": daily - 24hr", "both", sum(length.(g)), -1,
                  median(g[1]) - median(g[2]), NaN, pvalue(MannWhitneyUTest(g[1], g[2]))))
end
CSV.write(joinpath(OUT, "heldout_stats.csv"), stats)
println(stats)

# Within-session switching on held-out sessions (Viterbi under the fold's K = 4 fit)

rat_df = CSV.read(joinpath("data", "processed_rat_data.csv.gz"), DataFrame)
replace!(rat_df[!, :choose_right], 0 => -1)
rat_df.correct_side_numeric = [cs == "right" ? 1 : -1 for cs in rat_df.correct_side]

function sessions_for(rat, c)
    sub = rat_df[(rat_df.name .== rat) .& (rat_df.daily .== (c == "24hr" ? "24 hr" : "daily")), :]
    dates = [Date(split(dt)[1]) for dt in sub.trial_datetime]
    map(sort(unique(dates))) do d
        idx = findall(dates .== d)
        [DDMResult(rt, ch, st) for (rt, ch, st) in zip(sub.rt[idx], sub.choose_right[idx], sub.correct_side_numeric[idx])]
    end
end

"Held-out sessions of fold `f`, as in CrossValidateStatesDaily.jl."
function test_sessions(sessions, f)
    n = length(sessions)
    sz = div(n, N_FOLDS)
    sessions[((f - 1) * sz + 1):(f == N_FOLDS ? n : f * sz)]
end

"Run lengths of a state path; the first and last runs are censored by the session edges."
function runs(path)
    out = Tuple{Int,Int,Bool}[]  # (state, length, censored)
    start = 1
    for t in 2:(length(path) + 1)
        if t > length(path) || path[t] != path[start]
            push!(out, (path[start], t - start, start == 1 || t > length(path)))
            start = t
        end
    end
    out
end

sw_rows = DataFrame(; cohort=String[], rat=String[], fold=Int[], session=Int[], n_trials=Int[],
                    n_switches=Int[], n_states_used=Int[])
dwell_rows = DataFrame(; cohort=String[], rat=String[], state=Int[], dwell=Int[], censored=Bool[])
for c in COHORTS
    dir = cv_dir(c)
    for rat in unique(folds.rat[folds.cohort .== c])
        sessions = sessions_for(rat, c)
        for f in 1:N_FOLDS
            path = joinpath(dir, "$(rat)_K$(K_MAIN)_fold$(f)_$(c)_cv.bson")
            isfile(path) || continue
            @load path hmm
            for (j, s) in enumerate(test_sessions(sessions, f))
                z, _ = viterbi(hmm, s)
                r = runs(z)
                occ = [count(==(k), z) / length(z) for k in 1:K_MAIN]
                push!(sw_rows, (c, rat, f, j, length(s), length(r) - 1, count(>=(MIN_OCC), occ)))
                for (k, len, cens) in r
                    push!(dwell_rows, (c, rat, k, len, cens))
                end
            end
        end
    end
end
CSV.write(joinpath(OUT, "dwell_lengths.csv"), dwell_rows)

switching = combine(groupby(sw_rows, [:cohort, :rat]),
    :session => length => :n_sessions,
    [:n_switches, :n_trials] => ((s, n) -> 100 * sum(s) / sum(n)) => :switches_per_100_trials,
    :n_switches => median => :median_switches_per_session,
    :n_states_used => (x -> mean(x .>= 2)) => :frac_sessions_2plus_states,
    :n_states_used => mean => :mean_states_used,
)
dw = combine(groupby(dwell_rows[.!dwell_rows.censored, :], [:cohort, :rat]), :dwell => median => :median_dwell)
switching = leftjoin(switching, dw; on=[:cohort, :rat])
sort!(switching, [:cohort, :rat])
CSV.write(joinpath(OUT, "switching_by_rat.csv"), switching)
println(switching)
