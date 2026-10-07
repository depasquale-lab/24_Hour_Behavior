#=
How much trial-to-trial RT variance can each model predict from past trials?
Held-out sessions only (same folds as CrossValidateStatesDaily.jl). For trial t,
each model predicts E[RT_t] from trials 1..t-1 and the stimulus s_t:
  DDM-HMM         filtered P(z_t | y_1..t-1) × each state's E[RT | s_t]
  multilevel DDM  E[RT | s_t] under the population distribution (iid trials)
  running mean    mean of the previous w RTs; w picked per fold on training sessions
Score per rat: R² over all held-out trials, and R² after removing session means.

Writes results/session_cohort/predict_rt_by_rat.csv.
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Dates
using Statistics
using Random
using DriftDiffusionModels
using HiddenMarkovModels
using BSON: @load

const OUT = joinpath("results", "session_cohort")
const N_FOLDS = 5
const WINDOWS = [3, 5, 10, 20, 50]
const N_BANK = 4000
logistic(x) = 1 / (1 + exp(-x))

rat_df = CSV.read(joinpath("data", "processed_rat_data.csv.gz"), DataFrame)
replace!(rat_df[!, :choose_right], 0 => -1)
rat_df.s = [cs == "right" ? 1 : -1 for cs in rat_df.correct_side]

function sessions_for(rat, c)
    sub = rat_df[(rat_df.name .== rat) .& (rat_df.daily .== (c == "24hr" ? "24 hr" : "daily")), :]
    dates = [Date(split(dt)[1]) for dt in sub.trial_datetime]
    map(sort(unique(dates))) do d
        i = findall(==(d), dates)
        [DDMResult(rt, ch, s) for (rt, ch, s) in zip(sub.rt[i], sub.choose_right[i], sub.s[i])]
    end
end

function split_folds(sessions, f)
    n = length(sessions)
    sz = div(n, N_FOLDS)
    test = ((f - 1) * sz + 1):(f == N_FOLDS ? n : f * sz)
    sessions[setdiff(1:n, test)], sessions[test]
end

"Expected RT of a DDM (unit noise) given stimulus s."
function mean_rt(B, v, a₀, τ, s)
    μ, z = s * abs(v), a₀ * B
    abs(μ) < 1e-8 && return τ + z * (B - z)
    pup = (1 - exp(-2μ * z)) / (1 - exp(-2μ * B))
    τ + (B * pup - z) / μ
end

"One-step-ahead E[RT] for every trial of a session under the HMM."
function hmm_predict(hmm, sess)
    d = obs_distributions(hmm)
    A, K = transition_matrix(hmm), length(d)
    m = [mean_rt(x.B, x.v, x.a₀, x.τ, s) for x in d, s in (1, -1)]
    prior = copy(initialization(hmm))
    pred = similar(sess, Float64)
    for (t, y) in enumerate(sess)
        pred[t] = sum(prior[k] * m[k, y.s == 1 ? 1 : 2] for k in 1:K)
        ll = [logdensityof(d[k], y) for k in 1:K]
        post = prior .* exp.(ll .- maximum(ll))
        post ./= sum(post)
        prior = vec(post' * A)
    end
    pred
end

function running_mean_predict(sess, w, fallback)
    rts = [y.rt for y in sess]
    [t == 1 ? fallback : mean(rts[max(1, t - w):(t - 1)]) for t in eachindex(rts)]
end

r2(x, y) = (std(x) == 0 ? 0.0 : cor(x, y)^2)   # constant prediction explains nothing
demean(v, sid) = (m = Dict(i => mean(v[sid .== i]) for i in unique(sid)); [x - m[i] for (x, i) in zip(v, sid)])
within_r2(pred, obs, sid) = r2(demean(pred, sid), demean(obs, sid))

hyper(c) = CSV.read(joinpath("results", c == "24hr" ? "eddm_exact" : "eddm_exact_daily",
                             "eddm_hyperparams_by_rat.csv"), DataFrame; types=Dict(:rat_name => String))

function mlddm_means(h, rng)
    m = [h.m_uB, h.m_uτ, h.m_uv, h.m_ua0]
    σ = exp.([h.logσ0_uB, h.logσ0_uτ, h.logσ0_uv, h.logσ0_ua0])
    draws = [(exp(m[1] + σ[1] * randn(rng)), exp(m[2] + σ[2] * randn(rng)),
              exp(m[3] + σ[3] * randn(rng)), logistic(m[4] + σ[4] * randn(rng))) for _ in 1:N_BANK]
    Dict(s => mean(mean_rt(B, v, a, τ, s) for (B, τ, v, a) in draws) for s in (1, -1))
end

rows = DataFrame(; cohort=String[], rat=String[], n_test=Int[], window=Int[], window_within=Int[],
                 hmm=Float64[], mlddm=Union{Missing,Float64}[], running=Float64[],
                 hmm_within=Float64[], mlddm_within=Union{Missing,Float64}[], running_within=Float64[])
by_rat = CSV.read(joinpath(OUT, "heldout_by_rat.csv"), DataFrame; types=Dict(:rat => String))
rng = MersenneTwister(1)

for c in ["daily", "24hr"]
    cvdir = joinpath("results", "ddm_hmm_state_sweep", c == "24hr" ? "cv_24hr" : "cv")
    H = hyper(c)
    for rat in sort(unique(by_rat.rat[by_rat.cohort .== c]))
        sessions = sessions_for(rat, c)
        h = H[H.rat_name .== rat, :]
        ml = nrow(h) == 1 ? mlddm_means(h[1, :], rng) : nothing
        obs, p_hmm, p_ml, sid = Float64[], Float64[], Float64[], Int[]
        p_run = Dict(w => Float64[] for w in WINDOWS)
        fold_of = Int[]
        wins, wins_w = Int[], Int[]
        for f in 1:N_FOLDS
            path = joinpath(cvdir, "$(rat)_K4_fold$(f)_$(c)_cv.bson")
            isfile(path) || (@warn "$c $rat fold $f: no K4 fit"; continue)
            @load path hmm
            train, test = split_folds(sessions, f)
            fallback = mean(y.rt for s in train for y in s)
            # running-mean window chosen on training sessions, separately per metric
            tr_obs = [y.rt for s in train for y in s]
            tr_sid = reduce(vcat, [fill(i, length(s)) for (i, s) in enumerate(train)])
            tr_pred = [reduce(vcat, [running_mean_predict(s, w, fallback) for s in train]) for w in WINDOWS]
            push!(wins, WINDOWS[argmax([r2(p, tr_obs) for p in tr_pred])])
            push!(wins_w, WINDOWS[argmax([within_r2(p, tr_obs, tr_sid) for p in tr_pred])])
            for s in test
                append!(obs, [y.rt for y in s])
                append!(p_hmm, hmm_predict(hmm, s))
                for w in WINDOWS
                    append!(p_run[w], running_mean_predict(s, w, fallback))
                end
                ml === nothing || append!(p_ml, [ml[y.s == 1 ? 1 : -1] for y in s])
                append!(sid, fill(length(unique(sid)) + 1, length(s)))
                append!(fold_of, fill(f, length(s)))
            end
        end
        # each fold uses its own chosen window
        pick(ws) = [p_run[ws[findfirst(==(f), unique(fold_of))]][t] for (t, f) in enumerate(fold_of)]
        run_raw, run_within = pick(wins), pick(wins_w)
        isempty(obs) && continue
        push!(rows, (c, rat, length(obs), round(Int, median(wins)), round(Int, median(wins_w)),
                     r2(p_hmm, obs), ml === nothing ? missing : r2(p_ml, obs), r2(run_raw, obs),
                     within_r2(p_hmm, obs, sid), ml === nothing ? missing : within_r2(p_ml, obs, sid),
                     within_r2(run_within, obs, sid)))
        @info "$c $rat" rows[end, :]
    end
end
CSV.write(joinpath(OUT, "predict_rt_by_rat.csv"), rows)
show(rows; allrows=true)
println()
println(combine(groupby(rows, :cohort), [:hmm, :running, :hmm_within, :running_within] .=> median,
                [:hmm, :running] => ((a, b) -> count(a .> b)) => :hmm_beats_running,
                [:hmm_within, :running_within] => ((a, b) -> count(a .> b)) => :hmm_beats_running_within, nrow))
