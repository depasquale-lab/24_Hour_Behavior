#=
RT autocorrelation: real data vs. DDM-HMM vs. multilevel DDM, both cohorts.

Every session is re-simulated at its own length and stimulus sequence from each
rat's full-data fit:
  DDM-HMM (K = 4)  session  results/session_cohort/full_fits/<rat>_K4_daily.bson
                   24 hr    results/final_ddmhmms/final_ddmhmms/<rat>_K4_tied-full.bson
  multilevel DDM   results/eddm_exact{_daily,}/eddm_hyperparams_by_rat.csv
The HMM restarts from its initial distribution each session; the MLDDM draws
trial parameters iid. ACF is computed within session (lags 1..20) and averaged
over sessions per rat, identically for real and simulated data. RTs are sampled
exactly by inverting the first-passage CDF.

Writes results/session_cohort/:
  rt_acf_by_rat.csv    cohort, rat, source (real/hmm/mlddm), lag, acf
  rt_acf_error.csv     per rat: mean |model - real| ACF over lags 1..10, per model
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Dates
using Statistics
using StatsBase: autocor
using Random
logistic(x) = 1 / (1 + exp(-x))
using DriftDiffusionModels
using HiddenMarkovModels
using BSON
using BSON: @load

const OUT = joinpath("results", "session_cohort")
const LAGS = 1:20
const ERR_LAGS = 1:10
const N_REP = 20       # simulated replicates of each rat's full session set
const N_BANK = 1000    # MLDDM trial-parameter draws
const MIN_SESSION = maximum(LAGS) + 2

rat_df = CSV.read(joinpath("data", "processed_rat_data.csv.gz"), DataFrame)
replace!(rat_df[!, :choose_right], 0 => -1)
rat_df.s = [cs == "right" ? 1 : -1 for cs in rat_df.correct_side]

"Per session: (rt, stimulus) vectors, in date order."
function sessions_for(rat, cohort)
    sub = rat_df[(rat_df.name .== rat) .& (rat_df.daily .== (cohort == "24hr" ? "24 hr" : "daily")), :]
    dates = [Date(split(dt)[1]) for dt in sub.trial_datetime]
    [(sub.rt[i], sub.s[i]) for i in (findall(==(d), dates) for d in sort(unique(dates)))]
end

# Exact RT sampler: decision-time CDF on a grid, for one (B, v, a₀) and stimulus.

const U_GRID = vcat(range(1e-4, 2.0; length=1500), range(2.0, 30.0; length=1500)[2:end])

struct RTSampler
    u::Vector{Float64}
    cdf::Vector{Float64}
end

function RTSampler(B, v, a₀, s)
    vt = s * abs(v)
    f = [wfpt(u, vt, B, a₀, 0.0) + wfpt(u, -vt, B, 1 - a₀, 0.0) for u in U_GRID]
    c = cumsum(vcat(0.0, 0.5 .* (f[1:end-1] .+ f[2:end]) .* diff(U_GRID)))
    RTSampler(vcat(0.0, U_GRID[2:end]), c ./ c[end])
end

function Base.rand(rng::AbstractRNG, r::RTSampler)
    x = rand(rng)
    i = clamp(searchsortedfirst(r.cdf, x), 2, length(r.cdf))
    c0, c1 = r.cdf[i - 1], r.cdf[i]
    r.u[i - 1] + (c1 > c0 ? (x - c0) / (c1 - c0) : 0.0) * (r.u[i] - r.u[i - 1])
end

si(s) = s == 1 ? 1 : 2

# Simulators: given a session's stimulus sequence, return simulated RTs.

function hmm_simulator(hmm)
    d = obs_distributions(hmm)
    samp = [RTSampler(x.B, x.v, x.a₀, s) for x in d, s in (1, -1)]
    τ = [x.τ for x in d]
    init, A = initialization(hmm), transition_matrix(hmm)
    function (rng, stims)
        z = sample_cat(rng, init)
        map(enumerate(stims)) do (t, s)
            t > 1 && (z = sample_cat(rng, view(A, z, :)))
            τ[z] + rand(rng, samp[z, si(s)])
        end
    end
end

function mlddm_simulator(m, σ0, rng)
    draws = [(exp.(m[1:3] .+ σ0[1:3] .* randn(rng, 3))..., logistic(m[4] + σ0[4] * randn(rng)))
             for _ in 1:N_BANK]                      # (B, τ, v, a₀)
    samp = [RTSampler(B, v, a, s) for (B, τ, v, a) in draws, s in (1, -1)]
    τ = [d[2] for d in draws]
    (rng, stims) -> map(stims) do s
        j = rand(rng, 1:N_BANK)
        τ[j] + rand(rng, samp[j, si(s)])
    end
end

function sample_cat(rng, p)
    x, c = rand(rng), 0.0
    for k in eachindex(p)
        c += p[k]
        x <= c && return k
    end
    lastindex(p)
end

session_acf(rts_by_session) = vec(mean(reduce(hcat, [autocor(r, LAGS) for r in rts_by_session if length(r) >= MIN_SESSION]); dims=2))

# Fit loaders

function load_daily_hmm(rat)
    p = joinpath(OUT, "full_fits", "$(rat)_K4_daily.bson")
    isfile(p) || return nothing
    @load p hmm
    hmm
end

# final 24 hr fits: same three-schema loader as ExtractStateParameters.jl
struct ConstrainedDDMHMMFit
    hmm::Any
    tied::Vector{Symbol}
    logL::Float64
    logL_evolution::Vector{Float64}
    n_trials::Int
    n_free_params::Int
    bic::Float64
end
mutable struct TiedPriorHMM{T<:Real} <: HiddenMarkovModels.AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dists::Vector{DriftDiffusionModel}
    α_trans::Matrix{T}
    α_init::Vector{T}
    tied::Vector{Symbol}
end
struct DDMEmission{T<:Real}
    B::T
    v::T
    a₀::T
    τ::T
end
softmax1(x) = (e = exp.(x .- maximum(x)); e ./ sum(e))

function θ_to_hmm(θ, K, tied)
    params = (:B, :v, :a₀, :τ)
    free = [p for p in params if !(p in tied)]
    idx(k, p) = p in tied ? K * length(free) + findfirst(==(p), tied) : (k - 1) * length(free) + findfirst(==(p), free)
    init = softmax1(θ[1:K])
    tl = θ[(K + 1):(K + K^2)]
    trans = reduce(vcat, [softmax1(tl[((k - 1) * K + 1):(k * K)])' for k in 1:K])
    y = θ[(K + K^2 + 1):end]
    dists = [DriftDiffusionModel(; B=exp(y[idx(k, :B)]), v=exp(y[idx(k, :v)]),
                                 a₀=logistic(y[idx(k, :a₀)]), τ=exp(y[idx(k, :τ)])) for k in 1:K]
    PriorHMM(init, trans, dists, 1, 1)
end

function load_24hr_hmm(rat)
    p = joinpath("results", "final_ddmhmms", "final_ddmhmms", "$(rat)_K4_tied-full.bson")
    isfile(p) || return nothing
    ks = keys(BSON.parse(p))
    if :hmm_dists in ks
        @load p hmm_init hmm_trans hmm_dists
        PriorHMM(Vector{Float64}(hmm_init), Matrix{Float64}(hmm_trans),
                 [DriftDiffusionModel(; B=d.B, v=d.v, a₀=d.a₀, τ=d.τ) for d in hmm_dists], 1, 1)
    elseif :fit in ks
        @load p fit
        PriorHMM(Vector{Float64}(fit.hmm.init), Matrix{Float64}(fit.hmm.trans), deepcopy(fit.hmm.dists), 1, 1)
    else
        @load p best_θ K TIED
        θ_to_hmm(Vector{Float64}(best_θ), Int(K), Symbol[s for s in TIED])
    end
end

hyper(c) = CSV.read(joinpath("results", c == "24hr" ? "eddm_exact" : "eddm_exact_daily",
                             "eddm_hyperparams_by_rat.csv"), DataFrame; types=Dict(:rat_name => String))

# Run

by_rat = CSV.read(joinpath(OUT, "heldout_by_rat.csv"), DataFrame; types=Dict(:rat => String))
acf_rows = DataFrame(; cohort=String[], rat=String[], source=String[], lag=Int[], acf=Float64[])
rng = MersenneTwister(1)

for c in ["daily", "24hr"]
    H = hyper(c)
    for rat in sort(unique(by_rat.rat[by_rat.cohort .== c]))
        sess = sessions_for(rat, c)
        stims = last.(sess)
        push_acf!(src, a) = append!(acf_rows, DataFrame(; cohort=c, rat, source=src, lag=collect(LAGS), acf=a))
        push_acf!("real", session_acf(first.(sess)))
        sims = Pair{String,Any}[]
        hmm = c == "24hr" ? load_24hr_hmm(rat) : load_daily_hmm(rat)
        hmm === nothing ? @warn("$c $rat: no DDM-HMM fit") : push!(sims, "hmm" => hmm_simulator(hmm))
        h = H[H.rat_name .== rat, :]
        if nrow(h) == 1
            m = [h.m_uB[1], h.m_uτ[1], h.m_uv[1], h.m_ua0[1]]
            σ0 = exp.([h.logσ0_uB[1], h.logσ0_uτ[1], h.logσ0_uv[1], h.logσ0_ua0[1]])
            push!(sims, "mlddm" => mlddm_simulator(m, σ0, rng))
        else
            @warn "$c $rat: no MLDDM fit"
        end
        for (src, sim) in sims
            push_acf!(src, vec(mean(reduce(hcat, [session_acf([sim(rng, st) for st in stims]) for _ in 1:N_REP]); dims=2)))
        end
        @info "$c $rat: $(length(sess)) sessions, $(sum(length, stims)) trials, models: $(first.(sims))"
    end
end
CSV.write(joinpath(OUT, "rt_acf_by_rat.csv"), acf_rows)

err = combine(groupby(acf_rows[in.(acf_rows.lag, Ref(ERR_LAGS)), :], [:cohort, :rat])) do d
    real = d.acf[d.source .== "real"]
    e(src) = any(d.source .== src) ? mean(abs.(d.acf[d.source .== src] .- real)) : missing
    (hmm=e("hmm"), mlddm=e("mlddm"))
end
CSV.write(joinpath(OUT, "rt_acf_error.csv"), err)
println(err)
