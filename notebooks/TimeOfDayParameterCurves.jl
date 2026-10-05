#=
DDM parameters across the day: time-of-day DDM vs DDM-HMM (reviewer figure).

For each 24 hr rat:
  - DDM-HMM (final K = 4 tied-full fits): per-trial parameter = posterior-weighted
    state parameter, Σ_k γ_kt θ_k, from the smoothed state posteriors.
  - Time-of-day DDM (VARY_TAU=1, all-data fit, best-BIC H): θ(hour).
Both are binned by hour from lights-on (07:30). Also reports how much of the
HMM's trial-level parameter variance hour of day explains (between-hour variance
/ total variance, trial-weighted).

Writes results/tod_ddm_vary_tau/param_curves_long.csv and param_variance_by_hour.csv.
=#

using Pkg
Pkg.activate("notebooks")

using Dates, Statistics, CSV, DataFrames, Printf
using BSON
using BSON: @load
using DriftDiffusionModels
using HiddenMarkovModels

# --- struct stubs so BSON can rehydrate all three saved schemas ---
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
Base.length(h::TiedPriorHMM) = length(h.init)
HiddenMarkovModels.initialization(h::TiedPriorHMM) = h.init
HiddenMarkovModels.transition_matrix(h::TiedPriorHMM) = h.trans
HiddenMarkovModels.obs_distributions(h::TiedPriorHMM) = h.dists

struct DDMEmission{T<:Real}
    B::T
    v::T
    a₀::T
    τ::T
end

const DDM_PARAMS = (:B, :v, :a₀, :τ)
softmax1(x) = (e = exp.(x .- maximum(x)); e ./ sum(e))
sigmoid_(y) = 1 / (1 + exp(-y))

function build_tied_idx_map(K::Int, tied::Vector{Symbol})
    free = Symbol[p for p in DDM_PARAMS if !(p in tied)]
    nfree = length(free)
    idx = Matrix{Int}(undef, K, length(DDM_PARAMS))
    for (pi, p) in enumerate(DDM_PARAMS)
        if p in tied
            pos = K * nfree + findfirst(==(p), tied)
            for k in 1:K; idx[k, pi] = pos; end
        else
            pos = findfirst(==(p), free)
            for k in 1:K; idx[k, pi] = (k - 1) * nfree + pos; end
        end
    end
    return idx
end

function θ_to_priorhmm(θ::Vector{Float64}, K::Int, tied::Vector{Symbol})
    init = softmax1(θ[1:K])
    tl = θ[(K + 1):(K + K * K)]
    trans = Matrix{Float64}(undef, K, K)
    for k in 1:K
        trans[k, :] .= softmax1(tl[((k - 1) * K + 1):(k * K)])
    end
    ddm_y = θ[(K + K * K + 1):end]
    idx = build_tied_idx_map(K, tied)
    dists = [DriftDiffusionModel(;
        B=exp(ddm_y[idx[k, 1]]), v=exp(ddm_y[idx[k, 2]]),
        a₀=sigmoid_(ddm_y[idx[k, 3]]), τ=exp(ddm_y[idx[k, 4]])) for k in 1:K]
    return PriorHMM(init, trans, dists, 1, 1)
end

# Fit loading copied from ExtractStateParameters.jl.
_gamma(ret) = ret isa AbstractMatrix ? ret :
              ret isa Tuple ? _gamma(ret[1]) :
              hasproperty(ret, :γ) ? getfield(ret, :γ) : error("no γ")

fitdir = joinpath("results", "final_ddmhmms", "final_ddmhmms")
files = sort(filter(f -> endswith(f, ".bson"), readdir(fitdir)))

"Return (hmm, rat, K, logL) regardless of which of the three schemas was saved."
function load_fit(path::AbstractString)
    ks = keys(BSON.parse(path))
    if :hmm_dists in ks
        @load path hmm_init hmm_trans hmm_dists best_logL rat K_STATES
        dists = [DriftDiffusionModel(; B=d.B, v=d.v, a₀=d.a₀, τ=d.τ) for d in hmm_dists]
        hmm = PriorHMM(Vector{Float64}(hmm_init), Matrix{Float64}(hmm_trans), dists, 1, 1)
        return hmm, String(rat), Int(K_STATES), Float64(best_logL)
    elseif :fit in ks
        @load path fit rat K_STATES
        t = fit.hmm
        hmm = PriorHMM(
            Vector{Float64}(t.init), Matrix{Float64}(t.trans), deepcopy(t.dists), 1, 1
        )
        return hmm, String(rat), Int(K_STATES), Float64(fit.logL)
    elseif :best_direct in ks
        @load path best_θ best_direct K TIED RAT
        hmm = θ_to_priorhmm(Vector{Float64}(best_θ), Int(K), Symbol[s for s in TIED])
        return hmm, String(RAT), Int(K), Float64(best_direct)
    else
        error("unknown schema in $path: $(collect(ks))")
    end
end


const DATA_FILE = joinpath("data", "processed_rat_data.csv.gz")
rat_df = CSV.read(DATA_FILE, DataFrame; types=Dict(:name => String))
rat_df = rat_df[rat_df.daily .== "24 hr", :]
replace!(rat_df[!, :choose_right], 0 => -1)
rat_df.s = [cs == "right" ? 1 : -1 for cs in rat_df.correct_side]

const LIGHTS_ON = 7.5
clock_hour(dt) = (p = split(dt); length(p) < 2 ? 0.0 :
                  sum(parse.(Float64, split(p[2], ':')) .* (1, 1 / 60, 1 / 3600)))
from_lights_on(h) = mod(h - LIGHTS_ON, 24)

"Trials in session order (calendar days), matching the HMM fits."
function data_for_rat(rat)
    sub = rat_df[rat_df.name .== rat, :]
    dates = [Date(split(dt)[1]) for dt in sub.trial_datetime]
    groups = [findall(dates .== d) for d in sort(unique(dates))]
    ord = reduce(vcat, groups)
    obs = [DDMResult(sub.rt[i], sub.choose_right[i], sub.s[i]) for i in ord]
    hours = clock_hour.(String.(sub.trial_datetime[ord]))
    return obs, cumsum(length.(groups)), hours
end

const PARAMS = [:B, :v, :a₀, :τ]
tod = CSV.read(joinpath("results", "tod_ddm_vary_tau", "tod_ddm_summary.csv"), DataFrame;
               types=Dict(:rat => String))
const TOD_DIR = joinpath("results", "tod_ddm_vary_tau", "per_rat")

curves = DataFrame()
varrows = DataFrame()
for f in files
    hmm, rat, _, _ = load_fit(joinpath(fitdir, f))
    obs, seq_ends, hours = data_for_rat(rat)
    γ = _gamma(HiddenMarkovModels.forward_backward(hmm, obs; seq_ends=seq_ends))
    θk = [[getfield(d, p) for d in hmm.dists] for p in PARAMS]   # per param, K-vector
    θk[2] = abs.(θk[2])                                         # drift is a magnitude
    θt = [vec(θ' * γ) for θ in θk]                               # per param, per trial

    t = tod[(tod.rat .== rat) .& (tod.fold .== 0), :]
    H = t.H[argmin(t.bic)]
    fits = BSON.load(joinpath(TOD_DIR, "$(rat)_tod_fits.bson"))[:fits]
    β = fits[0][H].β

    hbin = floor.(Int, from_lights_on.(hours))                  # 0..23
    for (pi, p) in enumerate(PARAMS)
        x = θt[pi]
        μ = mean(x)
        between = sum(count(==(b), hbin) * (mean(x[hbin .== b]) - μ)^2 for b in unique(hbin))
        push!(varrows, (rat=rat, param=String(p), frac_var_hour=between / sum((x .- μ) .^ 2),
                        hmm_sd=std(x), tod_H=H))
        for b in 0:23
            m = hbin .== b
            any(m) || continue
            xb = x[m]
            hc = mod(b + 0.5 + LIGHTS_ON, 24)
            push!(curves, (rat=rat, param=String(p), hour=b + 0.5, n=count(m),
                           hmm_mean=mean(xb), hmm_q10=quantile(xb, 0.1), hmm_q90=quantile(xb, 0.9),
                           tod=regression_params(β, fourier_basis(hc; n_harmonics=H))[pi]))
        end
    end
    @info "rat $rat done (H = $H)"
end

out = joinpath("results", "tod_ddm_vary_tau")
CSV.write(joinpath(out, "param_curves_long.csv"), curves)
CSV.write(joinpath(out, "param_variance_by_hour.csv"), varrows)

println("Fraction of the HMM's trial-level parameter variance explained by hour of day")
for g in groupby(varrows, :param)
    @printf("  %-3s median %.3f  (range %.3f – %.3f)\n", g.param[1], median(g.frac_var_hour),
            minimum(g.frac_var_hour), maximum(g.frac_var_hour))
end
println("\nCorrelation across hours, HMM hourly mean vs time-of-day curve")
for g in groupby(curves, :param)
    r = [cor(s.hmm_mean, s.tod) for s in groupby(g, :rat)]
    @printf("  %-3s median r = %.2f  (%d/%d rats r > 0.5)\n", g.param[1], median(r), count(>(0.5), r), length(r))
end
