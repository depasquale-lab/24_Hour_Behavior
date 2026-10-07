#=
Hour-of-day occupancy of each K = 4 tied-full state, input to PlotFigure5.jl A-C.
Forward-backward posteriors per trial, binned by whole hours from lights-on
(07:30). States carry the paper-wide accuracy rank from state_parameters_long.csv.

Writes results/final_ddmhmms/state_occupancy_by_hour.csv:
  rat, state, acc_rank, hour (0-23 from lights-on), occupancy, n_trials
=#

using Pkg
Pkg.activate("notebooks")

using Dates, Statistics, StatsBase, Random, Printf
using CSV, DataFrames
using BSON
using BSON: @load
using DriftDiffusionModels
using HiddenMarkovModels

Random.seed!(20260915)

# Struct stubs so BSON can rehydrate all three saved schemas
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

_gamma(ret) = ret isa AbstractMatrix ? ret :
              ret isa Tuple ? _gamma(ret[1]) :
              hasproperty(ret, :γ) ? getfield(ret, :γ) : error("no γ")

const LIGHTS_ON = 7.5   # hours, clock time of lights-on

const DATA_FILE = joinpath("data", "processed_rat_data.csv.gz")
rat_df = CSV.read(DATA_FILE, DataFrame)
rat_df = rat_df[rat_df.daily .== "24 hr", :]
replace!(rat_df[!, :choose_right], 0 => -1)
side_mapping = Dict("right" => 1, "left" => -1)
DataFrames.transform!(rat_df,
    :correct_side => ByRow(cs -> get(side_mapping, cs, missing)) => :correct_side_numeric)
rat_df.dt = DateTime.(rat_df.trial_datetime, dateformat"yyyy-mm-dd HH:MM:SS")


"""
    data_for_rat(rat)

Same sequence construction the fits used: one sequence per calendar day, trials
in table order within a day. Returns the observations, the sequence ends, the
animal's rows and the permutation `ord` mapping posterior column t to row
`ord[t]` of `sub`.
"""
function data_for_rat(rat::AbstractString)
    sub = rat_df[rat_df.name .== rat, :]
    dates = [Date(split(dt)[1]) for dt in sub.trial_datetime]
    ud = sort(unique(dates))
    by_date = Vector{Vector{DDMResult}}(); rows = Vector{Vector{Int}}()
    for d in ud
        idx = findall(dates .== d); isempty(idx) && continue
        push!(by_date, [DDMResult(rt, ch, st) for (rt, ch, st) in
              zip(sub.rt[idx], sub.choose_right[idx], sub.correct_side_numeric[idx])])
        push!(rows, idx)
    end
    seq_ends = cumsum([length(s) for s in by_date])
    return reduce(vcat, by_date), seq_ends, sub, reduce(vcat, rows)
end

fitdir = joinpath("results", "final_ddmhmms", "final_ddmhmms")
files = sort(filter(f -> endswith(f, ".bson"), readdir(fitdir)))

params = CSV.read(joinpath("results", "final_ddmhmms", "state_parameters_long.csv"), DataFrame)
transform!(groupby(params, :rat), :acc => (a -> ordinalrank(a; rev=true)) => :acc_rank)
rank_of = Dict((String(r), s) => k for (r, s, k) in zip(params.rat, params.state, params.acc_rank))

rows = DataFrame()
for f in files
    hmm, rat, Ks, _ = load_fit(joinpath(fitdir, f))
    obs, seq_ends, sub, ord = data_for_rat(rat)
    γ = _gamma(HiddenMarkovModels.forward_backward(hmm, obs; seq_ends=seq_ends))
    hr = [floor(Int, mod(Dates.hour(t) - LIGHTS_ON, 24)) for t in sub.dt[ord]]
    for h in 0:23
        m = hr .== h
        n = count(m)
        for k in 1:Ks
            push!(rows, (rat=rat, state=k, acc_rank=rank_of[(rat, k)], hour=h,
                         occupancy=n > 0 ? mean(γ[k, m]) : NaN, n_trials=n))
        end
    end
    println(rat, ": ", length(obs), " trials")
end

CSV.write(joinpath("results", "final_ddmhmms", "state_occupancy_by_hour.csv"), rows)
