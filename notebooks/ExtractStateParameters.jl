#=
Extract per-rat, per-state DDM parameters and posterior-weighted behaviour from
the final K=4 tied-full DDM-HMM fits.

Writes `state_parameters_long.csv` (one row per rat × state) plus the
state-conditioned psychometric and chronometric curves and psychometric slopes,
which are the inputs to PlotStateComparability.jl and PlotStateBehavior.jl. The three saved BSON schemas in results/final_ddmhmms
(direct-gradient θ, constrained Baum–Welch `fit`, and the "rich" constrained form)
are all handled here so the fits do not need to be re-run or re-converted.
=#

using Pkg
Pkg.activate("notebooks")

using Dates, Statistics, CSV, DataFrames
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

const DATA_FILE = joinpath("data", "processed_rat_data.csv.gz")
rat_df = CSV.read(DATA_FILE, DataFrame)
rat_df = rat_df[rat_df.daily .== "24 hr", :]
replace!(rat_df[!, :choose_right], 0 => -1)
side_mapping = Dict("right" => 1, "left" => -1)
DataFrames.transform!(rat_df,
    :correct_side => ByRow(cs -> get(side_mapping, cs, missing)) => :correct_side_numeric)

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

# Evidence bins for the state-conditioned psychometric / chronometric curves.
# Signed bins for the psychometric, folded bins for the chronometric. Edges are
# fixed across animals so the curves can be averaged bin-by-bin.
const PSY_EDGES = [-Inf, -16.0, -11.0, -7.0, -4.0, -1.5, 1.5, 4.0, 7.0, 11.0, 16.0, Inf]
const CHR_EDGES = [0.0, 1.5, 4.0, 7.0, 11.0, 16.0, Inf]

"Index of the bin containing `x`, or 0 if it falls outside `edges`."
function bin_index(x::Real, edges::Vector{Float64})
    for i in 1:(length(edges) - 1)
        if x >= edges[i] && x < edges[i + 1]
            return i
        end
    end
    return 0
end

"""
    weighted_logistic(x, y, w)

Posterior-weighted logistic regression of `y` (0/1) on `x`, fit by IRLS.
Returns `(intercept, slope)`. The slope is the psychometric sensitivity in units
of P(right) per flash difference, computed from trial assignments rather than
from the fitted DDM parameters.
"""
function weighted_logistic(x::Vector{Float64}, y::Vector{Float64}, w::Vector{Float64})
    β = [0.0, 0.0]
    X = hcat(ones(length(x)), x)
    for _ in 1:50
        η = X * β
        p = 1.0 ./ (1.0 .+ exp.(-η))
        g = max.(p .* (1 .- p), 1e-9)          # guard against separation
        z = η .+ (y .- p) ./ g
        W = w .* g
        XtW = X' * (W .* X)
        XtWz = X' * (W .* z)
        βnew = try
            XtW \ XtWz
        catch
            return (β[1], β[2])
        end
        maximum(abs.(βnew .- β)) < 1e-8 && return (βnew[1], βnew[2])
        β = βnew
    end
    return (β[1], β[2])
end

rows = DataFrame()
transrows = DataFrame()
psyrows = DataFrame()
chrrows = DataFrame()
sloperows = DataFrame()
for f in files
    hmm, rat, Ks, logL = load_fit(joinpath(fitdir, f))
    dists = hmm.dists
    A = hmm.trans

    obs, seq_ends, sub, ord = data_for_rat(rat)
    γ = _gamma(HiddenMarkovModels.forward_backward(hmm, obs; seq_ends=seq_ends))

    correct = Float64.(sub.correct[ord])
    rt = Float64.(sub.rt[ord])
    chooseR = Float64.(sub.choose_right[ord] .== 1)
    absdf = Float64.(abs.(sub.delta_flashes[ord]))
    it = Float64.(sub.init_time[ord])
    dflash = Float64.(sub.delta_flashes[ord])
    psybin = [bin_index(d, PSY_EDGES) for d in dflash]
    chrbin = [bin_index(d, CHR_EDGES) for d in absdf]

    for k in 1:Ks
        d = dists[k]
        w = γ[k, :]
        W = sum(w)
        mrt = sum(w .* rt) / W
        push!(
            rows,
            (
                rat=rat,
                state=k,
                B=d.B,
                v=d.v,
                a0=d.a₀,
                tau=d.τ,
                occupancy=W / length(w),
                p_self=A[k, k],
                dwell=1 / (1 - A[k, k]),
                acc=sum(w .* correct) / W,
                rt_mean=mrt,
                rt_sd=sqrt(max(0.0, sum(w .* (rt .- mrt) .^ 2) / W)),
                p_right=sum(w .* chooseR) / W,
                abs_df=sum(w .* absdf) / W,
                init_time=sum(w .* it) / W,
                n_trials=length(w),
                logL=logL,
            );
            promote=true,
        )
        for j in 1:Ks
            push!(transrows, (rat=rat, from=k, to=j, p=A[k, j]); promote=true)
        end

        # State-conditioned psychometric curve: P(choose right) against signed
        # evidence, with each trial weighted by its posterior probability of
        # belonging to this state.
        for b in 1:(length(PSY_EDGES) - 1)
            m = psybin .== b
            any(m) || continue
            wb = w[m]
            sw = sum(wb)
            sw > 0 || continue
            push!(
                psyrows,
                (
                    rat=rat,
                    state=k,
                    bin=b,
                    delta_flashes=sum(wb .* dflash[m]) / sw,
                    p_right=sum(wb .* chooseR[m]) / sw,
                    n_eff=sw,
                );
                promote=true,
            )
        end

        # State-conditioned chronometric curve: mean RT against unsigned evidence.
        for b in 1:(length(CHR_EDGES) - 1)
            m = chrbin .== b
            any(m) || continue
            wb = w[m]
            sw = sum(wb)
            sw > 0 || continue
            push!(
                chrrows,
                (
                    rat=rat,
                    state=k,
                    bin=b,
                    abs_delta_flashes=sum(wb .* absdf[m]) / sw,
                    rt=sum(wb .* rt[m]) / sw,
                    n_eff=sw,
                );
                promote=true,
            )
        end

        b0, b1 = weighted_logistic(dflash, chooseR, w)
        push!(
            sloperows,
            (rat=rat, state=k, intercept=b0, slope=b1, n_eff=W);
            promote=true,
        )
    end
    println("done $rat K=$Ks")
end

CSV.write(joinpath("results", "final_ddmhmms", "state_parameters_long.csv"), rows)
CSV.write(joinpath("results", "final_ddmhmms", "state_transitions_long.csv"), transrows)
CSV.write(joinpath("results", "final_ddmhmms", "state_psychometric_long.csv"), psyrows)
CSV.write(joinpath("results", "final_ddmhmms", "state_chronometric_long.csv"), chrrows)
CSV.write(joinpath("results", "final_ddmhmms", "state_psychometric_slopes.csv"), sloperows)
println("wrote $(nrow(rows)) state rows")
