#=
External validation of the K = 4 tied-full DDM-HMM states against trial
initiation. The fits see only (RT, choice, correct side), so ITI and trial rate
are out-of-model predictions of the posterior.

Timing: `init_time[t]` is the latency after trial t, before trial t+1
(gap[t] = init_time[t] + rt[t+1] + c, c ≈ 5.8 s hardware overhead). So
    iti_pre[t]  = init_time[t-1]   wait before this trial
    iti_post[t] = init_time[t]     wait before the next one
Analyses use session-interior trials only, where both are defined. A session is
one calendar day, the HMM's sequence unit.

Writes to results/final_ddmhmms (inputs to PlotStateEngagementITI.jl):
  state_engagement_summary.csv   rat x state: posterior-weighted ITI and trial rate
  state_engagement_deciles.csv   rat x rank x ITI decile: P(state | ITI)
  state_engagement_stats.csv     rat-level correlations and AUC with a circular-shift null
  state_bout_profile.csv         rat x rank x position in a work bout
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

# Analysis constants
const PAUSE_S = 300.0   # a break of more than five minutes ends a work bout
const RATE_W = 5        # half-width, in trials, of the local trial-rate window
const BOUT_MIN = 20     # shortest bout entering the bout-position profile
const BOUT_POS = 10     # positions resolved from each end of a bout
const NPERM = 10_000    # circular shifts per rat; p floor 1/(NPERM+1) must sit below 0.05/54
const NDEC = 10         # ITI quantile bins

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

"Half-open session ranges (first:last) implied by cumulative `seq_ends`."
function session_ranges(seq_ends::Vector{Int})
    starts = [1; seq_ends[1:(end - 1)] .+ 1]
    return [starts[i]:seq_ends[i] for i in eachindex(seq_ends)]
end

"Fill `out[rng]` with trials per minute in a window of ±`RATE_W` trials, clipped
to the session so no window straddles a sequence boundary."
function local_rate!(out::Vector{Float64}, t_sec::Vector{Float64}, rng::UnitRange{Int})
    for i in rng
        lo = max(first(rng), i - RATE_W); hi = min(last(rng), i + RATE_W)
        hi > lo || continue
        span = max(t_sec[hi] - t_sec[lo], 1.0)   # datetimes are second-resolution
        out[i] = 60.0 * (hi - lo) / span
    end
    return out
end

"Weighted mean of `x`, ignoring non-finite entries."
function wmean(x::AbstractVector, w::AbstractVector)
    s = 0.0; sw = 0.0
    @inbounds for i in eachindex(x)
        isfinite(x[i]) || continue
        s += w[i] * x[i]; sw += w[i]
    end
    return sw > 0 ? s / sw : NaN
end

"Weighted `q`-quantile of `x` by the step-function definition."
function wquantile(x::AbstractVector, w::AbstractVector, q::Real)
    keep = findall(i -> isfinite(x[i]) && w[i] > 0, eachindex(x))
    isempty(keep) && return NaN
    p = sortperm(x[keep]); xs = x[keep][p]; ws = w[keep][p]
    c = cumsum(ws) ./ sum(ws)
    return xs[something(findfirst(>=(q), c), length(xs))]
end

"Mann-Whitney AUC of score `s` for the binary label `y`, computed from ranks."
function auc_ranks(s::AbstractVector{Float64}, y::BitVector)
    n1 = count(y); n0 = length(y) - n1
    (n1 == 0 || n0 == 0) && return NaN
    r = tiedrank(s)
    return (sum(r[y]) - n1 * (n1 + 1) / 2) / (n1 * n0)
end

"AUC where the score is a permutation `idx` of a precomputed rank vector."
function auc_from_rank(rs::Vector{Float64}, idx::Vector{Int}, y::BitVector, n1::Int, n0::Int)
    s = 0.0
    @inbounds for i in eachindex(y)
        y[i] && (s += rs[idx[i]])
    end
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)
end

"""
    shift_index(blocks, n)

Index vector that circularly shifts each block of `blocks` by an independent
random offset. Shifting within a session preserves the autocorrelation of the
state trajectory and the marginal distribution of the posterior, and destroys
only its alignment to the trial timeline, which is what the null requires.
"""
function shift_index(blocks::Vector{UnitRange{Int}}, n::Int)
    idx = Vector{Int}(undef, n)
    for b in blocks
        L = length(b); off = rand(0:(L - 1))
        @inbounds for (j, i) in enumerate(b)
            idx[i] = first(b) + mod(j - 1 + off, L)
        end
    end
    return idx
end

"Subtract the mean of each hour-of-day bin, so a correlation is within hour."
function detrend_hour(x::Vector{Float64}, hour::Vector{Int})
    r = copy(x)
    for h in 0:23
        m = hour .== h
        any(m) && (r[m] .-= mean(x[m]))
    end
    return r
end

fitdir = joinpath("results", "final_ddmhmms", "final_ddmhmms")
files = sort(filter(f -> endswith(f, ".bson"), readdir(fitdir)))

# Accuracy rank of each animal's states, the paper-wide labelling.
params = CSV.read(joinpath("results", "final_ddmhmms", "state_parameters_long.csv"), DataFrame)
transform!(groupby(params, :rat), :acc => (a -> ordinalrank(a; rev=true)) => :acc_rank)
rank_of = Dict((String(r), s) => k for (r, s, k) in zip(params.rat, params.state, params.acc_rank))

summary_rows = DataFrame()
decile_rows = DataFrame()
stat_rows = DataFrame()
bout_rows = DataFrame()

for f in files
    hmm, rat, Ks, logL = load_fit(joinpath(fitdir, f))
    obs, seq_ends, sub, ord = data_for_rat(rat)
    γ = _gamma(HiddenMarkovModels.forward_backward(hmm, obs; seq_ends=seq_ends))
    T = size(γ, 2)

    # Trial covariates in posterior order.
    init = Float64.(sub.init_time[ord])
    tsec = Float64.(Dates.value.(sub.dt[ord] .- DateTime(2000, 1, 1))) ./ 1000
    hour = Int.(Dates.hour.(sub.dt[ord]))
    sessions = session_ranges(seq_ends)

    iti_pre = fill(NaN, T); iti_post = fill(NaN, T); rate = fill(NaN, T)
    for rng in sessions
        length(rng) >= 3 || continue
        iti_pre[(first(rng) + 1):last(rng)] .= init[first(rng):(last(rng) - 1)]
        iti_post[first(rng):(last(rng) - 1)] .= init[first(rng):(last(rng) - 1)]
        local_rate!(rate, tsec, rng)
    end

    # Session-interior trials: both waits observed, same sequence.
    core = findall(t -> isfinite(iti_pre[t]) && isfinite(iti_post[t]), 1:T)
    incore = falses(T); incore[core] .= true
    lpre = log10.(max.(iti_pre[core], 0.01))
    lpost = log10.(max.(iti_post[core], 0.01))
    g = γ[:, core]
    ranks = [rank_of[(rat, k)] for k in 1:Ks]

    # Expected accuracy rank: 1 = certainly best state, K = certainly worst.
    erank = vec(sum(ranks .* g; dims=1))

    # Rank transforms: a circular shift permutes ranks, so Spearman and AUC are cheap.
    r_erank = tiedrank(erank)
    r_pre = tiedrank(lpre)
    r_post = tiedrank(lpost)
    r_rate = tiedrank(rate[core])
    rho_state = [cor(tiedrank(g[k, :]), r_pre) for k in 1:Ks]

    # Per-state summary
    for k in 1:Ks
        w = g[k, :]
        push!(summary_rows, (
            rat=rat, state=k, acc_rank=ranks[k],
            occupancy=mean(γ[k, :]),
            n_eff=sum(w),
            rho_pre=rho_state[k],
            iti_pre_geo=10.0^wmean(lpre, w),
            iti_post_geo=10.0^wmean(lpost, w),
            iti_pre_med=wquantile(iti_pre[core], w, 0.5),
            iti_post_med=wquantile(iti_post[core], w, 0.5),
            trial_rate=wmean(rate[core], w),
            p_pause_next=wmean(Float64.(iti_post[core] .> PAUSE_S), w),
        ); promote=true)
    end

    # P(state | ITI decile): the reverse conditional
    edges = quantile(lpre, range(0, 1; length=NDEC + 1))
    bin = clamp.(searchsortedlast.(Ref(edges[2:(end - 1)]), lpre) .+ 1, 1, NDEC)
    for b in 1:NDEC
        m = bin .== b
        any(m) || continue
        for k in 1:Ks
            push!(decile_rows, (
                rat=rat, state=k, acc_rank=ranks[k], decile=b,
                iti_pre=10.0^mean(lpre[m]), posterior=mean(g[k, m]), n=count(m),
            ); promote=true)
        end
    end

    # Correlations and AUC against a within-session circular-shift null
    # Both series are autocorrelated; the shift keeps that and breaks only alignment.
    blocks = UnitRange{Int}[]
    let pos = 1
        for rng in sessions
            n = count(@view incore[rng])     # core is sorted and session-contiguous
            n > 1 && push!(blocks, pos:(pos + n - 1))
            pos += n
        end
    end
    nc = length(core)
    pause = BitVector(iti_post[core] .> PAUSE_S)
    n1 = count(pause); n0 = nc - n1

    hpre = detrend_hour(r_pre, hour[core])
    herank = detrend_hour(r_erank, hour[core])

    obs_pre = cor(r_erank, r_pre)
    obs_post = cor(r_erank, r_post)
    obs_rate = cor(r_erank, r_rate)
    obs_tod = cor(herank, hpre)
    obs_auc = auc_ranks(erank, pause)

    null_pre = zeros(NPERM); null_post = zeros(NPERM); null_rate = zeros(NPERM)
    null_tod = zeros(NPERM); null_auc = zeros(NPERM)
    for p in 1:NPERM
        idx = shift_index(blocks, nc)
        se = @view r_erank[idx]
        null_pre[p] = cor(se, r_pre)
        null_post[p] = cor(se, r_post)
        null_rate[p] = cor(se, r_rate)
        null_tod[p] = cor(@view(herank[idx]), hpre)
        null_auc[p] = auc_from_rank(r_erank, idx, pause, n1, n0)
    end

    z(o, nl) = (o - mean(nl)) / max(std(nl), 1e-12)

    # Two-sided, centred on the null mean (the shift null is not centred at zero).
    pv(o, nl) = (1 + count(>=(abs(o - mean(nl))), abs.(nl .- mean(nl)))) / (NPERM + 1)

    push!(stat_rows, (
        rat=rat, n_trials=nc, n_sessions=length(blocks), K=Ks,
        rho_pre=obs_pre, rho_pre_null=mean(null_pre), rho_pre_null_sd=std(null_pre),
        rho_pre_z=z(obs_pre, null_pre), rho_pre_p=pv(obs_pre, null_pre),
        rho_post=obs_post, rho_post_null=mean(null_post), rho_post_null_sd=std(null_post),
        rho_post_z=z(obs_post, null_post), rho_post_p=pv(obs_post, null_post),
        rho_rate=obs_rate, rho_rate_null=mean(null_rate), rho_rate_null_sd=std(null_rate),
        rho_rate_z=z(obs_rate, null_rate), rho_rate_p=pv(obs_rate, null_rate),
        rho_pre_withinhour=obs_tod, rho_pre_withinhour_null=mean(null_tod),
        rho_pre_withinhour_null_sd=std(null_tod),
        rho_pre_withinhour_z=z(obs_tod, null_tod), rho_pre_withinhour_p=pv(obs_tod, null_tod),
        auc_pause=obs_auc, auc_pause_null=mean(null_auc), auc_pause_null_sd=std(null_auc),
        auc_pause_z=z(obs_auc, null_auc), auc_pause_p=pv(obs_auc, null_auc),
        frac_pause=n1 / nc,
    ); promote=true)

    # State occupancy along a work bout (bouts split by pauses > PAUSE_S)
    bsum = zeros(Ks, 2, BOUT_POS)      # state x (start, end) x position
    bcnt = zeros(Int, Ks, 2, BOUT_POS)
    nbout = 0
    for rng in sessions
        idx = [t for t in rng if incore[t]]
        isempty(idx) && continue
        starts = [1; findall(t -> iti_pre[idx[t]] > PAUSE_S, 2:length(idx)) .+ 1]
        stops = [starts[2:end] .- 1; length(idx)]
        for (s, e) in zip(starts, stops)
            L = e - s + 1
            L >= BOUT_MIN || continue
            nbout += 1
            for j in 1:min(BOUT_POS, L), k in 1:Ks
                bsum[k, 1, j] += γ[k, idx[s + j - 1]]; bcnt[k, 1, j] += 1
                bsum[k, 2, j] += γ[k, idx[e - j + 1]]; bcnt[k, 2, j] += 1
            end
        end
    end
    for k in 1:Ks, (ei, ename) in enumerate(("start", "end")), j in 1:BOUT_POS
        bcnt[k, ei, j] > 0 || continue
        push!(bout_rows, (rat=rat, state=k, acc_rank=ranks[k], edge=ename, pos=j,
                          posterior=bsum[k, ei, j] / bcnt[k, ei, j],
                          n_bouts=bcnt[k, ei, j]); promote=true)
    end

    @printf("%-8s n=%6d  rho(E[rank], log ITI_pre)=%+.3f (null %+.3f, z=%+.1f)  AUC pause=%.3f\n",
            rat, nc, obs_pre, mean(null_pre), z(obs_pre, null_pre), obs_auc)
end

# Multiple-comparison correction: family = 3 measures x n_rats tests
const FDR_FIELDS = [:rho_pre_p, :rho_rate_p, :rho_pre_withinhour_p]

"Benjamini-Hochberg step-up q-values."
function bh(p::AbstractVector{<:Real})
    m = length(p); ord = sortperm(p); q = zeros(Float64, m); run = 1.0
    for i in m:-1:1
        run = min(run, p[ord[i]] * m / i)
        q[ord[i]] = run
    end
    return q
end

"Holm step-down adjusted p-values."
function holm(p::AbstractVector{<:Real})
    m = length(p); ord = sortperm(p); a = zeros(Float64, m); run = 0.0
    for i in 1:m
        run = max(run, min(1.0, p[ord[i]] * (m - i + 1)))
        a[ord[i]] = run
    end
    return a
end

let nr = nrow(stat_rows)
    pooled = vcat([stat_rows[!, f] for f in FDR_FIELDS]...)
    qs, hs = bh(pooled), holm(pooled)
    for (i, f) in enumerate(FDR_FIELDS)
        rng = ((i - 1) * nr + 1):(i * nr)
        base = chopsuffix(String(f), "_p")
        stat_rows[!, Symbol(base * "_q")] = qs[rng]
        stat_rows[!, Symbol(base * "_pholm")] = hs[rng]
    end
    @printf("\nBH-FDR over %d pooled two-sided tests (%d rats x %d measures)\n",
            length(pooled), nr, length(FDR_FIELDS))
    for f in FDR_FIELDS
        base = chopsuffix(String(f), "_p")
        @printf("  %-22s q<0.05: %2d/%2d   Holm<0.05: %2d/%2d   (raw p<0.05: %2d/%2d)\n",
                base, count(<(0.05), stat_rows[!, Symbol(base * "_q")]), nr,
                count(<(0.05), stat_rows[!, Symbol(base * "_pholm")]), nr,
                count(<(0.05), stat_rows[!, f]), nr)
    end
end

outdir = joinpath("results", "final_ddmhmms")
CSV.write(joinpath(outdir, "state_engagement_summary.csv"), summary_rows)
CSV.write(joinpath(outdir, "state_engagement_deciles.csv"), decile_rows)
CSV.write(joinpath(outdir, "state_engagement_stats.csv"), stat_rows)
CSV.write(joinpath(outdir, "state_bout_profile.csv"), bout_rows)

println()
@printf("rats with rho(E[rank], log ITI_pre) > 0: %d / %d\n",
        count(>(0), stat_rows.rho_pre), nrow(stat_rows))
@printf("median rho = %+.3f,  median |z| vs shift null = %.1f,  rats with p < 0.05: %d\n",
        median(stat_rows.rho_pre), median(abs.(stat_rows.rho_pre_z)),
        count(<(0.05), stat_rows.rho_pre_p))
@printf("rats with rho(E[rank], trial rate) < 0: %d / %d, median %+.3f\n",
        count(<(0), stat_rows.rho_rate), nrow(stat_rows), median(stat_rows.rho_rate))
@printf("within hour of day, rats with rho > 0: %d / %d, median %+.3f\n",
        count(>(0), stat_rows.rho_pre_withinhour), nrow(stat_rows),
        median(stat_rows.rho_pre_withinhour))
@printf("median AUC for predicting a >%.0f s pause: %.3f (null %.3f)\n",
        PAUSE_S, median(stat_rows.auc_pause), median(stat_rows.auc_pause_null))
