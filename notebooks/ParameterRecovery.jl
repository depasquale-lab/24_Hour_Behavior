#=
Parameter recovery for the K=4 DDM-HMM, with ground truth matched to the animals.

Each synthetic dataset is a clone of one rat's final fitted model
(results/final_ddmhmms/final_ddmhmms), generated with that rat's session lengths and
its real sequence of correct sides. The clone is then refit blind with the same
pipeline used for the real data: 20 random initialisations from
`generate_ddmhmm_initialization`, Baum–Welch with nothing tied (TiedPriorHMM), each
then polished by L-BFGS on the marginal likelihood as in DirectGradientDDMHMM.jl, and
the best final logL kept. The truth is used only afterwards, to align recovered state labels with true
ones.

Trials are sampled exactly: choice and RT are drawn by inverse CDF from the same
WFPT density the fitter evaluates, so the test isolates the estimator rather than
simulator discretisation error.

Tasks (one per SGE array index; see `TASKS` below):
  full   every rat, all sessions
  scale  every rat × session fractions (0.1, 0.25, 0.5), 1 replicate

Each task also runs one extra fit initialised at the true parameters. It is not
used as the estimate; comparing its logL with the best blind fit tells us whether a
poor recovery is a local optimum or the likelihood genuinely preferring other values.

Output: one BSON file per task in results/parameter_recovery/tasks, merged by
PlotParameterRecovery.jl.
=#

include(joinpath(@__DIR__, "FitConstrainedDDMHMMs.jl"))

using BSON
using BSON: @load
using DensityInterface

const OUT_DIR = joinpath("results", "parameter_recovery", "tasks")
const FIT_DIR = joinpath("results", "final_ddmhmms", "final_ddmhmms")
const N_REPS_FULL = 1
const N_INITS = 20
const SCALE_FRACS = (0.1, 0.25, 0.5)

# Rehydrate the saved fits in all three schemas (mirrors ExtractStateParameters.jl).
struct DDMEmission{T<:Real}
    B::T
    v::T
    a₀::T
    τ::T
end

softmax1(x) = (e = exp.(x .- maximum(x)); e ./ sum(e))
sigmoid_(y) = 1 / (1 + exp(-y))

function build_tied_idx_map(K::Int, tied::Vector{Symbol})
    free = Symbol[p for p in DDM_PARAMS if !(p in tied)]
    nfree = length(free)
    idx = Matrix{Int}(undef, K, length(DDM_PARAMS))
    for (pi, p) in enumerate(DDM_PARAMS)
        if p in tied
            pos = K * nfree + findfirst(==(p), tied)
            for k in 1:K
                idx[k, pi] = pos
            end
        else
            pos = findfirst(==(p), free)
            for k in 1:K
                idx[k, pi] = (k - 1) * nfree + pos
            end
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
    dists = [
        DriftDiffusionModel(;
            B=exp(ddm_y[idx[k, 1]]),
            v=exp(ddm_y[idx[k, 2]]),
            a₀=sigmoid_(ddm_y[idx[k, 3]]),
            τ=exp(ddm_y[idx[k, 4]]),
        ) for k in 1:K
    ]
    return PriorHMM(init, trans, dists, 1, 1)
end

"Return (hmm, rat) for a saved final fit, whichever schema it was saved in."
function load_fit(path::AbstractString)
    ks = keys(BSON.parse(path))
    if :hmm_dists in ks
        @load path hmm_init hmm_trans hmm_dists rat
        dists = [DriftDiffusionModel(; B=d.B, v=d.v, a₀=d.a₀, τ=d.τ) for d in hmm_dists]
        return PriorHMM(Vector{Float64}(hmm_init), Matrix{Float64}(hmm_trans), dists, 1, 1),
        String(rat)
    elseif :fit in ks
        @load path fit rat
        t = fit.hmm
        return PriorHMM(
            Vector{Float64}(t.init), Matrix{Float64}(t.trans), deepcopy(t.dists), 1, 1
        ),
        String(rat)
    elseif :best_direct in ks
        @load path best_θ K TIED RAT
        return θ_to_priorhmm(Vector{Float64}(best_θ), Int(K), Symbol[s for s in TIED]),
        String(RAT)
    else
        error("unknown schema in $path: $(collect(ks))")
    end
end

# Exact trial sampler: inverse CDF of the WFPT density on a fine time grid.
const DT_GRID = 5e-4      # s
const T_MAX = 60.0        # s of decision time covered by the grid

struct StateSampler
    t::Vector{Float64}                            # grid of total RTs
    p_upper::Dict{Int,Float64}                    # P(choice = +1 | s)
    cdf::Dict{Tuple{Int,Int},Vector{Float64}}     # (s, choice) → normalised CDF on t
end

function StateSampler(d::DriftDiffusionModel)
    t = collect((d.τ + DT_GRID):DT_GRID:(d.τ + T_MAX))
    p_upper = Dict{Int,Float64}()
    cdf = Dict{Tuple{Int,Int},Vector{Float64}}()
    for s in (-1, 1)
        mass = Dict{Int,Float64}()
        for c in (-1, 1)
            f = [
                exp(DriftDiffusionModels.logdensityof(d.B, d.v, d.a₀, d.τ, ti, c, s)) for
                ti in t
            ]
            F = cumsum(f) .* DT_GRID
            mass[c] = F[end]
            cdf[(s, c)] = F ./ F[end]
        end
        p_upper[s] = mass[1] / (mass[1] + mass[-1])
        total = mass[1] + mass[-1]
        abs(total - 1) < 1e-3 ||
            @warn "WFPT mass on grid is $(round(total; digits=5)) for $d, s=$s"
    end
    return StateSampler(t, p_upper, cdf)
end

function sample_trial(rng::AbstractRNG, S::StateSampler, s::Int)
    c = rand(rng) < S.p_upper[s] ? 1 : -1
    F = S.cdf[(s, c)]
    u = rand(rng)
    i = searchsortedfirst(F, u)
    i = clamp(i, 1, length(F))
    # Linear interpolation of the CDF within the bin.
    if i == 1
        rt = S.t[1] - DT_GRID * (1 - u / F[1])
    else
        frac = (u - F[i - 1]) / max(F[i] - F[i - 1], eps())
        rt = S.t[i - 1] + frac * DT_GRID
    end
    return DDMResult(rt, c, s)
end

"""
    simulate_hmm(rng, hmm, stim_seqs)

Simulate one session per element of `stim_seqs` (vector of ±1 correct sides). Each
session starts from `hmm.init`, as in the fitted model. Returns observations, true
states and `seq_ends`.
"""
function simulate_hmm(rng::AbstractRNG, hmm, stim_seqs::Vector{Vector{Int}})
    K = length(hmm.init)
    samplers = [StateSampler(d) for d in hmm.dists]
    obs = DDMResult[]
    z = Int[]
    for stims in stim_seqs
        zt = rand(rng, Categorical(hmm.init))
        for (t, s) in enumerate(stims)
            t > 1 && (zt = rand(rng, Categorical(hmm.trans[zt, :])))
            push!(obs, sample_trial(rng, samplers[zt], s))
            push!(z, zt)
        end
    end
    seq_ends = cumsum(length.(stim_seqs))
    return obs, z, seq_ends
end

"Real correct-side sequence per session for `rat`, in session order."
function stimulus_sessions(rat::AbstractString)
    sub = rat_df[rat_df.name .== rat, :]
    dates = [Date(split(dt)[1]) for dt in sub.trial_datetime]
    return [Int.(sub.correct_side_numeric[dates .== d]) for d in sort(unique(dates))]
end

# Fitting: the real-data pipeline, applied to synthetic observations.
#
# Each initialisation runs Baum–Welch exactly as in FitConstrainedDDMHMMs.jl, then
# is polished by L-BFGS directly on the marginal log-likelihood (the approach of
# DirectGradientDDMHMM.jl, which produced several of the final fits). Baum–Welch
# alone often stops at its 100-iteration cap before converging, so the polished
# estimate is the one reported; the Baum–Welch-only estimate is kept for comparison.
function run_bw(prior, obs, seq_ends)
    hmm_est, evol = HiddenMarkovModels.baum_welch(
        TiedPriorHMM(prior, Symbol[]),
        obs;
        seq_ends=seq_ends,
        atol=1e-3,
        max_iterations=100,
        loglikelihood_increasing=false,
    )
    return hmm_est, last(evol), length(evol)
end

# Parametric emission/HMM so ForwardDiff Duals flow through the forward pass.
struct DualDDMHMM{T<:Real} <: HiddenMarkovModels.AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dists::Vector{DDMEmission{T}}
end
DensityInterface.DensityKind(::DDMEmission) = HasDensity()
DensityInterface.logdensityof(d::DDMEmission, x::DDMResult) =
    DriftDiffusionModels.logdensityof(d.B, d.v, d.a₀, d.τ, x.rt, x.choice, x.s)
Base.length(h::DualDDMHMM) = length(h.init)
HiddenMarkovModels.initialization(h::DualDDMHMM) = h.init
HiddenMarkovModels.transition_matrix(h::DualDDMHMM) = h.trans
HiddenMarkovModels.obs_distributions(h::DualDDMHMM) = h.dists

"""
Flatten an HMM into θ = [init logits | trans logits (row-major) | (log B, v, logit a₀, log τ) per state].

v is left unconstrained: the likelihood uses |v|, and under v = exp(y) a state whose
drift reaches ≈0 during Baum–Welch cannot recover because the gradient in y
vanishes with v.
"""
function hmm_to_θ(h)
    K = length(h.init)
    lg(x) = log(max(x, 1e-12))
    θ = Float64[lg.(h.init)...]
    for k in 1:K
        append!(θ, lg.(h.trans[k, :]))
    end
    for d in h.dists
        a = clamp(d.a₀, 1e-6, 1 - 1e-6)
        append!(θ, [lg(d.B), d.v, log(a / (1 - a)), lg(d.τ)])
    end
    return θ
end

function θ_to_dualhmm(θ, K::Int)
    init = softmax1(θ[1:K])
    trans = reduce(vcat, [softmax1(θ[(K + (k - 1) * K + 1):(K + k * K)])' for k in 1:K])
    off = K + K * K
    dists = [
        DDMEmission(
            exp(θ[off + 4(k - 1) + 1]),
            abs(θ[off + 4(k - 1) + 2]),
            sigmoid_(θ[off + 4(k - 1) + 3]),
            exp(θ[off + 4(k - 1) + 4]),
        ) for k in 1:K
    ]
    return DualDDMHMM(init, Matrix(trans), dists)
end

function polish(h, obs, seq_ends; max_iter::Int=1000)
    K = length(h.init)
    nll(θ) = -HiddenMarkovModels.logdensityof(θ_to_dualhmm(θ, K), obs; seq_ends=seq_ends)
    res = Optim.optimize(
        nll,
        hmm_to_θ(h),
        Optim.LBFGS(; linesearch=Optim.LineSearches.BackTracking()),
        Optim.Options(; iterations=max_iter, f_reltol=1e-9);
        autodiff=:forward,
    )
    hd = θ_to_dualhmm(Optim.minimizer(res), K)
    dists = [DriftDiffusionModel(; B=d.B, v=d.v, a₀=d.a₀, τ=d.τ) for d in hd.dists]
    return PriorHMM(hd.init, hd.trans, dists, 1, 1), -Optim.minimum(res), Optim.iterations(res)
end

function fit_one(prior, obs, seq_ends)
    bw_hmm, bw_ll, bw_iter = run_bw(prior, obs, seq_ends)
    hmm, ll, pol_iter = polish(bw_hmm, obs, seq_ends)
    # Keep the Baum–Welch point if polishing somehow made it worse.
    ll < bw_ll && ((hmm, ll) = (as_priorhmm(bw_hmm), bw_ll))
    return (hmm=hmm, logL=ll, bw_hmm=bw_hmm, bw_logL=bw_ll, bw_iter=bw_iter, polish_iter=pol_iter)
end

function fit_blind(obs, seq_ends, truth; n_inits::Int=N_INITS)
    K = length(truth.init)
    priors = [generate_ddmhmm_initialization(K) for _ in 1:n_inits]
    # Last slot: initialised at the truth, as an optimisation diagnostic only.
    push!(priors, PriorHMM(copy(truth.init), copy(truth.trans), deepcopy(truth.dists), 1, 1))
    results = Vector{Any}(nothing, length(priors))
    Threads.@threads for i in eachindex(priors)
        try
            results[i] = fit_one(priors[i], obs, seq_ends)
        catch e
            @warn "fit failed" i exception = (e, catch_backtrace())
        end
    end
    blind = [r for r in results[1:n_inits] if r !== nothing]
    isempty(blind) && error("every blind initialisation failed")
    best = blind[argmax([r.logL for r in blind])]
    best_bw = blind[argmax([r.bw_logL for r in blind])]
    truth_init = results[end]
    return (
        hmm=best.hmm,
        logL=best.logL,
        bw_hmm=best_bw.bw_hmm,
        bw_logL=best_bw.bw_logL,
        init_logLs=[r === nothing ? NaN : r.logL for r in results[1:n_inits]],
        init_bw_logLs=[r === nothing ? NaN : r.bw_logL for r in results[1:n_inits]],
        bw_iters=[r === nothing ? 0 : r.bw_iter for r in results[1:n_inits]],
        polish_iters=[r === nothing ? 0 : r.polish_iter for r in results[1:n_inits]],
        truthinit_logL=truth_init === nothing ? NaN : truth_init.logL,
    )
end

as_priorhmm(h) = PriorHMM(
    Vector{Float64}(h.init), Matrix{Float64}(h.trans), deepcopy(h.dists), 1, 1
)

_gamma(ret) =
    ret isa AbstractMatrix ? ret :
    ret isa Tuple ? _gamma(ret[1]) :
    hasproperty(ret, :γ) ? getfield(ret, :γ) : error("no γ")

posterior(hmm, obs, seq_ends) =
    _gamma(HiddenMarkovModels.forward_backward(as_priorhmm(hmm), obs; seq_ends=seq_ends))

"All permutations of 1:n (n is 4 here, so brute force is fine)."
function permutations_of(n::Int)
    n == 1 && return [[1]]
    out = Vector{Vector{Int}}()
    for p in permutations_of(n - 1), pos in 1:n
        push!(out, insert!(copy(p), pos, n))
    end
    return out
end

"""
    match_states(z, γ)

Permutation `perm` such that recovered state `perm[k]` corresponds to true state `k`,
chosen to maximise the posterior mass the recovered model puts on the true state.
"""
function match_states(z::Vector{Int}, γ::AbstractMatrix)
    K = size(γ, 1)
    M = zeros(K, K)                       # M[true, recovered] = Σ γ
    for t in eachindex(z)
        M[z[t], :] .+= @view γ[:, t]
    end
    perms = permutations_of(K)
    return perms[argmax([sum(M[k, p[k]] for k in 1:K) for p in perms])]
end

decode_acc(z, γ, perm) = mean(z[t] == findfirst(==(argmax(γ[:, t])), perm) for t in eachindex(z))

function params_table(hmm)
    return [
        (B=d.B, v=d.v, a0=d.a₀, tau=d.τ, p_self=hmm.trans[k, k]) for
        (k, d) in enumerate(hmm.dists)
    ]
end

# Task table.
fit_files = sort(filter(f -> endswith(f, ".bson"), readdir(FIT_DIR)))
const TASKS = vcat(
    [(file=f, rep=r, frac=1.0) for f in fit_files for r in 1:N_REPS_FULL],
    [(file=f, rep=1, frac=fr) for fr in SCALE_FRACS for f in fit_files],
)

function run_task(task_id::Int)
    task = TASKS[task_id]
    truth, rat = load_fit(joinpath(FIT_DIR, task.file))
    rng = Xoshiro(hash((rat, task.rep, task.frac)))
    Random.seed!(hash((rat, task.rep, task.frac, :inits)))   # random inits

    stims = stimulus_sessions(rat)
    if task.frac < 1
        n_keep = max(2, round(Int, task.frac * length(stims)))
        stims = stims[sort(randperm(rng, length(stims))[1:n_keep])]
    end

    obs, z, seq_ends = simulate_hmm(rng, truth, stims)
    @info "Task $task_id: rat=$rat rep=$(task.rep) frac=$(task.frac) " *
          "sessions=$(length(stims)) trials=$(length(obs)) threads=$(Threads.nthreads())"

    t0 = time()
    fit = fit_blind(obs, seq_ends, truth)
    elapsed = time() - t0

    logL_true = HiddenMarkovModels.logdensityof(truth, obs; seq_ends=seq_ends)
    γ_fit = posterior(fit.hmm, obs, seq_ends)
    γ_bw = posterior(fit.bw_hmm, obs, seq_ends)
    γ_true = posterior(truth, obs, seq_ends)
    perm = match_states(z, γ_fit)
    perm_bw = match_states(z, γ_bw)
    trans_rec = fit.hmm.trans[perm, perm]

    # Keep the first few sessions of posteriors for the example-trace panel.
    n_ex = min(3, length(seq_ends))
    ex_idx = 1:seq_ends[n_ex]

    result = Dict(
        :rat => rat,
        :rep => task.rep,
        :frac => task.frac,
        :n_sessions => length(stims),
        :n_trials => length(obs),
        :true_params => params_table(truth),
        :rec_params => params_table(fit.hmm)[perm],
        :true_trans => truth.trans,
        :rec_trans => trans_rec,
        :true_occ => [mean(z .== k) for k in 1:4],
        :rec_occ => vec(sum(γ_fit; dims=2))[perm] ./ length(z),
        :decode_acc_fit => decode_acc(z, γ_fit, perm),
        :decode_acc_true => decode_acc(z, γ_true, collect(1:4)),
        :logL_true => logL_true,
        :logL_fit => fit.logL,
        :logL_truthinit => fit.truthinit_logL,
        :init_logLs => fit.init_logLs,
        :init_bw_logLs => fit.init_bw_logLs,
        :bw_iters => fit.bw_iters,
        :polish_iters => fit.polish_iters,
        :bw_params => params_table(fit.bw_hmm)[perm_bw],
        :logL_bw => fit.bw_logL,
        :decode_acc_bw => decode_acc(z, γ_bw, perm_bw),
        :elapsed_s => elapsed,
        :example_z => z[ex_idx],
        :example_gamma_fit => γ_fit[perm, ex_idx],
        :example_gamma_true => γ_true[:, ex_idx],
        :example_seq_ends => seq_ends[1:n_ex],
    )

    isdir(OUT_DIR) || mkpath(OUT_DIR)
    fname = joinpath(OUT_DIR, "task_$(lpad(task_id, 3, '0'))_$(rat)_f$(task.frac)_r$(task.rep).bson")
    BSON.bson(fname, Dict(:result => result))
    @info "  done in $(round(elapsed / 60; digits=1)) min: decode fit=$(round(result[:decode_acc_fit]; digits=3)) " *
          "oracle=$(round(result[:decode_acc_true]; digits=3)) ΔlogL(fit−true)=$(round(fit.logL - logL_true; digits=1)) → $fname"
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    v = get(ENV, "SGE_TASK_ID", "")
    task_id = (isempty(v) || v == "undefined") ? parse(Int, ARGS[1]) : parse(Int, v)
    @assert 1 <= task_id <= length(TASKS) "task $task_id out of range 1:$(length(TASKS))"
    run_task(task_id)
end
