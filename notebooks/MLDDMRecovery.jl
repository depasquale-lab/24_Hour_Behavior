#=
eDDM recovery, truth = each rat's exact fit (eddm_hyperparams_by_rat.csv).
sim u_t ~ N(m, σ0²) per trial, exact WFPT; blind refit + truth-init refit
-> results/mlddm_recovery/tasks/*.csv; task = $SGE_TASK_ID or ARGS[1]
env: N_REP Q QTAU N_STARTS N_TRIALS (0 = all) ESCALATE OUT_DIR
=#

using Pkg
Pkg.activate("notebooks")

using Random
using DriftDiffusionModels
using CSV
using DataFrames
using Statistics
using Dates

const N_REP = parse(Int, get(ENV, "N_REP", "1"))
const Q = parse(Int, get(ENV, "Q", "10"))
const QTAU = parse(Int, get(ENV, "QTAU", "64"))
const N_STARTS = parse(Int, get(ENV, "N_STARTS", "3"))
const N_TRIALS = parse(Int, get(ENV, "N_TRIALS", "0"))
const ESCALATE = get(ENV, "ESCALATE", "0") == "1"
const N_FOLDS = 5                      # rat filter, as in eDDMExact.jl
const TRUTH_PATH = joinpath("results", "eddm_exact", "eddm_hyperparams_by_rat.csv")
const OUT_DIR = get(ENV, "OUT_DIR", joinpath("results", "mlddm_recovery"))
const TASK_DIR = joinpath(OUT_DIR, "tasks")

const PARAMS = ["B", "τ", "v", "a0"]

# Data: same cohort, filter and trial order as eDDMExact.jl.
rat_df = CSV.read(joinpath("data", "processed_rat_data.csv.gz"), DataFrame)
rat_df = rat_df[rat_df.daily .== "24 hr", :]
rat_df.s = [cs == "right" ? 1 : -1 for cs in rat_df.correct_side]
session_dates(sub) = [Date(split(dt)[1]) for dt in sub.trial_datetime]
const RATS = filter(
    r -> length(unique(session_dates(rat_df[rat_df.name .== r, :]))) >= N_FOLDS,
    String.(unique(rat_df.name)),
)

truth_df = CSV.read(TRUTH_PATH, DataFrame; types=Dict(:rat_name => String))
function truth_for(rat)
    row = truth_df[findfirst(==(rat), truth_df.rat_name), :]
    m = [row["m_u$p"] for p in PARAMS]
    σ0 = [exp(row["logσ0_u$p"]) for p in PARAMS]
    return m, σ0
end

to_natural(u) = (exp(u[1]), exp(u[2]), exp(u[3]), logistic(u[4]))

# sampler: grid log-spaced near 0 (spike when a₀ near bound), trapezoid CDF
const DT_GRID = vcat(10 .^ range(-7, log10(2.0); length=3000), range(2.0, 60.0; length=1500)[2:end])

function sample_trial(rng::AbstractRNG, m, σ0, s::Int)
    u = m .+ σ0 .* randn(rng, 4)
    B, τ, v, a0 = to_natural(u)
    f = [exp(DriftDiffusionModels.logdensityof(B, v, a0, τ, τ + t, c, s)) for t in DT_GRID, c in (-1, 1)]
    cdf = [cumsum(vcat(0.0, 0.5 .* (f[1:end-1, j] .+ f[2:end, j]) .* diff(DT_GRID))) for j in 1:2]
    mass = [cdf[j][end] for j in 1:2]
    total = sum(mass)
    total > 0.99 || @warn "WFPT mass on grid is $(round(total; digits=4)) for (B,τ,v,a0)=$(round.((B, τ, v, a0); digits=3)), s=$s"
    j = rand(rng) < mass[2] / total ? 2 : 1
    c = j == 2 ? 1 : -1
    x = rand(rng) * mass[j]
    F = cdf[j]
    i = clamp(searchsortedfirst(F, x), 2, length(F))
    c0, c1 = F[i-1], F[i]                      # cdf[k] is the mass up to DT_GRID[k]
    frac = c1 > c0 ? (x - c0) / (c1 - c0) : 0.0
    return DDMResult(τ + DT_GRID[i-1] + frac * (DT_GRID[i] - DT_GRID[i-1]), c, s)
end

function simulate(seed::Int, m, σ0, stims::Vector{Int})
    out = Vector{DDMResult}(undef, length(stims))
    Threads.@threads for t in eachindex(stims)
        out[t] = sample_trial(Xoshiro(seed + t), m, σ0, stims[t])
    end
    return out
end

const TASKS = [(rat=r, rep=rep) for rep in 1:N_REP for r in RATS]

function run_task(task_id::Int)
    task = TASKS[task_id]
    rat, rep = task.rat, task.rep
    out = joinpath(TASK_DIR, "$(rat)_rep$(rep).csv")
    isfile(out) && return @info "$rat rep $rep already done"

    m, σ0 = truth_for(rat)
    stims = rat_df.s[rat_df.name .== rat]
    seed = 10_000 * rep + task_id
    rng = MersenneTwister(seed)
    if N_TRIALS > 0 && N_TRIALS < length(stims)
        stims = stims[sort(randperm(rng, length(stims))[1:N_TRIALS])]
    end
    @info "== $rat rep $rep: $(length(stims)) trials, q=$Q qτ=$QTAU, threads=$(Threads.nthreads()) =="
    @info "truth m=$(round.(m; digits=3)) σ0=$(round.(σ0; digits=4))"

    t0 = time()
    data = simulate(seed, m, σ0, stims)
    @info "simulated in $(round(time() - t0; digits=1)) s; mean RT $(round(mean(d.rt for d in data); digits=3)), accuracy $(round(mean(d.choice == d.s for d in data); digits=3))"

    blind = fit_mlddm_exact(data; q=Q, qτ=QTAU, n_starts=N_STARTS, escalate=ESCALATE, rng=rng)
    show(stdout, blind)
    # Fit quadrature as used (escalation may have raised it).
    q, qτ = blind.q, blind.qτ
    warm = fit_mlddm_exact(data; q=q, qτ=qτ, n_starts=1, init=(m, σ0), escalate=false, rng=rng)
    show(stdout, warm)
    ll_true = marginal_loglik(data, m, log.(σ0); q=q, qτ=qτ)
    q_ref, qτ_ref = min(q + 2, 12), min(4qτ, 512)
    ll_blind_ref = marginal_loglik(data, blind.m, log.(blind.σ0); q=q_ref, qτ=qτ_ref)
    quad_err = abs(ll_blind_ref - blind.loglik) / length(data)
    @info "logL/trial: truth $(round(ll_true / length(data); digits=4)), blind $(round(blind.loglik / length(data); digits=4)), truth-init $(round(warm.loglik / length(data); digits=4)); quadrature error $(round(quad_err; sigdigits=3))"

    row = DataFrame(;
        rat=[rat], rep=[rep], seed=[seed], n_trials=[length(data)], q=[q], qτ=[qτ],
        n_starts=[blind.n_starts], n_converged=[blind.n_converged],
        # MLDDMFit.converged also demands a passed escalation check; count starts instead.
        blind_converged=[blind.n_converged > 0], warm_converged=[warm.n_converged > 0],
        true_logL=[ll_true], blind_logL=[blind.loglik], warm_logL=[warm.loglik],
        blind_quad_err=[quad_err], blind_boundary=[join(PARAMS[blind.boundary], ";")],
        true_boundary=[join(PARAMS[σ0 .<= 1e-3 * 1.5], ";")],
        seconds=[time() - t0],
    )
    for (d, p) in enumerate(PARAMS)
        row[!, "true_m_$p"] = [m[d]]
        row[!, "blind_m_$p"] = [blind.m[d]]
        row[!, "warm_m_$p"] = [warm.m[d]]
    end
    for (d, p) in enumerate(PARAMS)
        row[!, "true_s0_$p"] = [σ0[d]]
        row[!, "blind_s0_$p"] = [blind.σ0[d]]
        row[!, "warm_s0_$p"] = [warm.σ0[d]]
    end
    CSV.write(out, row)
    @info "$rat rep $rep done in $(round((time() - t0) / 60; digits=1)) min -> $out"
end

if abspath(PROGRAM_FILE) == @__FILE__
    mkpath(TASK_DIR)
    v = get(ENV, "SGE_TASK_ID", "")
    ids = if !isempty(v) && v != "undefined"
        [parse(Int, v)]
    elseif !isempty(ARGS)
        [parse(Int, ARGS[1])]
    else
        eachindex(TASKS)
    end
    foreach(run_task, ids)
end
