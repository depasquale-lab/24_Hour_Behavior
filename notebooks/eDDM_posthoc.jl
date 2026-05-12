### A Pluto.jl notebook ###
# v0.20.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ dade0002-0000-4000-8000-000000000002
begin
    import Pkg
    Pkg.activate(@__DIR__)

    using CSV
    using DataFrames
    using Distributions
    using PlotUtils
    using PlutoUI
    using Plots
    using Random
    using Statistics
    using StatsBase
    using StatsPlots
    nothing
end

# ╔═╡ dade0001-0000-4000-8000-000000000001
md"""
# eDDM posthoc

Posthoc diagnostics for the per-rat eDDM (extended DDM) fits stored under
`../results/`. Shows ELBO convergence, group-level hyperparameters,
trial-parameter autocorrelation, RT autocorrelation, and an RT-ACF posterior
predictive from simulated symmetric DDMs.

| § | Section |
|---|---------|
| 1 | Setup & data loading |
| 2 | ELBO history |
| 3 | Group hyperparameters |
| 4 | Trial-parameter ACFs |
| 5 | RT ACF (population) |
| 6 | RT-ACF posterior predictive |
| 7 | Save figures |
"""

# ╔═╡ dade0003-0000-4000-8000-000000000003
md"""
## §1 Setup & data loading
"""

# ╔═╡ dade0004-0000-4000-8000-000000000004
begin
    results_dir    = joinpath(@__DIR__, "..", "results")
    elbo_csv       = joinpath(results_dir, "eddm_elbo_history_by_rat.csv")
    trial_post_csv = joinpath(results_dir, "eddm_trial_posteriors_by_rat.csv.gz")
    hyper_csv      = joinpath(results_dir, "eddm_hyperparams_by_rat.csv")

    elbo_df       = CSV.read(elbo_csv,       DataFrame)
    trial_post_df = CSV.read(trial_post_csv, DataFrame)
    hyper_df      = CSV.read(hyper_csv,      DataFrame)
    (size(elbo_df), size(trial_post_df), size(hyper_df))
end

# ╔═╡ dade0005-0000-4000-8000-000000000005
md"""
## §2 ELBO history

ELBO trace per rat (rebased to its minimum) to confirm convergence.
"""

# ╔═╡ dade0006-0000-4000-8000-000000000006
elbo_plot = begin
    grouped = groupby(elbo_df, :rat_name)
    pElbo = plot(title  = "ELBO history by rat",
                 xlabel = "Iteration",
                 ylabel = "Δ ELBO",
                 fontfamily = "helvetica",
                 legend = false)
    for g in grouped
        rat = unique(g.rat_name)[1]
        plot!(pElbo, g.iter, g.elbo .- minimum(g.elbo);
              label = rat, lw = 2)
    end
    pElbo
end

# ╔═╡ dade0007-0000-4000-8000-000000000007
md"""
## §3 Group hyperparameters

Boxplots + dotplots of the rat-level group means for each eDDM parameter.
"""

# ╔═╡ dade0008-0000-4000-8000-000000000008
hyper_plot = begin
    pHyper = plot(legend = false, fontfamily = "helvetica")

    hp_labels = ["Bμ", "τμ", "vμ", "a0μ"]
    hp_cols   = [:B_group_mean, :τ_group_mean, :v_group_mean, :a0_group_mean]

    n_hp = nrow(hyper_df)
    for (lab, col) in zip(hp_labels, hp_cols)
        boxplot!(pHyper, fill(lab, n_hp), hyper_df[!, col])
        dotplot!(pHyper, fill(lab, n_hp), hyper_df[!, col])
    end

    ylabel!(pHyper, "θ")
    title!(pHyper, "eDDM rat hyperparameters")
    pHyper
end

# ╔═╡ dade0009-0000-4000-8000-000000000009
md"""
## §4 Trial-parameter ACFs

Per-rat ACFs of the posterior-mean trial parameters (B, τ, v, a0), then
averaged across rats with ±1.96 SEM ribbons.
"""

# ╔═╡ dade001b-0000-4000-8000-00000000001b
sem(x) = std(x) / sqrt(length(x))

# ╔═╡ dade000a-0000-4000-8000-00000000000a
function rat_param_acf(df::DataFrame, rat::AbstractString; max_lag::Int = 20)
    sub = filter(row -> row.rat_name == rat, df)
    DataFrame(
        B_acf  = autocor(sub.B_mean,  1:max_lag),
        τ_acf  = autocor(sub.τ_mean,  1:max_lag),
        v_acf  = autocor(sub.v_mean,  1:max_lag),
        a0_acf = autocor(sub.a0_mean, 1:max_lag),
    )
end

# ╔═╡ dade000b-0000-4000-8000-00000000000b
begin
    rats    = unique(trial_post_df.rat_name)
    max_lag = 20

    per_rat = [rat_param_acf(trial_post_df, r; max_lag = max_lag) for r in rats]

    combined = vcat(
        (hcat(d, DataFrame(rat = fill(r, nrow(d))))
         for (d, r) in zip(per_rat, rats))...;
        cols = :union,
    )
    combined.lag = repeat(1:max_lag, length(rats))

    mean_param_acf = combine(groupby(combined, :lag),
        :B_acf  => mean => :B_mean,
        :τ_acf  => mean => :τ_mean,
        :v_acf  => mean => :v_mean,
        :a0_acf => mean => :a0_mean,
    )

    sem_param_acf = combine(groupby(combined, :lag),
        :B_acf  => sem => :B_sem,
        :τ_acf  => sem => :τ_sem,
        :v_acf  => sem => :v_sem,
        :a0_acf => sem => :a0_sem,
    )
    (rats, size(mean_param_acf))
end

# ╔═╡ dade000c-0000-4000-8000-00000000000c
param_acf_plot = begin
    pParam = plot(title  = "Mean ACF of trial parameters",
                  xlabel = "Lag", ylabel = "ACF",
                  fontfamily = "helvetica")

    for (mean_col, sem_col, lab) in [
        (:B_mean,  :B_sem,  "B"),
        (:τ_mean,  :τ_sem,  "τ"),
        (:v_mean,  :v_sem,  "v"),
        (:a0_mean, :a0_sem, "a0"),
    ]
        plot!(pParam, mean_param_acf.lag, mean_param_acf[!, mean_col];
              ribbon = 1.96 .* sem_param_acf[!, sem_col],
              label  = lab, lw = 2)
    end
    pParam
end

# ╔═╡ dade000d-0000-4000-8000-00000000000d
md"""
## §5 RT ACF (population)

Per-rat RT autocorrelation, averaged across rats with ±1.96 SEM ribbon.
"""

# ╔═╡ dade000e-0000-4000-8000-00000000000e
function rat_rt_acf(df::DataFrame, rat::AbstractString; max_lag::Int = 20)
    sub = filter(row -> row.rat_name == rat, df)
    DataFrame(RT_acf = autocor(sub.rt, 1:max_lag))
end

# ╔═╡ dade000f-0000-4000-8000-00000000000f
begin
    per_rat_rt = [rat_rt_acf(trial_post_df, r; max_lag = max_lag) for r in rats]

    combined_rt = vcat(
        (hcat(d, DataFrame(rat = fill(r, nrow(d))))
         for (d, r) in zip(per_rat_rt, rats))...;
        cols = :union,
    )
    combined_rt.lag = repeat(1:max_lag, length(rats))

    mean_rt_acf = combine(groupby(combined_rt, :lag),
        :RT_acf => mean => :RT_mean)
    sem_rt_acf  = combine(groupby(combined_rt, :lag),
        :RT_acf => sem => :RT_sem)
    (size(mean_rt_acf), size(sem_rt_acf))
end

# ╔═╡ dade0010-0000-4000-8000-000000000010
rt_acf_plot = begin
    pRt = plot(title  = "Mean RT ACF across rats",
               xlabel = "Lag", ylabel = "ACF",
               fontfamily = "helvetica")
    plot!(pRt, mean_rt_acf.lag, mean_rt_acf.RT_mean;
        ribbon = 1.96 .* sem_rt_acf.RT_sem,
        label  = "RT",
        lw     = 2)
    pRt
end

# ╔═╡ dade0011-0000-4000-8000-000000000011
md"""
## §6 RT-ACF posterior predictive

Simulate a symmetric DDM per rat using the fitted group-mean parameters and
compare its population mean RT-ACF against the empirical RT-ACF from §5.
"""

# ╔═╡ dade0012-0000-4000-8000-000000000012
function sim_ddm_trial(rng::AbstractRNG, B, v, a0, τ;
                       dt = 1e-4, tmax = 5.0, drift_sign::Float64 = 1.0)
    x = a0 * B
    t = 0.0
    μ = drift_sign * v
    while t < tmax
        x += μ * dt + sqrt(dt) * randn(rng)
        t += dt
        if x >= B
            return (τ + t, 1)
        elseif x <= 0.0
            return (τ + t, 0)
        end
    end
    return (NaN, NaN)
end

# ╔═╡ dade0013-0000-4000-8000-000000000013
function sim_rts_symmetric(rng::AbstractRNG, B, v, a0, τ, N::Int;
                           dt = 1e-4, tmax = 5.0)
    rts = Vector{Float64}(undef, N)
    n_timeout = 0
    for t in 1:N
        sign = rand(rng, Bool) ? 1.0 : -1.0
        rt, _ = sim_ddm_trial(rng, B, v, a0, τ; dt = dt, tmax = tmax, drift_sign = sign)
        rts[t] = rt
        n_timeout += isfinite(rt) ? 0 : 1
    end
    good = isfinite.(rts)
    return rts[good], n_timeout / N
end

# ╔═╡ dade0014-0000-4000-8000-000000000014
function params_for_rat(df_params::DataFrame, rat)
    row = only(df_params[df_params.rat_name .== rat, :])
    (B = row.B_group_mean, v = row.v_group_mean,
     a0 = row.a0_group_mean, τ = row.τ_group_mean)
end

# ╔═╡ dade0015-0000-4000-8000-000000000015
function ppc_eddm_rt_acf(df_params::DataFrame;
                         N_per_rat::Int = 2000,
                         max_lag::Int   = 20,
                         n_rep::Int     = 400,
                         rng::AbstractRNG = MersenneTwister(1),
                         dt::Float64    = 1e-4,
                         tmax::Float64  = 8.0)
    rats   = sort(collect(df_params.rat_name))
    R      = length(rats)
    R == 0 && error("No rats in df_params.")

    acf_reps = Matrix{Float64}(undef, n_rep, max_lag + 1)
    timeout_sum = Dict(r => 0.0 for r in rats)

    for b in 1:n_rep
        rat_acf = Matrix{Float64}(undef, R, max_lag + 1)
        for (i, r) in enumerate(rats)
            p = params_for_rat(df_params, r)
            rts, to = sim_rts_symmetric(rng, p.B, p.v, p.a0, p.τ, N_per_rat;
                                        dt = dt, tmax = tmax)
            timeout_sum[r] += to

            if length(rts) <= max_lag + 2
                rat_acf[i, :] .= NaN
            else
                rat_acf[i, :] .= autocor(rts, 0:max_lag)
            end
        end
        for k in 1:(max_lag + 1)
            col = rat_acf[:, k]
            good = isfinite.(col)
            acf_reps[b, k] = mean(col[good])
        end
    end

    pred_mean = vec(mean(acf_reps; dims = 1))
    lo = [quantile(acf_reps[:, k], 0.025) for k in 1:(max_lag + 1)]
    hi = [quantile(acf_reps[:, k], 0.975) for k in 1:(max_lag + 1)]

    return (lag = collect(0:max_lag),
            pred_mean = pred_mean, lo = lo, hi = hi,
            timeout_rate_by_rat = Dict(r => timeout_sum[r] / n_rep for r in rats),
            rats = rats)
end

# ╔═╡ dade0016-0000-4000-8000-000000000016
ppc = ppc_eddm_rt_acf(hyper_df; N_per_rat = 2000, max_lag = 20, n_rep = 400, tmax = 8.0)

# ╔═╡ dade0017-0000-4000-8000-000000000017
ppc_plot = begin
    upper = ppc.hi .- ppc.pred_mean
    lower = ppc.pred_mean .- ppc.lo

    pPpc = plot(fontfamily = "helvetica",
                xlabel = "Lag", ylabel = "ACF",
                title  = "RT-ACF posterior predictive")

    plot!(pPpc, ppc.lag[2:end], ppc.pred_mean[2:end];
          ribbon = (upper[2:end], lower[2:end]),
          label  = "PPC mean ± 95%",
          lw     = 2)

    plot!(pPpc, mean_rt_acf.lag, mean_rt_acf.RT_mean;
          label = "True population ACF",
          lw    = 2)
    pPpc
end

# ╔═╡ dade0018-0000-4000-8000-000000000018
md"""
## §7 Save figures
"""

# ╔═╡ dade0019-0000-4000-8000-000000000019
md"""
Save all figures to `../results/`: $(@bind save_figs CheckBox(default = false))
"""

# ╔═╡ dade001a-0000-4000-8000-00000000001a
begin
    if save_figs
        out_dir = joinpath(@__DIR__, "..", "results")
        mkpath(out_dir)
        figs = [
            (elbo_plot,      "eddm_elbo_history_by_rat.svg"),
            (hyper_plot,     "eddm_hyperparams_by_rat.svg"),
            (param_acf_plot, "eddm_trial_param_acf.svg"),
            (rt_acf_plot,    "eddm_rt_acf.svg"),
            (ppc_plot,       "eddm_rt_acf_ppc.svg"),
        ]
        for (fig, name) in figs
            savefig(fig, joinpath(out_dir, name))
        end
        md"Saved $(length(figs)) figures to `$out_dir`."
    else
        md"_Tick the box above to save all figures._"
    end
end

# ╔═╡ Cell order:
# ╟─dade0001-0000-4000-8000-000000000001
# ╠═dade0002-0000-4000-8000-000000000002
# ╟─dade0003-0000-4000-8000-000000000003
# ╠═dade0004-0000-4000-8000-000000000004
# ╟─dade0005-0000-4000-8000-000000000005
# ╠═dade0006-0000-4000-8000-000000000006
# ╟─dade0007-0000-4000-8000-000000000007
# ╠═dade0008-0000-4000-8000-000000000008
# ╟─dade0009-0000-4000-8000-000000000009
# ╠═dade001b-0000-4000-8000-00000000001b
# ╠═dade000a-0000-4000-8000-00000000000a
# ╠═dade000b-0000-4000-8000-00000000000b
# ╠═dade000c-0000-4000-8000-00000000000c
# ╟─dade000d-0000-4000-8000-00000000000d
# ╠═dade000e-0000-4000-8000-00000000000e
# ╠═dade000f-0000-4000-8000-00000000000f
# ╠═dade0010-0000-4000-8000-000000000010
# ╟─dade0011-0000-4000-8000-000000000011
# ╠═dade0012-0000-4000-8000-000000000012
# ╠═dade0013-0000-4000-8000-000000000013
# ╠═dade0014-0000-4000-8000-000000000014
# ╠═dade0015-0000-4000-8000-000000000015
# ╠═dade0016-0000-4000-8000-000000000016
# ╠═dade0017-0000-4000-8000-000000000017
# ╟─dade0018-0000-4000-8000-000000000018
# ╟─dade0019-0000-4000-8000-000000000019
# ╠═dade001a-0000-4000-8000-00000000001a
