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

# ╔═╡ dada0003-0000-4000-8000-000000000003
begin
    import Pkg
    Pkg.activate(@__DIR__)

    using BSON
    using BSON: @load, @save
    using CSV
    using CategoricalArrays
    using Dates
    using DataFrames
    using Distributions
    using DriftDiffusionModels
    using HiddenMarkovModels
    using HypothesisTests
    using JLD2
    using LinearAlgebra
    using MCMCChains
    using PlotUtils
    using PlutoUI
    using Random
    using Serialization
    using Statistics
    using StatsBase
    using StatsFuns
    using StatsPlots
    using Turing
    using CairoMakie
    using Plots
    nothing
end

# ╔═╡ dada0001-0000-4000-8000-000000000001
md"""
# Population DDM-HMM analysis

This notebook reproduces every population-level figure used in the paper.
It is organised as **one section per figure** so you can run the section that
makes the figure you need without touching the rest.

## Notebook layout

| § | Section | Output(s) |
|---|---------|-----------|
| 1 | Setup & data loading | (none) |
| 2 | Per-rat × per-state DDM-parameter table | (drives everything below) |
| 3 | Parameter heatmaps (rat × state)        | `population_{v,B,a0,bias_mag,tau}_sat.svg` |
| 4 | Best-vs-worst-state paired differences  | `param_diff_plot_population.svg` |
| 5 | Within-animal speed-accuracy trade-off  | `population_sat_rtLine_accColorDots.svg` |
| 6 | Bayesian regression: accuracy ~ params  | `acc_pred_population_rankcolored.svg`, `beta_forest_population.svg` |
| 7 | Bayesian regression: RT ~ params        | `rt_pred_population_rankcolored.svg`, `beta_forest_population_rt.svg` |
| 8 | State occupancy by hour (population)    | `population_state_occupancy.svg`, `population_enrichment.svg` |
| 9 | Parameter boxswarm by accuracy rank     | `parameter_boxswarm_by_accRank_population.svg` |
| 10 | Repeated-measures ANOVA + post-hocs     | summary tables |
| 11 | Stylised trial timeline                 | `stylized_trial_timeline.svg` |
| 12 | Goodness of fit (QQ, RT histograms, accuracy) | `qq_population_rts.svg`, `rt_distributions_by_rat.svg`, `accuracy_fit_by_rat.svg` |

Toggle the **Save figures** checkbox below to write SVGs to `../results/`.
Bayesian chains are cached to `notebooks/cache/` so re-opens are fast — delete the cache directory to force a re-fit.
"""

# ╔═╡ dada0002-0000-4000-8000-000000000002
md"""
## §1 Setup & data loading
"""

# ╔═╡ dada0004-0000-4000-8000-000000000004
md"""
### Paths & global toggles
"""

# ╔═╡ dada0005-0000-4000-8000-000000000005
begin
    DATA_DIR          = joinpath(@__DIR__, "..", "ddmhmms")
    PROCESSED_RAT_CSV = joinpath(@__DIR__, "..", "data", "processed_rat_data.csv.gz")
    RESULTS_DIR       = joinpath(@__DIR__, "..", "results")
    CACHE_DIR         = joinpath(@__DIR__, "cache")
    isdir(CACHE_DIR) || mkpath(CACHE_DIR)
    nothing
end

# ╔═╡ dada0006-0000-4000-8000-000000000006
md"**Save figures to `../results/` when ticked:** $(@bind save_figs CheckBox(default=false))"

# ╔═╡ dada0007-0000-4000-8000-000000000007
md"""
### Load every K=4 fit

Glob the K=4 BSON fits in `../ddmhmms/`, exclude the daily-fit files, and load
each into a `DDMHMMFit` struct. The rat name is parsed from the filename
(`<name>_K4_tied-full_compat.bson`).
"""

# ╔═╡ dada0008-0000-4000-8000-000000000008
begin
    # BSON serialises type references as a literal module path starting at
    # :Main, so deserialisation tries to resolve `Main.DDMHMMFit`. In Pluto our
    # cells live in `Main.var"workspace#N"`, not `Main`, so a struct defined
    # here is invisible to BSON. Define it in the real `Main` module instead.
    # `Any`-typed fields are fine — BSON sets them directly to the deserialised
    # `PriorHMM`, `Float64`, and `Vector{Float64}` values without checking.
    if !isdefined(Main, :DDMHMMFit)
        Core.eval(Main, :(struct DDMHMMFit
            hmm
            logL
            logL_evolution
        end))
    end
    # Alias into this workspace so plain `DDMHMMFit` (and `Vector{DDMHMMFit}`
    # in downstream cells) resolves to the Main-scope struct.
    DDMHMMFit = Main.DDMHMMFit

    bson_files = filter(f -> occursin("K4", f) && !occursin("daily", f),
                        readdir(DATA_DIR; join=true))
    sort!(bson_files)

    ddmhmm_fits = Main.DDMHMMFit[]
    for path in bson_files
        @load path fit
        push!(ddmhmm_fits, fit)
    end
    animal_names = String.([split(basename(f), "_")[1] for f in bson_files])
    (n_rats = length(animal_names), rats = animal_names)
end

# ╔═╡ dada0009-0000-4000-8000-000000000009
md"""
## §2 Per-rat × per-state DDM-parameter table

For each (rat, state) pair we compute:
* the four fitted DDM parameters (`v`, `B`, `a0`, `tau`),
* simulated accuracy and mean RT (10 000 trials per state),
* a within-rat **state rank** (1 = lowest accuracy, K = highest),
* `bias_mag = |a0 − 0.5|`.

Most downstream sections work directly off `ddm_params_df`.
"""

# ╔═╡ dada000a-0000-4000-8000-00000000000a
function build_param_df(fits::Vector{DDMHMMFit}, names::Vector{<:AbstractString};
                        n_sim::Int = 10_000, K::Int = 4)
    rows = NamedTuple[]
    for (i, fit) in enumerate(fits)
        for k in 1:K
            ddm = fit.hmm.dists[k]
            sim = simulateDDM(ddm, n_sim)
            corrects = [d.s == d.choice for d in sim]
            rts      = [d.rt              for d in sim]
            push!(rows, (
                rat   = names[i],
                state = k,
                v     = ddm.v,  B = ddm.B, a0 = ddm.a₀, tau = ddm.τ,
                acc   = mean(corrects),
                mean_rt = mean(rts),
            ))
        end
    end
    df = DataFrame(rows)
    df.bias_mag = abs.(df.a0 .- 0.5)

    # within-rat rank by accuracy (1 = lowest acc, K = highest)
    DataFrames.transform!(groupby(df, :rat)) do sdf
        ord   = sortperm(sdf.acc)
        ranks = similar(ord)
        ranks[ord] = 1:length(ord)
        DataFrame(:state_rank => ranks)
    end
    df
end

# ╔═╡ dada000b-0000-4000-8000-00000000000b
ddm_params_df = build_param_df(ddmhmm_fits, animal_names)

# ╔═╡ dada000c-0000-4000-8000-00000000000c
md"""
### Centred SAT frame (`dfc`)

Adds within-rat mean-centred RT/accuracy and within-rat RT order. Used by §5
(within-animal SAT) and §9 (boxswarm).
"""

# ╔═╡ dada000d-0000-4000-8000-00000000000d
function add_within_animal_centering(df::DataFrame)
    g = groupby(df, :rat)
    out = DataFrames.transform(g,
        :mean_rt => (x -> x .- mean(x)) => :rt_c,
        :acc     => (x -> x .- mean(x)) => :acc_c,
    )
    out = DataFrames.transform(groupby(out, :rat),
        :rt_c  => (x -> invperm(sortperm(x))) => :rt_order,
        :acc_c => (x -> invperm(sortperm(x))) => :acc_rank_c,
    )
    out
end

# ╔═╡ dada000e-0000-4000-8000-00000000000e
dfc = add_within_animal_centering(ddm_params_df)

# ╔═╡ dada000f-0000-4000-8000-00000000000f
md"""
## §3 Figure — parameter heatmaps (rat × state)

Four 18 × 4 heatmaps (one per DDM parameter), with rats sorted top-to-bottom by
their mean drift `v`. Colour scale is robustly clipped to the 5–95th percentile
so a single outlier doesn't dominate.
"""

# ╔═╡ dada0010-0000-4000-8000-000000000010
begin
    function _robust_clims(M; q = (0.05, 0.95))
        v = vec(M); v = v[isfinite.(v)]
        lo, hi = quantile(v, q)
        (!isfinite(lo) || !isfinite(hi) || lo == hi) && return (minimum(v), maximum(v))
        return (lo, hi)
    end

    function rat_order_by_param(df::DataFrame, param::Symbol; rev=true)
        g = combine(groupby(df, :rat), param => mean => :stat)
        sort!(g, :stat; rev=rev)
        return string.(g.rat)
    end

    function param_heatmap(df::DataFrame, param::Symbol;
            title::AbstractString="$(param)", q=(0.05, 0.95),
            rats_order::Union{Nothing,Vector{String}}=nothing,
            cell_px::Int=26, left_pad::Int=140, bottom_pad::Int=90,
            top_pad::Int=70, right_pad::Int=40, cmap=:viridis)

        states = sort(unique(df.state))
        rats   = rats_order === nothing ? unique(string.(df.rat)) : rats_order
        state_to_j = Dict(states[j] => j for j in eachindex(states))
        rat_to_i   = Dict(rats[i]   => i for i in eachindex(rats))

        M = fill(NaN, length(rats), length(states))
        for r in eachrow(df)
            i = get(rat_to_i, string(r.rat), nothing); i === nothing && continue
            M[i, state_to_j[r.state]] = r[param]
        end

        cl = _robust_clims(M; q=q)
        nrows, ncols = size(M)
        W = left_pad + right_pad + cell_px*ncols
        H = top_pad  + bottom_pad + cell_px*nrows

        fig = CairoMakie.Figure(size=(W, H), figure_padding=(10, 10, 10, 10))
        ax  = CairoMakie.Axis(fig[2, 1];
            title=title, xlabel="State", ylabel="Rat",
            xticks=(1:ncols, string.(states)),
            yticks=(1:nrows, rats),
            aspect=CairoMakie.DataAspect(),
            xgridvisible=false, ygridvisible=false,
            topspinevisible=false, rightspinevisible=false,
        )
        hm = CairoMakie.heatmap!(ax, 1:ncols, 1:nrows, M';
            colormap=cmap, colorrange=cl, nan_color=(:gray90, 1.0),
        )
        CairoMakie.ylims!(ax, nrows + 0.5, 0.5)
        CairoMakie.xlims!(ax, 0.5, ncols + 0.5)

        cb = CairoMakie.Colorbar(fig[1, 1], hm; vertical=false)
        cb.height = 18
        fig
    end
end

# ╔═╡ dada0011-0000-4000-8000-000000000011
RAT_ORDER_V = rat_order_by_param(ddm_params_df, :v; rev=true)

# ╔═╡ dada0012-0000-4000-8000-000000000012
fig_v_heatmap = param_heatmap(ddm_params_df, :v; title="v (drift)", rats_order=RAT_ORDER_V)

# ╔═╡ dada0013-0000-4000-8000-000000000013
fig_tau_heatmap = param_heatmap(ddm_params_df, :tau; title="τ (non-decision)", rats_order=RAT_ORDER_V)

# ╔═╡ dada0014-0000-4000-8000-000000000014
fig_bias_heatmap = param_heatmap(ddm_params_df, :bias_mag; title="|a₀ − 0.5|  (bias magnitude)", rats_order=RAT_ORDER_V)

# ╔═╡ dada0015-0000-4000-8000-000000000015
fig_B_heatmap = param_heatmap(ddm_params_df, :B; title="B (boundary)", rats_order=RAT_ORDER_V)

# ╔═╡ dada0016-0000-4000-8000-000000000016
if save_figs
    CairoMakie.save(joinpath(RESULTS_DIR, "population_v_sat.svg"),        fig_v_heatmap)
    CairoMakie.save(joinpath(RESULTS_DIR, "population_tau_sat.svg"),      fig_tau_heatmap)
    CairoMakie.save(joinpath(RESULTS_DIR, "population_bias_mag_sat.svg"), fig_bias_heatmap)
    CairoMakie.save(joinpath(RESULTS_DIR, "population_B_sat.svg"),        fig_B_heatmap)
    "saved 4 heatmaps"
else
    "save_figs is OFF — tick the box at the top to write SVGs"
end

# ╔═╡ dada0017-0000-4000-8000-000000000017
md"""
## §4 Figure — best vs worst-state paired differences

For each rat, we pick the state with the highest simulated accuracy (`*_high`)
and the state with the lowest (`*_low`), then plot per-rat spaghetti lines for
each DDM parameter.
"""

# ╔═╡ dada0018-0000-4000-8000-000000000018
function high_low_by_rat(df::DataFrame; acc::Symbol=:acc)
    g    = groupby(df, :rat)
    high = combine(g) do sdf; sdf[argmax(sdf[!, acc]), :] end
    low  = combine(g) do sdf; sdf[argmin(sdf[!, acc]), :] end
    rename!(high, Symbol.(names(high)) .=> Symbol.(string.(names(high)) .* "_high"))
    rename!(low,  Symbol.(names(low))  .=> Symbol.(string.(names(low))  .* "_low"))
    out = innerjoin(high, low, on = :rat_high => :rat_low)
    out.Δv  = out.v_high   .- out.v_low
    out.ΔB  = out.B_high   .- out.B_low
    out.Δa0 = out.a0_high  .- out.a0_low
    out.Δτ  = out.tau_high .- out.tau_low
    out
end

# ╔═╡ dada0019-0000-4000-8000-000000000019
hl = high_low_by_rat(ddm_params_df)

# ╔═╡ dada001a-0000-4000-8000-00000000001a
function paired_spaghetti(hl::DataFrame, low::Symbol, high::Symbol; title="")
    lo, hi = hl[!, low], hl[!, high]
    p = Plots.plot(size=(500, 500), xticks=([1,2], ["Low", "High"]),
                   title=title, legend=false, fontfamily="helvetica")
    for i in eachindex(lo)
        Plots.plot!(p,    [1,2], [lo[i], hi[i]], lw=1, alpha=0.6)
        Plots.scatter!(p, [1,2], [lo[i], hi[i]], alpha=0.8, ms=4)
    end
    p
end

# ╔═╡ dada001b-0000-4000-8000-00000000001b
fig_param_diff = let
    a = paired_spaghetti(hl, :v_low,   :v_high;   title="v")
    b = paired_spaghetti(hl, :B_low,   :B_high;   title="B")
    c = paired_spaghetti(hl, :a0_low,  :a0_high;  title="a₀")
    d = paired_spaghetti(hl, :tau_low, :tau_high; title="τ")
    Plots.plot(a, b, c, d; layout=(2,2), size=(800, 800))
end

# ╔═╡ dada001c-0000-4000-8000-00000000001c
if save_figs
    Plots.savefig(fig_param_diff, joinpath(RESULTS_DIR, "param_diff_plot_population.svg"))
end

# ╔═╡ dada001d-0000-4000-8000-00000000001d
md"""
### Paired tests on the (high − low) differences

A paired *t*-test and Wilcoxon signed-rank test for each parameter, plus tests
of bias-magnitude against zero in the best vs worst-accuracy state.
"""

# ╔═╡ dada001e-0000-4000-8000-00000000001e
function paired_tests_table(hl::DataFrame)
    out = DataFrame(param=String[], n=Int[], mean=Float64[], median=Float64[],
                    t_p=Float64[], wilcoxon_p=Float64[])
    for (lab, col) in (("Δv", :Δv), ("ΔB", :ΔB), ("Δa₀", :Δa0), ("Δτ", :Δτ))
        x = collect(skipmissing(hl[!, col]))
        push!(out, (lab, length(x), mean(x), median(x),
                    pvalue(OneSampleTTest(x)), pvalue(SignedRankTest(x))))
    end
    out
end

# ╔═╡ dada001f-0000-4000-8000-00000000001f
paired_tests_results = paired_tests_table(hl)

# ╔═╡ dada0020-0000-4000-8000-000000000020
bias_tests = let
    bias_best   = hl.a0_high .- 0.5
    bias_worst  = hl.a0_low  .- 0.5
    DataFrame(
        what  = ["best-state |bias|", "worst-state |bias|"],
        mean  = [mean(bias_best), mean(bias_worst)],
        wilcoxon_p = [pvalue(SignedRankTest(bias_best)),
                      pvalue(SignedRankTest(bias_worst))],
    )
end

# ╔═╡ dada0021-0000-4000-8000-000000000021
md"""
## §5 Figure — within-animal speed-accuracy trade-off

Scatter of mean-centred (RT, accuracy) per (rat, state). The black line+ribbon
shows the mean across rats at each within-rat RT-rank position.
"""

# ╔═╡ dada0022-0000-4000-8000-000000000022
function plot_sat_dots(df::DataFrame; markersize=4, alpha=0.65)
    K   = maximum(df.acc_rank_c)
    pal = Plots.palette(:auto)
    palK = [pal[mod1(i, length(pal))] for i in 1:K]

    p = Plots.scatter(
        title="Within-animal SAT",
        xlabel="Mean-centred RT (s)",
        ylabel="Mean-centred accuracy",
        legend=false, fontfamily="helvetica",
    )
    for sdf in groupby(df, :rat)
        cols = palK[sdf.acc_rank_c]
        Plots.scatter!(p, sdf.rt_c, sdf.acc_c;
            markersize=markersize, alpha=alpha,
            markercolor=cols, markerstrokewidth=0)
    end
    p
end

# ╔═╡ dada0023-0000-4000-8000-000000000023
function overlay_mean_sat!(p, df::DataFrame; line_lw=5, ribbon_alpha=0.15)
    summ = combine(groupby(df, :rt_order),
        :rt_c  => mean => :rt_mean,
        :acc_c => mean => :acc_mean,
        :acc_c => (x -> std(x)/sqrt(length(x))) => :acc_sem,
    )
    sort!(summ, :rt_order)
    Plots.plot!(p, summ.rt_mean, summ.acc_mean; lw=line_lw, color=:black, alpha=1.0)
    Plots.plot!(p, summ.rt_mean, summ.acc_mean;
        ribbon=summ.acc_sem, fillalpha=ribbon_alpha, color=:black, lw=0)
    p
end

# ╔═╡ dada0024-0000-4000-8000-000000000024
fig_sat = let
    p = plot_sat_dots(dfc)
    overlay_mean_sat!(p, dfc)
end

# ╔═╡ dada0025-0000-4000-8000-000000000025
if save_figs
    Plots.savefig(fig_sat, joinpath(RESULTS_DIR, "population_sat_rtLine_accColorDots.svg"))
end

# ╔═╡ dada0026-0000-4000-8000-000000000026
md"""
## §6 Bayesian regression — accuracy ~ DDM params

Beta-likelihood regression with logit link.

```
acc_n ~ Beta(p_n · ϕ, (1 − p_n) · ϕ)
p_n   = logistic(x_n · β)
β     ~ Normal(0, I)
ϕ     ~ Exponential(1)
```

Predictors are z-scored within parameter, with an intercept appended.
The chain is **cached to JLD2** so re-opens skip NUTS.
"""

# ╔═╡ dada0027-0000-4000-8000-000000000027
begin
    zscore_cols(M) = (M .- mean(M; dims=1)) ./ std(M; dims=1)

    function design_matrix(df::DataFrame)
        X = Matrix(select(df, [:v, :B, :a0, :tau]))
        X = zscore_cols(X)
        hcat(X, ones(size(X, 1)))   # intercept last
    end

    PARAM_NAMES = ["v", "B", "a0", "tau", "intercept"]
    nothing
end

# ╔═╡ dada0028-0000-4000-8000-000000000028
@model function reg_accuracy(acc, X)
    N, P = size(X)
    β ~ MvNormal(zeros(P), I)
    ϕ ~ Exponential(1.0)
    η = X * β
    p = clamp.(StatsFuns.logistic.(η), 1e-6, 1 - 1e-6)
    for n in 1:N
        acc[n] ~ Beta(p[n] * ϕ, (1 - p[n]) * ϕ)
    end
end

# ╔═╡ dada0029-0000-4000-8000-000000000029
# `do`-block-friendly: the function is the first arg, so calls look like
# `cached_chain("chain_acc") do ... end`.
#
# Uses `Serialization` rather than `JLD2` because current Turing returns a
# `FlexiChain` from `sample()`, and JLD2 can't round-trip that type cleanly.
# Old `.jld2` caches are ignored; delete `notebooks/cache/*.jld2` if you want.
function cached_chain(sample_fn::Function, name::String)
    path = joinpath(CACHE_DIR, name * ".jls")
    if isfile(path)
        return Serialization.deserialize(path)
    end
    chain = sample_fn()
    Serialization.serialize(path, chain)
    chain
end

# ╔═╡ dada002a-0000-4000-8000-00000000002a
begin
    X_design = design_matrix(ddm_params_df)
    acc_obs  = clamp.(collect(ddm_params_df.acc), 1e-6, 1 - 1e-6)
    rt_obs   = clamp.(collect(ddm_params_df.mean_rt), 1e-9, Inf)
    nothing
end

# ╔═╡ dada002b-0000-4000-8000-00000000002b
chain_acc = cached_chain("chain_acc") do
    sample(reg_accuracy(acc_obs, X_design), NUTS(0.65), 5000; progress=true)
end

# ╔═╡ dada002c-0000-4000-8000-00000000002c
md"""
### Posterior predictive scatter (predicted vs observed accuracy)
"""

# ╔═╡ dada002f-0000-4000-8000-00000000002f
acc_pred = let N = length(acc_obs)
    miss = Vector{Union{Missing, Float64}}(missing, N)
    pp   = predict(reg_accuracy(miss, X_design), chain_acc)
    [mean(pp[@varname(acc[n])]) for n in 1:N]
end

# ╔═╡ dada0030-0000-4000-8000-000000000030
"""
Map continuous values to one of K rank-bins (1..K).
Used to colour scatter points by observed accuracy/RT rank.
"""
function rank_bins(x::AbstractVector; K::Int=4)
    N   = length(x)
    ord = sortperm(x)
    rk  = similar(ord); rk[ord] = 1:N
    clamp.(ceil.(Int, rk .* K ./ N), 1, K)
end

# ╔═╡ dada0031-0000-4000-8000-000000000031
fig_acc_scatter = let
    K = 4
    pal  = Plots.palette(:auto)
    palK = [pal[mod1(i, length(pal))] for i in 1:K]
    cols = palK[rank_bins(acc_obs; K=K)]

    p = Plots.scatter(acc_obs, acc_pred;
        xlabel="Observed accuracy", ylabel="Predicted accuracy",
        fontfamily="helvetica", aspect_ratio=:equal,
        xlims=(0.5, 1.0), ylims=(0.5, 1.0),
        markercolor=cols, markersize=4, alpha=0.75,
        markerstrokewidth=0, label="")
    Plots.plot!(p, [0.5, 1.0], [0.5, 1.0]; lc=:red, ls=:dash, label="")
    p
end

# ╔═╡ dada0032-0000-4000-8000-000000000032
md"""
### β forest plot (95 % CIs over coefficients)
"""

# ╔═╡ dada0033-0000-4000-8000-000000000033
function beta_forest(chain, param_names::Vector{String}; xlab="β")
    P = length(param_names)
    cols = [vec(chain[@varname(β[i])]) for i in 1:P]
    n_iter = length(cols[1])
    raw = Matrix{Float64}(undef, n_iter, P)
    for i in 1:P; raw[:, i] = cols[i]; end

    summ = DataFrame(
        param = param_names,
        idx   = 1:P,
        mean  = [mean(raw[:, i])              for i in 1:P],
        lo    = [quantile(raw[:, i], 0.025)   for i in 1:P],
        hi    = [quantile(raw[:, i], 0.975)   for i in 1:P],
    )

    p = Plots.plot()
    Plots.scatter!(p, summ.mean, summ.idx;
        xerror = (summ.mean .- summ.lo, summ.hi .- summ.mean),
        yticks = (summ.idx, summ.param),
        fontfamily = "helvetica",
        xlabel = xlab, ylabel = "",
        legend = false, markersize = 6,
    )
    Plots.vline!(p, [0.0]; ls=:dash, label="")
    p
end

# ╔═╡ dada0034-0000-4000-8000-000000000034
fig_acc_forest = beta_forest(chain_acc, PARAM_NAMES; xlab="βᵢ (accuracy regression)")

# ╔═╡ dada0035-0000-4000-8000-000000000035
if save_figs
    Plots.savefig(fig_acc_scatter, joinpath(RESULTS_DIR, "acc_pred_population_rankcolored.svg"))
    Plots.savefig(fig_acc_forest,  joinpath(RESULTS_DIR, "beta_forest_population.svg"))
end

# ╔═╡ dada0036-0000-4000-8000-000000000036
md"""
## §7 Bayesian regression — RT ~ DDM params

Gamma-likelihood regression with log link.

```
rt_n ~ Gamma(α, μ_n / α),   μ_n = exp(x_n · β)
β    ~ Normal(0, I)
α    ~ LogNormal(0, 0.5)
```
"""

# ╔═╡ dada0037-0000-4000-8000-000000000037
@model function reg_rt(rt::AbstractVector, X::AbstractMatrix)
    N, P = size(X)
    β ~ MvNormal(zeros(P), Matrix{Float64}(I, P, P))
    α ~ LogNormal(0.0, 0.5)
    μ = exp.(X * β)
    @inbounds for i in 1:N
        rt[i] ~ Gamma(α, μ[i] / α)
    end
end

# ╔═╡ dada0038-0000-4000-8000-000000000038
chain_rt = cached_chain("chain_rt") do
    sample(reg_rt(rt_obs, X_design), NUTS(0.65), 5000; progress=true)
end

# ╔═╡ dada0039-0000-4000-8000-000000000039
rt_pred = let N = length(rt_obs)
    miss = Vector{Union{Missing, Float64}}(missing, N)
    pp   = predict(reg_rt(miss, X_design), chain_rt)
    [mean(pp[@varname(rt[n])]) for n in 1:N]
end

# ╔═╡ dada003a-0000-4000-8000-00000000003a
fig_rt_scatter = let
    K = 4
    pal  = Plots.palette(:auto)
    palK = [pal[mod1(i, length(pal))] for i in 1:K]
    cols = palK[rank_bins(acc_obs; K=K)]   # colour by accuracy rank, matching paper

    p = Plots.scatter(rt_obs, rt_pred;
        xlabel="Observed mean RT (s)", ylabel="Predicted mean RT (s)",
        fontfamily="helvetica", aspect_ratio=:equal,
        xlims=(0.0, 3.0), ylims=(0.0, 3.0),
        markercolor=cols, markersize=4, alpha=0.75,
        markerstrokewidth=0, label="")
    Plots.plot!(p, [0.0, 3.0], [0.0, 3.0]; lc=:red, ls=:dash, label="")
    p
end

# ╔═╡ dada003b-0000-4000-8000-00000000003b
fig_rt_forest = beta_forest(chain_rt, PARAM_NAMES; xlab="βᵢ (RT regression)")

# ╔═╡ dada003c-0000-4000-8000-00000000003c
if save_figs
    Plots.savefig(fig_rt_scatter, joinpath(RESULTS_DIR, "rt_pred_population_rankcolored.svg"))
    Plots.savefig(fig_rt_forest,  joinpath(RESULTS_DIR, "beta_forest_population_rt.svg"))
end

# ╔═╡ dada003d-0000-4000-8000-00000000003d
md"""
## §8 State occupancy by hour

For each rat, run forward-backward to get posterior state probabilities `γ`,
align states across rats by **per-rat accuracy rank**, then bin by hour-from-lights-on.

The hour-aggregation bug from the original notebook (`hours`, `mean_occ`,
`sem_occ` were referenced but never computed) is fixed here.
"""

# ╔═╡ dada003e-0000-4000-8000-00000000003e
begin
    const _DT_FORMATS = (
        dateformat"yyyy-mm-dd HH:MM:SS.s",
        dateformat"yyyy-mm-dd HH:MM:SS",
        dateformat"yyyy-mm-ddTHH:MM:SS.s",
        dateformat"yyyy-mm-ddTHH:MM:SS",
    )

    parse_dt(x) = x isa DateTime ? x :
                  x isa Date     ? DateTime(x) :
                  begin
                      s = String(x)
                      for fmt in _DT_FORMATS
                          try; return DateTime(s, fmt); catch; end
                      end
                      error("Couldn't parse trial_datetime: $x")
                  end

    nothing
end

# ╔═╡ dada003f-0000-4000-8000-00000000003f
function load_rat_sequences(rat::AbstractString;
        data_file::AbstractString = PROCESSED_RAT_CSV)
    rat_df = CSV.read(data_file, DataFrame)
    rat_df = rat_df[rat_df.daily .== "24 hr", :]
    replace!(rat_df[!, :choose_right], 0 => -1)
    side_map = Dict("right" => 1, "left" => -1)
    DataFrames.transform!(rat_df,
        :correct_side => ByRow(cs -> get(side_map, cs, missing)) => :correct_side_numeric)

    sub = rat_df[rat_df.name .== rat, :]
    dts = parse_dt.(sub.trial_datetime)
    dates = Date.(dts)
    unique_dates = sort(unique(dates))

    by_date = [
        [DDMResult(rt, ch, ss) for (rt, ch, ss) in zip(
                sub.rt[idx], sub.choose_right[idx], sub.correct_side_numeric[idx])]
        for idx in (findall(dates .== d) for d in unique_dates)
    ]

    all_results = vcat(by_date...)
    seq_ends    = cumsum(length.(by_date))
    (; sub, dates, unique_dates, by_date, all_results, seq_ends)
end

# ╔═╡ dada0040-0000-4000-8000-000000000040
function state_occupancy_by_hour(γ, sub, dates, unique_dates;
                                 light_on_hour::Real = 7.5)
    K = size(γ, 1)
    concat_idx = reduce(vcat, [findall(dates .== d) for d in unique_dates])
    hours_concat = [
        mod(hour(parse_dt(sub.trial_datetime[i])) - light_on_hour, 24)
        for i in concat_idx
    ]
    occ        = fill(NaN, 24, K)
    n_per_hour = zeros(Int, 24)
    for h in 0:23
        idx = findall(x -> floor(Int, x) == h, hours_concat)
        n_per_hour[h+1] = length(idx)
        if !isempty(idx)
            occ[h+1, :] .= vec(mean(γ[:, idx]; dims=2))
        end
    end
    (; occ, n_per_hour)
end

# ╔═╡ dada0041-0000-4000-8000-000000000041
function state_accuracy(γ, all_results)
    K = size(γ, 1)
    correct = [r.choice == r.s for r in all_results]
    [sum(γ[k, :] .* correct) / sum(γ[k, :]) for k in 1:K]
end

# ╔═╡ dada0042-0000-4000-8000-000000000042
"""
Per-rat 24×K occupancy stack, with states reordered by within-rat accuracy
rank (so column k corresponds to the k-th lowest-accuracy state for every rat).
"""
function compute_occ_by_rank(animal_names, ddmhmm_fits; K::Int=4)
    n = length(animal_names)
    occ_by_rank = Array{Float64}(undef, n, 24, K)
    for (i, rat) in enumerate(animal_names)
        seqs = load_rat_sequences(rat)
        γ, _ = forward_backward(ddmhmm_fits[i].hmm, seqs.all_results;
                                seq_ends=seqs.seq_ends)
        order = sortperm(state_accuracy(γ, seqs.all_results))   # low → high
        out = state_occupancy_by_hour(γ, seqs.sub, seqs.dates, seqs.unique_dates)
        occ_by_rank[i, :, :] .= out.occ[:, order]
    end
    occ_by_rank
end

# ╔═╡ dada0043-0000-4000-8000-000000000043
occ_by_rank = compute_occ_by_rank(animal_names, ddmhmm_fits)

# ╔═╡ dada0044-0000-4000-8000-000000000044
"""
Aggregate `occ_by_rank` (rats × 24 × K) to NaN-aware hour-by-rank mean and SEM.
"""
function aggregate_occupancy(occ_by_rank::Array{<:Real,3})
    _, H, K = size(occ_by_rank)
    mean_occ = fill(NaN, H, K)
    sem_occ  = fill(NaN, H, K)
    for h in 1:H, k in 1:K
        v = occ_by_rank[:, h, k]
        v = v[.!isnan.(v)]
        if !isempty(v)
            mean_occ[h, k] = mean(v)
            sem_occ[h, k]  = std(v) / sqrt(length(v))
        end
    end
    (; hours = 0:23, mean_occ, sem_occ)
end

# ╔═╡ dada0045-0000-4000-8000-000000000045
occ_summary = aggregate_occupancy(occ_by_rank)

# ╔═╡ dada0046-0000-4000-8000-000000000046
fig_occupancy = let
    labels = ["Lowest accuracy", "Low-mid accuracy", "High-mid accuracy", "Highest accuracy"]
    p = Plots.plot(legend=:topleft, fontfamily="helvetica",
                   xlabel="Hour from lights-on (binned)",
                   ylabel="State occupancy",
                   title="Population state occupancy")
    for k in 1:size(occ_summary.mean_occ, 2)
        Plots.plot!(p, occ_summary.hours, occ_summary.mean_occ[:, k];
                    ribbon = occ_summary.sem_occ[:, k], lw=2, label=labels[k])
    end
    p
end

# ╔═╡ dada0047-0000-4000-8000-000000000047
md"""
### Enrichment in the dark cycle and the daily feeding window
"""

# ╔═╡ dada0048-0000-4000-8000-000000000048
function enrichment_per_rat(occ_by_rank::Array{<:Real,3}, hour_mask::AbstractVector{Bool})
    n_rats, _, K = size(occ_by_rank)
    enrich = fill(NaN, n_rats, K)
    for i in 1:n_rats, k in 1:K
        all_h = occ_by_rank[i, :, k];           all_h = all_h[.!isnan.(all_h)]
        win   = occ_by_rank[i, hour_mask, k];   win   = win[.!isnan.(win)]
        enrich[i, k] = mean(win) / mean(all_h)
    end
    enrich
end

# ╔═╡ dada0049-0000-4000-8000-000000000049
begin
    dark_mask    = falses(24); dark_mask[13:24] .= true     # hours 12–23 from lights-on
    feeding_mask = falses(24); feeding_mask[7:9]  .= true   # hours  6–8

    enrich_dark = enrichment_per_rat(occ_by_rank, dark_mask)
    enrich_feed = enrichment_per_rat(occ_by_rank, feeding_mask)
    nothing
end

# ╔═╡ dada004a-0000-4000-8000-00000000004a
fig_enrichment = let
    labels = ["Lowest", "Low-mid", "High-mid", "Highest"]
    function _msem(x)
        x = x[.!isnan.(x)]
        return mean(x), std(x) / sqrt(length(x))
    end
    K = size(enrich_dark, 2)
    md_, sd_ = first.(_msem.(eachcol(enrich_dark))), last.(_msem.(eachcol(enrich_dark)))
    mf_, sf_ = first.(_msem.(eachcol(enrich_feed))), last.(_msem.(eachcol(enrich_feed)))

    p1 = Plots.bar(labels, md_; yerror=sd_, legend=false,
        title="Dark-cycle enrichment", ylabel="Enrichment (× baseline)")
    Plots.hline!(p1, [1.0]; alpha=0.4)
    p2 = Plots.bar(labels, mf_; yerror=sf_, legend=false,
        title="Feeding-time enrichment", ylabel="Enrichment (× baseline)")
    Plots.hline!(p2, [1.0]; alpha=0.4)
    Plots.plot(p1, p2; link=:y, layout=(1,2), size=(700, 320), fontfamily="helvetica")
end

# ╔═╡ dada004b-0000-4000-8000-00000000004b
if save_figs
    Plots.savefig(fig_occupancy,  joinpath(RESULTS_DIR, "population_state_occupancy.svg"))
    Plots.savefig(fig_enrichment, joinpath(RESULTS_DIR, "population_enrichment.svg"))
end

# ╔═╡ dada004c-0000-4000-8000-00000000004c
md"""
## §9 Figure — parameter boxswarm by within-rat accuracy rank

Box-and-jittered-dot for each DDM parameter, grouped by within-rat accuracy
rank (1 = worst, K = best within that rat).
"""

# ╔═╡ dada004d-0000-4000-8000-00000000004d
fig_boxswarm = let
    df = copy(dfc)
    df.acc_rank = categorical(df.state_rank; ordered=true)

    long = stack(df, [:v, :B, :bias_mag, :tau];
                 variable_name=:param, value_name=:value)

    p = StatsPlots.@df long Plots.boxplot(
        :acc_rank, :value;
        group=:param, layout=(2,2), legend=false,
        xlabel="state_rank", ylabel="parameter value",
        outliers=false, whisker_width=0.6, fontfamily="helvetica",
    )
    StatsPlots.@df long StatsPlots.dotplot!(
        :acc_rank, :value;
        group=:param, layout=(2,2), legend=false,
        jitter=0.25, alpha=0.45, markersize=3,
    )
    Plots.plot!(p[1]; ylims=(0, 6))     # v
    Plots.plot!(p[4]; ylims=(0, 1.5))   # tau
    p
end

# ╔═╡ dada004e-0000-4000-8000-00000000004e
if save_figs
    Plots.savefig(fig_boxswarm,
        joinpath(RESULTS_DIR, "parameter_boxswarm_by_accRank_population.svg"))
end

# ╔═╡ dada004f-0000-4000-8000-00000000004f
md"""
## §10 Repeated-measures ANOVA + paired post-hocs

One-way RM-ANOVA per parameter (factor = state, subject = rat) followed by
pairwise paired *t*-tests with Holm correction.
"""

# ╔═╡ dada0050-0000-4000-8000-000000000050
begin
    function wide_by_state(df::DataFrame, param::Symbol)
        # Pivot on within-rat accuracy rank, NOT raw :state. Raw state IDs are
        # arbitrary across rats (the HMM fitter labels states in random order),
        # so unstacking on :state aligns rats by a meaningless axis and the
        # condition effect averages to ~0 → spuriously non-significant ANOVA.
        w = unstack(df, :rat, :state_rank, param)
        sort!(w, :rat)
        w
    end

    function holm_adjust(p::Vector{Float64})
        m = length(p)
        ord = sortperm(p)
        adj = similar(p)
        for (rank, idx) in enumerate(ord)
            adj[idx] = min(1.0, (m - rank + 1) * p[idx])
        end
        for i in (m-1):-1:1
            adj[ord[i]] = min(adj[ord[i]], adj[ord[i+1]])
        end
        adj
    end

    function rm_anova_oneway(wide::DataFrame)
        cols = filter(!=("rat"), DataFrames.names(wide))
        Y = Matrix{Float64}(wide[:, cols])
        n, k = size(Y)
        grand = mean(Y)
        SS_total      = sum((Y .- grand).^2)
        SS_subjects   = k * sum((mean(Y, dims=2) .- grand).^2)
        SS_conditions = n * sum((mean(Y, dims=1) .- grand).^2)
        SS_error      = SS_total - SS_subjects - SS_conditions
        df1 = k - 1
        df2 = (n - 1) * (k - 1)
        F   = (SS_conditions / df1) / (SS_error / df2)
        p   = 1 - cdf(FDist(df1, df2), F)
        (F=F, df1=df1, df2=df2, p=p)
    end

    function paired_posthoc(df::DataFrame, param::Symbol)
        wide = wide_by_state(df, param)
        cols = filter(!=("rat"), DataFrames.names(wide))
        states = parse.(Int, String.(cols))

        rows = DataFrame(state_i=Int[], state_j=Int[],
                         mean_diff=Float64[], t=Float64[], df=Int[], p=Float64[])
        ps = Float64[]
        for a in 1:(length(cols)-1), b in (a+1):length(cols)
            x = Float64.(wide[:, cols[a]])
            y = Float64.(wide[:, cols[b]])
            d  = y .- x
            tt = OneSampleTTest(d)
            push!(rows, (states[a], states[b], mean(d), tt.t, Int(tt.df), pvalue(tt)))
            push!(ps, pvalue(tt))
        end
        rows.p_holm = holm_adjust(ps)
        sort!(rows, :p_holm)
        rows
    end
    nothing
end

# ╔═╡ dada0051-0000-4000-8000-000000000051
anova_table = let
    rows = NamedTuple[]
    for p in (:v, :B, :bias_mag, :tau)
        out = rm_anova_oneway(wide_by_state(ddm_params_df, p))
        push!(rows, (param=String(p), F=out.F, df1=out.df1, df2=out.df2, p=out.p))
    end
    DataFrame(rows)
end

# ╔═╡ dada0100-0000-4000-8000-000000000100
# Partial eta squared from RM-ANOVA F-stats:
#   η²_p = (F·df1) / (F·df1 + df2)   ⇔   SS_cond / (SS_cond + SS_error)
# Cohen's f benchmarks: 0.10 small, 0.25 medium, 0.40 large.
anova_effect_sizes = let
    rows = NamedTuple[]
    for r in eachrow(anova_table)
        eta2p = (r.F * r.df1) / (r.F * r.df1 + r.df2)
        f     = sqrt(eta2p / (1 - eta2p))
        push!(rows, (param=r.param, F=r.F, df1=r.df1, df2=r.df2,
                     partial_eta2=eta2p, cohens_f=f))
    end
    DataFrame(rows)
end

# ╔═╡ dada0052-0000-4000-8000-000000000052
posthoc_v   = paired_posthoc(ddm_params_df, :v)

# ╔═╡ dada0053-0000-4000-8000-000000000053
posthoc_tau = paired_posthoc(ddm_params_df, :tau)

# ╔═╡ dada0054-0000-4000-8000-000000000054
md"""
## §11 Figure — stylised trial timeline

Schematic of one trial used in the methods figure. Side flashes (probabilistic)
are dropped in between center-poke and decision time.
"""

# ╔═╡ dada0055-0000-4000-8000-000000000055
function trial_timeline(; tmax=4.0,
        center_light_on=0.0, center_light_off=0.6,
        center_poke_on=0.6,  center_poke_off=0.7,
        decision_on=2.10, reward_off=3.20,
        flash_dt=0.10, flash_start_offset=0.10,
        p_right=0.75, rng=MersenneTwister(1234),
        labels=("center light","center poke","right flash","left flash","decision/reward"),
        task_band_alpha=0.12, step_height=0.62)

    n = length(labels)
    y = collect(n:-1:1)
    p = Plots.plot(
        xlim=(0, tmax), ylim=(0.5, n + 0.5),
        yticks=(1:n, reverse(collect(labels))),
        xlabel="time in trial (s)",
        legend=false, framestyle=:box, grid=false,
        color=:black, size=(760, 300),
    )

    function add_rect!(x1, x2, y1, y2; α=0.1)
        Plots.plot!(p, Plots.Shape([x1, x2, x2, x1], [y1, y1, y2, y2]);
            seriestype=:shape, linealpha=0, fillalpha=α, color=:black)
    end

    add_rect!(clamp(center_poke_on, 0, tmax), clamp(decision_on, 0, tmax),
             0.5, n + 0.5; α=task_band_alpha)
    Plots.plot!(p, [0.0, 0.0], [0.5, n + 0.5]; lw=1, color=:black)

    function add_step!(t_on, t_off, yrow; height=step_height, lw=2)
        t_on  = clamp(t_on,  0, tmax)
        t_off = clamp(t_off, 0, tmax)
        v = ([0.0, 0.0, 1.0, 0.0] .* height) .+ (yrow - height/2)
        Plots.plot!(p, [0.0, t_on, t_off, tmax], v;
            seriestype=:steppre, lw=lw, color=:black)
    end
    add_tick!(t, yrow; tick=0.30, lw=2) =
        Plots.plot!(p, [t, t], [yrow - tick, yrow + tick]; lw=lw, color=:black)

    flash_start = center_poke_on + flash_start_offset
    flash_times = (decision_on > flash_start) ?
                  collect(flash_start:flash_dt:(decision_on - 1e-9)) : Float64[]

    right_flash, left_flash = Float64[center_poke_on], Float64[center_poke_on]
    for t in flash_times
        (rand(rng) < p_right ? right_flash : left_flash) |> v -> push!(v, t)
    end

    add_step!(center_light_on, center_light_off, y[1]; lw=2)
    add_step!(center_poke_on,  center_poke_off,  y[2]; lw=2)
    for t in right_flash; add_tick!(t, y[3]; tick=0.34, lw=2); end
    for t in left_flash;  add_tick!(t, y[4]; tick=0.24, lw=2); end
    add_step!(decision_on, reward_off, y[5]; height=0.70, lw=3)
    p
end

# ╔═╡ dada0056-0000-4000-8000-000000000056
fig_timeline = trial_timeline(p_right=0.75, rng=MersenneTwister(1234))

# ╔═╡ dada0057-0000-4000-8000-000000000057
if save_figs
    Plots.savefig(fig_timeline, joinpath(RESULTS_DIR, "stylized_trial_timeline.svg"))
end

# ╔═╡ dada0058-0000-4000-8000-000000000058
md"""
## §12 Goodness of fit

Per rat, simulate `n_trials_observed` trials from the fitted HMM, then compare
to the real data.
"""

# ╔═╡ dada0059-0000-4000-8000-000000000059
function simulate_per_rat(animal_names::Vector{String}, ddmhmm_fits::Vector{DDMHMMFit};
                          data_file::AbstractString = PROCESSED_RAT_CSV,
                          rng_seed::Int = 0)
    rat_df = CSV.read(data_file, DataFrame)
    rat_df = rat_df[rat_df.daily .== "24 hr", :]
    replace!(rat_df[!, :choose_right], 0 => -1)
    side_map = Dict("right" => 1, "left" => -1)
    DataFrames.transform!(rat_df,
        :correct_side => ByRow(cs -> get(side_map, cs, missing)) => :correct_side_numeric)

    n = length(animal_names)
    out = Vector{NamedTuple}(undef, n)
    Threads.@threads for i in 1:n
        sub = rat_df[rat_df.name .== animal_names[i], :]
        n_trials = nrow(sub)
        _, sim = rand(MersenneTwister(rng_seed + i),
                      ddmhmm_fits[i].hmm, n_trials)

        out[i] = (
            name = animal_names[i],
            real_rts      = collect(sub.rt),
            sim_rts       = [s.rt for s in sim],
            real_corrects = collect(sub.correct),
            sim_corrects  = [s.choice == s.s for s in sim],
        )
    end
    out
end

# ╔═╡ dada005a-0000-4000-8000-00000000005a
gof_results = simulate_per_rat(animal_names, ddmhmm_fits)

# ╔═╡ dada005b-0000-4000-8000-00000000005b
md"""
### Pooled QQ plot — population RTs
"""

# ╔═╡ dada005c-0000-4000-8000-00000000005c
fig_qq = let
    real_all = reduce(vcat, (r.real_rts for r in gof_results))
    sim_all  = reduce(vcat, (r.sim_rts  for r in gof_results))
    real_all = real_all[isfinite.(real_all) .& (real_all .>= 0)]
    sim_all  = sim_all[isfinite.(sim_all)   .& (sim_all   .>= 0)]

    ps = range(0.001, 0.999; length=400)
    q_real = [quantile(real_all, p) for p in ps]
    q_sim  = [quantile(sim_all,  p) for p in ps]

    p = Plots.scatter(q_real, q_sim;
        ms=3, alpha=0.6,
        xlabel="Real RT quantiles (s)", ylabel="Sim RT quantiles (s)",
        title="Pooled QQ: population RTs",
        fontfamily="helvetica", legend=false, size=(400, 400),
    )
    lo = min(minimum(q_real), minimum(q_sim))
    hi = max(maximum(q_real), maximum(q_sim))
    Plots.plot!(p, [lo, hi], [lo, hi]; lw=2)
    p
end

# ╔═╡ dada005d-0000-4000-8000-00000000005d
md"""
### Per-rat RT histograms
"""

# ╔═╡ dada005e-0000-4000-8000-00000000005e
fig_rt_per_rat = let
    n = length(gof_results)
    subps = Vector{Plots.Plot}(undef, n)
    for i in 1:n
        p = Plots.plot(title=gof_results[i].name, legend=false)
        Plots.histogram!(p, gof_results[i].real_rts; bins=100, normalize=:pdf, alpha=0.5)
        Plots.density!(p,    gof_results[i].sim_rts; lw=2)
        Plots.xlims!(p, 0, 8)
        subps[i] = p
    end
    grid_rows = ceil(Int, n / 3)
    big = Plots.plot(subps...; layout=(grid_rows, 3),
                     size=(900, 200 * grid_rows),
                     fontfamily="helvetica", link=:both)
    Plots.xlabel!(big, "Reaction Time (s)")
    Plots.ylabel!(big, "Density")
    big
end

# ╔═╡ dada005f-0000-4000-8000-00000000005f
md"""
### Real vs simulated mean accuracy
"""

# ╔═╡ dada0060-0000-4000-8000-000000000060
fig_acc_fit = let
    real_acc = [mean(r.real_corrects) for r in gof_results]
    sim_acc  = [mean(r.sim_corrects)  for r in gof_results]
    p = Plots.scatter(real_acc, sim_acc;
        xlabel="Real accuracy", ylabel="Simulated accuracy",
        title="Model fit: accuracy by rat",
        fontfamily="helvetica", markersize=6, alpha=0.7, size=(400, 400),
        label="")
    Plots.plot!(p, [0.0, 1.0], [0.0, 1.0]; lw=2, ls=:dash, color=:black, label="")
    Plots.xlims!(p, 0.65, 0.9)
    Plots.ylims!(p, 0.65, 0.9)
    p
end

# ╔═╡ dada0061-0000-4000-8000-000000000061
if save_figs
    Plots.savefig(fig_qq,         joinpath(RESULTS_DIR, "qq_population_rts.svg"))
    Plots.savefig(fig_rt_per_rat, joinpath(RESULTS_DIR, "rt_distributions_by_rat.svg"))
    Plots.savefig(fig_acc_fit,    joinpath(RESULTS_DIR, "accuracy_fit_by_rat.svg"))
end

# ╔═╡ dada0062-0000-4000-8000-000000000062
md"""
## §13 Optional — smoothed conditional accuracy function

Helper retained from the original notebook. Not called by any figure here, but
exposed so it can be reused interactively. Pass it RTs and 0/1 correctness.
"""

# ╔═╡ dada0063-0000-4000-8000-000000000063
"""
    smooth_caf(rts, corrects, grid; h=0.2, α=0.05) -> NamedTuple

Kernel-smoothed estimate of `p(correct | RT)` on `grid`, with an approximate
Wald confidence band based on the Kish effective sample size.
"""
function smooth_caf(rts, corrects, grid; h::Real=0.2, α::Real=0.05)
    @assert α == 0.05 "only 95% CI hard-coded"
    z = 1.959963984540054
    y = Float64.(corrects)
    n = length(rts)

    p̂  = Vector{Float64}(undef, length(grid))
    lo = similar(p̂); hi = similar(p̂)
    @inline _gauss(u) = exp(-0.5 * u * u)

    for (j, t) in enumerate(grid)
        w = [_gauss((rts[i] - t) / h) for i in 1:n]
        sw = sum(w)
        if sw == 0
            p̂[j] = lo[j] = hi[j] = NaN
            continue
        end
        phat = sum(w .* y) / sw
        neff = sw^2 / sum(w .^ 2)
        se   = sqrt(max(phat*(1 - phat), 0.0) / max(neff, 1.0))
        p̂[j]  = phat
        lo[j] = clamp(phat - z*se, 0.0, 1.0)
        hi[j] = clamp(phat + z*se, 0.0, 1.0)
    end
    (rt=collect(grid), acc=p̂, acc_lo=lo, acc_hi=hi)
end

# ╔═╡ Cell order:
# ╟─dada0001-0000-4000-8000-000000000001
# ╟─dada0002-0000-4000-8000-000000000002
# ╠═dada0003-0000-4000-8000-000000000003
# ╟─dada0004-0000-4000-8000-000000000004
# ╠═dada0005-0000-4000-8000-000000000005
# ╟─dada0006-0000-4000-8000-000000000006
# ╟─dada0007-0000-4000-8000-000000000007
# ╠═dada0008-0000-4000-8000-000000000008
# ╟─dada0009-0000-4000-8000-000000000009
# ╠═dada000a-0000-4000-8000-00000000000a
# ╠═dada000b-0000-4000-8000-00000000000b
# ╟─dada000c-0000-4000-8000-00000000000c
# ╠═dada000d-0000-4000-8000-00000000000d
# ╠═dada000e-0000-4000-8000-00000000000e
# ╟─dada000f-0000-4000-8000-00000000000f
# ╠═dada0010-0000-4000-8000-000000000010
# ╠═dada0011-0000-4000-8000-000000000011
# ╠═dada0012-0000-4000-8000-000000000012
# ╠═dada0013-0000-4000-8000-000000000013
# ╠═dada0014-0000-4000-8000-000000000014
# ╠═dada0015-0000-4000-8000-000000000015
# ╠═dada0016-0000-4000-8000-000000000016
# ╟─dada0017-0000-4000-8000-000000000017
# ╠═dada0018-0000-4000-8000-000000000018
# ╠═dada0019-0000-4000-8000-000000000019
# ╠═dada001a-0000-4000-8000-00000000001a
# ╠═dada001b-0000-4000-8000-00000000001b
# ╠═dada001c-0000-4000-8000-00000000001c
# ╟─dada001d-0000-4000-8000-00000000001d
# ╠═dada001e-0000-4000-8000-00000000001e
# ╠═dada001f-0000-4000-8000-00000000001f
# ╠═dada0020-0000-4000-8000-000000000020
# ╟─dada0021-0000-4000-8000-000000000021
# ╠═dada0022-0000-4000-8000-000000000022
# ╠═dada0023-0000-4000-8000-000000000023
# ╠═dada0024-0000-4000-8000-000000000024
# ╠═dada0025-0000-4000-8000-000000000025
# ╟─dada0026-0000-4000-8000-000000000026
# ╠═dada0027-0000-4000-8000-000000000027
# ╠═dada0028-0000-4000-8000-000000000028
# ╠═dada0029-0000-4000-8000-000000000029
# ╠═dada002a-0000-4000-8000-00000000002a
# ╠═dada002b-0000-4000-8000-00000000002b
# ╟─dada002c-0000-4000-8000-00000000002c
# ╠═dada002f-0000-4000-8000-00000000002f
# ╠═dada0030-0000-4000-8000-000000000030
# ╠═dada0031-0000-4000-8000-000000000031
# ╟─dada0032-0000-4000-8000-000000000032
# ╠═dada0033-0000-4000-8000-000000000033
# ╠═dada0034-0000-4000-8000-000000000034
# ╠═dada0035-0000-4000-8000-000000000035
# ╟─dada0036-0000-4000-8000-000000000036
# ╠═dada0037-0000-4000-8000-000000000037
# ╠═dada0038-0000-4000-8000-000000000038
# ╠═dada0039-0000-4000-8000-000000000039
# ╠═dada003a-0000-4000-8000-00000000003a
# ╠═dada003b-0000-4000-8000-00000000003b
# ╠═dada003c-0000-4000-8000-00000000003c
# ╟─dada003d-0000-4000-8000-00000000003d
# ╠═dada003e-0000-4000-8000-00000000003e
# ╠═dada003f-0000-4000-8000-00000000003f
# ╠═dada0040-0000-4000-8000-000000000040
# ╠═dada0041-0000-4000-8000-000000000041
# ╠═dada0042-0000-4000-8000-000000000042
# ╠═dada0043-0000-4000-8000-000000000043
# ╠═dada0044-0000-4000-8000-000000000044
# ╠═dada0045-0000-4000-8000-000000000045
# ╠═dada0046-0000-4000-8000-000000000046
# ╟─dada0047-0000-4000-8000-000000000047
# ╠═dada0048-0000-4000-8000-000000000048
# ╠═dada0049-0000-4000-8000-000000000049
# ╠═dada004a-0000-4000-8000-00000000004a
# ╠═dada004b-0000-4000-8000-00000000004b
# ╟─dada004c-0000-4000-8000-00000000004c
# ╠═dada004d-0000-4000-8000-00000000004d
# ╠═dada004e-0000-4000-8000-00000000004e
# ╟─dada004f-0000-4000-8000-00000000004f
# ╠═dada0050-0000-4000-8000-000000000050
# ╠═dada0051-0000-4000-8000-000000000051
# ╠═dada0100-0000-4000-8000-000000000100
# ╠═dada0052-0000-4000-8000-000000000052
# ╠═dada0053-0000-4000-8000-000000000053
# ╟─dada0054-0000-4000-8000-000000000054
# ╠═dada0055-0000-4000-8000-000000000055
# ╠═dada0056-0000-4000-8000-000000000056
# ╠═dada0057-0000-4000-8000-000000000057
# ╟─dada0058-0000-4000-8000-000000000058
# ╠═dada0059-0000-4000-8000-000000000059
# ╠═dada005a-0000-4000-8000-00000000005a
# ╟─dada005b-0000-4000-8000-00000000005b
# ╠═dada005c-0000-4000-8000-00000000005c
# ╟─dada005d-0000-4000-8000-00000000005d
# ╠═dada005e-0000-4000-8000-00000000005e
# ╟─dada005f-0000-4000-8000-00000000005f
# ╠═dada0060-0000-4000-8000-000000000060
# ╠═dada0061-0000-4000-8000-000000000061
# ╟─dada0062-0000-4000-8000-000000000062
# ╠═dada0063-0000-4000-8000-000000000063
