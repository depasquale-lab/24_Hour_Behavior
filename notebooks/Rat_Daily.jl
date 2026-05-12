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

# ╔═╡ bada0002-0000-4000-8000-000000000002
begin
    import Pkg
    Pkg.activate(@__DIR__)

    using BSON: @load
    using CSV
    using DataFrames
    using Dates
    using DriftDiffusionModels
    using HiddenMarkovModels
    using PlotUtils
    using PlutoUI
    using Random
    using Statistics
    using StatsBase
    using StatsPlots
    nothing
end

# ╔═╡ bada0001-0000-4000-8000-000000000001
md"""
# Daily DDM-HMM analysis

Posthoc analyses of K=4 DDM-HMMs fit on the *daily* (non-24-hour) data:
population ACF posterior predictive check, an example posterior decoding,
and per-state weighted RT histograms using the joint-normalised scheme
from `PosthocDDMHMM.jl`.

| § | Section |
|---|---------|
| 1 | Setup & data loading |
| 2 | Population RT-ACF posterior predictive |
| 3 | Example posterior decoding |
| 4 | Per-state RT histograms |
| 5 | Marginal RT distribution |
| 6 | Save figures |
"""

# ╔═╡ bada0003-0000-4000-8000-000000000003
md"""
## §1 Setup & data loading
"""

# ╔═╡ bada0004-0000-4000-8000-000000000004
begin
    # Rats with K=4 daily fits available on disk.
    rat_names   = ["Daenerys", "Dobby", "Dory", "Regina", "Rhubarb"]
    example_rat = "Dobby"
end

# ╔═╡ bada0005-0000-4000-8000-000000000005
begin
    # BSON serialises type references as a literal module path starting at
    # :Main, so deserialisation tries to resolve `Main.DDMHMMFit`. In Pluto
    # cells live in `Main.var"workspace#N"`, so define the struct in the real
    # `Main` module to make it visible to BSON.
    if !isdefined(Main, :DDMHMMFit)
        Core.eval(Main, :(struct DDMHMMFit
            hmm
            logL
            logL_evolution
        end))
    end
    DDMHMMFit = Main.DDMHMMFit

    data_dir = joinpath(@__DIR__, "..", "data")
    bson_files = filter(f -> occursin("K4", f) && occursin("daily", f),
                        readdir(data_dir; join = true))

    # Rat name is the first underscore-separated token of the filename.
    fit_names = [String(split(basename(f), "_")[1]) for f in bson_files]

    ddmhmm_fits = Main.DDMHMMFit[]
    for f in bson_files
        @load f fit
        push!(ddmhmm_fits, fit)
    end
    (fit_names, length(ddmhmm_fits))
end

# ╔═╡ bada0006-0000-4000-8000-000000000006
begin
    real_data = CSV.read(joinpath(@__DIR__, "..", "data", "processed_rat_data.csv.gz"),
                         DataFrame)
    real_data = real_data[real_data.daily .== "daily", :]
    real_data = real_data[in.(real_data.name, Ref(rat_names)), :]

    replace!(real_data[!, :choose_right], 0 => -1)
    mapping = Dict("right" => 1, "left" => -1)
    DataFrames.transform!(real_data,
        :correct_side => ByRow(cs -> get(mapping, cs, missing)) => :correct_side_numeric)
    real_data
end

# ╔═╡ bada0007-0000-4000-8000-000000000007
md"""
## §2 Population RT-ACF posterior predictive

Empirical grand-average RT autocorrelation across rats vs. RT-ACFs simulated
from each fitted DDM-HMM.
"""

# ╔═╡ bada0008-0000-4000-8000-000000000008
function per_rat_mean_acf(df::DataFrame, name::AbstractString; lags = 1:20)
    sub = df[df.name .== name, :]
    dates = [Date(split(dt)[1]) for dt in sub.trial_datetime]
    unique_dates = sort(unique(dates))

    day_acfs = Vector{Vector{Float64}}()
    for date in unique_dates
        idx = findall(dates .== date)
        isempty(idx) && continue
        rts = sub.rt[idx]
        length(rts) <= maximum(lags) + 1 && continue
        push!(day_acfs, autocor(rts, lags))
    end
    isempty(day_acfs) && return fill(NaN, length(lags))
    return vec(mean(reduce(hcat, day_acfs); dims = 2))
end

# ╔═╡ bada0009-0000-4000-8000-000000000009
begin
    lags          = 1:20
    per_rat_acfs  = [per_rat_mean_acf(real_data, n; lags = lags) for n in fit_names]
    grand_avg_acf = vec(mean(reduce(hcat, per_rat_acfs); dims = 2))
    grand_avg_acf
end

# ╔═╡ bada000a-0000-4000-8000-00000000000a
function mean_acf_over_rats(hmms::Vector; n_tsteps::Int = 1000, lags = 1:20, burn::Int = 0)
    A = Matrix{Float64}(undef, length(lags), length(hmms))
    for (j, hmm) in enumerate(hmms)
        _, o = rand(hmm, n_tsteps + burn)
        rts  = [res.rt for res in o]
        burn > 0 && (rts = rts[(burn + 1):end])
        A[:, j] = autocor(rts, lags)
    end
    return vec(mean(A; dims = 2))
end

# ╔═╡ bada000b-0000-4000-8000-00000000000b
function ppc_population_acf(hmms::Vector;
                            n_rep::Int = 100, n_tsteps::Int = 1000,
                            lags = 1:20, burn::Int = 0)
    reps = Matrix{Float64}(undef, length(lags), n_rep)
    for r in 1:n_rep
        reps[:, r] = mean_acf_over_rats(hmms; n_tsteps = n_tsteps, lags = lags, burn = burn)
    end
    pred_mean = vec(mean(reps; dims = 2))
    lo = [quantile(view(reps, i, :), 0.025) for i in 1:size(reps, 1)]
    hi = [quantile(view(reps, i, :), 0.975) for i in 1:size(reps, 1)]
    return pred_mean, lo, hi, reps
end

# ╔═╡ bada000c-0000-4000-8000-00000000000c
begin
    hmms = [f.hmm for f in ddmhmm_fits]
    pred_mean, lo, hi, _M = ppc_population_acf(hmms; n_rep = 100, n_tsteps = 1000, lags = lags)
    (pred_mean[1:3], length(hmms))
end

# ╔═╡ bada000d-0000-4000-8000-00000000000d
acf_ppc_plot = begin
    p = plot(collect(lags), pred_mean;
        ribbon     = (pred_mean .- lo, hi .- pred_mean),
        label      = "DDM-HMM PPC (95%)",
        alpha      = 0.5,
        linewidth  = 3,
        fontfamily = "helvetica",
        xlabel     = "Lag",
        ylabel     = "ACF",
    )
    plot!(p, collect(lags), grand_avg_acf;
        label     = "Real data",
        color     = :black,
        linewidth = 3)
    p
end

# ╔═╡ bada000e-0000-4000-8000-00000000000e
md"""
## §3 Example posterior decoding

Forward-backward posterior for **$(example_rat)** over the trial-concatenated
session sequence; one short window is shown.
"""

# ╔═╡ bada000f-0000-4000-8000-00000000000f
function build_results_by_date(df::DataFrame, name::AbstractString)
    sub = df[df.name .== name, :]
    dates = [Date(split(dt)[1]) for dt in sub.trial_datetime]
    unique_dates = sort(unique(dates))

    results_by_date = Vector{Vector{DDMResult}}()
    for date in unique_dates
        idx = findall(dates .== date)
        isempty(idx) && continue
        rts   = sub.rt[idx]
        outs  = sub.choose_right[idx]
        stims = sub.correct_side_numeric[idx]
        push!(results_by_date,
              [DDMResult(rt, c, s) for (rt, c, s) in zip(rts, outs, stims)])
    end
    return results_by_date
end

# ╔═╡ bada0010-0000-4000-8000-000000000010
begin
    example_idx  = findfirst(==(example_rat), fit_names)
    example_idx === nothing && error("$example_rat has no K4 daily fit.")
    example_hmm  = ddmhmm_fits[example_idx].hmm

    results_by_date = build_results_by_date(real_data, example_rat)
    all_results     = reduce(vcat, results_by_date)
    seq_ends        = cumsum(length.(results_by_date))

    γ, _ll = forward_backward(example_hmm, all_results; seq_ends = seq_ends)
    K, T   = size(γ)
    (example_idx, K, T)
end

# ╔═╡ bada0011-0000-4000-8000-000000000011
posterior_window_plot = begin
    # Show one day's worth (use the 3rd day if available, else the first).
    day_idx = min(3, length(results_by_date))
    start_t = day_idx == 1 ? 1 : (cumsum(length.(results_by_date))[day_idx - 1] + 1)
    stop_t  = cumsum(length.(results_by_date))[day_idx]

    p2 = plot(
        xlabel     = "Trial",
        ylabel     = "P(State)",
        fontfamily = "helvetica",
        lw         = 3,
        legend     = :outerright,
        title      = "$example_rat — day $day_idx posterior",
    )
    for s in 1:K
        plot!(p2, start_t:stop_t, γ[s, start_t:stop_t]; label = "State $s")
    end
    p2
end

# ╔═╡ bada0012-0000-4000-8000-000000000012
md"""
## §4 Per-state RT histograms

Joint-normalised empirical RTs (weighted by posterior γ) overlaid with
simulated RT KDEs from the fitted HMM, split by state and correctness —
same scheme as `PosthocDDMHMM.jl`.
"""

# ╔═╡ bada0013-0000-4000-8000-000000000013
begin
    rts        = [res.rt for res in all_results]
    is_correct = [res.choice == res.s for res in all_results]
    correct_idx   = findall(is_correct)
    incorrect_idx = findall(.!is_correct)

    n_sims = 30_000
    sim_states, sim_data = rand(example_hmm, n_sims)

    simulated_rts = Dict{Int, Dict{String, Vector{Float64}}}()
    for s in 1:K
        simulated_rts[s] = Dict("correct" => Float64[], "incorrect" => Float64[])
        for trial_idx in findall(==(s), sim_states)
            res = sim_data[trial_idx]
            category = res.choice == res.s ? "correct" : "incorrect"
            push!(simulated_rts[s][category], res.rt)
        end
    end

    state_colors = distinguishable_colors(K)
    nothing
end

# ╔═╡ bada0014-0000-4000-8000-000000000014
# Joint normalisation: per state s, ∫correct + ∫incorrect = 1.
# Each panel's integral equals the (soft) probability of that outcome in state s.
rt_hist_plot = begin
    pA = plot(layout = (K, 2),
              size   = (900, 300 * K),
              link   = :x,
              fontfamily = "helvetica")

    edges = range(0, 8; length = 51)

    for s in 1:K
        idx_correct   = 2 * (s - 1) + 1
        idx_incorrect = 2 * (s - 1) + 2

        γs = vec(γ[s, :])
        Ws = sum(γs)

        rts_c = rts[correct_idx]
        if !isempty(rts_c) && Ws > 0
            histogram!(pA[idx_correct], rts_c;
                weights   = γs[correct_idx] ./ Ws,
                bins      = edges,
                normalize = :density,
                fillalpha = 0.4,
                linealpha = 0.0,
                color     = state_colors[s],
                label     = s == 1 ? "Real (joint-norm)" : "")
        end

        rts_i = rts[incorrect_idx]
        if !isempty(rts_i) && Ws > 0
            histogram!(pA[idx_incorrect], rts_i;
                weights   = γs[incorrect_idx] ./ Ws,
                bins      = edges,
                normalize = :density,
                fillalpha = 0.4,
                linealpha = 0.0,
                color     = state_colors[s],
                label     = "")
        end

        sim_rts_c = get(simulated_rts[s], "correct",   Float64[])
        sim_rts_i = get(simulated_rts[s], "incorrect", Float64[])
        N_state   = length(sim_rts_c) + length(sim_rts_i)

        if length(sim_rts_c) > 1 && N_state > 0
            kc = StatsPlots.KernelDensity.kde(sim_rts_c)
            p_sim_c = length(sim_rts_c) / N_state
            plot!(pA[idx_correct], kc.x, kc.density .* p_sim_c;
                linewidth = 2,
                color     = :black,
                label     = s == 1 ? "Sim" : "")
        end

        if length(sim_rts_i) > 1 && N_state > 0
            ki = StatsPlots.KernelDensity.kde(sim_rts_i)
            p_sim_i = length(sim_rts_i) / N_state
            plot!(pA[idx_incorrect], ki.x, ki.density .* p_sim_i;
                linewidth = 2,
                color     = :black,
                label     = "")
        end

        plot!(pA[idx_correct];
            xlabel = "RT (s)",
            ylabel = "State $s\nDensity",
            title  = s == 1 ? "Correct -- $example_rat" : "",
            legend = (s == 1))
        plot!(pA[idx_incorrect];
            xlabel = "RT (s)",
            ylabel = "",
            title  = s == 1 ? "Incorrect -- $example_rat" : "",
            legend = false)
    end

    # Match y-limits across each row
    for s in 1:K
        ic = 2 * (s - 1) + 1
        ii = 2 * (s - 1) + 2
        y1_min, y1_max = Plots.ylims(pA[ic])
        y2_min, y2_max = Plots.ylims(pA[ii])
        row_min = min(y1_min, y2_min)
        row_max = max(y1_max, y2_max)
        ylims!(pA[ic], row_min, row_max)
        ylims!(pA[ii], row_min, row_max)
    end

    xlims!(0, 8)
    pA
end

# ╔═╡ bada0015-0000-4000-8000-000000000015
md"""
## §5 Marginal RT distribution

Whole-session empirical RTs vs. simulated RT KDE — same scheme as the
per-state panels, just collapsed across states.
"""

# ╔═╡ bada0016-0000-4000-8000-000000000016
marginal_rt_plot = begin
    sim_rts_all = [res.rt for res in sim_data]
    edges_m = range(0, 8; length = 51)

    pM = plot(fontfamily = "helvetica",
              xlabel = "RT (s)",
              ylabel = "Density",
              title  = "$example_rat — marginal RT")

    histogram!(pM, rts;
        bins      = edges_m,
        normalize = :density,
        fillalpha = 0.4,
        linealpha = 0.0,
        color     = state_colors[1],
        label     = "Real")

    if length(sim_rts_all) > 1
        km = StatsPlots.KernelDensity.kde(sim_rts_all)
        plot!(pM, km.x, km.density;
            linewidth = 2,
            color     = :black,
            label     = "DDM-HMM sim")
    end
    xlims!(pM, 0, 8)
    pM
end

# ╔═╡ bada0017-0000-4000-8000-000000000017
md"""
## §6 Save figures
"""

# ╔═╡ bada0018-0000-4000-8000-000000000018
md"""
Save all figures to `../results/`: $(@bind save_figs CheckBox(default = false))
"""

# ╔═╡ bada0019-0000-4000-8000-000000000019
begin
    if save_figs
        results_dir = joinpath(@__DIR__, "..", "results")
        mkpath(results_dir)
        tag = lowercase(String(example_rat))
        figs = [
            (acf_ppc_plot,         "rat_daily_acf_ppc.svg"),
            (posterior_window_plot,"rat_daily_$(tag)_posterior.svg"),
            (rt_hist_plot,         "rat_daily_$(tag)_rt_distributions.svg"),
            (marginal_rt_plot,     "rat_daily_$(tag)_marginal_rt.svg"),
        ]
        for (fig, name) in figs
            savefig(fig, joinpath(results_dir, name))
        end
        md"Saved $(length(figs)) figures to `$results_dir`."
    else
        md"_Tick the box above to save all figures._"
    end
end

# ╔═╡ Cell order:
# ╟─bada0001-0000-4000-8000-000000000001
# ╠═bada0002-0000-4000-8000-000000000002
# ╟─bada0003-0000-4000-8000-000000000003
# ╠═bada0004-0000-4000-8000-000000000004
# ╠═bada0005-0000-4000-8000-000000000005
# ╠═bada0006-0000-4000-8000-000000000006
# ╟─bada0007-0000-4000-8000-000000000007
# ╠═bada0008-0000-4000-8000-000000000008
# ╠═bada0009-0000-4000-8000-000000000009
# ╠═bada000a-0000-4000-8000-00000000000a
# ╠═bada000b-0000-4000-8000-00000000000b
# ╠═bada000c-0000-4000-8000-00000000000c
# ╠═bada000d-0000-4000-8000-00000000000d
# ╟─bada000e-0000-4000-8000-00000000000e
# ╠═bada000f-0000-4000-8000-00000000000f
# ╠═bada0010-0000-4000-8000-000000000010
# ╠═bada0011-0000-4000-8000-000000000011
# ╟─bada0012-0000-4000-8000-000000000012
# ╠═bada0013-0000-4000-8000-000000000013
# ╠═bada0014-0000-4000-8000-000000000014
# ╟─bada0015-0000-4000-8000-000000000015
# ╠═bada0016-0000-4000-8000-000000000016
# ╟─bada0017-0000-4000-8000-000000000017
# ╟─bada0018-0000-4000-8000-000000000018
# ╠═bada0019-0000-4000-8000-000000000019
