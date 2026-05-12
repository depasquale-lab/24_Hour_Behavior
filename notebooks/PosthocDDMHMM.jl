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

# ╔═╡ aada0002-0000-4000-8000-000000000002
begin
    import Pkg
    Pkg.activate(@__DIR__)

    using BSON: @load
    using CSV
    using DataFrames
    using Dates
    using Distributions
    using DriftDiffusionModels
    using HiddenMarkovModels
    using PlotUtils
    using PlutoUI
    using Printf
    using Random
    using Statistics
    using StatsBase
    using StatsPlots
    using Base.Threads: @threads
    nothing
end

# ╔═╡ aada0001-0000-4000-8000-000000000001
md"""
# Posthoc DDM-HMM analysis

Per-rat posthoc analyses of a fitted DDM-HMM:
posterior decoding, hour-of-day state occupancy, predicted vs empirical accuracy,
weighted RT histograms, per-state DDM parameters, and an autocorrelation PPC.

| § | Section |
|---|---------|
| 1 | Setup & data loading |
| 2 | Posterior decoding (forward-backward) |
| 3 | Posterior plots |
| 4 | State occupancy by hour |
| 5 | RT histograms |
| 6 | Per-state parameter plots |
| 7 | ACF posterior predictive check |
| 8 | Save figures |
"""

# ╔═╡ aada0003-0000-4000-8000-000000000003
md"""
## §1 Setup & data loading
"""

# ╔═╡ aada0004-0000-4000-8000-000000000004
rat = "1062"

# ╔═╡ aada0006-0000-4000-8000-000000000006
begin
    # BSON serialises type references as a literal module path starting at
    # :Main, so deserialisation tries to resolve `Main.DDMHMMFit`. In Pluto our
    # cells live in `Main.var"workspace#N"`, not `Main`, so a struct defined
    # here is invisible to BSON. Define it in the real `Main` module instead.
    if !isdefined(Main, :DDMHMMFit)
        Core.eval(Main, :(struct DDMHMMFit
            hmm
            logL
            logL_evolution
        end))
    end
    DDMHMMFit = Main.DDMHMMFit

    path_to_rat = joinpath(@__DIR__, "..", "data", rat * "_K4_best.bson")
    @load path_to_rat fit
    hmm = fit.hmm
end

# ╔═╡ aada0007-0000-4000-8000-000000000007
begin
    data_file = joinpath(@__DIR__, "..", "data", "processed_rat_data.csv.gz")

    rat_df = CSV.read(data_file, DataFrame)
    rat_df = rat_df[rat_df.daily .== "24 hr", :]   # only keep 24-hour data

    # numerics for choices / sides
    replace!(rat_df[!, :choose_right], 0 => -1)
    mapping = Dict("right" => 1, "left" => -1)
    DataFrames.transform!(rat_df,
        :correct_side => ByRow(cs -> get(mapping, cs, missing)) => :correct_side_numeric)
    rat_df
end

# ╔═╡ aada0008-0000-4000-8000-000000000008
begin
    rat_of_interest = rat_df[rat_df.name .== rat, :]
    dates = [Date(split(dt)[1]) for dt in rat_of_interest.trial_datetime]

    unique_dates = sort(unique(dates))

    results_by_date = Vector{Vector{DDMResult}}()
    for date in unique_dates
        day_indices = findall(dates .== date)
        if isempty(day_indices)
            continue
        end

        day_rts       = rat_of_interest.rt[day_indices]
        day_outcomes  = rat_of_interest.choose_right[day_indices]
        day_stim_side = rat_of_interest.correct_side_numeric[day_indices]

        day_results = [DDMResult(rt, choice, stim) for (rt, choice, stim) in
                       zip(day_rts, day_outcomes, day_stim_side)]
        push!(results_by_date, day_results)
    end

    all_results = vcat(results_by_date...)
    seq_ends    = cumsum(length.(results_by_date))
    seq_ends
end

# ╔═╡ aada0009-0000-4000-8000-000000000009
md"""
## §2 Posterior decoding
"""

# ╔═╡ aada000a-0000-4000-8000-00000000000a
γ, ll = forward_backward(hmm, all_results; seq_ends=seq_ends)

# ╔═╡ aada000b-0000-4000-8000-00000000000b
md"""
## §3 Posterior plots

Two short windows from the concatenated trial sequence (one near a "night"
chunk and one near a "day" chunk) to illustrate state dynamics.
"""

# ╔═╡ aada000c-0000-4000-8000-00000000000c
posterior_plot_night = begin
    pn = plot(
        xlabel="Trial",
        ylabel="P(State)",
        fontfamily="helvetica",
        lw=3,
        legend=false,
    )
    for s in 1:size(γ, 1)
        plot!(pn, γ[s, 1109:1109+500])
    end
    pn
end

# ╔═╡ aada000d-0000-4000-8000-00000000000d
posterior_plot_day = begin
    pd = plot(
        xlabel="Trial",
        ylabel="P(State)",
        fontfamily="helvetica",
        lw=3,
        legend=false,
    )
    for s in 1:size(γ, 1)
        plot!(pd, γ[s, 1109+1500:1109+2000])
    end
    pd
end

# ╔═╡ aada000e-0000-4000-8000-00000000000e
md"""
## §4 State occupancy by hour
"""

# ╔═╡ aada000f-0000-4000-8000-00000000000f
const _dt_formats = (
    dateformat"yyyy-mm-dd HH:MM:SS.s",
    dateformat"yyyy-mm-dd HH:MM:SS",
    dateformat"yyyy-mm-ddTHH:MM:SS.s",
    dateformat"yyyy-mm-ddTHH:MM:SS",
)

# ╔═╡ aada0010-0000-4000-8000-000000000010
parse_dt(x) = x isa DateTime ? x :
              x isa Date     ? DateTime(x) :
              begin
                  s = String(x)
                  for df in _dt_formats
                      try
                          return DateTime(s, df)
                      catch
                      end
                  end
                  error("Couldn't parse trial_datetime string: $x. Add your format to _dt_formats.")
              end

# ╔═╡ aada0011-0000-4000-8000-000000000011
begin
    # Build the index order used when concatenating trials (aligns with γ).
    idxs_by_date = Vector{Vector{Int}}()
    for date in unique_dates
        day_indices = findall(dates .== date)
        push!(idxs_by_date, day_indices)
    end
    concat_idxs  = reduce(vcat, idxs_by_date)
    trial_dt_col = rat_of_interest.trial_datetime
    hours_concat = [mod(hour(parse_dt(trial_dt_col[i])) - 7.5, 24) for i in concat_idxs]
    hours_concat
end

# ╔═╡ aada0012-0000-4000-8000-000000000012
begin
    occ_soft   = zeros(24, 4)
    n_per_hour = zeros(Int, 24)

    for h in 0:23
        idx_h = findall(x -> floor(Int, x) == h, hours_concat)
        n = length(idx_h)
        n_per_hour[h+1] = n
        if n > 0
            m = dropdims(mean(γ[:, idx_h]; dims=2); dims=2)
            occ_soft[h+1, :] = permutedims(m)
        else
            occ_soft[h+1, :] .= 0.0
        end
    end

    println("max deviation from 1: ",
            maximum(abs.(sum(occ_soft, dims=2) .- 1)))

    # Mask empty hours so they don't show partial stacks
    y = occ_soft .* 100
    for h in 1:24
        if n_per_hour[h] == 0
            y[h, :] .= NaN
        end
    end
    y
end

# ╔═╡ aada0013-0000-4000-8000-000000000013
state_occupancy_plot = begin
    state_labels = ["S1", "S2", "S3", "S4"]
    hours        = 0:23
    order        = [4, 2, 3, 1]

    p = plot(fontfamily="helvetica")
    groupedbar!(p, hours, occ_soft[:, order] .* 100;
        bar_position = :stack,
        xlabel       = "Time From Light On (H)",
        ylabel       = "State occupancy (%)",
        labels       = permutedims(state_labels[order]),
        title        = "HMM-DDM soft state occupancy by hour — $rat",
        fontfamily   = "helvetica")
    p
end

# ╔═╡ aada0014-0000-4000-8000-000000000014
y

# ╔═╡ aada0015-0000-4000-8000-000000000015
function pred_accuracy_by_hour(hmm::PriorHMM, state_occ_hour::AbstractMatrix, trials_per_bin::Int=1000)
    K = length(hmm.dists)
    n_hours = size(state_occ_hour, 1)

    sim_acc_hour = zeros(trials_per_bin, n_hours)
    @threads for i in 1:n_hours
        ddm_cat = Categorical(state_occ_hour[i, :])
        for j in 1:trials_per_bin
            ddm_result = simulateDDM(hmm.dists[rand(ddm_cat)])
            sim_acc_hour[j, i] = ddm_result.choice == ddm_result.s
        end
    end
    mean_acc_hour = dropdims(mean(sim_acc_hour; dims=1); dims=1)
    return mean_acc_hour, sim_acc_hour
end

# ╔═╡ aada0016-0000-4000-8000-000000000016
begin
    # Trial-aligned vectors (match γ columns)
    stim_concat   = rat_of_interest.correct_side_numeric[concat_idxs]   # -1/1
    choice_concat = rat_of_interest.choose_right[concat_idxs]           # -1/1
    hour_bin      = [floor(Int, h) for h in hours_concat]               # 0..23

    is_correct_concat = choice_concat .== stim_concat

    emp_acc = fill(NaN, 24)
    for h in 0:23
        idx = findall(==(h), hour_bin)
        if !isempty(idx)
            emp_acc[h+1] = mean(is_correct_concat[idx])
        end
    end
    emp_acc
end

# ╔═╡ aada0017-0000-4000-8000-000000000017
begin
    mean_acc, sim_acc = pred_accuracy_by_hour(hmm, occ_soft, 10_000)
    mean_acc
end

# ╔═╡ aada0018-0000-4000-8000-000000000018
md"""
## §5 RT histograms

Empirical RTs (soft-weighted by posterior) overlaid with simulated RT
distributions from the fitted HMM, split by state and correctness.
"""

# ╔═╡ aada0019-0000-4000-8000-000000000019
begin
    rat_str = String(rat)
    K, T    = size(γ)
    @assert length(all_results) == T "γ and all_results must have same length"

    rts        = [res.rt for res in all_results]
    is_correct = [res.choice == res.s for res in all_results]

    correct_idx   = findall(is_correct)
    incorrect_idx = findall(.!is_correct)
    (rat_str, K, T)
end

# ╔═╡ aada001a-0000-4000-8000-00000000001a
begin
    n_sims = 30_000
    sim_states, sim_data = rand(hmm, n_sims)

    simulated_rts = Dict{Int, Dict{String, Vector{Float64}}}()
    for s in 1:K
        simulated_rts[s] = Dict("correct" => Float64[], "incorrect" => Float64[])
        state_s_trials = findall(x -> x == s, sim_states)
        for trial_idx in state_s_trials
            res = sim_data[trial_idx]
            category = (res.choice == res.s) ? "correct" : "incorrect"
            push!(simulated_rts[s][category], res.rt)
        end
    end

    state_colors = distinguishable_colors(K)
    simulated_rts
end

# ╔═╡ aada001b-0000-4000-8000-00000000001b
# Joint normalization: per state s, ∫correct + ∫incorrect = 1.
# Each panel's integral equals the (soft) probability of that outcome in state s.
rt_hist_plot = begin
    pA = plot(layout = (K, 2),
              size   = (900, 300*K),
              link   = :x,
              fontfamily = "helvetica")

    # shared bin edges so correct/incorrect panels are directly comparable
    edges = range(0, 8; length = 51)

    for s in 1:K
        idx_correct   = 2*(s-1) + 1
        idx_incorrect = 2*(s-1) + 2

        γs = vec(γ[s, :])
        Ws = sum(γs)                      # joint state weight (denominator)

        # empirical: weights γ/Ws + normalize=:density -> integrals sum to 1 
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

        # simulated: KDE scaled by p(correct|state) / p(incorrect|state)
        # so ∫sim_correct + ∫sim_incorrect = 1, matching the empirical scheme.
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
            title  = s == 1 ? "Correct -- $rat_str" : "",
            legend = (s == 1))
        plot!(pA[idx_incorrect];
            xlabel = "RT (s)",
            ylabel = "",
            title  = s == 1 ? "Incorrect -- $rat_str" : "",
            legend = false)
    end

    # Match y-limits across each row
    for s in 1:K
        ic = 2*(s-1) + 1
        ii = 2*(s-1) + 2
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

# ╔═╡ aada001c-0000-4000-8000-00000000001c
md"""
## §6 Per-state parameter plots
"""

# ╔═╡ aada001d-0000-4000-8000-00000000001d
begin
    accuracies_soft = Float64[]
    N_soft          = Float64[]

    for s in 1:K
        γs = vec(γ[s, :])
        Ns = sum(γs)                    # expected # trials in state s
        Cs = sum(γs[correct_idx])       # expected # correct
        push!(N_soft, Ns)
        push!(accuracies_soft, Ns > 0 ? 100 * Cs / Ns : NaN)
    end
    accuracies_soft
end

# ╔═╡ aada001e-0000-4000-8000-00000000001e
soft_accuracy_plot = begin
    xs = 1:K
    xtick_labels = ["State $s\n(nH$(round(Int, N_soft[s])))" for s in 1:K]

    pB = bar(xs, accuracies_soft;
        bar_width  = 0.7,
        color      = state_colors,
        ylim       = (0, 100),
        ylabel     = "Accuracy (%)",
        xticks     = (xs, xtick_labels),
        legend     = false,
        fontfamily = "helvetica",
        title      = "Soft-assigned accuracy -- $rat_str")

    for s in 1:K
        if !isnan(accuracies_soft[s])
            annotate!(pB, (s, accuracies_soft[s] + 3,
                text(@sprintf("%.1f%%", accuracies_soft[s]),
                     8, :black, :center)))
        end
    end
    ylims!(pB, 50, 90)
    pB
end

# ╔═╡ aada001f-0000-4000-8000-00000000001f
function plot_ddm_params(hmm_est;
                         accuracies_soft = nothing,
                         N_soft          = nothing,
                         state_colors    = nothing,
                         rat_str         = "",
                         fontfamily      = "helvetica")

    ddms   = hmm_est.dists
    K      = length(ddms)
    states = 1:K

    drifts = [d.v  for d in ddms]
    bounds = [d.B  for d in ddms]
    biases = [d.a₀ for d in ddms]
    t0s    = [d.τ  for d in ddms]

    if state_colors === nothing
        state_colors = distinguishable_colors(K)
    end

    xticks_states = (states, ["S$s" for s in states])

    p_v = bar(states, drifts; color=state_colors, legend=false,
              xticks=xticks_states, ylabel="Drift v",
              title="Drift by state - $rat_str", fontfamily=fontfamily)
    p_a = bar(states, bounds; color=state_colors, legend=false,
              xticks=xticks_states, ylabel="Boundary a",
              title="Boundary by state", fontfamily=fontfamily)
    p_z = bar(states, biases; color=state_colors, legend=false,
              xticks=xticks_states, ylabel="Bias z",
              title="Starting point bias", fontfamily=fontfamily)
    p_t = bar(states, t0s; color=state_colors, legend=false,
              xticks=xticks_states, ylabel="τ (s)",
              title="Non-decision time", fontfamily=fontfamily)

    if accuracies_soft !== nothing || N_soft !== nothing
        for s in states
            label_str = ""
            if N_soft !== nothing
                label_str *= "nH$(round(Int, N_soft[s]))"
            end
            if accuracies_soft !== nothing
                if !isempty(label_str); label_str *= ", "; end
                label_str *= @sprintf("%.1f%%", accuracies_soft[s])
            end

            if !isempty(label_str)
                annotate!(p_v, (s, drifts[s] + 0.05*maximum(abs.(drifts))),
                          text(label_str, 7, :black, :center))
                annotate!(p_a, (s, bounds[s] + 0.05*maximum(abs.(bounds))),
                          text(label_str, 7, :black, :center))
                annotate!(p_z, (s, biases[s] + 0.05*maximum(abs.(biases))),
                          text(label_str, 7, :black, :center))
                annotate!(p_t, (s, t0s[s] + 0.05*maximum(abs.(t0s))),
                          text(label_str, 7, :black, :center))
            end
        end
    end

    plot(p_v, p_a, p_z, p_t; layout = (2, 2))
end

# ╔═╡ aada0020-0000-4000-8000-000000000020
ddm_params_plot = plot_ddm_params(hmm;
    accuracies_soft = accuracies_soft,
    N_soft          = N_soft,
    state_colors    = state_colors,
    rat_str         = rat_str)

# ╔═╡ aada0021-0000-4000-8000-000000000021
md"""
## §7 ACF posterior predictive check
"""

# ╔═╡ aada0022-0000-4000-8000-000000000022
begin
    acfs = Vector{Vector{Float64}}(undef, length(results_by_date))
    for i in eachindex(results_by_date)
        rts_i      = [x.rt for x in results_by_date[i]]
        acf_vals   = autocor(rts_i, 1:20)
        acfs[i]    = acf_vals
    end
    mean_acfs_real = mean(acfs, dims=1)
    mean_acfs_real
end

# ╔═╡ aada0023-0000-4000-8000-000000000023
begin
    nsims    = 10_000
    sim_acfs = Matrix{Float64}(undef, 20, nsims)
    @threads for i in 1:nsims
        _states, _data = rand(hmm, 1000)
        rts_sim        = [x.rt for x in _data]
        sim_acfs[:, i] = autocor(rts_sim, 1:20)
    end

    mean_sim_acfs = mean(sim_acfs, dims=2)
    ci_low  = [quantile(x, 0.03) for x in eachrow(sim_acfs)]
    ci_high = [quantile(x, 0.97) for x in eachrow(sim_acfs)]
    (mean_sim_acfs, ci_low, ci_high)
end

# ╔═╡ aada0024-0000-4000-8000-000000000024
auto_corr_ppc = begin
    lags = 1:20
    acp  = plot(
        lags, mean_sim_acfs;
        ribbon = (mean_sim_acfs .- ci_low, ci_high .- mean_sim_acfs),
        label  = "PPC mean ± 94% CI",
        xlabel = "Lag",
        ylabel = "ACF",
        fontfamily = "helvetica",
    )
    plot!(acp, lags, mean_acfs_real;
        seriestype = :scatter,
        marker     = :circle,
        label      = "data mean ACF")
    acp
end

# ╔═╡ aada0027-0000-4000-8000-000000000027
md"""
## §8 Save figures
"""

# ╔═╡ aada0028-0000-4000-8000-000000000028
md"""
Save all figures to `../results/`: $(@bind save_figs CheckBox(default=false))
"""

# ╔═╡ aada0029-0000-4000-8000-000000000029
begin
    if save_figs
        results_dir = joinpath(@__DIR__, "..", "results")
        mkpath(results_dir)
        rat_tag = lowercase(String(rat))
        figs = [
            (posterior_plot_night, "posterior_night"),
            (posterior_plot_day,   "posterior_day"),
            (state_occupancy_plot, "state_occupancy"),
            (rt_hist_plot,         "rt_hist"),
            (soft_accuracy_plot,   "soft_accuracy"),
            (ddm_params_plot,      "ddm_params"),
            (auto_corr_ppc,        "acf_ppc"),
        ]
        for (fig, name) in figs
            savefig(fig, joinpath(results_dir, "posthoc_$(rat_tag)_$(name).svg"))
        end
        md"Saved $(length(figs)) figures to `$results_dir`."
    else
        md"_Tick the box above to save all figures._"
    end
end

# ╔═╡ 98a31a09-62b6-4f47-baa9-c005613f48da
hmm.trans

# ╔═╡ Cell order:
# ╟─aada0001-0000-4000-8000-000000000001
# ╠═aada0002-0000-4000-8000-000000000002
# ╟─aada0003-0000-4000-8000-000000000003
# ╠═aada0004-0000-4000-8000-000000000004
# ╠═aada0006-0000-4000-8000-000000000006
# ╠═aada0007-0000-4000-8000-000000000007
# ╠═aada0008-0000-4000-8000-000000000008
# ╟─aada0009-0000-4000-8000-000000000009
# ╠═aada000a-0000-4000-8000-00000000000a
# ╟─aada000b-0000-4000-8000-00000000000b
# ╠═aada000c-0000-4000-8000-00000000000c
# ╠═aada000d-0000-4000-8000-00000000000d
# ╟─aada000e-0000-4000-8000-00000000000e
# ╠═aada000f-0000-4000-8000-00000000000f
# ╠═aada0010-0000-4000-8000-000000000010
# ╠═aada0011-0000-4000-8000-000000000011
# ╠═aada0012-0000-4000-8000-000000000012
# ╠═aada0013-0000-4000-8000-000000000013
# ╠═aada0014-0000-4000-8000-000000000014
# ╠═aada0015-0000-4000-8000-000000000015
# ╠═aada0016-0000-4000-8000-000000000016
# ╠═aada0017-0000-4000-8000-000000000017
# ╟─aada0018-0000-4000-8000-000000000018
# ╠═aada0019-0000-4000-8000-000000000019
# ╠═aada001a-0000-4000-8000-00000000001a
# ╠═aada001b-0000-4000-8000-00000000001b
# ╟─aada001c-0000-4000-8000-00000000001c
# ╠═aada001d-0000-4000-8000-00000000001d
# ╠═aada001e-0000-4000-8000-00000000001e
# ╠═aada001f-0000-4000-8000-00000000001f
# ╠═aada0020-0000-4000-8000-000000000020
# ╟─aada0021-0000-4000-8000-000000000021
# ╠═aada0022-0000-4000-8000-000000000022
# ╠═aada0023-0000-4000-8000-000000000023
# ╠═aada0024-0000-4000-8000-000000000024
# ╟─aada0027-0000-4000-8000-000000000027
# ╟─aada0028-0000-4000-8000-000000000028
# ╠═aada0029-0000-4000-8000-000000000029
# ╠═98a31a09-62b6-4f47-baa9-c005613f48da
