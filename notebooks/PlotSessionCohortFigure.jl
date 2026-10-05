#=
Does the DDM-HMM help on session-based data too? Held-out fits for the session
(2 h) and 24 hr cohorts. Run SessionCohortSummary.jl first.

  A  Δ held-out logL per trial, DDM-HMM (K = 4) vs multilevel DDM, per rat
  B  Δ held-out logL per trial vs K (vs K = 1), session cohort; MLDDM level dashed
  C  same, 24 hr cohort

Writes results/session_cohort/session_cohort_figure.{svg,png}.
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Statistics
using Random
using Plots

const DIR = joinpath("results", "session_cohort")
const COHORTS = ["daily", "24hr"]
const LABEL = Dict("daily" => "Session (2 h)", "24hr" => "24 h")
const C_HMM = "#2a78d6"     # DDM-HMM (as in Fig. 6B)
const C_ML = "#eb6834"      # multilevel DDM
const C_RAT = :grey65

default(; fontfamily="helvetica", grid=false, framestyle=:axes, legend=false,
        titlefontsize=10, guidefontsize=9, tickfontsize=8, legendfontsize=7)

readcsv(f) = CSV.read(joinpath(DIR, f), DataFrame; types=Dict(:rat => String), validate=false)
by_rat = readcsv("heldout_by_rat.csv")
vs_K = readcsv("heldout_vs_K.csv")
stats = readcsv("heldout_stats.csv")

pfmt(p) = ismissing(p) || isnan(p) ? "" : p < 1e-3 ? "p < 0.001" : "p = $(round(p; sigdigits=2))"
stat_p(label, c) = let r = stats[(stats.contrast .== label) .& (stats.cohort .== c), :]
    nrow(r) == 0 ? NaN : r.wilcoxon_p[1]
end
gain(c, a, b) = collect(skipmissing(by_rat[by_rat.cohort .== c, a] .- by_rat[by_rat.cohort .== c, b]))

# A: DDM-HMM vs multilevel DDM, per rat
pA = let g = [gain(c, :hmm_K4, :mlddm) for c in COHORTS]
    p = plot(; title="DDM-HMM vs multilevel DDM", ylabel="Δ held-out logL / trial",
             xticks=(1:2, [LABEL[c] for c in COHORTS]), xlim=(0.4, 2.6))
    hline!(p, [0.0]; color=:grey70, ls=:dash, lw=1)
    rng = MersenneTwister(1)
    for (i, d) in enumerate(g)
        isempty(d) && continue
        scatter!(p, i .+ 0.12 .* (rand(rng, length(d)) .- 0.5), d; color=C_RAT, markersize=4, markerstrokewidth=0)
        plot!(p, [i - 0.25, i + 0.25], fill(median(d), 2); color=:black, lw=3)
        annotate!(p, i, maximum(d),
                  Plots.text("$(count(>(0), d))/$(length(d)) rats, $(pfmt(stat_p("DDM-HMM vs multilevel DDM", COHORTS[i])))", 7, :bottom))
    end
    all_d = reduce(vcat, g)
    pad = 0.15 * (maximum(all_d) - min(0, minimum(all_d)))
    ylims!(p, (min(0, minimum(all_d)) - pad, maximum(all_d) + pad))
    p
end

# B, C: gain vs K, one panel per cohort, shared y
pB = map(COHORTS) do c
    sub = vs_K[vs_K.cohort .== c, :]
    p = plot(; title=LABEL[c], xlabel="states (K)", xticks=1:5,
             ylabel=c == "daily" ? "Δ held-out logL / trial (vs K = 1)" : "")
    hline!(p, [0.0]; color=:grey70, ls=:dash, lw=1)
    for r in unique(sub.rat)
        s = sort(sub[sub.rat .== r, :], :K)
        plot!(p, s.K, s.Δ; color=C_RAT, lw=1)
    end
    med = combine(groupby(dropmissing(sub, :Δ), :K), :Δ => median => :m)
    plot!(p, med.K, med.m; color=C_HMM, lw=3, marker=:circle, markersize=4, markerstrokewidth=0)
    ml = gain(c, :mlddm, :ddm)
    if !isempty(ml)
        hline!(p, [median(ml)]; color=C_ML, lw=2, ls=:dash)
        annotate!(p, 5, median(ml), Plots.text("multilevel DDM", 7, C_ML, :right, :bottom))
    end
    p
end
lims = extrema(skipmissing(vs_K.Δ))
foreach(p -> ylims!(p, (min(0, lims[1]) - 0.02, lims[2] + 0.05)), pB)

fig = plot(pA, pB...; layout=(1, 3), size=(1150, 360), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
savefig(fig, joinpath(DIR, "session_cohort_figure.svg"))
savefig(fig, joinpath(DIR, "session_cohort_figure.png"))
@info "wrote $(joinpath(DIR, "session_cohort_figure.svg"))"
