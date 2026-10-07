#=
State-conditioned behaviour from posterior state assignments.

  A  psychometric functions, P(choose right) against signed evidence
  B  mean RT against accuracy rank, one line per rat
  C  psychometric slope against accuracy rank, one line per rat

No chronometric function: the task is free response (a flash every 100 ms until
the animal commits), so |Δ flashes| is an outcome of RT (r = 0.80), not a
difficulty level, and RT vs |Δ flashes| is positive by construction. The
generative Bernoulli probability is not recorded per trial, so B reports RT by state.

Inputs: state_psychometric_long.csv, state_parameters_long.csv and
state_psychometric_slopes.csv from ExtractStateParameters.jl.
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Statistics
using StatsBase
using Printf
using Plots

gr()

# Set a standard font family so text stays editable in Illustrator after SVG export.
const FONT_FAMILY = "Helvetica"
default(;
    fontfamily=FONT_FAMILY,
    titlefontfamily=FONT_FAMILY,
    guidefontfamily=FONT_FAMILY,
    tickfontfamily=FONT_FAMILY,
    legendfontfamily=FONT_FAMILY,
    guidefontsize=9,
    tickfontsize=8,
    titlefontsize=10,
    legendfontsize=7,
    grid=false,
    framestyle=:axes,
)

"savefig both .png and .svg alongside each other."
function savefig_both(p, stem::AbstractString)
    savefig(p, stem * ".png")
    return savefig(p, stem * ".svg")
end

results_dir = joinpath("results", "final_ddmhmms")
params = CSV.read(joinpath(results_dir, "state_parameters_long.csv"), DataFrame)
psy = CSV.read(joinpath(results_dir, "state_psychometric_long.csv"), DataFrame)
slopes = CSV.read(joinpath(results_dir, "state_psychometric_slopes.csv"), DataFrame)

rats = sort(unique(params.rat))
const K = 4
const NRAT = length(rats)

# Accuracy rank within rat (1 = best), carried onto the behavioural tables.
transform!(groupby(params, :rat), :acc => (a -> ordinalrank(a; rev=true)) => :acc_rank)
key = Dict((r, s) => k for (r, s, k) in zip(params.rat, params.state, params.acc_rank))
for tbl in (psy, slopes)
    tbl.acc_rank = [key[(r, s)] for (r, s) in zip(tbl.rat, tbl.state)]
end

rank_colors = palette(:viridis, K)

# Drop bins with negligible posterior mass, which are unstable in a single animal.
psy = psy[psy.n_eff .>= 20, :]

# Panel A: psychometric functions

pA = plot(;
    xlabel="Δ flashes  (right − left)",
    ylabel="P(choose right)",
    legend=:topleft,
    foreground_color_legend=nothing,
    background_color_legend=nothing,
    ylims=(0, 1),
)
for rank in 1:K
    sub = psy[psy.acc_rank .== rank, :]
    g = combine(
        groupby(sub, :bin),
        :delta_flashes => mean => :x,
        :p_right => mean => :y,
        :p_right => (v -> std(v) / sqrt(length(v))) => :sem,
        nrow => :n,
    )
    g = sort(g[g.n .>= NRAT ÷ 2, :], :x)
    plot!(pA, g.x, g.y; yerror=g.sem, color=rank_colors[rank], linewidth=2,
          marker=(:circle, 3.5, stroke(0)), markercolor=rank_colors[rank],
          markerstrokecolor=rank_colors[rank], label="rank $rank")
end
hline!(pA, [0.5]; color=:black, linestyle=:dash, linewidth=0.7, label="")

# Panel B: mean reaction time against accuracy rank

pB = plot(;
    xlabel="state rank (by accuracy)",
    ylabel="mean reaction time (s)",
    legend=false,
    xticks=(1:K, string.(1:K)),
    xlims=(0.7, K + 0.3),
)
for rat in rats
    sub = sort(params[params.rat .== rat, :], :acc_rank)
    plot!(pB, sub.acc_rank, sub.rt_mean; color=:gray75, linewidth=0.8,
          marker=(:circle, 2, 0.5, stroke(0)), markercolor=:gray65)
end
mB = [mean(params[params.acc_rank .== k, :rt_mean]) for k in 1:K]
sB = [std(params[params.acc_rank .== k, :rt_mean]) / sqrt(NRAT) for k in 1:K]
plot!(pB, 1:K, mB; yerror=sB, color=:black, linewidth=2.5,
      marker=(:circle, 5, stroke(0)), markercolor=:black)

# Panel C: psychometric slope against accuracy rank

pC = plot(;
    xlabel="state rank (by accuracy)",
    ylabel="psychometric slope  (per flash)",
    legend=false,
    xticks=(1:K, string.(1:K)),
    xlims=(0.7, K + 0.3),
)
for rat in rats
    sub = sort(slopes[slopes.rat .== rat, :], :acc_rank)
    plot!(pC, sub.acc_rank, sub.slope; color=:gray75, linewidth=0.8,
          marker=(:circle, 2, 0.5, stroke(0)), markercolor=:gray65)
end
mC = [mean(slopes[slopes.acc_rank .== k, :slope]) for k in 1:K]
sC = [std(slopes[slopes.acc_rank .== k, :slope]) / sqrt(NRAT) for k in 1:K]
plot!(pC, 1:K, mC; yerror=sC, color=:black, linewidth=2.5,
      marker=(:circle, 5, stroke(0)), markercolor=:black)

# Assemble

fig = plot(
    pA, pB, pC;
    layout=@layout([a{0.40w} b{0.30w} c{0.30w}]),
    size=(1000, 345),
    left_margin=6Plots.mm,
    bottom_margin=7Plots.mm,
    top_margin=8Plots.mm,
)
for (i, lab) in enumerate(["A", "B", "C"])
    annotate!(fig[i], (-0.23, 1.09), text(lab, 12, FONT_FAMILY, :left, :black))
end

savefig_both(fig, joinpath(results_dir, "state_behavior"))

# Console summary

println("\nmean reaction time by accuracy rank (mean ± SEM across $NRAT rats)")
for k in 1:K
    @printf("  rank %d: %.3f ± %.3f s\n", k, mB[k], sB[k])
end
τrt = [
    corkendall(
        Float64.(sort(params[params.rat .== r, :], :acc_rank).acc_rank),
        sort(params[params.rat .== r, :], :acc_rank).rt_mean,
    ) for r in rats
]
@printf(
    "  Kendall τ(rank, RT): median %+.2f, negative in %d/%d rats\n",
    median(τrt), count(<(0), τrt), NRAT
)

println("\npsychometric slope by accuracy rank (mean ± SEM across $NRAT rats)")
for k in 1:K
    @printf("  rank %d: %.3f ± %.3f per flash\n", k, mC[k], sC[k])
end

mono = count(
    all(diff(sort(slopes[slopes.rat .== r, :], :acc_rank).slope) .< 0) for r in rats
)
@printf("\nslope decreases monotonically with rank in %d/%d rats\n", mono, NRAT)

ord = count(
    let s = sort(slopes[slopes.rat .== r, :], :acc_rank).slope
        s[1] == maximum(s)
    end for r in rats
)
@printf("rank-1 state has the steepest slope in %d/%d rats\n", ord, NRAT)

τs = [
    corkendall(
        Float64.(sort(slopes[slopes.rat .== r, :], :acc_rank).acc_rank),
        sort(slopes[slopes.rat .== r, :], :acc_rank).slope,
    ) for r in rats
]
@printf(
    "Kendall τ(rank, slope): median %+.2f, negative in %d/%d rats\n",
    median(τs), count(<(0), τs), NRAT
)
println("\nFigure written to $(joinpath(results_dir, "state_behavior"))")
