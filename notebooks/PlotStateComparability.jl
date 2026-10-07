#=
Cross-animal comparability of the hidden states.

Three panels:
  A  drift rate against accuracy rank, one line per rat
  B  within-rat Kendall τ between each DDM parameter and state quality
  C  posterior-weighted accuracy against the DDM discriminability term v·B

Input is `state_parameters_long.csv` from ExtractStateParameters.jl: one row per
rat × state with the fitted DDM parameters and posterior-weighted behaviour.
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Statistics
using StatsBase
using Printf
using Plots
using StatsPlots

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
df = CSV.read(joinpath(results_dir, "state_parameters_long.csv"), DataFrame)

rats = sort(unique(df.rat))
const K = 4
const NRAT = length(rats)

# vB: P(correct) = 1 / (1 + exp(-2vB)) for an unbiased DDM, the model-internal accuracy.
df.absbias = abs.(df.a0 .- 0.5)
df.vB = df.v .* df.B

# Rank states 1..4 within each rat by posterior-weighted accuracy, best = rank 1.
transform!(groupby(df, :rat), :acc => (a -> ordinalrank(a; rev=true)) => :acc_rank)
sort!(df, [:rat, :acc_rank])

rank_colors = palette(:viridis, K)

# Panel A: drift rate against accuracy rank, one line per rat

pA = plot(;
    xlabel="state rank (by accuracy)",
    ylabel="v  (drift rate)",
    legend=false,
    xticks=(1:K, string.(1:K)),
    xlims=(0.7, K + 0.3),
)
for rat in rats
    sub = df[df.rat .== rat, :]
    plot!(
        pA,
        sub.acc_rank,
        sub.v;
        color=:gray75,
        linewidth=0.8,
        marker=(:circle, 2, 0.5, stroke(0)),
        markercolor=:gray65,
    )
end
mA = [mean(df[df.acc_rank .== k, :v]) for k in 1:K]
sA = [std(df[df.acc_rank .== k, :v]) / sqrt(NRAT) for k in 1:K]
plot!(
    pA,
    1:K,
    mA;
    yerror=sA,
    color=:black,
    linewidth=2.5,
    marker=(:circle, 5, stroke(0)),
    markercolor=:black,
)

# Panel B: which parameter orders the states the same way in every rat

# Within-rat Kendall τ against accuracy rank, signed so τ > 0 = higher in the better state.
feature_order = [:v, :tau, :occupancy, :B, :a0, :absbias]
feature_labels = ["v", "τ", "occ.", "B", "a0", "|bias|"]

tau_rows = DataFrame()
for (j, f) in enumerate(feature_order), rat in rats
    sub = df[df.rat .== rat, :]
    push!(
        tau_rows,
        (
            feature=f,
            feature_idx=j,
            τ=-corkendall(Float64.(sub.acc_rank), Float64.(sub[!, f])),
        );
        promote=true,
    )
end

# Consistency = rats sharing the majority sign, excluding τ = 0 ties (B has five);
# exact two-sided sign test over the non-tied rats.
"Exact two-sided binomial test of `k` successes in `n` trials against p = 0.5."
function sign_test_p(k::Int, n::Int)
    n == 0 && return 1.0
    k = max(k, n - k)
    tail = sum(binomial(n, i) for i in k:n) / 2.0^n
    return min(1.0, 2 * tail)
end

consistency = NamedTuple[]
for f in feature_order
    ts = tau_rows[tau_rows.feature .== f, :τ]
    npos, nneg = count(>(0), ts), count(<(0), ts)
    k, n = max(npos, nneg), npos + nneg
    push!(consistency, (k=k, n=n, ties=length(ts) - n, p=sign_test_p(k, n)))
end

"Conventional significance stars: *** < .001, ** < .01, * < .05, otherwise n.s."
function stars(p::Float64)
    p < 0.001 && return "***"
    p < 0.01 && return "**"
    p < 0.05 && return "*"
    return "n.s."
end

pB = boxplot(
    tau_rows.feature_idx,
    tau_rows.τ;
    fillalpha=0.4,
    linecolor=:black,
    color=:steelblue,
    legend=false,
    xlabel="DDM parameter / state property",
    ylabel="Kendall τ with state quality",
    xticks=(1:length(feature_order), feature_labels),
    ylims=(-1.18, 1.78),
    xlims=(0.4, length(feature_order) + 0.6),
)
dotplot!(
    pB,
    tau_rows.feature_idx,
    tau_rows.τ;
    marker=(:circle, 3, 0.6, stroke(0)),
    color=:steelblue,
)
hline!(pB, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
for j in 1:length(feature_order)
    c = consistency[j]
    annotate!(pB, j, 1.56, text("$(c.k)/$(c.n)", 7, FONT_FAMILY, :center, :black))
    st = stars(c.p)
    annotate!(
        pB, j, st == "n.s." ? 1.34 : 1.28,
        text(st, st == "n.s." ? 6 : 9, FONT_FAMILY, :center, :gray25),
    )
end

# Panel C: the accuracy ordering is an ordering on v·B

ρ_all = corspearman(df.vB, df.acc)

pC = plot(;
    xlabel="v · B  (DDM discriminability)",
    ylabel="accuracy",
    legend=:bottomright,
    foreground_color_legend=nothing,
    background_color_legend=nothing,
)
for rat in rats
    sub = sort(df[df.rat .== rat, :], :vB)
    plot!(pC, sub.vB, sub.acc; color=:gray80, linewidth=0.7, label="")
end
for k in 1:K
    sub = df[df.acc_rank .== k, :]
    scatter!(
        pC,
        sub.vB,
        sub.acc;
        marker=(:circle, 4, 0.9, stroke(0)),
        color=rank_colors[k],
        label="rank $k",
    )
end
annotate!(
    pC,
    0.15,
    0.925,
    text(@sprintf("ρ = %.2f", ρ_all), 8, FONT_FAMILY, :left, :black),
)

# Assemble

fig = plot(
    pA,
    pB,
    pC;
    layout=@layout([a{0.26w} b{0.40w} c{0.34w}]),
    size=(1000, 345),
    left_margin=6Plots.mm,
    bottom_margin=7Plots.mm,
    top_margin=8Plots.mm,
)
for (i, lab) in enumerate(["A", "B", "C"])
    annotate!(fig[i], (-0.23, 1.09), text(lab, 12, FONT_FAMILY, :left, :black))
end

savefig_both(fig, joinpath(results_dir, "state_comparability"))

# Console summary

println("\nKendall τ with state quality (ties excluded from the denominator)")
for (j, f) in enumerate(feature_order)
    ts = tau_rows[tau_rows.feature .== f, :τ]
    c = consistency[j]
    @printf(
        "  %-10s median τ = %+.2f   %2d/%-2d rats (%d tied)   sign test p = %.4f\n",
        String(f), median(ts), c.k, c.n, c.ties, c.p
    )
end

v_rank = combine(groupby(df, :rat), [:v, :acc_rank] => ((v, r) -> begin
    pr = ordinalrank(Float64.(v); rev=true)
    (n_exact=count(pr .== r), all_match=all(pr .== r))
end) => AsTable)
@printf(
    "\nRanking by v reproduces the accuracy ordering exactly in %d/%d rats (%.0f%% of states)\n",
    count(v_rank.all_match), NRAT, 100 * sum(v_rank.n_exact) / nrow(df)
)
@printf("Spearman(accuracy, v·B) over %d states = %.3f\n", nrow(df), ρ_all)
println("\nFigure written to $(joinpath(results_dir, "state_comparability"))")
