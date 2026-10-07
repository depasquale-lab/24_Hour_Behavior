#=
Does the DDM-HMM also improve fits on session-level data?
Session (2 h) and 24 h cohorts side by side. Run SessionCohortSummary.jl and
SessionCohortACF.jl first.

  A–C  held-out gain over a single DDM per rat, DDM-HMM (y) vs. A multilevel DDM,
       B per-session DDM (first 80 % / last 20 %), C shuffled-order mixture.
       Circles = session, squares = 24 h; above the diagonal = DDM-HMM better.
  D    simulated vs. real RT ACF (mean of lags 1–5), both models
  E    mean |simulated − real| RT ACF (lags 1–20), DDM-HMM vs MLDDM
  F    RT ACF explained: 1 − Σ(data − model)² / Σ data² over lags 1–20

Writes results/session_cohort/session_cohort_figure.{svg,png} and one SVG per
panel to results/session_cohort/panels/.
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Statistics
using Random
using Plots
using HypothesisTests

const DIR = joinpath("results", "session_cohort")
const PANELS = joinpath(DIR, "panels")
mkpath(PANELS)
const COHORTS = ["daily", "24hr"]
const LABEL = Dict("daily" => "Session (2 h)", "24hr" => "24 h")

default(; fontfamily="helvetica", grid=false, framestyle=:axes, legend=false,
        titlefontsize=10, guidefontsize=9, tickfontsize=8, legendfontsize=7,
        foreground_color_legend=nothing, background_color_legend=nothing)

readcsv(f) = CSV.read(joinpath(DIR, f), DataFrame; types=Dict(:rat => String), validate=false)
by_rat = readcsv("heldout_by_rat.csv")
split_by_rat = readcsv("split_by_rat.csv")
stats = readcsv("heldout_stats.csv")
acf = readcsv("rt_acf_by_rat.csv")

stars(p) = p < 1e-4 ? "****" : p < 1e-3 ? "***" : p < 1e-2 ? "**" : p < 0.05 ? "*" : "n.s."
signrank_p(d) = pvalue(SignedRankTest(Float64.(d)))
stat_row(label, c) = stats[(stats.contrast .== label) .& (stats.cohort .== c), :]

# A–C: held-out gain over a single DDM, DDM-HMM (y) vs. competitor (x), per rat.

const COHORT_STYLE = Dict("daily" => (:circle, "#1f9e89"), "24hr" => (:square, "#8e6bbf"))

function gain_scatter(df, xcol, ycol, base, title, xlabel, statlabel)
    g = [(c, dropmissing(df[df.cohort .== c, :], [xcol, ycol])) for c in COHORTS]
    vals = vcat([vcat(d[!, xcol] .- d[!, base], d[!, ycol] .- d[!, base]) for (_, d) in g]...)
    lo, hi = min(0, minimum(vals)) - 0.03, maximum(vals) + 0.05
    R = hi - lo
    p = plot(; title, xlabel, ylabel="DDM-HMM gain over DDM\n(held-out nats / trial)",
             xlim=(lo, hi), ylim=(lo, hi), aspect_ratio=:equal)
    plot!(p, [lo, hi], [lo, hi]; color=:grey70, ls=:dash, lw=1)
    for (c, d) in g
        shape, col = COHORT_STYLE[c]
        scatter!(p, d[!, xcol] .- d[!, base], d[!, ycol] .- d[!, base]; markershape=shape, color=col,
                 markersize=4, markerstrokewidth=0, alpha=0.9)
    end

    # inset (lower right, below the diagonal): per-rat DDM-HMM advantage = distance above the diagonal
    Δ = [d[!, ycol] .- d[!, xcol] for (_, d) in g]
    dlo, dhi = min(0, minimum(minimum.(Δ))), maximum(maximum.(Δ))
    dhi += 0.25 * (dhi - dlo)                                   # headroom for the stars
    bx0, bx1, by0, by1 = lo + 0.56R, hi - 0.03R, lo + 0.07R, lo + 0.40R
    Y(v) = by0 + (v - dlo) / (dhi - dlo) * (by1 - by0)
    plot!(p, [bx0, bx0, bx1], [by1, by0, by0]; color=:black, lw=0.8)          # inset axes
    plot!(p, [bx0, bx1], fill(Y(0), 2); color=:grey70, ls=:dash, lw=0.8)
    annotate!(p, bx0 - 0.01R, Y(0), Plots.text("0", 6, :right))
    tick = round(maximum(maximum.(Δ)); sigdigits=1)
    plot!(p, [bx0 - 0.008R, bx0], fill(Y(tick), 2); color=:black, lw=0.8)
    annotate!(p, bx0 - 0.01R, Y(tick), Plots.text(string(tick), 6, :right))
    annotate!(p, (bx0 + bx1) / 2, by1 + 0.015R, Plots.text("DDM-HMM advantage", 7, :bottom))
    rng = MersenneTwister(3)
    for (j, ((c, _), δ)) in enumerate(zip(g, Δ))
        shape, col = COHORT_STYLE[c]
        xc = bx0 + (j - 0.5) / 2 * (bx1 - bx0)
        w = 0.12 * (bx1 - bx0)
        scatter!(p, xc .+ w .* (rand(rng, length(δ)) .- 0.5), Y.(δ); markershape=shape, color=col,
                 markersize=2.5, markerstrokewidth=0, alpha=0.9)
        plot!(p, [xc - 0.7w, xc + 0.7w], fill(Y(median(δ)), 2); color=:black, lw=2)
        annotate!(p, xc, Y(maximum(δ)) + 0.01R, Plots.text(stars(stat_row(statlabel, c).wilcoxon_p[1]), 8, :bottom))
        annotate!(p, xc, by0 - 0.012R, Plots.text(c == "daily" ? "2 h" : "24 h", 7, col, :top))
    end
    p
end

panel_fit = [
    gain_scatter(by_rat, :mlddm, :hmm_K4, :ddm, "vs. multilevel DDM",
                 "multilevel DDM gain over DDM", "DDM-HMM vs multilevel DDM"),
    gain_scatter(split_by_rat, :split_session_ddm, :split_hmm, :split_ddm, "vs. a DDM refit every session",
                 "per-session DDM gain over DDM", "DDM-HMM vs per-session DDM (split)"),
]
panel_split = gain_scatter(by_rat, :mixture, :hmm_K4, :ddm, "vs. same states, no dynamics",
                           "shuffled-order mixture gain over DDM", "DDM-HMM vs mixture (dynamics)")

# D: model vs. data, per rat; E, F: RT ACF error and R², DDM-HMM vs. MLDDM

model_marker(src, col) = src == "hmm" ?
    (markercolor=col, markerstrokewidth=0, markersize=4, alpha=0.9) :
    (markercolor=:white, markerstrokecolor=col, markerstrokewidth=1.2, markersize=3.5)

const SCATTER_LAGS = 1:5
panel_scatter = let
    s = combine(groupby(acf[in.(acf.lag, Ref(SCATTER_LAGS)), :], [:cohort, :rat, :source]), :acf => mean => :m)
    w = unstack(s, [:cohort, :rat], :source, :m)
    hi = 1.1 * maximum(skipmissing(vcat(w.real, w.hmm, w.mlddm)))
    lo = min(-0.02, minimum(skipmissing(vcat(w.hmm, w.mlddm))) - 0.01)
    p = plot(; title="Each rat: model vs. data", xlabel="data RT ACF (lags 1–5)",
             ylabel="simulated RT ACF (lags 1–5)", xlim=(lo, hi), ylim=(lo, hi),
             aspect_ratio=:equal, legend=:topleft)
    plot!(p, [lo, hi], [lo, hi]; color=:grey70, ls=:dash, lw=1, label="")
    for c in COHORTS
        d = w[w.cohort .== c, :]
        shape, col = COHORT_STYLE[c]
        for (src, lab) in [("hmm", "DDM-HMM"), ("mlddm", "multilevel DDM")]
            dd = dropmissing(d, [:real, Symbol(src)])
            scatter!(p, dd.real, dd[!, src]; markershape=shape, label="$lab, $(LABEL[c])", model_marker(src, col)...)
        end
    end
    p
end

"Paired per-rat panel: DDM-HMM vs MLDDM, one pair of columns per cohort."
function paired_models(df, title, ylabel; top=nothing, ref=nothing)
    p = plot(; title, ylabel, xlim=(0.4, 5.6),
             xticks=([1, 2, 4, 5], ["DDM-\nHMM", "multilevel\nDDM", "DDM-\nHMM", "multilevel\nDDM"]))
    ref === nothing || hline!(p, [ref]; color=:grey80, lw=1)
    vals = skipmissing(vcat(df.hmm, df.mlddm))
    lo, hi = min(0, minimum(vals)), maximum(vals)
    top = something(top, hi + 0.45 * (hi - lo))
    for (j, c) in enumerate(COHORTS)
        d = dropmissing(df[df.cohort .== c, :], [:hmm, :mlddm])
        x0 = 3j - 2
        shape, col = COHORT_STYLE[c]
        for r in eachrow(d)
            plot!(p, [x0, x0 + 1], [r.hmm, r.mlddm]; color=col, lw=0.8, alpha=0.35)
        end
        scatter!(p, fill(x0, nrow(d)), d.hmm; markershape=shape, model_marker("hmm", col)...)
        scatter!(p, fill(x0 + 1, nrow(d)), d.mlddm; markershape=shape, model_marker("mlddm", col)...)
        yb = top - 0.12 * (top - lo)
        plot!(p, [x0, x0, x0 + 1, x0 + 1], [yb - 0.03 * (top - lo), yb, yb, yb - 0.03 * (top - lo)]; color=:black, lw=1)
        annotate!(p, x0 + 0.5, yb, Plots.text(stars(signrank_p(d.hmm .- d.mlddm)), 9, :bottom))
        annotate!(p, x0 + 0.5, top, Plots.text(LABEL[c], 8, col, :top))
    end
    ylims!(p, (lo - 0.03 * (hi - lo), top + 0.02 * (hi - lo)))
    p
end

const ACF_LAGS = 1:20   # E and F
acf_wide = unstack(acf[in.(acf.lag, Ref(ACF_LAGS)), :], [:cohort, :rat, :lag], :source, :acf)
acf_score(f) = combine(groupby(acf_wide, [:cohort, :rat])) do d
    g(m) = all(ismissing, d[!, m]) ? missing : f(d.real, d[!, m])
    (hmm=g(:hmm), mlddm=g(:mlddm))
end
acf_err = acf_score((r, m) -> mean(abs.(r .- m)))
panel_err = paired_models(acf_err, "RT ACF error", "mean |model − data| ACF\n(lags 1–20)")

# R² = 1 − Σ(data − model)² / Σ data² over ACF_LAGS
acf_r2 = acf_score((r, m) -> 1 - sum((r .- m) .^ 2) / sum(r .^ 2))
CSV.write(joinpath(DIR, "rt_acf_r2.csv"), acf_r2)
panel_r2 = paired_models(acf_r2, "RT ACF explained", "R² of RT ACF\n(lags 1–20)"; ref=0.0, top=1.35)

# Assemble

letters = ["A", "B", "C", "D", "E", "F"]
panels = [panel_fit..., panel_split, panel_scatter, panel_err, panel_r2]
names = ["A_vs_mlddm", "B_vs_per_session_ddm", "C_vs_mixture", "D_acf_model_vs_data", "E_acf_error", "F_acf_r2"]
for (p, n) in zip(panels, names)
    savefig(plot(p; size=(420, 330), left_margin=4Plots.mm, bottom_margin=4Plots.mm), joinpath(PANELS, n * ".svg"))
end
for (p, l) in zip(panels, letters)
    plot!(p; title=l * "   " * p[1][:title], titlelocation=:left)
end

l = @layout grid(2, 3)
fig = plot(panels...; layout=l, size=(1250, 800), left_margin=6Plots.mm, bottom_margin=5Plots.mm, top_margin=2Plots.mm)
savefig(fig, joinpath(DIR, "session_cohort_figure.svg"))
savefig(fig, joinpath(DIR, "session_cohort_figure.png"))
@info "wrote $(joinpath(DIR, "session_cohort_figure.svg")) and panels/ "
