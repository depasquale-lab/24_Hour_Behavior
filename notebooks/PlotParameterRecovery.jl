#=
Figure for the animal-matched DDM-HMM parameter recovery (ParameterRecovery.jl).

  A–D  true vs recovered DDM parameters (v, B, a₀, τ) for every state of every
       synthetic rat, coloured by the state's accuracy rank within its rat
  E    true vs recovered self-transition probability
  F    example synthetic session: true state vs recovered posterior
  G    per-trial state decoding: recovered model vs the true generative model
  H    within-rat ordering of each parameter preserved (Kendall τ, true vs recovered)
  I    recovery error against dataset size, with the real animals' sizes marked

Also writes recovery_long.csv (one row per task × state × parameter) and
recovery_summary.csv (one row per task).
=#

using Pkg
Pkg.activate("notebooks")

using BSON
using CSV
using DataFrames
using Statistics
using StatsBase
using Printf
using Plots
using StatsPlots
using Plots.PlotMeasures

gr()

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

const K = 4
out_dir = joinpath("results", "parameter_recovery")
task_dir = joinpath(out_dir, "tasks")
results = [
    BSON.load(joinpath(task_dir, f))[:result] for
    f in sort(readdir(task_dir)) if endswith(f, ".bson")
]
@info "loaded $(length(results)) recovery tasks"

# Long table: one row per task × state × parameter.
const PARAMS = (:v, :B, :a0, :tau, :p_self)
long = DataFrame()
summary = DataFrame()
for r in results
    tp, rp = r[:true_params], r[:rec_params]
    # Rank states within the rat by the DDM's discriminability v·B (the model-internal
    # counterpart of the accuracy ranking used in the paper; 1 = best).
    acc_rank = ordinalrank([p.v * p.B for p in tp]; rev=true)
    for k in 1:K, p in PARAMS
        push!(
            long,
            (
                rat=r[:rat],
                rep=r[:rep],
                frac=r[:frac],
                n_trials=r[:n_trials],
                state=k,
                rank=acc_rank[k],
                occupancy=r[:true_occ][k],
                param=String(p),
                truth=getfield(tp[k], p),
                recovered=getfield(rp[k], p),
            ),
        )
    end
    push!(
        summary,
        (
            rat=r[:rat],
            rep=r[:rep],
            frac=r[:frac],
            n_sessions=r[:n_sessions],
            n_trials=r[:n_trials],
            decode_acc_fit=r[:decode_acc_fit],
            decode_acc_true=r[:decode_acc_true],
            logL_true=r[:logL_true],
            logL_fit=r[:logL_fit],
            logL_truthinit=r[:logL_truthinit],
            logL_bw=r[:logL_bw],
            decode_acc_bw=r[:decode_acc_bw],
            elapsed_min=r[:elapsed_s] / 60,
        ),
    )
end
CSV.write(joinpath(out_dir, "recovery_long.csv"), long)
CSV.write(joinpath(out_dir, "recovery_summary.csv"), summary)

full = long[long.frac .== 1.0, :]
rank_colors = palette(:viridis, K)

# Panels A–E: true vs recovered
labels = Dict(
    "v" => "drift rate v",
    "B" => "boundary B",
    "a0" => "start point a0",
    "tau" => "non-decision time τ",
    "p_self" => "self-transition p(stay)",
)

function scatter_panel(param)
    sub = full[full.param .== param, :]
    lo = min(minimum(sub.truth), minimum(sub.recovered))
    hi = max(maximum(sub.truth), maximum(sub.recovered))
    pad = 0.05 * (hi - lo)
    lims = (lo - pad, hi + pad)
    p = plot(;
        xlabel="true",
        ylabel="recovered",
        title=labels[param],
        legend=false,
        xlims=lims,
        ylims=lims,
        aspect_ratio=:equal,
    )
    plot!(p, [lims...], [lims...]; color=:gray60, ls=:dash, lw=1, label="")
    for rk in K:-1:1
        s = sub[sub.rank .== rk, :]
        scatter!(
            p,
            s.truth,
            s.recovered;
            color=rank_colors[rk],
            markersize=2 .+ 10 .* sqrt.(s.occupancy),
            markerstrokewidth=0.3,
            markerstrokecolor=:white,
            alpha=0.85,
            label="",
        )
    end
    ρ = cor(sub.truth, sub.recovered)
    annotate!(
        p,
        lims[1] + 0.04 * (lims[2] - lims[1]),
        lims[2] - 0.06 * (lims[2] - lims[1]),
        text(@sprintf("r = %.2f", ρ), 8, :left),
    )
    return p
end

pA, pB, pC, pD, pE = (scatter_panel(p) for p in ("v", "B", "a0", "tau", "p_self"))
# Rank legend on panel A.
for rk in 1:K
    scatter!(
        pA,
        [NaN],
        [NaN];
        color=rank_colors[rk],
        markerstrokewidth=0,
        label="state rank $rk",
        legend=:bottomright,
        legendtitle="",
    )
end

# Panel F: example session (median-decoding full-data task, first session)
full_results = [r for r in results if r[:frac] == 1.0]
order = sortperm([r[:decode_acc_fit] for r in full_results])
ex = full_results[order[cld(length(order), 2)]]
ex_end = ex[:example_seq_ends][1]
n_show = min(ex_end, 400)
tp_ex = ex[:true_params]
ex_rank = ordinalrank([p.v * p.B for p in tp_ex]; rev=true)
ex_colors = [rank_colors[ex_rank[k]] for k in 1:K]
γ = ex[:example_gamma_fit][:, 1:n_show]
z = ex[:example_z][1:n_show]

pF = plot(;
    xlabel="trial",
    ylabel="P(state)",
    title="example synthetic session (rat $(ex[:rat]) clone)",
    legend=false,
    ylims=(0, 1.28),
    yticks=[0, 0.5, 1],
    xlims=(0, n_show),
)
# Stack posterior areas in rank order so the best state sits on top.
cum = zeros(n_show)
for k in sortperm(ex_rank; rev=true)
    top = cum .+ γ[k, :]
    plot!(
        pF,
        1:n_show,
        top;
        fillrange=copy(cum),
        color=ex_colors[k],
        lw=0,
        fillalpha=0.9,
        seriestype=:path,
    )
    cum .= top
end
# True state as a colour strip above the stack.
for t in 1:n_show
    plot!(
        pF,
        [t - 0.5, t + 0.5],
        [1.14, 1.14];
        color=ex_colors[z[t]],
        lw=6,
    )
end
annotate!(pF, 2, 1.22, text("true state", 7, :left))
annotate!(pF, 2, 0.96, text("recovered posterior", 7, :left, :white))

# Panel G: decoding accuracy, recovered vs oracle
fs = summary[summary.frac .== 1.0, :]
lims_g = (floor(min(minimum(fs.decode_acc_true), minimum(fs.decode_acc_fit)) * 20) / 20, 1.0)
pG = plot(;
    xlabel="decoding with true model",
    ylabel="decoding with recovered model",
    title="per-trial state decoding",
    legend=false,
    xlims=lims_g,
    ylims=lims_g,
    aspect_ratio=:equal,
)
plot!(pG, [lims_g...], [lims_g...]; color=:gray60, ls=:dash, lw=1)
scatter!(
    pG,
    fs.decode_acc_true,
    fs.decode_acc_fit;
    color=:black,
    markersize=4,
    markerstrokewidth=0,
    alpha=0.7,
)
annotate!(
    pG,
    lims_g[1] + 0.03 * (1 - lims_g[1]),
    0.97,
    text(
        @sprintf(
            "median gap = %.1f pts",
            100 * median(fs.decode_acc_true .- fs.decode_acc_fit)
        ),
        8,
        :left,
    ),
)

# Panel H: within-rat ordering preserved (Kendall τ between true and recovered)
τrows = DataFrame()
for g in groupby(full, [:rat, :rep, :param])
    push!(τrows, (param=g.param[1], tau=corkendall(g.truth, g.recovered)))
end
hparams = ["v", "B", "a0", "tau", "p_self"]
hlabels = ["v", "B", "a0", "τ", "p(stay)"]
pH = plot(;
    ylabel="Kendall τ (true, recovered)",
    title="within-rat state ordering",
    legend=false,
    xticks=(1:length(hparams), hlabels),
    ylims=(-1.05, 1.1),
    xlims=(0.4, length(hparams) + 0.6),
)
hline!(pH, [0]; color=:gray70, lw=1)
for (i, p) in enumerate(hparams)
    vals = τrows.tau[τrows.param .== p]
    # Kendall τ on 4 states takes few values, so jitter horizontally.
    jit = 0.25 .* (rand(length(vals)) .- 0.5)
    scatter!(
        pH,
        i .+ jit,
        vals;
        color=:gray40,
        markersize=2.5,
        markerstrokewidth=0,
        alpha=0.6,
    )
    plot!(pH, [i - 0.3, i + 0.3], fill(median(vals), 2); color=:black, lw=2.5)
    annotate!(
        pH,
        i,
        1.07,
        text(@sprintf("%d/%d", count(==(1.0), vals), length(vals)), 7, :center),
    )
end

# Panel I: error vs dataset size.
# Error in each state is scaled by the spread of that parameter across the rat's
# four true states, so a value below 1 means the estimate is closer to its own state
# than the states are to each other.
spread = combine(groupby(long, [:rat, :rep, :frac, :param]), :truth => std => :spread)
lj = leftjoin(long, spread; on=[:rat, :rep, :frac, :param])
lj.scaled_err = abs.(lj.recovered .- lj.truth) ./ lj.spread
err = combine(
    groupby(lj, [:rat, :rep, :frac, :n_trials, :param]),
    :scaled_err => median => :err,
)
pI = plot(;
    xlabel="trials in synthetic dataset",
    ylabel="median |error| / between-state SD",
    title="recovery error vs data size",
    xscale=:log10,
    yscale=:log10,
    legend=:topright,
    xticks=([1e3, 1e4, 1e5], ["1k", "10k", "100k"]),
)
param_colors = Dict(
    "v" => palette(:tab10)[1],
    "B" => palette(:tab10)[2],
    "a0" => palette(:tab10)[3],
    "tau" => palette(:tab10)[4],
)
for p in ("v", "B", "a0", "tau")
    sub = err[err.param .== p, :]
    scatter!(
        pI,
        sub.n_trials,
        sub.err;
        color=param_colors[p],
        markersize=2,
        markerstrokewidth=0,
        alpha=0.35,
        label="",
    )
    # Binned median trend.
    edges = 10 .^ range(log10(minimum(sub.n_trials)), log10(maximum(sub.n_trials)); length=7)
    xs, ys = Float64[], Float64[]
    for i in 1:(length(edges) - 1)
        m = (sub.n_trials .>= edges[i]) .& (sub.n_trials .<= edges[i + 1])
        count(m) >= 3 || continue
        push!(xs, sqrt(edges[i] * edges[i + 1]))
        push!(ys, median(sub.err[m]))
    end
    plot!(pI, xs, ys; color=param_colors[p], lw=2, label=hlabels[findfirst(==(p), hparams)])
end
hline!(pI, [1.0]; color=:gray50, ls=:dash, lw=1, label="")
# Real animals' dataset sizes as a rug.
real_n = unique(summary[summary.frac .== 1.0, [:rat, :n_trials]]).n_trials
yl = ylims(pI)
for n in real_n
    plot!(pI, [n, n], [yl[1], yl[1] * 1.25]; color=:black, lw=1, label="")
end
annotate!(pI, minimum(real_n), yl[1] * 1.45, text("real rats", 7, :left))

fig = plot(
    pA,
    pB,
    pC,
    pD,
    pE,
    pF,
    pG,
    pH,
    pI;
    layout=@layout([a b c d e; f{0.36w} g h i]),
    size=(1600, 680),
    left_margin=6mm,
    bottom_margin=6mm,
    top_margin=2mm,
)
savefig_both(fig, joinpath(out_dir, "parameter_recovery"))
@info "wrote $(joinpath(out_dir, "parameter_recovery")).{png,svg}"

# Console summary for the text.
for p in hparams
    s = full[full.param .== p, :]
    @printf(
        "%-7s r = %.3f   median |err| = %.3f   within-rat τ=1 in %d/%d\n",
        p,
        cor(s.truth, s.recovered),
        median(abs.(s.recovered .- s.truth)),
        count(==(1.0), τrows.tau[τrows.param .== p]),
        count(τrows.param .== p),
    )
end
@printf(
    "decoding: recovered median %.3f, oracle median %.3f\n",
    median(fs.decode_acc_fit),
    median(fs.decode_acc_true)
)
@printf(
    "logL(best blind fit) − logL(truth-init fit): median %.2f, min %.2f\n",
    median(fs.logL_fit .- summary.logL_truthinit[summary.frac .== 1.0]),
    minimum(fs.logL_fit .- summary.logL_truthinit[summary.frac .== 1.0])
)
