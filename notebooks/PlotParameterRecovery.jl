#=
Figure for the animal-matched DDM-HMM parameter recovery (ParameterRecovery.jl).

  A–D  true vs recovered DDM parameters (v, B, a₀, τ) for every state of every
       synthetic rat, coloured by the state's accuracy rank within its rat
  E    true vs recovered self-transition probability
  F    example synthetic session: true state vs recovered posterior
  G    per-trial state decoding: recovered model vs the true generative model
  H    how fast recovery error shrinks with data: within-rat power-law exponent
       per parameter, with a rat-bootstrap 95% CI, against the 1/√n rate
  I    recovery error against dataset size with the fitted power laws, and the
       real animals' sizes marked

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
using Random
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
    # Rank states within rat by v·B (model-internal accuracy; 1 = best).
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

# Within-rat ordering (Kendall τ between true and recovered), console summary only.
τrows = DataFrame()
for g in groupby(full, [:rat, :rep, :param])
    push!(τrows, (param=g.param[1], tau=corkendall(g.truth, g.recovered)))
end
hparams = ["v", "B", "a0", "tau", "p_self"]
hlabels = ["v", "B", "a0", "τ", "p(stay)"]

# Panel I: error vs dataset size, scaled by the parameter's spread across the
# rat's true states (< 1 = closer to its own state than the states are to each other).
spread = combine(groupby(long, [:rat, :rep, :frac, :param]), :truth => std => :spread)
lj = leftjoin(long, spread; on=[:rat, :rep, :frac, :param])
lj.scaled_err = abs.(lj.recovered .- lj.truth) ./ lj.spread
param_colors = Dict(
    "v" => palette(:tab10)[1],
    "B" => palette(:tab10)[2],
    "a0" => palette(:tab10)[3],
    "tau" => palette(:tab10)[4],
)
err = combine(
    groupby(lj, [:rat, :rep, :frac, :n_trials, :param]),
    :scaled_err => median => :err,
)
err = err[err.err .> 0, :]

# Panel H: within-rat power law log10(err) = α_rat + β·log10(n); β = −0.5 is the 1/√n rate.
err.logn = log10.(err.n_trials)
err.logerr = log10.(err.err)
transform!(
    groupby(err, [:rat, :rep, :param]),
    :logn => (x -> x .- mean(x)) => :dx,
    :logerr => (y -> y .- mean(y)) => :dy,
)
within_slope(d) = sum(d.dx .* d.dy) / sum(d.dx .^ 2)

rng_boot = MersenneTwister(1)
N_BOOT = 2000
scaling = DataFrame()
for p in hparams
    sub = err[err.param .== p, :]
    β = within_slope(sub)
    # α such that the fitted line passes through the rats' average (logn, logerr)
    α = mean(combine(groupby(sub, :rat), [:logn, :logerr] => ((x, y) -> mean(y) - β * mean(x)) => :a).a)
    by_rat = [g for g in groupby(sub, :rat)]
    boots = [within_slope(reduce(vcat, rand(rng_boot, by_rat, length(by_rat)))) for _ in 1:N_BOOT]
    lo, hi = quantile(boots, (0.025, 0.975))
    push!(scaling, (param=p, beta=β, lo=lo, hi=hi, alpha=α, n_rats=length(by_rat)))
end
CSV.write(joinpath(out_dir, "recovery_scaling.csv"), scaling)

pH = plot(;
    ylabel="error scaling exponent β",
    title="error scaling with data",
    legend=false,
    xticks=(1:length(hparams), hlabels),
    xlims=(0.4, length(hparams) + 0.6),
    ylims=(-1.0, 0.15),
)
hline!(pH, [0.0]; color=:gray70, lw=1)
hline!(pH, [-0.5]; color=:gray40, ls=:dash, lw=1)
annotate!(pH, 0.45, -0.46, text("1/√n", 7, :left, :gray30))
for (i, r) in enumerate(eachrow(scaling))
    c = r.param == "p_self" ? :gray30 : param_colors[r.param]
    plot!(pH, [i, i], [r.lo, r.hi]; color=c, lw=2.5)
    scatter!(pH, [i], [r.beta]; color=c, markersize=6, markerstrokewidth=0)
    annotate!(pH, i, r.hi + 0.07, text(@sprintf("%.2f", r.beta), 7, :center))
end
pI = plot(;
    xlabel="trials in synthetic dataset",
    ylabel="median |error| / between-state SD",
    title="recovery error vs data size",
    xscale=:log10,
    yscale=:log10,
    legend=:topright,
    xticks=([1e3, 1e4, 1e5], ["1k", "10k", "100k"]),
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
    # Fitted within-rat power law (panel H).
    r = scaling[scaling.param .== p, :][1, :]
    xs = 10 .^ range(minimum(sub.logn), maximum(sub.logn); length=50)
    plot!(pI, xs, 10 .^ (r.alpha .+ r.beta .* log10.(xs)); color=param_colors[p], lw=2,
        label=hlabels[findfirst(==(p), hparams)])
end
hline!(pI, [1.0]; color=:gray50, ls=:dash, lw=1, label="")
# Real animals' dataset sizes as a rug.
real_n = unique(summary[summary.frac .== 1.0, [:rat, :n_trials]]).n_trials
yl = ylims(pI)
for n in real_n
    plot!(pI, [n, n], [yl[1], yl[1] * 1.25]; color=:black, lw=1, label="")
end
annotate!(pI, minimum(real_n), yl[1] * 1.45, text("real rats", 7, :left))

# Panel letters, prepended to each title and left-aligned.
for (pp, L) in zip((pA, pB, pC, pD, pE, pF, pG, pH, pI), 'A':'I')
    t = pp.subplots[1].attr[:title]
    plot!(pp; title="$(L)    $(t)", titlelocation=:left)
end

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
for r in eachrow(scaling)
    @printf("%-7s error ∝ n^%.2f  (95%% CI %.2f to %.2f, %d rats)\n", r.param, r.beta, r.lo, r.hi, r.n_rats)
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
