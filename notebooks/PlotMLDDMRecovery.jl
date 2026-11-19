#=
MLDDM recovery figure + tables from MLDDMRecovery.jl task CSVs.
true vs blind per hyperparam, m top, log σ0 bottom; hollow = true σ0 near floor
-> mlddm_recovery{,_reviewer}, recovery_{summary,long,stats}.csv
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Statistics
using Printf
using Plots
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

function savefig_both(p, stem::AbstractString)
    savefig(p, stem * ".png")
    return savefig(p, stem * ".svg")
end

const PARAMS = ["B", "τ", "v", "a0"]
const LABELS = Dict("B" => "log B", "τ" => "log τ", "v" => "log v", "a0" => "logit a0")
# true σ0 near optimizer floor (1e-3) = nothing to recover; loose cut, bimodal
const SIGMA0_FLOOR = 1e-2

out_dir = get(ENV, "OUT_DIR", joinpath("results", "mlddm_recovery"))
task_dir = joinpath(out_dir, "tasks")
files = filter(f -> endswith(f, ".csv"), readdir(task_dir; join=true))
isempty(files) && error("no task CSVs in $task_dir")
summary = sort!(reduce(vcat, [CSV.read(f, DataFrame; types=Dict(:rat => String)) for f in files]), [:rep, :rat])
@info "loaded $(nrow(summary)) recovery tasks"
CSV.write(joinpath(out_dir, "recovery_summary.csv"), summary)

# blind - truth-init logL < 0 = optimizer missed the basin
summary.blind_minus_warm_per_trial = (summary.blind_logL .- summary.warm_logL) ./ summary.n_trials
summary.blind_minus_true_per_trial = (summary.blind_logL .- summary.true_logL) ./ summary.n_trials
println("\nlogL per trial, blind − truth-initialised (negative = blind fit missed the optimum):")
for r in eachrow(summary)
    @printf("  %-8s rep %d  n=%6d  blind−warm %+.5f  blind−true %+.5f  conv %d/%d  quad_err %.2e\n",
            r.rat, r.rep, r.n_trials, r.blind_minus_warm_per_trial, r.blind_minus_true_per_trial,
            r.n_converged, r.n_starts, r.blind_quad_err)
end

long = DataFrame(rat=String[], rep=Int[], n_trials=Int[], level=String[], param=String[],
                 true_value=Float64[], blind=Float64[], warm=Float64[], true_at_floor=Bool[])
for r in eachrow(summary), p in PARAMS
    floor = r["true_s0_$p"] <= SIGMA0_FLOOR
    push!(long, (r.rat, r.rep, r.n_trials, "m", p, r["true_m_$p"], r["blind_m_$p"], r["warm_m_$p"], floor))
    push!(long, (r.rat, r.rep, r.n_trials, "s0", p, r["true_s0_$p"], r["blind_s0_$p"], r["warm_s0_$p"], floor))
end
CSV.write(joinpath(out_dir, "recovery_long.csv"), long)

# σ0 on log scale, floor cases dropped (true value only a bound)
stats = DataFrame(level=String[], param=String[], n=Int[], r=Float64[], median_abs_err=Float64[],
                  median_rel_err=Float64[], frac_within_10pct=Float64[])
for lvl in ("m", "s0"), p in PARAMS
    d = long[(long.level .== lvl) .& (long.param .== p), :]
    lvl == "s0" && (d = d[.!d.true_at_floor, :])
    nrow(d) < 2 && continue
    x, y = d.true_value, d.blind
    if lvl == "s0"
        x, y = log.(x), log.(y)
    end
    err = abs.(y .- x)
    rel = lvl == "s0" ? abs.(d.blind ./ d.true_value .- 1) : abs.(y .- x) ./ max.(abs.(x), 1e-6)
    push!(stats, (lvl, p, nrow(d), cor(x, y), median(err), median(rel), mean(rel .< 0.1)))
end
CSV.write(joinpath(out_dir, "recovery_stats.csv"), stats)
println("\nper-parameter recovery (blind):")
println(stats)

# Figure
panels = Plots.Plot[]
for lvl in ("m", "s0"), p in PARAMS
    d = long[(long.level .== lvl) .& (long.param .== p), :]
    islog = lvl == "s0"
    tf = islog ? log : identity
    full, floor = d[.!d.true_at_floor, :], d[d.true_at_floor, :]
    vals = tf.(vcat(d.true_value, d.blind))
    lo, hi = extrema(vals)
    pad = 0.08 * max(hi - lo, 0.2); lo -= pad; hi += pad
    ttl = islog ? "log σ0  $(LABELS[p])" : "m  $(LABELS[p])"
    pl = plot(; title=ttl, xlabel="true", ylabel="recovered", xlims=(lo, hi), ylims=(lo, hi),
              aspect_ratio=1, legend=false)
    plot!(pl, [lo, hi], [lo, hi]; color=:grey70, linestyle=:dash)
    scatter!(pl, tf.(full.true_value), tf.(full.blind); color=:black, markersize=4, markerstrokewidth=0)
    nrow(floor) > 0 && scatter!(pl, tf.(floor.true_value), tf.(floor.blind); color=:white,
                                markerstrokecolor=:black, markersize=4, markerstrokewidth=1)
    s = stats[(stats.level .== lvl) .& (stats.param .== p), :]
    nrow(s) == 1 && annotate!(pl, lo + 0.03 * (hi - lo), hi - 0.03 * (hi - lo),
                              text(@sprintf("r = %.2f", s.r[1]), 8, :left, :top, FONT_FAMILY))
    push!(panels, pl)
end
fig = plot(panels...; layout=(2, 4), size=(1150, 560), margin=4mm, left_margin=6mm, bottom_margin=5mm)
savefig_both(fig, joinpath(out_dir, "mlddm_recovery"))
@info "wrote $(joinpath(out_dir, "mlddm_recovery")).{png,svg}"

# reviewer figure: A means m, B log σ0 (floor rats dropped)
# colours as in PlotParameterRecovery.jl
param_colors = Dict("v" => palette(:tab10)[1], "B" => palette(:tab10)[2],
                    "a0" => palette(:tab10)[3], "τ" => palette(:tab10)[4])
panel_dir = joinpath(out_dir, "panels")
mkpath(panel_dir)

function recovery_panel(lvl, ttl; ymin=-Inf)
    d = long[long.level .== lvl, :]
    lvl == "s0" && (d = d[.!d.true_at_floor, :])
    tf = lvl == "s0" ? log : identity
    vals = tf.(vcat(d.true_value, d.blind))
    lo, hi = extrema(vals)
    pad = 0.07 * (hi - lo); lo -= pad; hi += pad
    # Points below ymin (one failed a₀ spread) are drawn at the axis edge, labelled.
    clipped = d[tf.(d.blind) .< ymin, :]
    lo = max(lo, ymin)
    pl = plot(; title=ttl, titlelocation=:left, xlabel="true", ylabel="recovered",
              xlims=(lo, hi), ylims=(lo, hi), aspect_ratio=1, legend=:topleft,
              foreground_color_legend=nothing, background_color_legend=nothing)
    plot!(pl, [lo, hi], [lo, hi]; color=:gray60, linestyle=:dash, lw=1, label="")
    for p in PARAMS
        dp = d[d.param .== p, :]
        s = stats[(stats.level .== lvl) .& (stats.param .== p), :]
        lab = @sprintf("%s  (r = %.2f, n = %d)", LABELS[p], s.r[1], nrow(dp))
        scatter!(pl, tf.(dp.true_value), max.(tf.(dp.blind), lo); color=param_colors[p],
                 markersize=4.5, markerstrokewidth=0, alpha=0.9, label=lab)
    end
    for r in eachrow(clipped)
        annotate!(pl, tf(r.true_value) + 0.04 * (hi - lo), lo + 0.015 * (hi - lo),
                  text(@sprintf("%s %s: %.1f", r.rat, LABELS[r.param], tf(r.blind)), 7, :left, :bottom, FONT_FAMILY))
    end
    pl
end

pA = recovery_panel("m", "hyperparameter means m")
pB = recovery_panel("s0", "trial-level spreads, log σ0"; ymin=-2.6)
for (p, n) in zip((pA, pB), ("A_means", "B_spreads"))
    savefig(plot(p; size=(400, 400), left_margin=4mm, bottom_margin=4mm), joinpath(panel_dir, n * ".svg"))
end
plot!(pA; title="A   " * pA[1][:title]); plot!(pB; title="B   " * pB[1][:title])
rev = plot(pA, pB; layout=(1, 2), size=(820, 400), left_margin=5mm, bottom_margin=4mm, top_margin=2mm)
savefig_both(rev, joinpath(out_dir, "mlddm_recovery_reviewer"))
@info "wrote $(joinpath(out_dir, "mlddm_recovery_reviewer")).{png,svg} and panels/"
