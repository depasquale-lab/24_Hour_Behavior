#=
LL-vs-number-of-states curves for the session-based ("daily") trained animals.

Reads results/ddm_hmm_state_sweep/state_sweep_summary.csv (produced by
StateSweepDaily.jl + MergeStateSweep.jl) and writes:
  - ll_vs_K_per_rat        per-trial logL curve, one line per rat
  - ll_gain_vs_K           ΔlogL per trial relative to K = 1 (rats aligned)
  - bic_vs_K_per_rat       ΔBIC relative to each rat's best K
  - state_sweep_winners.csv  the BIC-optimal K per rat
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Statistics
using Plots
using StatsPlots

gr()

# Standard font family so text stays editable in Illustrator after SVG export.
const FONT_FAMILY = "Helvetica"
default(;
    fontfamily=FONT_FAMILY,
    titlefontfamily=FONT_FAMILY,
    guidefontfamily=FONT_FAMILY,
    tickfontfamily=FONT_FAMILY,
    legendfontfamily=FONT_FAMILY,
)

"savefig both .png and .svg alongside each other."
function savefig_both(p, stem::AbstractString)
    savefig(p, stem * ".png")
    return savefig(p, stem * ".svg")
end

results_dir = joinpath("results", "ddm_hmm_state_sweep")
df = CSV.read(joinpath(results_dir, "state_sweep_summary.csv"), DataFrame)
sort!(df, [:rat, :K])

# Per-rat normalizations: ΔBIC from each rat's best K, ΔlogL/trial from K = 1.
transform!(groupby(df, :rat), :bic => (b -> b .- minimum(b)) => :ΔBIC)
transform!(groupby(df, :rat)) do sub
    base = sub.logL_per_trial[sub.K .== minimum(sub.K)][1]
    sub.Δ_logL_per_trial = sub.logL_per_trial .- base
    sub
end

rat_order = sort(unique(df[:, [:rat, :n_trials]]), :n_trials).rat
K_ticks = sort(unique(df.K))

# Plot 1: per-trial logL vs K, one line per rat

p_ll = plot(;
    xlabel="number of states (K)",
    ylabel="log-likelihood per trial",
    title="DDM-HMM fit vs K — session-based animals",
    size=(750, 520),
    xticks=K_ticks,
    legend=:outerright,
    legendfontsize=7,
)
for rat in rat_order
    sub = df[df.rat .== rat, :]
    plot!(p_ll, sub.K, sub.logL_per_trial; label=rat, marker=:circle, markersize=3, lw=1.5)
end
savefig_both(p_ll, joinpath(results_dir, "ll_vs_K_per_rat"))

# Plot 2: ΔlogL per trial relative to K = 1 (puts all rats on a common scale)

p_gain = plot(;
    xlabel="number of states (K)",
    ylabel="Δ log-likelihood per trial (vs K = $(minimum(df.K)))",
    title="Improvement in fit as states are added",
    size=(750, 520),
    xticks=K_ticks,
    legend=:outerright,
    legendfontsize=7,
)
for rat in rat_order
    sub = df[df.rat .== rat, :]
    plot!(p_gain, sub.K, sub.Δ_logL_per_trial; label=rat, marker=:circle, markersize=3, lw=1.5)
end

# Group mean ± SEM across rats, overlaid in black.
grp = combine(groupby(df, :K)) do sub
    (mean_Δ=mean(sub.Δ_logL_per_trial), sem_Δ=std(sub.Δ_logL_per_trial) / sqrt(nrow(sub)))
end
sort!(grp, :K)
plot!(
    p_gain,
    grp.K,
    grp.mean_Δ;
    ribbon=grp.sem_Δ,
    label="mean ± SEM",
    color=:black,
    lw=3,
    marker=:square,
    markersize=4,
)
savefig_both(p_gain, joinpath(results_dir, "ll_gain_vs_K"))

# Plot 3: ΔBIC vs K (raw logL always improves with K — BIC is the selection curve)

p_bic = plot(;
    xlabel="number of states (K)",
    ylabel="ΔBIC (relative to each rat's best K)",
    title="BIC vs K — session-based animals",
    size=(750, 520),
    xticks=K_ticks,
    legend=:outerright,
    legendfontsize=7,
)
for rat in rat_order
    sub = df[df.rat .== rat, :]
    plot!(p_bic, sub.K, sub.ΔBIC; label=rat, marker=:circle, markersize=3, lw=1.5)
end
hline!(p_bic, [0.0]; color=:black, linestyle=:dash, linewidth=1, label="")
savefig_both(p_bic, joinpath(results_dir, "bic_vs_K_per_rat"))

# Winning K per rat

winners = combine(groupby(df, :rat)) do sub
    best = sub[argmin(sub.bic), :]
    (
        best_K=best.K,
        best_bic=best.bic,
        ΔBIC_to_next=length(sub.ΔBIC) > 1 ? sort(sub.ΔBIC)[2] : missing,
        logL_per_trial_at_best=best.logL_per_trial,
        n_trials=best.n_trials,
        n_sessions=best.n_sessions,
    )
end
sort!(winners, :n_trials)
CSV.write(joinpath(results_dir, "state_sweep_winners.csv"), winners)

@info "Wrote plots to $results_dir"
@info "BIC-optimal K per rat:"
show(winners; allrows=true, allcols=true)
println()
