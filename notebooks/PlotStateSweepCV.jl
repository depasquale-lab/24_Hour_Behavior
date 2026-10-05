#=
Cross-validated logL-vs-K curves for the session-based ("daily") animals.

Reads results/ddm_hmm_state_sweep/cv/cv_state_sweep_summary.csv (from
CrossValidateStatesDaily.jl + MergeStateSweepCV.jl) and writes:
  - cv_test_ll_vs_K          held-out logL/trial vs K, one line per rat (mean over folds)
  - cv_test_ll_gain_vs_K     Δ held-out logL/trial vs K = 1, with group mean ± SEM
  - cv_train_vs_test_ll      train vs held-out logL/trial, to show the overfitting gap
  - cv_state_sweep_winners.csv  the CV-optimal K per rat
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

out_dir = joinpath("results", "ddm_hmm_state_sweep", "cv")
raw = CSV.read(
    joinpath(out_dir, "cv_state_sweep_summary.csv"), DataFrame; types=Dict(:rat => String)
)

# Average over folds within each (rat, K). SEM is across folds.
df = combine(groupby(raw, [:rat, :K])) do sub
    (
        n_folds=nrow(sub),
        n_params=sub.n_params[1],
        train_ll_per_trial=mean(sub.train_logL_per_trial),
        test_ll_per_trial=mean(sub.test_logL_per_trial),
        test_ll_sem=nrow(sub) > 1 ? std(sub.test_logL_per_trial) / sqrt(nrow(sub)) : 0.0,
        n_trials=sub.n_train_trials[1] + sub.n_test_trials[1],
    )
end
sort!(df, [:rat, :K])

# ΔlogL per trial relative to each rat's smallest K.
transform!(groupby(df, :rat)) do sub
    base = sub.test_ll_per_trial[sub.K .== minimum(sub.K)][1]
    sub.Δ_test_ll = sub.test_ll_per_trial .- base
    sub
end

rat_order = sort(unique(df[:, [:rat, :n_trials]]), :n_trials).rat
K_ticks = sort(unique(df.K))

# Plot 1: held-out logL per trial vs K, one line per rat (± SEM across folds)

p_test = plot(;
    xlabel="number of states (K)",
    ylabel="held-out log-likelihood per trial",
    title="Cross-validated DDM-HMM fit vs K — session-based animals",
    size=(750, 520),
    xticks=K_ticks,
    legend=:outerright,
    legendfontsize=7,
)
for rat in rat_order
    sub = df[df.rat .== rat, :]
    plot!(
        p_test,
        sub.K,
        sub.test_ll_per_trial;
        ribbon=sub.test_ll_sem,
        fillalpha=0.15,
        label=rat,
        marker=:circle,
        markersize=3,
        lw=1.5,
    )
end
savefig_both(p_test, joinpath(out_dir, "cv_test_ll_vs_K"))

# Plot 2: Δ held-out logL per trial vs K = 1; individual rats in grey, group mean in black

p_gain = plot(;
    xlabel="number of states (K)",
    ylabel="Δ held-out logL per trial (vs K = $(minimum(df.K)))",
    size=(420, 380),
    xticks=K_ticks,
    legend=false,
    grid=false,
    framestyle=:axes,
)
hline!(p_gain, [0.0]; color=:grey70, linestyle=:dash, linewidth=1)
for rat in rat_order
    sub = df[df.rat .== rat, :]
    plot!(p_gain, sub.K, sub.Δ_test_ll; color=:grey65, lw=1.2, alpha=0.9)
end

grp = combine(groupby(df, :K), :Δ_test_ll => mean => :mean_Δ)
sort!(grp, :K)
plot!(
    p_gain,
    grp.K,
    grp.mean_Δ;
    color=:black,
    lw=3,
    marker=:circle,
    markersize=5,
    markerstrokewidth=0,
)
savefig_both(p_gain, joinpath(out_dir, "cv_test_ll_gain_vs_K"))

# Plot 3: train vs held-out logL per trial — the overfitting gap

p_gap = plot(;
    xlabel="number of states (K)",
    ylabel="log-likelihood per trial",
    title="Train (dashed) vs held-out (solid) fit vs K",
    size=(750, 520),
    xticks=K_ticks,
    legend=:outerright,
    legendfontsize=7,
)
for (i, rat) in enumerate(rat_order)
    sub = df[df.rat .== rat, :]
    c = palette(:default)[mod1(i, 16)]
    plot!(p_gap, sub.K, sub.test_ll_per_trial; label=rat, color=c, marker=:circle, markersize=3, lw=1.5)
    plot!(p_gap, sub.K, sub.train_ll_per_trial; label="", color=c, linestyle=:dash, lw=1.2)
end
savefig_both(p_gap, joinpath(out_dir, "cv_train_vs_test_ll"))

# Winning K per rat (highest mean held-out logL per trial)

winners = combine(groupby(df, :rat)) do sub
    best = sub[argmax(sub.test_ll_per_trial), :]
    sorted = sort(sub.test_ll_per_trial; rev=true)
    (
        best_K=best.K,
        best_test_ll_per_trial=best.test_ll_per_trial,
        sem=best.test_ll_sem,
        margin_over_next=length(sorted) > 1 ? sorted[1] - sorted[2] : missing,
        n_folds=best.n_folds,
        n_trials=best.n_trials,
    )
end
sort!(winners, :n_trials)
CSV.write(joinpath(out_dir, "cv_state_sweep_winners.csv"), winners)

@info "Wrote plots to $out_dir"
@info "CV-optimal K per rat:"
show(winners; allrows=true, allcols=true)
println()
