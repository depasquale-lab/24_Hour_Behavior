using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Statistics
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
    framestyle=:box,
)

"savefig both .png and .svg alongside each other."
function savefig_both(p, stem::AbstractString)
    savefig(p, stem * ".png")
    savefig(p, stem * ".svg")
end

results_dir = joinpath("results")
summary_dir = joinpath(results_dir, "bic_summary.csv")
df = CSV.read(summary_dir, DataFrame)

# Fixed plotting order for the tied configurations (must match fit script).
const CONFIG_ORDER = ["full", "τ", "a₀", "τ_a₀", "v", "B"]
const CONFIG_RANK = Dict(c => i for (i, c) in enumerate(CONFIG_ORDER))

# Some (rat, tied) runs may have NaN-ed out of the fitter and be absent from
# the CSV. Drop unknown tied labels defensively, then keep only finite BICs so
# downstream groupby/argmin calls don't pick up missing-data ghosts.
df = df[in.(df.tied, Ref(CONFIG_ORDER)), :]
df = df[isfinite.(df.bic), :]

df.tied_rank = [CONFIG_RANK[t] for t in df.tied]

# Sort rats by total trial count so the heatmap y-axis has an interpretable order.
rat_order = sort(unique(df[:, [:rat, :n_trials]]), :n_trials).rat

# ΔBIC relative to the best (lowest) BIC for each rat.
transform!(groupby(df, :rat), :bic => (b -> b .- minimum(b)) => :ΔBIC)

# ΔBIC relative to the full model for each rat. NaN when the rat has no
# "full" entry (e.g., the full-model run failed and didn't make it into the CSV).
transform!(groupby(df, :rat)) do sub
    full_rows = sub[sub.tied .== "full", :]
    sub.ΔBIC_from_full = isempty(full_rows) ?
        fill(NaN, nrow(sub)) :
        sub.bic .- full_rows.bic[1]
    sub
end

# Flag the winning (ΔBIC == 0) configuration for each rat.
df.is_best = df.ΔBIC .== 0

# Plot 1: ΔBIC heatmap (rat × config), relative to full model
M = Matrix{Float64}(undef, length(rat_order), length(CONFIG_ORDER))
for (i, rat) in enumerate(rat_order)
    for (j, cfg) in enumerate(CONFIG_ORDER)
        row = df[(df.rat .== rat) .& (df.tied .== cfg), :]
        M[i, j] = isempty(row) ? NaN : row.ΔBIC_from_full[1]
    end
end

# Symmetric color range around 0 so improvements/worsenings are visually balanced.
finite_vals = M[isfinite.(M)]
absmax_val  = isempty(finite_vals) ? 1.0 : maximum(abs.(finite_vals))

# RdYlBu reversed: blue = lower BIC (better than full), red = higher BIC (worse),
# yellow at zero so the neutral row is visible (not white as with :RdBu).
p_heat = heatmap(
    CONFIG_ORDER,
    rat_order,
    M;
    xlabel="Tied DDM parameters",
    ylabel="Rat (sorted by # trials, ascending)",
    title="ΔBIC relative to full model per rat",
    color=:Blues,
    colorbar_title="ΔBIC from full",
    clims=(0, 15000),
    grid=false,
    size=(820, 620),
    left_margin=10Plots.mm,
    bottom_margin=8Plots.mm,
)

# Mark the best (lowest-BIC) config per rat — skip rats with no finite entries.
for (i, rat) in enumerate(rat_order)
    row = M[i, :]
    valid_idx = findall(isfinite, row)
    isempty(valid_idx) && continue
    j = valid_idx[argmin(row[valid_idx])]
    scatter!(
        p_heat,
        [CONFIG_ORDER[j]], [rat];
        marker=:star5,
        markersize=10,
        color=:gold,
        markerstrokecolor=:black,
        markerstrokewidth=1.2,
        label="",
    )
end

savefig_both(p_heat, joinpath(results_dir, "bic_heatmap_from_full"))

bic_from_full_summary = combine(groupby(df_dist, :tied),
    :ΔBIC_from_full => mean => :mean_ΔBIC,
    :ΔBIC_from_full => (x -> std(x) / sqrt(length(x))) => :sem_ΔBIC,
)
bic_from_full_summary.tied_rank = [CONFIG_RANK[t] for t in bic_from_full_summary.tied]
sort!(bic_from_full_summary, :tied_rank)

p_bar = bar(
    bic_from_full_summary.tied_rank,
    bic_from_full_summary.mean_ΔBIC;
    yerror=bic_from_full_summary.sem_ΔBIC,
    xlabel="Tied DDM parameters",
    ylabel="Mean ΔBIC from full model",
    title="Mean ΔBIC from full across rats (K = 4)",
    legend=false,
    size=(720, 520),
    xticks=(1:length(CONFIG_ORDER), CONFIG_ORDER),
    linecolor=:black,
    fillcolor=:steelblue,
    fillalpha=0.85,
    bar_width=0.65,
    grid=:y,
)

hline!(p_bar, [0.0]; color=:black, linestyle=:dash, linewidth=1)

savefig_both(p_bar, joinpath(results_dir, "bic_change_from_full_bar"))

# Summary table of which config wins for each rat. Handles rats whose CSV is
# missing entries (e.g., only one config survived NaN-ing) — ΔBIC_to_next is
# NaN when there's no second-best to compare against.

winners_rows = NamedTuple[]
for sub in groupby(df, :rat)
    isempty(sub) && continue
    best_row = sub[argmin(sub.bic), :]
    deltas_sorted = sort(sub.ΔBIC)
    push!(winners_rows, (
        rat          = best_row.rat,
        winning_tied = best_row.tied,
        winning_bic  = best_row.bic,
        ΔBIC_to_next = length(deltas_sorted) >= 2 ? deltas_sorted[2] : NaN,
        n_trials     = best_row.n_trials,
    ))
end
winners = DataFrame(winners_rows)
sort!(winners, :n_trials)

CSV.write(joinpath(results_dir, "bic_winners.csv"), winners)

@info "Wrote plots to $results_dir"
@info "Winning config per rat:"
show(winners; allrows=true, allcols=true)
println()
