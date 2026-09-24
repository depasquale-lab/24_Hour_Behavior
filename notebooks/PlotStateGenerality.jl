#=
Response-to-reviewers figure: the hidden states recur across animals.

Addresses "was one animal's State 2 the same as another's?". States are ranked
1..4 within each animal by posterior-weighted accuracy. Every cross-animal test
below matches states on the fitted DDM parameters alone (log v, log B, log τ,
|a0 - 0.5|), so accuracy only supplies the labels being tested, never the match.

  A  mean cross-animal distance between states, by accuracy rank
  B  that distance against rank separation, one line per animal, with shuffle null
  C  leave-one-animal-out: rank assigned from the other 17 animals' templates
     against the animal's own accuracy rank
  D  leave-one-animal-out hit rate against the label-shuffle null
  E  agreement of the accuracy ordering with orderings by other state properties

Input is `state_parameters_long.csv` from ExtractStateParameters.jl and
`state_psychometric_slopes.csv` from PlotStateBehavior.jl.
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Statistics
using StatsBase
using Random
using Printf
using Plots
using StatsPlots

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
    titlefontsize=9,
    legendfontsize=7,
    grid=false,
    framestyle=:axes,
)

function savefig_both(p, stem::AbstractString)
    savefig(p, stem * ".png")
    return savefig(p, stem * ".svg")
end

Random.seed!(67)

results_dir = joinpath("results", "final_ddmhmms")
df = CSV.read(joinpath(results_dir, "state_parameters_long.csv"), DataFrame)
slopes = CSV.read(joinpath(results_dir, "state_psychometric_slopes.csv"), DataFrame)
df = innerjoin(df, slopes[:, [:rat, :state, :slope]]; on=[:rat, :state])
df.rat = string.(df.rat)

rats = sort(unique(df.rat))
const K = 4
const NRAT = length(rats)
const EPS = 1e-3
const NNULL = 2000

df.logv = log.(df.v .+ EPS)
df.logB = log.(df.B)
df.logτ = log.(df.tau .+ EPS)
df.absbias = abs.(df.a0 .- 0.5)
df.vB = df.v .* df.B
transform!(groupby(df, :rat), :acc => (a -> ordinalrank(a; rev=true)) => :acc_rank)
sort!(df, [:rat, :acc_rank])

FEATS = [:logv, :logB, :logτ, :absbias]

# Each animal's four states z-scored within that animal, so matching is on the
# shape of the state structure and not on between-animal parameter offsets.
# Rows are in accuracy-rank order because df is sorted that way.
P = Dict{String,Matrix{Float64}}()
for r in rats
    Z = Matrix{Float64}(df[df.rat .== r, FEATS])
    P[r] = (Z .- mean(Z; dims=1)) ./ (std(Z; dims=1) .+ 1e-9)
end

identity_labels = Dict(r => collect(1:K) for r in rats)
shuffled_labels() = Dict(r => shuffle(collect(1:K)) for r in rats)

dist(x, y) = sqrt(sum((x .- y) .^ 2))

"Mean distance between rank-i states of one animal and rank-j states of another."
function rank_distance_matrix(lab)
    M, n = zeros(K, K), zeros(K, K)
    for a in rats, b in rats
        a == b && continue
        for i in 1:K, j in 1:K
            M[lab[a][i], lab[b][j]] += dist(P[a][i, :], P[b][j, :])
            n[lab[a][i], lab[b][j]] += 1
        end
    end
    return M ./ n
end

offdiag_minus_diag(M) =
    mean(M[i, j] for i in 1:K, j in 1:K if i != j) - mean(M[i, i] for i in 1:K)

"Per animal: mean distance from its states to other animals' states at |Δrank| = d."
function distance_by_separation(lab)
    D = zeros(NRAT, K)
    for (ai, a) in enumerate(rats)
        s, n = zeros(K), zeros(K)
        for b in rats
            b == a && continue
            for i in 1:K, j in 1:K
                d = abs(lab[a][i] - lab[b][j]) + 1
                s[d] += dist(P[a][i, :], P[b][j, :])
                n[d] += 1
            end
        end
        D[ai, :] = s ./ n
    end
    return D
end

function all_permutations(n::Int)
    n == 1 && return [[1]]
    out = Vector{Int}[]
    for p in all_permutations(n - 1), i in 1:n
        push!(out, vcat(p[1:(i - 1)], n, p[i:end]))
    end
    return out
end
const ALL_PERMS = all_permutations(K)

"""
Hold out each animal, average the other animals' state profiles by rank into four
templates, and give the held-out animal's states the one-to-one template
assignment with the smallest total distance (exhaustive over K! assignments).
"""
function leave_one_out(lab)
    assigned = Dict{String,Vector{Int}}()
    for h in rats
        T = zeros(K, length(FEATS))
        for r in rats
            r == h && continue
            for i in 1:K
                T[lab[r][i], :] .+= P[r][i, :]
            end
        end
        T ./= NRAT - 1
        best, bc = ALL_PERMS[1], Inf
        for σ in ALL_PERMS
            c = sum(sum((P[h][i, :] .- T[σ[i], :]) .^ 2) for i in 1:K)
            c < bc && (bc = c; best = σ)
        end
        assigned[h] = best
    end
    exact = mean(vcat([assigned[r] .== lab[r] for r in rats]...))
    within1 = mean(vcat([abs.(assigned[r] .- lab[r]) .<= 1 for r in rats]...))
    return assigned, exact, within1
end

# Panel A / B statistics
M_obs = rank_distance_matrix(identity_labels)
contrast_obs = offdiag_minus_diag(M_obs)
D_obs = distance_by_separation(identity_labels)

contrast_null = Float64[]
D_null = zeros(NNULL, K)
for i in 1:NNULL
    lab = shuffled_labels()
    push!(contrast_null, offdiag_minus_diag(rank_distance_matrix(lab)))
    D_null[i, :] = vec(mean(distance_by_separation(lab); dims=1))
end
p_contrast = (1 + count(>=(contrast_obs), contrast_null)) / (NNULL + 1)
ρ_sep = [corspearman(collect(0.0:(K - 1)), D_obs[a, :]) for a in 1:NRAT]

# Panel C / D statistics
assigned, loo_exact, loo_within1 = leave_one_out(identity_labels)
loo_null = [leave_one_out(shuffled_labels())[2] for _ in 1:NNULL]
p_loo = (1 + count(>=(loo_exact), loo_null)) / (NNULL + 1)
τ_loo = [corkendall(Float64.(assigned[r]), Float64.(1:K)) for r in rats]

C = zeros(Int, K, K)
for r in rats, i in 1:K
    C[i, assigned[r][i]] += 1
end

# Panel E statistics: agreement between the accuracy ordering and orderings by
# other state properties, all oriented so larger = better state.
criteria = [
    (:vB, "v·B"),
    (:v, "v"),
    (:tau, "τ"),
    (:slope, "psych.\nslope"),
    (:occupancy, "occup."),
]
agree = DataFrame(; crit=String[], idx=Int[], rat=String[], τ=Float64[], exact=Int[])
for (j, (f, lab)) in enumerate(criteria), r in rats
    sub = df[df.rat .== r, :]
    rk = ordinalrank(Float64.(sub[!, f]); rev=true)
    push!(agree, (lab, j, r, corkendall(Float64.(rk), Float64.(sub.acc_rank)), count(rk .== sub.acc_rank)))
end

# Figure

rank_ticks = (1:K, string.(1:K))

pA = heatmap(
    1:K, 1:K, M_obs;
    color=cgrad(:Blues; rev=true),
    title="A  cross-animal distance", titlelocation=:left,
    xlabel="state rank, animal j",
    ylabel="state rank, animal i",
    xticks=rank_ticks, yticks=rank_ticks,
    yflip=true, aspect_ratio=:equal,
    xlims=(0.5, K + 0.5), ylims=(0.5, K + 0.5),
    colorbar_title="distance",
)
for i in 1:K, j in 1:K
    annotate!(
        pA, j, i,
        text(@sprintf("%.2f", M_obs[i, j]), 7, FONT_FAMILY, :center,
            M_obs[i, j] < mean(extrema(M_obs)) ? :white : :black),
    )
end

null_lo = [quantile(D_null[:, d], 0.025) for d in 1:K]
null_hi = [quantile(D_null[:, d], 0.975) for d in 1:K]
null_mid = vec(mean(D_null; dims=1))
pB = plot(
    0:(K - 1), null_mid;
    ribbon=(null_mid .- null_lo, null_hi .- null_mid),
    color=:gray60, fillalpha=0.3, linestyle=:dash, linewidth=1,
    label="shuffled ranks (95%)",
    title="B  distance grows with rank gap", titlelocation=:left,
    xlabel="|rank difference| between states",
    ylabel="cross-animal distance",
    xticks=0:(K - 1),
    legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing,
)
for a in 1:NRAT
    plot!(pB, 0:(K - 1), D_obs[a, :]; color=:steelblue, alpha=0.35, linewidth=0.8, label="")
end
plot!(
    pB, 0:(K - 1), vec(mean(D_obs; dims=1));
    yerror=vec(std(D_obs; dims=1)) ./ sqrt(NRAT),
    color=:black, linewidth=2.5, marker=(:circle, 4, stroke(0)), label="observed",
)
annotate!(
    pB, K - 1, minimum(D_obs) + 0.05,
    text(@sprintf("increasing in %d/%d animals", count(>(0), ρ_sep), NRAT), 7, FONT_FAMILY, :right, :gray25),
)

pC = heatmap(
    1:K, 1:K, C;
    color=:Blues, clims=(0, NRAT),
    title="C  held-out animal, rank from others", titlelocation=:left,
    xlabel="rank assigned from other 17 animals",
    ylabel="rank by own accuracy",
    xticks=rank_ticks, yticks=rank_ticks,
    yflip=true, aspect_ratio=:equal,
    xlims=(0.5, K + 0.5), ylims=(0.5, K + 0.5),
    colorbar=false,
)
for i in 1:K, j in 1:K
    annotate!(pC, j, i, text(string(C[i, j]), 8, FONT_FAMILY, :center, C[i, j] > 0.55 * NRAT ? :white : :black))
end

pD = histogram(
    100 .* loo_null;
    bins=range(0, 70; step=100 / (K * NRAT) * 2),
    color=:gray70, linecolor=:gray50, fillalpha=0.8,
    normalize=:probability,
    label="shuffled ranks",
    title="D  transfer to a new animal", titlelocation=:left,
    xlabel="% states assigned the correct rank",
    ylabel="fraction of shuffles",
    xlims=(0, 100),
    legend=:topright, foreground_color_legend=nothing, background_color_legend=nothing,
)
vline!(pD, [100 * loo_exact]; color=:black, linewidth=2.5, label="observed")
vline!(pD, [25]; color=:gray40, linestyle=:dot, linewidth=1, label="chance")
annotate!(
    pD, 100 * loo_exact + 3, 0.5 * ylims(pD)[2],
    text(
        @sprintf("%.0f%% exact\n%.0f%% within ±1\np %s %.4f", 100 * loo_exact, 100 * loo_within1,
            p_loo <= 1 / (NNULL + 1) ? "<" : "=", max(p_loo, 1 / (NNULL + 1))),
        7, FONT_FAMILY, :left, :black,
    ),
)

pE = boxplot(
    agree.idx, agree.τ;
    fillalpha=0.4, linecolor=:black, color=:steelblue, legend=false, outliers=false,
    title="E  accuracy ordering vs. other orderings", titlelocation=:left,
    xlabel="state property used to rank",
    ylabel="Kendall τ with accuracy rank",
    xticks=(1:length(criteria), last.(criteria)),
    ylims=(-1.15, 1.55),
    xlims=(0.4, length(criteria) + 0.6),
)
dotplot!(pE, agree.idx, agree.τ; marker=(:circle, 3, 0.6, stroke(0)), color=:steelblue)
hline!(pE, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
for j in 1:length(criteria)
    s = agree[agree.idx .== j, :]
    annotate!(pE, j, 1.38, text("$(count(>(0), s.τ))/$(NRAT)", 7, FONT_FAMILY, :center, :black))
end

fig = plot(
    pA, pB, pC, pD, pE;
    layout=@layout([a b c; d{0.4w} e]),
    size=(1100, 700),
    left_margin=6Plots.mm,
    bottom_margin=7Plots.mm,
    top_margin=4Plots.mm,
)
savefig_both(fig, joinpath(results_dir, "reviewer_state_generality"))

# Console summary

println("\nCross-animal distance by accuracy rank (features: $(join(FEATS, ", ")))")
display(round.(M_obs; digits=2))
@printf(
    "off-diagonal minus diagonal = %.3f   null %.3f ± %.3f   p = %.4f\n",
    contrast_obs, mean(contrast_null), std(contrast_null), p_contrast
)
@printf("distance increases with rank gap in %d/%d animals\n", count(>(0), ρ_sep), NRAT)
@printf(
    "\nleave-one-animal-out: %.0f%% exact, %.0f%% within ±1   null %.0f%% ± %.0f%%   p = %.4f\n",
    100 * loo_exact, 100 * loo_within1, 100 * mean(loo_null), 100 * std(loo_null), p_loo
)
@printf("  per-animal Kendall τ median %+.2f, positive in %d/%d\n", median(τ_loo), count(>(0), τ_loo), NRAT)
println("\nAgreement of accuracy ordering with other orderings")
for (j, (f, lab)) in enumerate(criteria)
    s = agree[agree.idx .== j, :]
    @printf(
        "  %-10s median τ %+.2f   positive %2d/%d   identical rank %2.0f%% of states\n",
        replace(lab, "\n" => " "), median(s.τ), count(>(0), s.τ), NRAT, 100 * sum(s.exact) / (K * NRAT)
    )
end
println("\nFigure written to $(joinpath(results_dir, "reviewer_state_generality"))")
