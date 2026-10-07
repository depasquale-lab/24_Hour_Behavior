#=
Can unsupervised analyses of the fitted DDM parameters recover a cross-animal
state taxonomy? Not a manuscript figure.

  A  PCA of all 72 states under three normalisations (population z, within-animal
     centred with pooled SD, within-animal z), coloured by accuracy rank
  B  per normalisation: leave-one-animal-out template rank recovery vs shuffle
     null, and animals whose PC1 orders states by accuracy
  C  k-means: how many clusters each animal's four states occupy
  D  unsupervised consensus alignment (never uses accuracy) vs the accuracy labels

Input: state_parameters_long.csv from ExtractStateParameters.jl.
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Statistics
using StatsBase
using LinearAlgebra
using Random
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

function savefig_both(p, stem::AbstractString)
    savefig(p, stem * ".png")
    return savefig(p, stem * ".svg")
end

Random.seed!(67)

results_dir = joinpath("results", "final_ddmhmms")
df = CSV.read(joinpath(results_dir, "state_parameters_long.csv"), DataFrame)
rats = sort(unique(df.rat))
const K = 4
const NRAT = length(rats)
const EPS = 1e-6

df.logB = log.(df.B)
df.logv = log.(df.v .+ EPS)
df.logτ = log.(df.tau .+ EPS)
df.absbias = abs.(df.a0 .- 0.5)
transform!(groupby(df, :rat), :acc => (a -> ordinalrank(a; rev=true)) => :acc_rank)
sort!(df, [:rat, :state])

FEATS = [:logB, :logv, :logτ, :absbias]
X = Matrix{Float64}(df[!, FEATS])

# Three normalisations: population z; within-animal centred, pooled SD (used by
# k-means); within-animal z.
Xp = (X .- mean(X; dims=1)) ./ std(X; dims=1)
Xw = copy(X)
Xz = copy(X)
for rat in rats
    m = df.rat .== rat
    Xw[m, :] .-= mean(Xw[m, :]; dims=1)
    Xz[m, :] = (X[m, :] .- mean(X[m, :]; dims=1)) ./ (std(X[m, :]; dims=1) .+ 1e-9)
end
Xw ./= std(X; dims=1)
NORMS = [
    ("z-scored across population", Xp),
    ("centred within animal", Xw),
    ("z-scored within animal", Xz),
]

"Exact two-sided binomial test of `k` successes in `n` trials against p = 0.5."
function sign_test_p(k::Int, n::Int)
    n == 0 && return 1.0
    k = max(k, n - k)
    return min(1.0, 2 * sum(binomial(n, i) for i in k:n) / 2.0^n)
end

"Principal component scores and variance explained."
function pca_scores(Z)
    E = eigen(Symmetric(cov(Z)))
    ord = sortperm(E.values; rev=true)
    return (Z .- mean(Z; dims=1)) * E.vectors[:, ord], E.values[ord] ./ sum(E.values)
end

"""
Hold out each animal, average the other animals' states by rank into K templates,
and give each held-out state the rank of its nearest template.
"""
function loo_nearest(Z, ranks)
    hits = 0
    for h in rats
        tr = df.rat .!= h
        T = reduce(vcat, [mean(Z[tr .& (ranks .== k), :]; dims=1) for k in 1:K])
        for i in findall(df.rat .== h)
            hits += argmin([sum((Z[i, :] .- T[k, :]) .^ 2) for k in 1:K]) == ranks[i]
        end
    end
    return hits / length(ranks)
end

function shuffled_ranks()
    r = copy(df.acc_rank)
    for rat in rats
        m = findall(df.rat .== rat)
        r[m] = shuffle(r[m])
    end
    return r
end

const NNULL = 2000
null_ranks = [shuffled_ranks() for _ in 1:NNULL]

# Panel A / B statistics; τ = 0 animals are excluded from the sign test.
norm_stats = map(NORMS) do (name, Z)
    S, ve = pca_scores(Z)
    ts = [corkendall(Float64.(df[df.rat .== r, :acc_rank]), S[df.rat .== r, 1]) for r in rats]
    npos, nneg = count(>(0), ts), count(<(0), ts)
    loo = loo_nearest(Z, df.acc_rank)
    null = [loo_nearest(Z, r) for r in null_ranks]
    (;
        name, S, ve,
        pc1_k=max(npos, nneg), pc1_n=npos + nneg, pc1_p=sign_test_p(max(npos, nneg), npos + nneg),
        loo, null, loo_p=(1 + count(>=(loo), null)) / (NNULL + 1),
    )
end

rank_colors = [:firebrick, :orange, :mediumseagreen, :royalblue]
pA = map(enumerate(norm_stats)) do (j, ns)
    p = plot(;
        title=(j == 1 ? "A  " : "") * ns.name, titlelocation=:left, titlefontsize=9,
        xlabel=@sprintf("PC1 (%.0f%%)", 100 * ns.ve[1]),
        ylabel=@sprintf("PC2 (%.0f%%)", 100 * ns.ve[2]),
        legend=j == 1 ? :topright : false, legendtitle="accuracy rank",
        legendtitlefontsize=7, foreground_color_legend=nothing, background_color_legend=nothing,
    )
    for k in 1:K
        m = df.acc_rank .== k
        scatter!(p, ns.S[m, 1], ns.S[m, 2]; color=rank_colors[k], marker=(:circle, 4, stroke(0)), alpha=0.85, label=string(k))
    end
    return p
end

pstr(p) = p <= 1 / (NNULL + 1) ? @sprintf("p < %.4f", 1 / (NNULL + 1)) : @sprintf("p = %.3f", p)
short = ["population", "centred", "within-\nanimal z"]
pD = plot(;
    title="B  relative structure recovers rank", titlelocation=:left, titlefontsize=12,
    ylabel="% states given the correct rank\n(held-out animal, nearest template)",
    xticks=(1:3, short), xlims=(0.4, 3.6), ylims=(0, 80), legend=false,
)
hline!(pD, [25]; color=:gray40, linestyle=:dot, linewidth=1)
annotate!(pD, 3.55, 27.5, text("chance", 6, FONT_FAMILY, :right, :gray35))
annotate!(pD, 3.55, 3, text("*animals with τ = 0 excluded", 6, FONT_FAMILY, :right, :gray35))
for (x, ns) in enumerate(norm_stats)
    violin!(pD, fill(x, NNULL), 100 .* ns.null; color=:gray80, linewidth=0)
    scatter!(pD, [x], [100 * ns.loo]; color=:black, marker=(:diamond, 7, stroke(0)))
    annotate!(pD, x, 100 * ns.loo + 5, text(@sprintf("%.0f%%, %s", 100 * ns.loo, pstr(ns.loo_p)), 7, FONT_FAMILY, :center, :black))
    annotate!(pD, x, 73, text("PC1 orders\n$(ns.pc1_k)/$(ns.pc1_n) animals*", 6, FONT_FAMILY, :center, :gray25))
end

# Panel C: does k-means recover a four-state taxonomy?

"Lloyd's algorithm with random restarts; returns the best labelling found."
function kmeans_best(Z::Matrix{Float64}, k::Int; restarts::Int=500, iters::Int=100)
    n = size(Z, 1)
    best_lab, best_cost = zeros(Int, n), Inf
    for _ in 1:restarts
        C = Z[randperm(n)[1:k], :]
        lab = zeros(Int, n)
        for _ in 1:iters
            changed = false
            for i in 1:n
                d = [sum((Z[i, :] .- C[c, :]) .^ 2) for c in 1:k]
                a = argmin(d)
                a != lab[i] && (lab[i] = a; changed = true)
            end
            for c in 1:k
                m = lab .== c
                any(m) && (C[c, :] = vec(mean(Z[m, :]; dims=1)))
            end
            changed || break
        end
        cost = sum(sum((Z[i, :] .- C[lab[i], :]) .^ 2) for i in 1:n)
        cost < best_cost && (best_cost = cost; best_lab = copy(lab))
    end
    return best_lab
end

clab = kmeans_best(Xw, K)
df.cluster = clab
spanned = [length(unique(df[df.rat .== r, :cluster])) for r in rats]
cluster_sizes = [count(==(c), clab) for c in 1:K]

counts = [count(==(j), spanned) for j in 1:K]
pB = bar(
    1:K,
    counts;
    fillalpha=0.7,
    linecolor=:black,
    color=:indianred,
    legend=false,
    title="C",
    titlelocation=:left,
    titlefontsize=12,
    xlabel="distinct clusters occupied by an animal's 4 states",
    ylabel="number of animals",
    xticks=(1:K, string.(1:K)),
    ylims=(0, maximum(counts) * 1.25),
)
for j in 1:K
    counts[j] > 0 && annotate!(
        pB, j, counts[j] + maximum(counts) * 0.06,
        text(string(counts[j]), 8, FONT_FAMILY, :center, :black),
    )
end
annotate!(
    pB,
    2.5,
    maximum(counts) * 1.15,
    text(
        "cluster sizes " * join(sort(cluster_sizes; rev=true), "/"),
        7, FONT_FAMILY, :center, :gray25,
    ),
)

# Panel D: unsupervised consensus alignment against the accuracy labelling

# Alignment is exhaustive over all K! relabellings.
P = Dict(r => Xz[df.rat .== r, :] for r in rats)
"All permutations of 1:n (matches the helper in CompareVTiedVsFull.jl)."
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
    consensus_alignment(P; restarts)

Align every animal's K states to a shared template by alternating between
(a) relabelling each animal to best match the template and (b) recomputing the
template. Uses only the DDM parameters; accuracy is never consulted.
"""
function consensus_alignment(P::AbstractDict; restarts::Int=400)
    best_perm, best_cost = nothing, Inf
    for _ in 1:restarts
        perm = Dict(r => shuffle(collect(1:K)) for r in rats)
        T = mean([P[r][perm[r], :] for r in rats])
        local cost = Inf
        for _ in 1:200
            newperm = Dict{eltype(rats),Vector{Int}}()
            for r in rats
                bp, bc = perm[r], Inf
                for σ in ALL_PERMS
                    c = sum((P[r][σ, :] .- T) .^ 2)
                    c < bc && (bc = c; bp = copy(σ))
                end
                newperm[r] = bp
            end
            stable = all(newperm[r] == perm[r] for r in rats)
            perm = newperm
            T = mean([P[r][perm[r], :] for r in rats])
            cost = mean([sum((P[r][perm[r], :] .- T) .^ 2) for r in rats])
            stable && break
        end
        cost < best_cost && (best_cost = cost; best_perm = deepcopy(perm))
    end
    return best_perm, best_cost
end

perm, consensus_cost = consensus_alignment(P)

# Null: same procedure on random profiles.
null_costs = Float64[]
for _ in 1:200
    Pn = Dict(
        r => (
            let Z = randn(K, length(FEATS))
                (Z .- mean(Z; dims=1)) ./ (std(Z; dims=1) .+ 1e-9)
            end
        ) for r in rats
    )
    push!(null_costs, consensus_alignment(Pn; restarts=25)[2])
end

# Slots are arbitrary; map them onto accuracy ranks by the best relabelling.
slot = Dict{Tuple{eltype(rats),Int},Int}()
for r in rats
    states = df[df.rat .== r, :state]
    for (sl, idx) in enumerate(perm[r])
        slot[(r, states[idx])] = sl
    end
end
df.consensus_slot = [slot[(r, s)] for (r, s) in zip(df.rat, df.state)]

"Map consensus slots onto accuracy ranks by the best-matching relabelling."
function best_slot_map(slots, ranks)
    bm, ba = collect(1:K), -1.0
    for σ in ALL_PERMS
        a = mean([σ[c] for c in slots] .== ranks)
        if a > ba
            ba, bm = a, copy(σ)
        end
    end
    return bm, ba
end

best_map, best_agree = best_slot_map(df.consensus_slot, df.acc_rank)
df.consensus_rank = [best_map[c] for c in df.consensus_slot]

M = zeros(Int, K, K)
for (ar, cr) in zip(df.acc_rank, df.consensus_rank)
    M[ar, cr] += 1
end
τ_cons = [
    corkendall(
        Float64.(df[df.rat .== r, :acc_rank]), Float64.(df[df.rat .== r, :consensus_rank])
    ) for r in rats
]

pC = heatmap(
    1:K,
    1:K,
    M;
    color=:Blues,
    clims=(0, maximum(M)),
    title="D",
    titlelocation=:left,
    titlefontsize=12,
    xlabel="rank from unsupervised consensus",
    ylabel="rank by accuracy",
    xticks=1:K,
    yticks=1:K,
    yflip=true,
    colorbar=false,
)
for a in 1:K, b in 1:K
    annotate!(
        pC, b, a,
        text(
            string(M[a, b]), 8, FONT_FAMILY, :center,
            M[a, b] > 0.6 * maximum(M) ? :white : :black,
        ),
    )
end
annotate!(
    pC,
    0.55,
    0.35,
    text(
        @sprintf("%.0f%% agreement, median τ = %+.2f", 100 * best_agree, median(τ_cons)),
        7, FONT_FAMILY, :left, :gray25,
    ),
)

fig = plot(
    pA..., pD, pB, pC;
    layout=@layout([a b c; d e f]),
    size=(1100, 720),
    left_margin=7Plots.mm,
    bottom_margin=8Plots.mm,
    top_margin=5Plots.mm,
)
savefig_both(fig, joinpath(results_dir, "state_structure"))

println("\nPCA and held-out rank recovery by normalisation (chance 25%)")
for ns in norm_stats
    @printf(
        "  %-28s PC1 %4.1f%% var   orders %2d/%-2d animals (p = %.4f)   held-out %2.0f%%  null %2.0f%% ± %.0f%%  %s\n",
        ns.name, 100 * ns.ve[1], ns.pc1_k, ns.pc1_n, ns.pc1_p,
        100 * ns.loo, 100 * mean(ns.null), 100 * std(ns.null), pstr(ns.loo_p)
    )
end

@printf("\nk-means (k=%d): cluster sizes %s\n", K, join(sort(cluster_sizes; rev=true), "/"))
for j in 1:K
    @printf("  %2d animals span %d distinct clusters\n", counts[j], j)
end

@printf(
    "\nconsensus alignment cost %.2f vs null %.2f ± %.2f  (p = %.3f)\n",
    consensus_cost, mean(null_costs), std(null_costs),
    mean(null_costs .<= consensus_cost)
)
@printf(
    "consensus vs accuracy labelling: %.0f%% of states, median τ = %+.2f, %d/%d animals positive\n",
    100 * best_agree, median(τ_cons), count(>(0), τ_cons), NRAT
)
println("\nFigure written to $(joinpath(results_dir, "state_structure"))")
