#=
Response-to-reviewers figure: is there a data-driven cross-animal state taxonomy?

Not intended for the manuscript. It documents what unsupervised analyses of the
fitted DDM parameters do and do not recover, in support of the accuracy-based
state labelling.

  A  variance explained by each principal component of the within-animal state
     structure, against how often that component orders the states consistently
  B  k-means clustering of all 72 states: how many distinct clusters each
     animal's four states occupy
  C  unsupervised consensus alignment (never uses accuracy) against the
     accuracy-based labelling

Input is `state_parameters_long.csv` from ExtractStateParameters.jl.
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

# Centre each animal on its own mean so the analysis describes within-animal state
# structure rather than between-animal parameter offsets, then scale by the pooled
# standard deviation so no parameter dominates by units alone.
Xw = copy(X)
for rat in rats
    m = df.rat .== rat
    Xw[m, :] .-= mean(Xw[m, :]; dims=1)
end
Xw ./= std(X; dims=1)

# Panel A: variance explained against cross-animal conservation

E = eigen(Symmetric(cov(Xw)))
ord = sortperm(E.values; rev=true)
eigvals, eigvecs = E.values[ord], E.vectors[:, ord]
varexp = eigvals ./ sum(eigvals)
scores = Xw * eigvecs

"Exact two-sided binomial test of `k` successes in `n` trials against p = 0.5."
function sign_test_p(k::Int, n::Int)
    n == 0 && return 1.0
    k = max(k, n - k)
    return min(1.0, 2 * sum(binomial(n, i) for i in k:n) / 2.0^n)
end

"Conventional significance stars: *** < .001, ** < .01, * < .05, otherwise n.s."
function stars(p::Float64)
    p < 0.001 && return "***"
    p < 0.01 && return "**"
    p < 0.05 && return "*"
    return "n.s."
end

# Animals whose τ is exactly zero show no ordering and carry no directional
# information, so they are excluded from the denominator rather than counted
# against the effect.
pc_k, pc_n, pc_p = Int[], Int[], Float64[]
for i in 1:length(FEATS)
    ts = [
        corkendall(Float64.(df[df.rat .== r, :acc_rank]), scores[df.rat .== r, i]) for
        r in rats
    ]
    npos, nneg = count(>(0), ts), count(<(0), ts)
    k, n = max(npos, nneg), npos + nneg
    push!(pc_k, k); push!(pc_n, n); push!(pc_p, sign_test_p(k, n))
end
pc_consistency = pc_k

pretty_feat = Dict(:logB => "log B", :logv => "log v", :logτ => "log τ", :absbias => "|bias|")
pc_names = [
    "PC$i\n($(pretty_feat[FEATS[argmax(abs.(eigvecs[:, i]))]]))" for i in 1:length(FEATS)
]
xs = repeat(1:length(FEATS); outer=2)
grp = repeat(["variance explained", "animals ordered consistently"]; inner=length(FEATS))
ys = vcat(100 .* varexp, 100 .* pc_k ./ pc_n)

pA = groupedbar(
    xs,
    ys;
    group=grp,
    bar_position=:dodge,
    fillalpha=0.75,
    linecolor=:black,
    color=[:steelblue :gray60],   # groups are ordered alphabetically by `group`
    title="A",
    titlelocation=:left,
    titlefontsize=12,
    xlabel="principal component (dominant loading)",
    ylabel="percent",
    xticks=(1:length(FEATS), pc_names),
    ylims=(0, 148),
    legend=:topleft,
    foreground_color_legend=nothing,
    background_color_legend=nothing,
)
# Chance for a majority sign among n non-tied animals sits above 50%; use the
# median non-tied n across components for the reference line.
nref = Int(round(median(pc_n)))
chance = 100 * mean([
    let s = sum(rand((-1, 1), nref) .> 0)
        max(s, nref - s)
    end for _ in 1:20000
]) / nref
hline!(pA, [chance]; color=:black, linestyle=:dash, linewidth=0.8, label="")
annotate!(pA, 4.48, chance + 4, text("chance", 6, FONT_FAMILY, :right, :gray35))
for i in 1:length(FEATS)
    annotate!(
        pA, i, 120,
        text("$(pc_k[i])/$(pc_n[i]) $(stars(pc_p[i]))", 6, FONT_FAMILY, :center, :gray25),
    )
end
# Panel B: does k-means recover a four-state taxonomy?

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
    title="B",
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

# Panel C: unsupervised consensus alignment against the accuracy labelling

# Per-animal z-scored state profiles; alignment is over all K! relabellings, which
# is exhaustive for K = 4 and so needs no assignment heuristic.
P = Dict{eltype(rats),Matrix{Float64}}()
for r in rats
    Z = Xw[df.rat .== r, :]
    P[r] = (Z .- mean(Z; dims=1)) ./ (std(Z; dims=1) .+ 1e-9)
end
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

# Null: the same procedure on parameter-independent random profiles, which gives
# the alignment cost achievable when no real cross-animal correspondence exists.
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

# Slot 1..K is arbitrary, so map slots onto accuracy ranks by the best-matching
# relabelling before cross-tabulating.
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
    title="C",
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
    pA, pB, pC;
    layout=@layout([a{0.36w} b{0.32w} c{0.32w}]),
    size=(1050, 350),
    left_margin=6Plots.mm,
    bottom_margin=8Plots.mm,
    top_margin=8Plots.mm,
)
savefig_both(fig, joinpath(results_dir, "reviewer_state_structure"))

println("\nPCA of within-animal state structure")
for i in 1:length(FEATS)
    j = argmax(abs.(eigvecs[:, i]))
    @printf(
        "  PC%d  %4.1f%% var   consistent %2d/%-2d animals (p = %.4f)   dominant loading %s\n",
        i, 100 * varexp[i], pc_k[i], pc_n[i], pc_p[i], String(FEATS[j])
    )
end
@printf("  chance level for consistency ≈ %.0f%% (n = %d non-tied)\n", chance, nref)

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
println("\nFigure written to $(joinpath(results_dir, "reviewer_state_structure"))")
