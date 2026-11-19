#=
Are states comparable across animals? Ranks each rat's 4 states by accuracy and
checks whether other orderings recover that rank.
A: accuracy rank vs rank from each ordering, + recovery vs shuffle. B-D: PCA.
in: state_parameters_long.csv, state_psychometric_slopes.csv
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Statistics
using StatsBase
using Random
using LinearAlgebra
using Printf
using Plots
using StatsPlots

gr()

const FONT_FAMILY = "Helvetica"
const BOLD = "Helvetica Bold"
default(;
    fontfamily=FONT_FAMILY,
    titlefontfamily=FONT_FAMILY,
    guidefontfamily=FONT_FAMILY,
    tickfontfamily=FONT_FAMILY,
    legendfontfamily=FONT_FAMILY,
    guidefontsize=10,
    tickfontsize=9,
    titlefontsize=11,
    legendfontsize=8,
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
sort!(df, [:rat, :acc_rank])   # row i of each rat = accuracy rank i

FEATS = [:logv, :logB, :logτ, :absbias]

# Within-animal z-scored profiles (rows in accuracy-rank order).
P = Dict{String,Matrix{Float64}}()
for r in rats
    Z = Matrix{Float64}(df[df.rat .== r, FEATS])
    P[r] = (Z .- mean(Z; dims=1)) ./ (std(Z; dims=1) .+ 1e-9)
end
# Population-scaled alternative: keeps between-animal offsets.
Xall = Matrix{Float64}(df[!, FEATS])
Xpop = (Xall .- mean(Xall; dims=1)) ./ std(Xall; dims=1)

identity_labels = Dict(r => collect(1:K) for r in rats)
shuffled_labels() = Dict(r => shuffle(collect(1:K)) for r in rats)
exact_rate(rk, lab) = mean(rk[r][i] == lab[r][i] for r in rats for i in 1:K)
near_rate(rk, lab) = mean(abs(rk[r][i] - lab[r][i]) <= 1 for r in rats for i in 1:K)
ptail(obs, null) = (1 + count(>=(obs), null)) / (NNULL + 1)
rank_by(v) = ordinalrank(v; rev=true)   # larger value = better rank
by_rat(col) = Dict(r => rank_by(col[df.rat .== r]) for r in rats)

function all_permutations(n::Int)
    n == 1 && return [[1]]
    out = Vector{Int}[]
    for p in all_permutations(n - 1), i in 1:n
        push!(out, vcat(p[1:(i - 1)], n, p[i:end]))
    end
    return out
end
const ALL_PERMS = all_permutations(K)

"One-to-one assignment of X's rows onto Y's rows with the smallest total squared distance."
function best_perm(X, Y)
    best, bc = ALL_PERMS[1], Inf
    for σ in ALL_PERMS
        c = sum(sum((X[i, :] .- Y[σ[i], :]) .^ 2) for i in 1:K)
        c < bc && (bc = c; best = σ)
    end
    return best
end

"Average the given animals' state profiles by rank into K templates."
function template(lab, animals)
    T = zeros(K, length(FEATS))
    for r in animals, i in 1:K
        T[lab[r][i], :] .+= P[r][i, :]
    end
    return T ./ length(animals)
end

"Hold out each rat, average the other rats' profiles by rank into templates, and match the held-out rat's states one-to-one."
heldout(lab) = Dict(h => best_perm(P[h], template(lab, filter(!=(h), rats))) for h in rats)
assigned = heldout(identity_labels)
loo_nulls = map(1:NNULL) do _
    lab = shuffled_labels()
    a = heldout(lab)
    (exact_rate(a, lab), near_rate(a, lab))
end
loo_null, loo_null1 = first.(loo_nulls), last.(loo_nulls)

"Label-free consensus: alternate relabelling and template updates; never uses accuracy."
function consensus_alignment(; restarts=400)
    best, best_cost = nothing, Inf
    for _ in 1:restarts
        perm = Dict(r => shuffle(collect(1:K)) for r in rats)
        for _ in 1:200
            T = mean([P[r][perm[r], :] for r in rats])
            newperm = Dict(r => ALL_PERMS[argmin([sum((P[r][σ, :] .- T) .^ 2) for σ in ALL_PERMS])] for r in rats)
            newperm == perm && break
            perm = newperm
        end
        T = mean([P[r][perm[r], :] for r in rats])
        cost = mean([sum((P[r][perm[r], :] .- T) .^ 2) for r in rats])
        cost < best_cost && (best_cost = cost; best = perm)
    end
    return best
end
"""
Consensus slots are arbitrary names, so give them the rank names that best match
accuracy overall (one global relabelling). Returns each rat's consensus rank for
its accuracy-rank-1..K states.
"""
function name_slots(slot_of)
    σb = argmax(σ -> mean(σ[slot_of[r][i]] == i for r in rats for i in 1:K), ALL_PERMS)
    return Dict(r => [σb[slot_of[r][i]] for i in 1:K] for r in rats)
end
cons_perm = consensus_alignment()
cons_rank = name_slots(Dict(r => invperm(cons_perm[r]) for r in rats))
# Slots are named to match accuracy, so the null gets the same advantage.
cons_nulls = map(1:NNULL) do _
    cr = name_slots(shuffled_labels())
    (exact_rate(cr, identity_labels), near_rate(cr, identity_labels))
end
cons_null, cons_null1 = first.(cons_nulls), last.(cons_nulls)

"PCA of the given states, with PC1 and PC2 signs fixed by their log v loadings (never by accuracy)."
function pca_oriented(Z)
    E = eigen(Symmetric(cov(Z)))
    ord = sortperm(E.values; rev=true)
    V = E.vectors[:, ord]
    for j in 1:2   # eigenvector signs are arbitrary; pin PC1 and PC2 by their log v loading
        V[:, j] .*= sign(V[findfirst(==(:logv), FEATS), j])
    end
    return (; S=(Z .- mean(Z; dims=1)) * V, ve=E.values[ord] ./ sum(E.values), V)
end
Xwz = reduce(vcat, [P[r] for r in rats])   # same row order as df
pca_pop, pca_wz = pca_oriented(Xpop), pca_oriented(Xwz)

"Per animal: Kendall τ between its states' accuracy and their PC1 scores."
pc1_tau(pc) = [corkendall(df.acc[df.rat .== r], pc.S[df.rat .== r, 1]) for r in rats]
τ_pop, τ_wz = pc1_tau(pca_pop), pc1_tau(pca_wz)

"Exact two-sided binomial test of `k` successes in `n` trials against p = 0.5."
function sign_test_p(k::Int, n::Int)
    n == 0 && return 1.0
    k = max(k, n - k)
    return min(1.0, 2 * sum(binomial(n, i) for i in k:n) / 2.0^n)
end
p_sign(τs) = sign_test_p(count(>(0), τs), count(!=(0), τs))

dist(x, y) = sqrt(sum((x .- y) .^ 2))
M_obs = [mean(dist(P[a][i, :], P[b][j, :]) for a in rats, b in rats if a != b) for i in 1:K, j in 1:K]

"Lloyd's algorithm with random restarts; returns the best labelling found."
function kmeans_best(Z, k; restarts=500, iters=100)
    n = size(Z, 1)
    best_lab, best_cost = zeros(Int, n), Inf
    for _ in 1:restarts
        C = Z[randperm(n)[1:k], :]
        lab = zeros(Int, n)
        for _ in 1:iters
            changed = false
            for i in 1:n
                a = argmin([sum((Z[i, :] .- C[c, :]) .^ 2) for c in 1:k])
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
km = kmeans_best(Xwz, K)
km_span = [length(unique(km[df.rat .== r])) for r in rats]
km_sizes = sort([count(==(c), km) for c in 1:K]; rev=true)

# orderings vs accuracy rank. simple orderings get a random-order null
simple_nulls = map(1:NNULL) do _
    perms = [shuffle(1:K) for _ in rats]
    (mean(mean(q .== 1:K) for q in perms), mean(mean(abs.(q .- (1:K)) .<= 1) for q in perms))
end
simple_exact_null, simple_within1_null = first.(simple_nulls), last.(simple_nulls)
# (title, short name, ranks, exact null, within-one null)
routes = [
    ("Within-animal PC1", "Within-animal PC1", by_rat(pca_wz.S[:, 1]), simple_exact_null, simple_within1_null),
    ("Consensus (no labels)", "Consensus", cons_rank, cons_null, cons_null1),
    ("Held-out template", "Held-out template", assigned, loo_null, loo_null1),
    ("Absolute PC1 (control)", "Absolute PC1", by_rat(pca_pop.S[:, 1]), simple_exact_null, simple_within1_null),
    ("Non-decision time", "Non-decision time", by_rat(df.tau), simple_exact_null, simple_within1_null),
    ("Occupancy", "Occupancy", by_rat(df.occupancy), simple_exact_null, simple_within1_null),
    ("Psychometric slope", "Psych. slope", by_rat(df.slope), simple_exact_null, simple_within1_null),
]
const CONTROL = 4
route_counts = map(routes) do rt
    N = zeros(Int, K, K)
    for r in rats, i in 1:K
        N[i, rt[3][r][i]] += 1
    end
    N
end
route_exact = [exact_rate(rt[3], identity_labels) for rt in routes]
route_within1 = [near_rate(rt[3], identity_labels) for rt in routes]
route_p = [ptail(route_exact[j], routes[j][4]) for j in eachindex(routes)]
route_p1 = [ptail(route_within1[j], routes[j][5]) for j in eachindex(routes)]

# figure

rank_colors = [colorant"#9E2A2B", colorant"#E29578", colorant"#83C5BE", colorant"#006D77"]   # rank 1 (best) -> 4
pfmt(p) = p < 0.001 ? "p < .001" : replace(@sprintf("p = %.3f", p), "0." => ".")
"Large bold panel letter at the top-left corner, outside the axes (fractions of the axis span)."
function letter!(p, L; dx=-0.09, dy=1.13)
    (x0, x1), (y0, y1) = xlims(p), ylims(p)
    annotate!(p, x0 + dx * (x1 - x0), y0 + dy * (y1 - y0), text(L, 20, BOLD, :left, :black))
end
spacer(; kw...) = plot(; framestyle=:none, ticks=false, grid=false, legend=false, kw...)

# A: confusion matrices
CMAX = maximum(maximum.(route_counts))
const MAT_MARGINS = (; left_margin=2Plots.mm, right_margin=1Plots.mm, top_margin=2Plots.mm, bottom_margin=2Plots.mm)
route_panels = map(enumerate(routes)) do (j, rt)
    N = route_counts[j]
    pl = heatmap(
        1:K, 1:K, N;
        c=:viridis, clims=(0, CMAX), colorbar=false, legend=false, yflip=true, aspect_ratio=1,
        xlims=(0.5, K + 0.5), ylims=(0.5, K + 0.5), xticks=1:K, yticks=1:K, framestyle=:box,
        title=rt[1], titlefontfamily=BOLD, titlefontsize=9, tickfontsize=8, guidefontsize=8,
        xlabel="rank by ordering", ylabel=j in (1, 5) ? "accuracy rank" : "",
        MAT_MARGINS...,
    )
    for i in 1:K, k in 1:K
        annotate!(pl, k, i, text(string(N[i, k]), 9, FONT_FAMILY, :center, N[i, k] > 0.6CMAX ? :black : :white))
    end
    for k in 1:K
        plot!(pl, Shape(k .+ [-0.5, 0.5, 0.5, -0.5], k .+ [-0.5, -0.5, 0.5, 0.5]); fillalpha=0, linecolor=:white, linewidth=1.2, label="")
    end
    pl
end
route_cb = heatmap(
    [1], range(0, CMAX; length=60), reshape(collect(range(0, CMAX; length=60)), :, 1);
    c=:viridis, clims=(0, CMAX), colorbar=false, legend=false, xticks=false, yticks=unique([0, CMAX ÷ 2, CMAX]),
    ymirror=true, framestyle=:box, title="animals", titlefontsize=9, tickfontsize=8, guidefontsize=8,
    xlabel=" ", MAT_MARGINS...,
)

# A, last panel: exact and within-1 recovery, dashed = shuffle mean
BAR_EXACT, BAR_NEAR = colorant"#3E5C7E", colorant"#A9BCD0"
nr = length(routes)
pmax = maximum(vcat(route_p, route_p1)[setdiff(1:2nr, (CONTROL, nr + CONTROL))])
pbars = plot(;
    title="Rank recovery", titlefontfamily=BOLD, titlefontsize=9, tickfontsize=8, guidefontsize=8,
    xlims=(0, 112), xticks=0:25:100, ylims=(0.45, nr + 1.75), yflip=false,
    yticks=(collect(nr:-1:1), [rt[2] for rt in routes]), legend=false,
    xlabel="states recovered (% of 72)",
    MAT_MARGINS...,
)
for (j, rt) in enumerate(routes)
    y = nr + 1 - j
    for (obs, null, p, c, y0, y1) in (
        (route_exact[j], rt[4], route_p[j], BAR_EXACT, y + 0.02, y + 0.38),
        (route_within1[j], rt[5], route_p1[j], BAR_NEAR, y - 0.38, y - 0.02),
    )
        plot!(pbars, Shape([0, 100obs, 100obs, 0], [y0, y0, y1, y1]); fillcolor=c, linewidth=0, label="")
        plot!(pbars, fill(100mean(null), 2), [y0 - 0.03, y1 + 0.03]; color=:black, linestyle=:dash, linewidth=0.9, label="")
        lab = @sprintf("%.0f", 100obs) * (p >= 0.05 ? " n.s." : "")
        annotate!(pbars, 100obs + 1.5, (y0 + y1) / 2, text(lab, 7, FONT_FAMILY, :left, :black))
    end
end
let y = nr + 0.85
    for (x, c, lab) in ((1, BAR_EXACT, "exact"), (33, BAR_NEAR, "within ±1"))
        plot!(pbars, Shape([x, x + 6, x + 6, x], [y - 0.15, y - 0.15, y + 0.15, y + 0.15]); fillcolor=c, linewidth=0, label="")
        annotate!(pbars, x + 8, y, text(lab, 7, FONT_FAMILY, :left, :black))
    end
    plot!(pbars, [74, 82], [y, y]; color=:black, linestyle=:dash, linewidth=0.9, label="")
    annotate!(pbars, 84, y, text("shuffle", 7, FONT_FAMILY, :left, :black))
    annotate!(pbars, 1, y + 0.5, text(pmax < 0.001 ? "all p < .001 except the control" : @sprintf("all p ≤ %.3f except the control", pmax),
        7, FONT_FAMILY, :left, :black))
end

# narrow column left of each row of matrices; the first carries the panel letter
function letter_col(letter="")
    sp = spacer(; xlims=(0, 1), ylims=(0, 1), title=" ", titlefontsize=9, xlabel="rank by ordering", guidefontsize=8,
        xguidefontcolor=:white, MAT_MARGINS...)
    letter == "" || annotate!(sp, 0.0, 1.12, text(letter, 20, BOLD, :left, :black))
    return sp
end
sA1 = letter_col("A")
sA2 = letter_col()

# B, C: all 72 states on PC1/PC2, absolute versus within-animal parameters
acc_wz = vcat([(a = df.acc[df.rat .== r]; (a .- mean(a)) ./ std(a)) for r in rats]...)
function pca_panel(pc, ttl; leg=false)
    p = plot(;
        title=ttl,
        xlabel=@sprintf("PC1 (%.0f%%)", 100 * pc.ve[1]), ylabel=@sprintf("PC2 (%.0f%%)", 100 * pc.ve[2]),
        legend=leg ? :topleft : false, legendtitle="accuracy rank", legendtitlefontsize=8,
        foreground_color_legend=nothing, background_color_legend=nothing,
    )
    for k in 1:K
        m = df.acc_rank .== k
        scatter!(p, pc.S[m, 1], pc.S[m, 2]; color=rank_colors[k], marker=(:circle, 4, stroke(0)), alpha=0.9, label=string(k))
    end
    # Accuracy regressed on PC1/PC2 after the PCA, drawn as a biplot arrow.
    Xs = pc.S[:, 1:2]
    b = Xs \ acc_wz
    R2 = 1 - sum((acc_wz .- Xs * b) .^ 2) / sum(acc_wz .^ 2)
    L = 0.45 * minimum(maximum(abs.(Xs); dims=1))
    u = L .* b ./ norm(b)
    plot!(p, [-u[1], u[1]], [-u[2], u[2]]; color=:black, linewidth=1.5, arrow=arrow(:closed, :head, 0.4, 0.3), label="")
    annotate!(p, 1.12u[1], 1.12u[2] + 0.12L, text(@sprintf("accuracy\nR² = %.2f", R2), 8, FONT_FAMILY, u[1] >= 0 ? :left : :right, :black))
    return p
end
pC = pca_panel(pca_pop, "PCA, absolute parameters"; leg=true)
pD = pca_panel(pca_wz, "PCA, within-animal parameters")

# D: the label-free axis
pE = plot(;
    title="Within-animal PC1 vs. accuracy rank",
    xlabel="state accuracy rank (within animal)", ylabel=@sprintf("PC1 score (%.0f%% of variance)", 100 * pca_wz.ve[1]),
    xticks=(1:K, ["1\nbest", "2", "3", "4\nworst"]), xlims=(0.7, K + 0.3), legend=false,
)
for r in rats
    m = df.rat .== r
    plot!(pE, df.acc_rank[m], pca_wz.S[m, 1]; color=:gray80, linewidth=0.6)
    scatter!(pE, df.acc_rank[m], pca_wz.S[m, 1]; color=rank_colors[df.acc_rank[m]], marker=(:circle, 4, stroke(0)), alpha=0.9)
end
mk = [mean(pca_wz.S[df.acc_rank .== k, 1]) for k in 1:K]
sk = [std(pca_wz.S[df.acc_rank .== k, 1]) / sqrt(NRAT) for k in 1:K]
plot!(pE, 1:K, mk; yerror=sk, color=:black, linewidth=1.6, marker=(:circle, 4.5, stroke(0)))

const PCA_MARGINS = (; left_margin=6Plots.mm, right_margin=2Plots.mm, top_margin=6Plots.mm, bottom_margin=3Plots.mm)
for pl in (pC, pD, pE)
    plot!(pl; PCA_MARGINS...)
end
letter!(pC, "B"; dx=-0.22)
letter!(pD, "C"; dx=-0.22)
letter!(pE, "D"; dx=-0.22)

rowA1 = plot(sA1, route_panels[1:4]..., route_cb; layout=grid(1, 6; widths=[0.03, 0.225, 0.225, 0.225, 0.225, 0.07]))
# bar chart labels pad this row more, so wider matrix cells to match the top row
rowA2 = plot(sA2, route_panels[5:7]..., pbars; layout=grid(1, 5; widths=[0.03, 0.248, 0.248, 0.248, 0.226]))
rowP = plot(pC, pD, pE; layout=(1, 3))
topgap = spacer(; top_margin=0Plots.mm, bottom_margin=0Plots.mm)   # room above A's titles and letter
fig = plot(topgap, rowA1, rowA2, rowP; layout=grid(4, 1; heights=[0.025, 0.265, 0.265, 0.445]), size=(1100, 960))
savefig_both(fig, joinpath(results_dir, "state_generality"))

# summary

println("\nexact / within-1 rank recovered (chance, p)")
for (j, rt) in enumerate(routes)
    @printf("  %-24s %3.0f%% (%2.0f%%, p = %.4f)   %3.0f%% (%2.0f%%, p = %.4f)\n",
        rt[1], 100route_exact[j], 100mean(rt[4]), route_p[j], 100route_within1[j], 100mean(rt[5]), route_p1[j])
end
# accuracy is mostly set by v and B, so these are partly built in
for (name, col) in (("v", df.v), ("v*B", df.vB))
    @printf("  %-24s %3.0f%%\n", name, 100exact_rate(by_rat(col), identity_labels))
end
println("\nPC1 vs accuracy, per-rat Kendall τ")
for (name, pc, τs) in (("within-animal z", pca_wz, τ_wz), ("population z", pca_pop, τ_pop))
    @printf("  %-16s PC1 %.0f%% var  positive %d/%d  median τ %+.2f  sign test p = %.4f  loadings %s\n",
        name, 100 * pc.ve[1], count(>(0), τs), NRAT, median(τs), p_sign(τs),
        join(["$(f)=$(round(v; digits=2))" for (f, v) in zip(FEATS, pc.V[:, 1])], " "))
end
println("\ncross-rat distance by accuracy rank (within-animal z)")
display(round.(M_obs; digits=2))
@printf("k-means: cluster sizes %s; rats spanning 1..4 clusters %s\n", join(km_sizes, "/"), join([count(==(j), km_span) for j in 1:K], "/"))
println("\nfigure: $(joinpath(results_dir, "state_generality"))")
