#=
Response-to-reviewers figure: how states are matched across animals, and how well
the accuracy-ranked states align.

Addresses (R1) "was one animal's State 2 the same as another's?" and (R2) how
states are matched across animals, how well they align, and whether a more
data-driven approach than sorting on accuracy would do better. States are ranked
1..4 within each animal by posterior-weighted accuracy. Every test below works on
each state's DDM parameter profile (log v, log B, log τ, |a0 - 0.5|) z-scored
within its animal; accuracy only supplies the labels being compared.

  A  flow chart of the matching and each analysis (with the absolute-parameter
     control), and Ai-Aiv worked examples on real animals: ranking by accuracy,
     the parameter profile, held-out matching and label-free consensus
  B  per-animal rank correlation with accuracy for every route, each against its
     own shuffle null: label-free PCA axis, label-free consensus, held-out
     template matching, non-decision time alone, occupancy, psychometric slope,
     and PCA on absolute (population-scaled) parameters. Median with bootstrap
     CI over animals, beside the count of animals agreeing / tied / opposed
  C  PCA of all 72 states on absolute (population-scaled) parameters
  D  PCA of all 72 states on within-animal z-scored parameters
     (C, D: arrow = within-animal accuracy regressed on PC1/PC2 after the PCA)
  E  PC1 of D against accuracy rank, one line per animal

PC1's sign is fixed by its log v loading, never by accuracy. Consensus slots are
named by the single global relabelling that best matches accuracy, and its null
gets the same advantage.

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
using LinearAlgebra
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

# The alternative: each parameter z-scored once over all 72 states, so absolute
# parameter values (and between-animal offsets) are kept.
Xall = Matrix{Float64}(df[!, FEATS])
Xpop = (Xall .- mean(Xall; dims=1)) ./ std(Xall; dims=1)
Ppop = Dict(r => Xpop[df.rat .== r, :] for r in rats)

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

"One-to-one assignment of X's rows onto Y's rows with the smallest total squared distance."
function best_perm(X, Y)
    best, bc = ALL_PERMS[1], Inf
    for σ in ALL_PERMS
        c = sum(sum((X[i, :] .- Y[σ[i], :]) .^ 2) for i in 1:K)
        c < bc && (bc = c; best = σ)
    end
    return best
end

# Pairwise assignments depend only on the profiles, so compute them once; the
# shuffle null only relabels which rank each matched state carries.
const PAIR_PERMS = Dict((a, b) => best_perm(P[a], P[b]) for a in rats, b in rats if a != b)

"""
Match every animal's states one-to-one to every other animal's states. Returns the
row-normalised rank confusion matrix, the overall exact and within-±1 rates, and
each animal's exact and within-±1 rates averaged over its partners.
"""
function pairwise_match(lab)
    Cm = zeros(K, K)
    hits = zeros(NRAT)
    near = zeros(NRAT)
    for (ai, a) in enumerate(rats), b in rats
        a == b && continue
        σ = PAIR_PERMS[(a, b)]
        for i in 1:K
            Cm[lab[a][i], lab[b][σ[i]]] += 1
            hits[ai] += lab[a][i] == lab[b][σ[i]]
            near[ai] += abs(lab[a][i] - lab[b][σ[i]]) <= 1
        end
    end
    exact, within1 = hits ./ (K * (NRAT - 1)), near ./ (K * (NRAT - 1))
    return Cm ./ sum(Cm; dims=2), mean(exact), mean(within1), exact, within1
end

"Average the given animals' state profiles by rank into K templates."
function template(lab, animals; Pm=P)
    T = zeros(K, length(FEATS))
    for r in animals, i in 1:K
        T[lab[r][i], :] .+= Pm[r][i, :]
    end
    return T ./ length(animals)
end

"""
Hold out each animal, average the other animals' state profiles by rank into four
templates, and give the held-out animal's states the one-to-one template
assignment with the smallest total distance (exhaustive over K! assignments).
"""
function leave_one_out(lab; Pm=P)
    assigned = Dict(h => best_perm(Pm[h], template(lab, filter(!=(h), rats); Pm)) for h in rats)
    exact = mean(vcat([assigned[r] .== lab[r] for r in rats]...))
    within1 = mean(vcat([abs.(assigned[r] .- lab[r]) .<= 1 for r in rats]...))
    return assigned, exact, within1
end

# Distance statistics (panel I)
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

# Matching statistics (panels E, F)
assigned, loo_exact, loo_within1 = leave_one_out(identity_labels)
loo_nulls = map(1:NNULL) do _
    lab = shuffled_labels()
    asg, ex, w1 = leave_one_out(lab)
    (ex, w1, median([corkendall(Float64.(asg[r]), Float64.(lab[r])) for r in rats]))
end
loo_null, loo_null1 = getindex.(loo_nulls, 1), getindex.(loo_nulls, 2)
loo_τnull = getindex.(loo_nulls, 3)
p_loo = (1 + count(>=(loo_exact), loo_null)) / (NNULL + 1)
p_loo1 = (1 + count(>=(loo_within1), loo_null1)) / (NNULL + 1)
τ_loo = [corkendall(Float64.(assigned[r]), Float64.(1:K)) for r in rats]

C = zeros(Int, K, K)
for r in rats, i in 1:K
    C[i, assigned[r][i]] += 1
end
C_loo = C ./ NRAT
loo_animal = [mean(assigned[r] .== 1:K) for r in rats]
loo_animal1 = [mean(abs.(assigned[r] .- (1:K)) .<= 1) for r in rats]

C_pair, pair_exact, pair_within1, pair_animal, pair_animal1 = pairwise_match(identity_labels)
pair_nulls = [pairwise_match(shuffled_labels())[2:3] for _ in 1:NNULL]
pair_null, pair_null1 = first.(pair_nulls), last.(pair_nulls)
p_pair = (1 + count(>=(pair_exact), pair_null)) / (NNULL + 1)
p_pair1 = (1 + count(>=(pair_within1), pair_null1)) / (NNULL + 1)

# Normalisation statistics (panels B-D): the same held-out matching on
# population-scaled parameters, and PCA of all 72 states under each scaling.
# Two matching rules. One-to-one forces an animal's four states onto four
# different ranks, which is itself a within-animal comparison; nearest gives each
# state the rank of its closest template independently, so only the scaling
# decides how much of the rank is recoverable.
"Held-out exact rate when each state takes the rank of its nearest template."
function loo_nearest(lab; Pm=P)
    hits = 0
    for h in rats
        T = template(lab, filter(!=(h), rats); Pm)
        hits += sum(argmin([sum((Pm[h][i, :] .- T[k, :]) .^ 2) for k in 1:K]) == lab[h][i] for i in 1:K)
    end
    return hits / (K * NRAT)
end
scaling_tests = Dict(
    (sc, rule) => begin
        Pm = sc == :pop ? Ppop : P
        f = rule == :near ? (l -> loo_nearest(l; Pm)) : (l -> leave_one_out(l; Pm)[2])
        obs = f(identity_labels)
        null = [f(shuffled_labels()) for _ in 1:NNULL]
        (; obs, null, p=(1 + count(>=(obs), null)) / (NNULL + 1))
    end for sc in (:pop, :wz), rule in (:near, :one)
)

"""
PCA of the given states. PC1's sign is fixed so it loads positively on log v, a
choice that never uses accuracy.
"""
function pca_oriented(Z)
    E = eigen(Symmetric(cov(Z)))
    ord = sortperm(E.values; rev=true)
    V = E.vectors[:, ord]
    V[:, 1] .*= sign(V[findfirst(==(:logv), FEATS), 1])
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

# Null for any fixed ordering compared with accuracy: median over animals of the
# Kendall τ between a random ordering of 4 states and the accuracy ordering.
simple_τnull = [median([corkendall(Float64.(shuffle(1:K)), Float64.(1:K)) for _ in rats]) for _ in 1:NNULL]

# Label-free consensus alignment: relabel every animal's states onto a shared
# template by alternating relabelling and template updates, never using accuracy.
function consensus_alignment(Pm; restarts=400)
    best, best_cost = nothing, Inf
    for _ in 1:restarts
        perm = Dict(r => shuffle(collect(1:K)) for r in rats)
        for _ in 1:200
            T = mean([Pm[r][perm[r], :] for r in rats])
            newperm = Dict(r => ALL_PERMS[argmin([sum((Pm[r][σ, :] .- T) .^ 2) for σ in ALL_PERMS])] for r in rats)
            newperm == perm && break
            perm = newperm
        end
        T = mean([Pm[r][perm[r], :] for r in rats])
        cost = mean([sum((Pm[r][perm[r], :] .- T) .^ 2) for r in rats])
        cost < best_cost && (best_cost = cost; best = perm)
    end
    return best
end

"""
Consensus slots are arbitrary names, so give them the rank names that best match
accuracy overall (one global relabelling). Returns each animal's consensus rank for
its accuracy-rank-1..K states.
"""
function name_slots(slot_of)
    σb = argmax(σ -> mean(σ[slot_of[r][i]] == i for r in rats for i in 1:K), ALL_PERMS)
    return Dict(r => [σb[slot_of[r][i]] for i in 1:K] for r in rats)
end
cons_perm = consensus_alignment(P)
cons_rank = name_slots(Dict(r => invperm(cons_perm[r]) for r in rats))
C_cons = zeros(K, K)
for r in rats, i in 1:K
    C_cons[i, cons_rank[r][i]] += 1
end
C_cons ./= NRAT
cons_exact = mean(cons_rank[r][i] == i for r in rats for i in 1:K)
cons_within1 = mean(abs(cons_rank[r][i] - i) <= 1 for r in rats for i in 1:K)
τ_cons = [corkendall(Float64.(cons_rank[r]), Float64.(1:K)) for r in rats]
# The global slot naming is chosen to match accuracy, so the null gets the same
# advantage: random per-animal labellings, named by the best global relabelling.
cons_nulls = map(1:NNULL) do _
    cr = name_slots(Dict(r => shuffle(collect(1:K)) for r in rats))
    (mean(cr[r][i] == i for r in rats for i in 1:K), median([corkendall(Float64.(cr[r]), Float64.(1:K)) for r in rats]),
        mean(abs(cr[r][i] - i) <= 1 for r in rats for i in 1:K))
end
cons_null, cons_τnull, cons_null1 = getindex.(cons_nulls, 1), getindex.(cons_nulls, 2), getindex.(cons_nulls, 3)
p_cons = (1 + count(>=(cons_exact), cons_null)) / (NNULL + 1)
p_cons1 = (1 + count(>=(cons_within1), cons_null1)) / (NNULL + 1)

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

# Panel H statistics: agreement between the accuracy ordering and orderings by
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

# Converging-evidence rows (panel B): per-animal agreement of each ordering with
# accuracy, oriented so positive = agrees, each against its own null.
crit_τ(lab) = agree[agree.crit .== lab, :τ]
evidence = [
    ("PCA, within-animal parameters (PC1)", τ_wz, simple_τnull),
    ("consensus matching", τ_cons, cons_τnull),
    ("held-out template matching", τ_loo, loo_τnull),
    ("non-decision time", crit_τ("τ"), simple_τnull),
    ("state occupancy", crit_τ("occup."), simple_τnull),
    ("psychometric slope", crit_τ("psych.\nslope"), simple_τnull),
    ("PCA, absolute parameters (PC1)", τ_pop, simple_τnull),
]

# Figure

feat_labels = ["log v", "log B", "log τ", "|bias|"]
rank_colors = [:firebrick, :orange, :mediumseagreen, :royalblue]
pfmt(p) = p < 0.001 ? "p < .001" : replace(@sprintf("p = %.3f", p), "0." => ".")
GRAYBOX, BLUEBOX, ORANGEBOX = RGB(0.95, 0.95, 0.95), RGB(0.85, 0.91, 0.97), RGB(0.99, 0.90, 0.80)

# A: how states are matched across animals
pA = plot(;
    framestyle=:none, grid=false, legend=false,
    title="A  Methods overview", titlelocation=:left,
    xlims=(0, 10), ylims=(-0.03, 1),
)
function box!(p, x0, x1, y0, y1, hdr, body, fill; tag="", tagcolor=:black, ls=:solid)
    plot!(p, Shape([x0, x1, x1, x0], [y0, y0, y1, y1]); fillcolor=fill, linecolor=:gray40, linewidth=0.8, linestyle=ls)
    xc, h = (x0 + x1) / 2, y1 - y0
    annotate!(p, xc, y1 - 0.22h, text(hdr, 8, FONT_FAMILY, :center, :black))
    annotate!(p, xc, y0 + 0.36h, text(body, 6, FONT_FAMILY, :center, :black))
    tag == "" || annotate!(p, x1 - 0.06, y1 - 0.07, text(tag, 6, FONT_FAMILY, :right, tagcolor))
end
arr!(p, x0, y0, x1, y1; ls=:solid) =
    plot!(p, [x0, x1], [y0, y1]; arrow=arrow(:closed, :head, 0.3, 0.2), color=:gray30, linewidth=0.9, linestyle=ls)

box!(pA, 0.05, 1.75, 0.33, 0.67, "Fit", "HMM-DDM per animal\n18 animals × 4 states", GRAYBOX)
box!(pA, 2.3, 4.3, 0.66, 0.98, "Parameter profile", "each state's v, B, τ, |bias|\nz-scored within its animal", GRAYBOX)
box!(pA, 2.3, 4.3, 0.36, 0.58, "Control profile (absolute)", "z-scored across all 72 states;\nfeeds B (last row), C", :white; ls=:dash)
box!(pA, 2.3, 4.3, 0.02, 0.28, "Accuracy label", "rank 1-4 by each state's\nown accuracy", ORANGEBOX)
box!(pA, 5.0, 7.6, 0.70, 0.98, "Shared axis  (B, D, E)", "PCA on all 72 profiles", BLUEBOX; tag="no labels", tagcolor=:steelblue)
box!(pA, 5.0, 7.6, 0.37, 0.64, "Label-free matching  (B)", "consensus: relabel every animal\nonto one shared template", BLUEBOX; tag="no labels", tagcolor=:steelblue)
box!(pA, 5.0, 7.6, 0.03, 0.31, "Transfer to a new animal  (B)", "template = mean profile per rank (other 17);\nmatch held-out animal's states one-to-one", GRAYBOX; tag="uses labels", tagcolor=RGB(0.8, 0.45, 0.1))
box!(pA, 8.2, 9.95, 0.30, 0.70, "Compare", "with each state's\naccuracy rank,\nagainst shuffled labels", ORANGEBOX)
arr!(pA, 1.75, 0.58, 2.28, 0.80)
arr!(pA, 1.75, 0.50, 2.28, 0.47)
arr!(pA, 1.75, 0.42, 2.28, 0.16)
for (y, yend) in ((0.84, 0.62), (0.50, 0.50), (0.17, 0.38))
    arr!(pA, 4.3, 0.82, 4.98, y)
    arr!(pA, 7.6, y, 8.18, yend)
end
arr!(pA, 4.3, 0.50, 4.98, 0.76; ls=:dash)
arr!(pA, 4.3, 0.12, 4.98, 0.12)
plot!(pA, [4.3, 4.6, 4.6], [0.06, 0.06, 0.0]; color=:gray30, linewidth=0.9)
plot!(pA, [4.6, 9.1], [0.0, 0.0]; color=:gray30, linewidth=0.9)
arr!(pA, 9.1, 0.0, 9.1, 0.285)

# A (lower row): worked example of each step on real animals.
# Example animal: held-out and consensus labels both match accuracy, accuracies
# well separated, HMM numbering not already in accuracy order.
ex = "Robert"
# Consensus examples: two other animals whose consensus labels match accuracy.
cons_examples = ["1064", "1054"]
hmm_state(r) = df.state[df.rat .== r]          # HMM index of each accuracy-rank row
zgrad = cgrad(:RdBu; rev=true)
zcol(v) = get(zgrad, (clamp(v, -2, 2) + 2) / 4)
short_feats = ["v", "B", "τ", "bias"]

"Draw profile rows `rows` (top to bottom) of matrix Z as a grid; returns row centre y values."
function grid!(p, Z, rows, x0, ytop; cw=0.55, rh=0.9, header=false)
    ys = Float64[]
    for (k, i) in enumerate(rows)
        y1 = ytop - (k - 1) * rh
        for j in 1:size(Z, 2)
            xa = x0 + (j - 1) * cw
            plot!(p, Shape([xa, xa + cw, xa + cw, xa], [y1 - rh, y1 - rh, y1, y1]); fillcolor=zcol(Z[i, j]), linecolor=:white, linewidth=0.5)
        end
        push!(ys, y1 - rh / 2)
    end
    if header
        for j in 1:size(Z, 2)
            annotate!(p, x0 + (j - 0.5) * cw, ytop + 0.3, text(short_feats[j], 6, FONT_FAMILY, :center, :black))
        end
    end
    return ys
end
blank(ttl) = plot(; framestyle=:none, grid=false, legend=false, title=ttl, titlelocation=:left, titlefontsize=8, xlims=(0, 10), ylims=(0, 10))
link!(p, x0, y0, x1, y1, col) = plot!(p, [x0, x1], [y0, y1]; color=col, linewidth=1.2)

# a1: ranking
sub = df[df.rat .== ex, :]
o = sortperm(sub.state)
a1 = scatter(
    1:K, 100 .* sub.acc[o];
    color=rank_colors[sub.acc_rank[o]], marker=(:circle, 9, stroke(0)), legend=false,
    title="Ai  Accuracy ranking", titlelocation=:left, titlefontsize=8,
    xlabel="HMM state (arbitrary number)", ylabel="state accuracy (%)",
    xticks=1:K, xlims=(0.5, K + 0.5), ylims=(100 * minimum(sub.acc) - 6, 100 * maximum(sub.acc) + 6),
)
for (x, k, a) in zip(1:K, sub.acc_rank[o], sub.acc[o])
    annotate!(a1, x, 100a + 2.2, text("rank $k", 7, FONT_FAMILY, :center, :black))
end

# a2: profile
a2 = blank("Aii  Parameter profile")
ys = grid!(a2, P[ex], 1:K, 3.7, 8.8; cw=0.65, rh=1.5, header=true)
for (k, y) in enumerate(ys)
    annotate!(a2, 3.6, y, text("rank $k", 7, FONT_FAMILY, :right, :black))
end

# a3: held-out matching
a3 = blank("Aiii  Held-out template matching")
Tex = template(identity_labels, filter(!=(ex), rats))
yt = grid!(a3, Tex, 1:K, 1.6, 8.8; cw=0.65, rh=1.5, header=true)
hmm_rows = sortperm(hmm_state(ex))              # rows in HMM order = no labels
yh = grid!(a3, P[ex], hmm_rows, 6.3, 8.8; cw=0.65, rh=1.5, header=true)
for k in 1:K
    annotate!(a3, 1.1, yt[k], text("rank $k", 6, FONT_FAMILY, :right, :black))
end
for (pos, i) in enumerate(hmm_rows)
    k = assigned[ex][i]
    ok = k == i
    link!(a3, 1.6 + 4 * 0.65, yt[k], 6.3, yh[pos], ok ? :black : :firebrick)
    annotate!(a3, 6.3 + 4 * 0.65 + 0.12, yh[pos], text("HMM $(hmm_state(ex)[i]): rank $i " * (ok ? "✓" : "×"), 6, FONT_FAMILY, :left, :black))
end
annotate!(a3, 2.9, 1.9, text("template\n(other 17 animals)", 6, FONT_FAMILY, :center, :black))
annotate!(a3, 7.6, 1.9, text("held-out animal", 6, FONT_FAMILY, :center, :black))

# a4: label-free consensus
a4 = blank("Aiv  Consensus matching")
cons_T = reduce(vcat, [mean([P[r][findfirst(==(k), cons_rank[r]), :] for r in rats])' for k in 1:K])
ytc = grid!(a4, cons_T, 1:K, 6.2, 8.8; cw=0.65, rh=1.5, header=true)
for k in 1:K
    annotate!(a4, 6.2 + 4 * 0.65 + 0.12, ytc[k], text("slot $k", 6, FONT_FAMILY, :left, :black))
end
slot_cols = [:gray20, :gray40, :gray60, :gray80]
for (b, r) in enumerate(cons_examples)
    rows = sortperm(hmm_state(r))
    top = 8.6 - (b - 1) * 4.0
    yr = grid!(a4, P[r], rows, 1.3, top; cw=0.55, rh=0.8, header=b == 1)
    for (pos, i) in enumerate(rows)
        link!(a4, 1.3 + 4 * 0.55, yr[pos], 6.2, ytc[cons_rank[r][i]], RGBA(0.3, 0.3, 0.3, 0.7))
    end
    annotate!(a4, 1.2, top - 1.6, text("animal $b", 6, FONT_FAMILY, :right, :black))
end
annotate!(a4, 7.5, 1.9, text("shared template", 6, FONT_FAMILY, :center, :black))

# B: every analysis on one metric. Left: median per-animal τ with a bootstrap 95%
# CI over animals, against the 95% range of medians under shuffled labels. Right:
# how many animals' orderings agree with, tie with, or oppose accuracy.
nr = length(evidence)
rowy = [nr + 1 - j for j in 1:nr]
boot_ci(τs; n=NNULL) = quantile([median(rand(τs, length(τs))) for _ in 1:n], (0.025, 0.975))
GREEN, RED = RGB(0.30, 0.60, 0.40), RGB(0.80, 0.35, 0.30)
pB = plot(;
    title="B  Rank correlation with accuracy, by method", titlelocation=:left,
    xlabel="median per-animal rank correlation with accuracy (Kendall τ)",
    yticks=false, ylims=(0.4, nr + 1.3),
    xlims=(-1.05, 1.05), xticks=-1:0.5:1, legend=false,
)
plot!(pB, Shape([-1.05, 1.05, 1.05, -1.05], [0.5, 0.5, 1.5, 1.5]); fillcolor=:gray93, linewidth=0, label="")
hline!(pB, [nr - 3.5, nr - 5.5]; color=:gray60, linewidth=0.5, label="")
vline!(pB, [0.0]; color=:black, linestyle=:dash, linewidth=0.8, label="")
pBn = plot(;
    title="animals agreeing", titlelocation=:left, titlefontsize=8,
    xlabel="animals", yticks=false, ylims=(0.4, nr + 1.3), xlims=(0, NRAT + 9),
    xticks=[0, 9, NRAT], legend=false, framestyle=:axes,
)
plot!(pBn, Shape([0, NRAT + 9, NRAT + 9, 0], [0.5, 0.5, 1.5, 1.5]); fillcolor=:gray93, linewidth=0)
hline!(pBn, [nr - 3.5, nr - 5.5]; color=:gray60, linewidth=0.5)
for (j, (name, τs, null)) in enumerate(evidence)
    y = rowy[j]
    lo, hi = quantile(null, 0.025), quantile(null, 0.975)
    plot!(pB, Shape([lo, hi, hi, lo], [y - 0.28, y - 0.28, y + 0.28, y + 0.28]); fillcolor=:gray78, linewidth=0)
    annotate!(pB, -1.02, y + 0.4, text(name, 7, FONT_FAMILY, :left, :black))
    clo, chi = boot_ci(τs)
    plot!(pB, [clo, chi], [y, y]; color=:black, linewidth=1.5)
    scatter!(pB, [median(τs)], [y]; color=:black, marker=(:circle, 5, stroke(0)))
    npos, nzero = count(>(0), τs), count(==(0), τs)
    for (x0, w, c) in ((0, npos, GREEN), (npos, nzero, :gray80), (npos + nzero, NRAT - npos - nzero, RED))
        w > 0 && plot!(pBn, Shape([x0, x0 + w, x0 + w, x0], [y - 0.28, y - 0.28, y + 0.28, y + 0.28]); fillcolor=c, linecolor=:white, linewidth=0.5)
    end
    p = (1 + count(>=(median(τs)), null)) / (NNULL + 1)
    annotate!(pBn, NRAT + 0.6, y, text(@sprintf("%d/%d\n%s", npos, NRAT, pfmt(p)), 6, FONT_FAMILY, :left, :black))
end
ky = nr + 0.95
plot!(pB, Shape([-0.55, -0.45, -0.45, -0.55], [ky - 0.15, ky - 0.15, ky + 0.15, ky + 0.15]); fillcolor=:gray78, linewidth=0)
annotate!(pB, -0.42, ky, text("shuffled labels: 95% of medians", 7, FONT_FAMILY, :left, :black))
plot!(pB, [0.25, 0.37], [ky, ky]; color=:black, linewidth=1.5)
scatter!(pB, [0.31], [ky]; color=:black, marker=(:circle, 5, stroke(0)))
annotate!(pB, 0.40, ky, text("median across animals, bootstrap 95% CI", 7, FONT_FAMILY, :left, :black))
annotate!(pBn, 0.3, nr + 0.75, text("τ > 0", 7, FONT_FAMILY, :left, GREEN))
annotate!(pBn, 6.0, nr + 0.75, text("τ = 0", 7, FONT_FAMILY, :left, :gray45))
annotate!(pBn, 11.7, nr + 0.75, text("τ < 0", 7, FONT_FAMILY, :left, RED))

# C, D: all 72 states on PC1/PC2, absolute versus within-animal parameters
acc_wz = vcat([(a = df.acc[df.rat .== r]; (a .- mean(a)) ./ std(a)) for r in rats]...)
function pca_panel(pc, ttl; leg=false)
    p = plot(;
        title=ttl, titlelocation=:left,
        xlabel=@sprintf("PC1 (%.0f%%)", 100 * pc.ve[1]), ylabel=@sprintf("PC2 (%.0f%%)", 100 * pc.ve[2]),
        legend=leg ? :topleft : false, legendtitle="accuracy rank", legendtitlefontsize=7,
        foreground_color_legend=nothing, background_color_legend=nothing,
    )
    for k in 1:K
        m = df.acc_rank .== k
        scatter!(p, pc.S[m, 1], pc.S[m, 2]; color=rank_colors[k], marker=(:circle, 3.5, stroke(0)), alpha=0.85, label=string(k))
    end
    # Accuracy axis, fitted after the PCA: within-animal z-scored accuracy
    # regressed on PC1 and PC2, drawn as a biplot arrow through the centroid.
    Xs = pc.S[:, 1:2]
    b = Xs \ acc_wz
    R2 = 1 - sum((acc_wz .- Xs * b) .^ 2) / sum(acc_wz .^ 2)
    L = 0.45 * minimum(maximum(abs.(Xs); dims=1))
    u = L .* b ./ norm(b)
    plot!(p, [-u[1], u[1]], [-u[2], u[2]]; color=:black, linewidth=1.5, arrow=arrow(:closed, :head, 0.4, 0.3), label="")
    annotate!(p, 1.12u[1], 1.12u[2] + 0.12L, text(@sprintf("accuracy\nR² = %.2f", R2), 7, FONT_FAMILY, u[1] >= 0 ? :left : :right, :black))
    return p
end
pC = pca_panel(pca_pop, "C  PCA, absolute parameters"; leg=true)
pD = pca_panel(pca_wz, "D  PCA, within-animal parameters")

# E: the label-free axis
pE = plot(;
    title="E  Within-animal PC1 vs. accuracy rank", titlelocation=:left,
    xlabel="state accuracy rank (within animal)", ylabel=@sprintf("PC1 score (%.0f%% of variance)", 100 * pca_wz.ve[1]),
    xticks=(1:K, ["1\nbest", "2", "3", "4\nworst"]), xlims=(0.7, K + 0.3), legend=false,
)
for r in rats
    m = df.rat .== r
    plot!(pE, df.acc_rank[m], pca_wz.S[m, 1]; color=:gray75, linewidth=0.7)
    scatter!(pE, df.acc_rank[m], pca_wz.S[m, 1]; color=rank_colors[df.acc_rank[m]], marker=(:circle, 3.5, stroke(0)), alpha=0.8)
end
mk = [mean(pca_wz.S[df.acc_rank .== k, 1]) for k in 1:K]
sk = [std(pca_wz.S[df.acc_rank .== k, 1]) / sqrt(NRAT) for k in 1:K]
plot!(pE, 1:K, mk; yerror=sk, color=:black, linewidth=2.5, marker=(:circle, 5, stroke(0)))

rowA = plot(pA)
rowA2 = plot(a1, a2, a3, a4; layout=(1, 4))
rowB = plot(pB, pBn; layout=grid(1, 2; widths=[0.78, 0.22]))
rowC = plot(pC, pD, pE; layout=(1, 3))
fig = plot(
    rowA, rowA2, rowB, rowC;
    layout=grid(4, 1; heights=[0.19, 0.23, 0.27, 0.31]),
    size=(1100, 1380),
    left_margin=6Plots.mm,
    bottom_margin=7Plots.mm,
    top_margin=5Plots.mm,
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
@printf(
    "pairwise matching: %.0f%% exact, %.0f%% within ±1   null %.0f%% ± %.0f%%   p = %.4f\n",
    100 * pair_exact, 100 * pair_within1, 100 * mean(pair_null), 100 * std(pair_null), p_pair
)
@printf("  per-animal Kendall τ median %+.2f, positive in %d/%d\n", median(τ_loo), count(>(0), τ_loo), NRAT)
println("\nheld-out rank recovery by scaling and matching rule")
for sc in (:pop, :wz), rule in (:near, :one)
    t = scaling_tests[(sc, rule)]
    @printf("  %-4s %-5s %.0f%% exact   null %.0f%%   p = %.4f\n", sc, rule, 100 * t.obs, 100 * mean(t.null), t.p)
end
println("\nPC1 vs accuracy (PC1 oriented by log v loading, never by accuracy)")
for (name, pc, τs) in (("within-animal z", pca_wz, τ_wz), ("population z", pca_pop, τ_pop))
    @printf(
        "  %-16s PC1 %.0f%% var   positive %d/%d   median τ %+.2f   sign test p = %.4f   loadings %s\n",
        name, 100 * pc.ve[1], count(>(0), τs), NRAT, median(τs), p_sign(τs),
        join(["$(f)=$(round(v; digits=2))" for (f, v) in zip(FEATS, pc.V[:, 1])], " ")
    )
end
println("\nConverging evidence (per-animal Kendall agreement with accuracy)")
for (name, τs, null) in evidence
    @printf("  %-36s median %+.2f  positive %2d/%d  null median %+.2f  p = %.4f\n",
        name, median(τs), count(>(0), τs), NRAT, mean(null), (1 + count(>=(median(τs)), null)) / (NNULL + 1))
end
@printf("held-out: exact %.0f%% (null %.0f%%, p = %.4f), within ±1 %.0f%% (null %.0f%%, p = %.4f)\n",
    100 * loo_exact, 100 * mean(loo_null), p_loo, 100 * loo_within1, 100 * mean(loo_null1), p_loo1)
@printf("consensus: exact %.0f%% (null %.0f%%, p = %.4f), within ±1 %.0f%% (null %.0f%%, p = %.4f)\n",
    100 * cons_exact, 100 * mean(cons_null), p_cons, 100 * cons_within1, 100 * mean(cons_null1), p_cons1)
@printf("PC1 loadings: %s (sign set by log v)\n", join(["$(f) $(round(v; digits=2))" for (f, v) in zip(FEATS, pca_wz.V[:, 1])], ", "))
@printf("k-means on within-animal profiles: sizes %s; animals spanning 1..4 clusters %s\n", join(km_sizes, "/"), join([count(==(j), km_span) for j in 1:K], "/"))
println("\nAgreement of accuracy ordering with other orderings")
for (j, (f, lab)) in enumerate(criteria)
    s = agree[agree.idx .== j, :]
    @printf(
        "  %-10s median τ %+.2f   positive %2d/%d   identical rank %2.0f%% of states\n",
        replace(lab, "\n" => " "), median(s.τ), count(>(0), s.τ), NRAT, 100 * sum(s.exact) / (K * NRAT)
    )
end
println("\nFigure written to $(joinpath(results_dir, "reviewer_state_generality"))")
