### A Pluto.jl notebook ###
# v0.20.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ cafe0002-0000-4000-8000-000000000002
begin
    import Pkg
    Pkg.activate(@__DIR__)

    using BSON: @load
    using CSV
    using CairoMakie
    using DataFrames
    using Dates
    using Distributions
    using DriftDiffusionModels
    using FFTW
    using HiddenMarkovModels
    using JLD2
    using LinearAlgebra
    using Plots
    using PlutoUI
    using Printf
    using Random
    using Statistics
    nothing
end

# ╔═╡ cafe0001-0000-4000-8000-000000000001
md"""
# Model comparison: GLM-HMM vs DDM-HMM

Cross-model alignment between fitted GLM-HMM and DDM-HMM posteriors.

| § | Section |
|---|---------|
| 1 | Setup, paths, and posterior loading |
| 2 | GLM-HMM posterior sorting |
| 3 | DDM-HMM data prep & posterior sorting |
| 4 | Confusion / co-occupancy & lift |
| 5 | Lift heatmap figure |
| 6 | NMI permutation tests |
| 7 | Per-animal alignment figures |
| 8 | Fraction-of-entropy-explained |
| 9 | Posterior-only choice prediction |
| 10 | Save figures |
"""

# ╔═╡ cafe0003-0000-4000-8000-000000000003
md"""
## §1 Setup, paths, and posterior loading
"""

# ╔═╡ cafe0004-0000-4000-8000-000000000004
begin
    GLM_GAMMAS_PATH = "/Users/senne/Downloads/best_gammas2.jld2"
    GLM_PARAMS_PATH = "/Users/senne/Downloads/FINAL_fit3.csv"
    DATA_DIR        = joinpath(@__DIR__, "..", "data")
	DDMS_DIR        = joinpath(@__DIR__, "..", "ddmhmms")
    RAT_DATA_FILE   = joinpath(DATA_DIR, "processed_rat_data.csv.gz")
    nothing
end

# ╔═╡ cafe0005-0000-4000-8000-000000000005
begin
    post_file       = load(GLM_GAMMAS_PATH)
    gammas_dict     = sort(post_file["gammas_dict"])
    glmhmm_params   = CSV.read(GLM_PARAMS_PATH, DataFrame)

    unique_animals = sort(unique([
        split_string[1]
        for split_string in [split(rat_str, "_") for rat_str in glmhmm_params.rat]
    ]))

    best_run = [parse(Int, split(String(k), "_")[2][4:end]) for k in keys(gammas_dict)]
    nothing
end

# ╔═╡ cafe0006-0000-4000-8000-000000000006
best_run

# ╔═╡ cafe0007-0000-4000-8000-000000000007
gammas_dict

# ╔═╡ cafe0008-0000-4000-8000-000000000008
md"""
## §2 GLM-HMM posterior sorting
"""

# ╔═╡ cafe0009-0000-4000-8000-000000000009
"""
    parse_betas_string(s) -> Vector{Matrix{Float64}}

Parse a CSV field that looks like:
`Any[[-0.1; 0.2; 0.3;;], [1.0; 2.0;;]]`
into `Vector{Matrix{Float64}}` (each inner is n×1).
"""
function parse_betas_string(s::AbstractString)::Vector{Matrix{Float64}}
    mats = Matrix{Float64}[]
    for m in eachmatch(r"\[\s*(.*?)\s*;;\s*\]"s, s)
        payload = m.captures[1]
        toks = eachmatch(r"[-+]?(?:\d+\.\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?", payload)
        nums = [parse(Float64, t.match) for t in toks]
        push!(mats, reshape(nums, :, 1))
    end
    return mats
end

# ╔═╡ cafe000a-0000-4000-8000-00000000000a
function sort_glmmhmm_posterior(name::AbstractString, run::Int)
    gammas = gammas_dict["$(name)_run$(run)"]

    glm_params = glmhmm_params[(glmhmm_params.rat .== "$(name)_run$(run)"), :]
    betas_str  = glm_params.B[1]
    betas      = parse_betas_string(betas_str)

    # sort on beta 2 (stimulus gain), descending
    perm           = sortperm(betas; by = B -> B[2], rev = true)
    gammas_sorted  = gammas[perm, :]
    return gammas_sorted
end

# ╔═╡ cafe000b-0000-4000-8000-00000000000b
md"""
## §3 DDM-HMM data prep & posterior sorting
"""

# ╔═╡ cafe000d-0000-4000-8000-00000000000d
begin
    # BSON serialises type references as a literal module path starting at
    # :Main, so deserialisation tries to resolve `Main.DDMHMMFit`. In Pluto our
    # cells live in `Main.var"workspace#N"`, not `Main`, so a struct defined
    # here is invisible to BSON. Define it in the real `Main` module instead.
    if !isdefined(Main, :DDMHMMFit)
        Core.eval(Main, :(struct DDMHMMFit
            hmm
            logL
            logL_evolution
        end))
    end
    DDMHMMFit = Main.DDMHMMFit

    bson_files = filter(f -> occursin("K4", f), readdir(DATA_DIR; join = true))

    ddmhmm_fits = Main.DDMHMMFit[]
    for bson_file in bson_files
        @load bson_file fit
        push!(ddmhmm_fits, fit)
    end

    rat_df = CSV.read(RAT_DATA_FILE, DataFrame)
    rat_df = rat_df[rat_df.daily .== "24 hr", :]

    replace!(rat_df[!, :choose_right], 0 => -1)
    side_mapping = Dict("right" => 1, "left" => -1)
    DataFrames.transform!(rat_df,
        :correct_side => ByRow(cs -> get(side_mapping, cs, missing)) => :correct_side_numeric)

    rat_names = sort(unique(rat_df[!, "name"]))
    nothing
end

# ╔═╡ cafe000e-0000-4000-8000-00000000000e
"""
Build all_results, seq_ends, results_by_date for a given rat index in `rat_names`.
"""
function data_for_ddmhmm(rat_idx::Int)
    rat = rat_names[rat_idx]

    rat_of_interest = rat_df[rat_df.name .== rat, :]
    dates = [Date(split(dt)[1]) for dt in rat_of_interest.trial_datetime]

    unique_dates    = sort(unique(dates))
    results_by_date = Vector{Vector{DDMResult}}()

    for date in unique_dates
        day_indices = findall(dates .== date)
        if isempty(day_indices); continue; end

        day_rts       = rat_of_interest.rt[day_indices]
        day_outcomes  = rat_of_interest.choose_right[day_indices]
        day_stim_side = rat_of_interest.correct_side_numeric[day_indices]

        day_results = [DDMResult(rt, choice, stim) for (rt, choice, stim) in
                       zip(day_rts, day_outcomes, day_stim_side)]
        push!(results_by_date, day_results)
    end

    seq_ends    = cumsum([length(seq) for seq in results_by_date])
    all_results = reduce(vcat, results_by_date)
    return all_results, seq_ends, results_by_date
end

# ╔═╡ cafe000f-0000-4000-8000-00000000000f
function sort_ddmhmm_posterior(rat_idx::Int)
    hmm           = ddmhmm_fits[rat_idx].hmm
    all_results, seq_ends, _ = data_for_ddmhmm(rat_idx)
    gammas, _ll   = forward_backward(hmm, all_results; seq_ends = seq_ends)

    accs = Vector{Float64}(undef, length(hmm.dists))
    for (i, model) in enumerate(hmm.dists)
        data       = simulateDDM(model, 10_000)
        is_correct = [d.s == d.choice for d in data]
        accs[i]    = mean(is_correct)
    end

    perm          = sortperm(accs, rev = true)
    gammas_sorted = gammas[perm, :]
    return gammas_sorted
end

# ╔═╡ cafe0010-0000-4000-8000-000000000010
md"""
## §4 Confusion / co-occupancy & lift
"""

# ╔═╡ cafe0011-0000-4000-8000-000000000011
"""
    confusion_matrix(glmhmm_post, ddmhmm_post; mode=:soft, normalize=:rows, weights=nothing)

K₁×K₂ confusion / co-occupancy matrix between two HMM posteriors.

`mode`     : `:soft` expected joint occupancy or `:hard` MAP counts.
`normalize`: `nothing`, `:rows`, `:cols`, `:all`.
`weights`  : optional length-T nonnegative weights.
"""
function confusion_matrix(glmhmm_post::AbstractMatrix, ddmhmm_post::AbstractMatrix;
                          mode::Symbol = :soft,
                          normalize::Union{Nothing,Symbol} = :rows,
                          weights::Union{Nothing,AbstractVector} = nothing)

    K1, T1 = size(glmhmm_post)
    K2, T2 = size(ddmhmm_post)
    T1 == T2 || throw(ArgumentError("Time dimension mismatch: $(T1) vs $(T2)"))

    if weights !== nothing
        length(weights) == T1 || throw(ArgumentError("weights must have length T=$T1"))
        any(<(0), weights)    && throw(ArgumentError("weights must be nonnegative"))
    end

    C = if mode == :soft
        if weights === nothing
            glmhmm_post * transpose(ddmhmm_post)
        else
            (glmhmm_post .* reshape(weights, 1, :)) * transpose(ddmhmm_post)
        end

    elseif mode == :hard
        if weights === nothing
            Cint = zeros(Int, K1, K2)
            @inbounds for t in 1:T1
                i = argmax(@view glmhmm_post[:, t])
                j = argmax(@view ddmhmm_post[:, t])
                Cint[i, j] += 1
            end
            Cint
        else
            Cw = zeros(Float64, K1, K2)
            @inbounds for t in 1:T1
                i = argmax(@view glmhmm_post[:, t])
                j = argmax(@view ddmhmm_post[:, t])
                Cw[i, j] += weights[t]
            end
            Cw
        end
    else
        throw(ArgumentError("mode must be :soft or :hard, got $mode"))
    end

    if normalize === nothing
        return C
    elseif normalize == :rows
        Cn = C ./ sum(C, dims = 2)
        Cn[.!isfinite.(Cn)] .= 0
        return Cn
    elseif normalize == :cols
        Cn = C ./ sum(C, dims = 1)
        Cn[.!isfinite.(Cn)] .= 0
        return Cn
    elseif normalize == :all
        s = sum(C)
        return s == 0 ? zero.(C) : (C ./ s)
    else
        throw(ArgumentError("normalize must be nothing, :rows, :cols, or :all"))
    end
end

# ╔═╡ cafe0012-0000-4000-8000-000000000012
function lift_matrix(C::AbstractMatrix; eps::Float64 = 0.0, loglift::Bool = false)
    N = sum(C)
    if N == 0
        return zero.(float.(C))
    end
    r     = sum(C, dims = 2)
    c     = sum(C, dims = 1)
    denom = r * c

    C2     = float.(C) .+ eps
    denom2 = float.(denom) .+ eps

    L = (C2 .* N) ./ denom2
    return loglift ? log.(L) : L
end

# ╔═╡ cafe0013-0000-4000-8000-000000000013
"""
    aggregate_lifts(unique_animals, best_run; eps=1e-12, weighted=true)

Returns `(avg_lift, avg_loglift, n_used)` averaged over animals.
"""
function aggregate_lifts(unique_animals, best_run; eps::Float64 = 1e-12, weighted::Bool = true)
    sumL    = nothing
    sumLL   = nothing
    total_w = 0.0
    n_used  = 0

    for (idx, animal_id) in pairs(unique_animals)
        run_idx = best_run[idx]

        p_glm = sort_glmmhmm_posterior(animal_id, run_idx)
        p_ddm = sort_ddmhmm_posterior(idx)[:, 2:end]   # always off by one

        C = confusion_matrix(p_glm, p_ddm; mode = :soft, normalize = nothing)

        if sumL === nothing
            sumL  = zeros(Float64, size(C)...)
            sumLL = zeros(Float64, size(C)...)
        elseif size(C) != size(sumL)
            @warn "Skipping animal $animal_id: size(C)=$(size(C)) != $(size(sumL))"
            continue
        end

        L  = lift_matrix(C; eps = eps, loglift = false)
        LL = lift_matrix(C; eps = eps, loglift = true)

        w = weighted ? sum(C) : 1.0
        sumL  .+= w .* L
        sumLL .+= w .* LL
        total_w += w
        n_used  += 1
    end

    n_used == 0 && error("No animals aggregated.")

    avg_lift    = sumL  ./ total_w
    avg_loglift = sumLL ./ total_w
    return avg_lift, avg_loglift, n_used
end

# ╔═╡ cafe0014-0000-4000-8000-000000000014
begin
    avg_lift, avg_loglift, n_used =
        aggregate_lifts(String.(unique_animals), best_run; weighted = true)
    (avg_lift, n_used)
end

# ╔═╡ cafe0015-0000-4000-8000-000000000015
md"""
## §5 Lift heatmap figure
"""

# ╔═╡ cafe0016-0000-4000-8000-000000000016
function pretty_lift_heatmap(avg_lift; title = "Average lift", fontsize = 14, figsize = (950, 520))
    K1, K2 = size(avg_lift)

    fig = Figure(size = figsize, figure_padding = (25, 35, 25, 20))
    ax  = Axis(fig[1, 1];
        title = title,
        xlabel = "DDM state",
        ylabel = "GLM state",
        aspect = DataAspect(),
        xgridvisible = false, ygridvisible = false,
        topspinevisible = false, rightspinevisible = false,
    )

    hm = CairoMakie.heatmap!(ax, avg_lift')

    ax.xticks = (1:K2, ["D$(j)" for j in 1:K2])
    ax.yticks = (1:K1, ["G$(i)" for i in 1:K1])

    CairoMakie.xlims!(ax, 0.5, K2 + 0.5)
    CairoMakie.ylims!(ax, 0.5, K1 + 0.5)

    Colorbar(fig[1, 2], hm, width = 18)

    lo, hi = extrema(avg_lift)
    mid = (lo + hi) / 2

    for j in 1:K2, i in 1:K1
        v  = avg_lift[i, j]
        tc = (v < mid) ? :white : :black
        text!(ax, j, i;
            text     = @sprintf("%.2f", v),
            align    = (:center, :center),
            color    = tc,
            fontsize = fontsize,
        )
    end

    return fig
end

# ╔═╡ cafe0017-0000-4000-8000-000000000017
lift_heatmap = pretty_lift_heatmap(avg_lift; title = "Average lift (n=$n_used)")

# ╔═╡ cafe0018-0000-4000-8000-000000000018
# (Heatmap saving is handled by the §10 checkbox cell.)

# ╔═╡ cafe0019-0000-4000-8000-000000000019
md"""
## §6 NMI permutation tests
"""

# ╔═╡ cafe001a-0000-4000-8000-00000000001a
function nmi_from_C(C::AbstractMatrix; eps::Float64 = 1e-12)
    P = float.(C)
    Z = sum(P)
    Z == 0 && return 0.0
    P ./= Z

    p1 = vec(sum(P, dims = 2))
    p2 = vec(sum(P, dims = 1))

    H1 = -sum(p1 .* log.(p1 .+ eps))
    H2 = -sum(p2 .* log.(p2 .+ eps))

    denom = (p1 .+ eps) * (p2 .+ eps)'
    I     = sum(P .* log.((P .+ eps) ./ denom))

    return (H1 <= eps || H2 <= eps) ? 0.0 : I / sqrt(H1 * H2)
end

# ╔═╡ cafe001b-0000-4000-8000-00000000001b
"""
    confusion_shifted!(C, p1, p2, s)

Update `C` (K1×K2) with the circular-shift confusion at lag `s`, in-place,
without allocating a shifted posterior matrix.
"""
function confusion_shifted!(C::AbstractMatrix, p1::AbstractMatrix, p2::AbstractMatrix, s::Int)
    K1, T  = size(p1)
    K2, T2 = size(p2)
    T == T2 || throw(ArgumentError("T mismatch"))

    s = mod(s, T)
    fill!(C, 0.0)

    if s == 0
        mul!(C, p1, transpose(p2), 1.0, 0.0)
        return C
    end

    t1 = 1:(T - s)
    t2 = (T - s + 1):T

    mul!(C,
         @view(p1[:, t1]),
         transpose(@view(p2[:, (1+s):T])),
         1.0, 1.0)

    mul!(C,
         @view(p1[:, t2]),
         transpose(@view(p2[:, 1:s])),
         1.0, 1.0)

    return C
end

# ╔═╡ cafe001c-0000-4000-8000-00000000001c
"""
    perm_test_alignment_circular(p1, p2; nperm=2000, eps=1e-12, rng=...)

Per-animal alignment test using a circular-shift null distribution of NMI.
Returns NamedTuple with `S_obs, p, z, null_mean, null_std, null_q, S_perm`.
"""
function perm_test_alignment_circular(p1::AbstractMatrix, p2::AbstractMatrix;
                                      nperm::Int = 2000,
                                      eps::Float64 = 1e-12,
                                      rng::AbstractRNG = Random.default_rng(),
                                      store_perm::Bool = true)

    K1, T  = size(p1)
    K2, T2 = size(p2)
    T == T2 || throw(ArgumentError("T mismatch"))

    C = zeros(Float64, K1, K2)

    confusion_shifted!(C, p1, p2, 0)
    S_obs = nmi_from_C(C; eps = eps)

    S_perm = store_perm ? Vector{Float64}(undef, nperm) : Float64[]
    μ  = 0.0
    m2 = 0.0

    @inbounds for b in 1:nperm
        s  = rand(rng, 0:(T - 1))
        confusion_shifted!(C, p1, p2, s)
        sb = nmi_from_C(C; eps = eps)

        δ  = sb - μ
        μ += δ / b
        m2 += δ * (sb - μ)

        if store_perm; S_perm[b] = sb; end
    end

    σ = nperm > 1 ? sqrt(m2 / (nperm - 1)) : 0.0

    store_perm || error("store_perm=false disables exact p-value.")
    p  = (1 + count(>=(S_obs), S_perm)) / (nperm + 1)
    z  = (σ == 0.0) ? 0.0 : (S_obs - μ) / σ
    qs = quantile(S_perm, [0.025, 0.5, 0.975])

    return (S_obs = S_obs, p = p, z = z, null_mean = μ, null_std = σ,
            null_q = (q025 = qs[1], q50 = qs[2], q975 = qs[3]),
            S_perm = S_perm)
end

# ╔═╡ cafe001d-0000-4000-8000-00000000001d
function block_permute_cols!(idxbuf::Vector{Int}, T::Int, B::Int, rng::AbstractRNG)
    empty!(idxbuf)
    B    = max(1, B)
    nb   = cld(T, B)
    perm = randperm(rng, nb)
    for b in perm
        lo = (b - 1) * B + 1
        hi = min(b * B, T)
        append!(idxbuf, lo:hi)
    end
    return idxbuf
end

# ╔═╡ cafe001e-0000-4000-8000-00000000001e
"""
    perm_test_alignment_block(p1, p2; nperm=1000, B=50, eps=1e-12, rng=...)

Slower block-permutation alternative to circular shift.
"""
function perm_test_alignment_block(p1::AbstractMatrix, p2::AbstractMatrix;
                                   nperm::Int = 1000,
                                   B::Int = 50,
                                   eps::Float64 = 1e-12,
                                   rng::AbstractRNG = Random.default_rng())

    _K1, T = size(p1)
    _K2, T2 = size(p2)
    T == T2 || throw(ArgumentError("T mismatch"))

    Cobs   = p1 * transpose(p2)
    S_obs  = nmi_from_C(Cobs; eps = eps)

    S_perm = Vector{Float64}(undef, nperm)
    idxbuf = Int[]

    @inbounds for b in 1:nperm
        cols = block_permute_cols!(idxbuf, T, B, rng)
        p2b  = @view p2[:, cols]
        C    = p1 * transpose(p2b)
        S_perm[b] = nmi_from_C(C; eps = eps)
    end

    p  = (1 + count(>=(S_obs), S_perm)) / (nperm + 1)
    μ, σ = mean(S_perm), std(S_perm)
    z  = (σ == 0.0) ? 0.0 : (S_obs - μ) / σ
    qs = quantile(S_perm, [0.025, 0.5, 0.975])

    return (S_obs = S_obs, p = p, z = z, null_mean = μ, null_std = σ,
            null_q = (q025 = qs[1], q50 = qs[2], q975 = qs[3]),
            S_perm = S_perm)
end

# ╔═╡ cafe001f-0000-4000-8000-00000000001f
"""
    load_posterior_cache(unique_animals, best_run)

Cache GLM-HMM and DDM-HMM posteriors per animal so permutation tests can
reuse them without recomputing.
"""
function load_posterior_cache(unique_animals, best_run)
    cache = NamedTuple[]
    for (idx, animal_id) in pairs(unique_animals)
        run_idx    = best_run[idx]
        p_glm      = sort_glmmhmm_posterior(animal_id, run_idx)
        p_ddm_full = sort_ddmhmm_posterior(idx)
        p_ddm      = p_ddm_full[:, 2:end]

        if size(p_glm, 2) != size(p_ddm, 2)
            @warn "Skipping animal $animal_id: T mismatch glm=$(size(p_glm,2)) ddm=$(size(p_ddm,2))"
            continue
        end

        push!(cache, (animal_id = animal_id, idx = idx, run_idx = run_idx,
                      p_glm = Matrix{Float64}(p_glm),
                      p_ddm = Matrix{Float64}(p_ddm)))
    end
    return cache
end

# ╔═╡ cafe0020-0000-4000-8000-000000000020
function group_perm_from_peranimal(per_animal::Vector{NamedTuple})
    keep = [r for r in per_animal if haskey(r, :S_perm) && !isempty(r.S_perm)]
    n    = length(keep)
    n == 0 && error("No animals with permutation distributions.")

    nperm = length(keep[1].S_perm)
    for r in keep
        length(r.S_perm) == nperm || error("All animals must have same nperm.")
    end

    w        = [r.w for r in keep]
    wsum     = sum(w)
    mean_obs = sum(w .* [r.S_obs for r in keep]) / wsum

    null_means = Vector{Float64}(undef, nperm)
    @inbounds for b in 1:nperm
        null_means[b] = sum(w .* [r.S_perm[b] for r in keep]) / wsum
    end

    p  = (1 + count(>=(mean_obs), null_means)) / (nperm + 1)
    μ, σ = mean(null_means), std(null_means)
    z  = (σ == 0.0) ? 0.0 : (mean_obs - μ) / σ
    qs = quantile(null_means, [0.025, 0.5, 0.975])

    return (mean_obs = mean_obs, p = p, z = z,
            null_mean = μ, null_std = σ,
            null_q = (q025 = qs[1], q50 = qs[2], q975 = qs[3]),
            n_used = n)
end

# ╔═╡ cafe0021-0000-4000-8000-000000000021
"""
    run_all_alignment_tests(unique_animals, best_run; nulls=[:circular], nperm=2000, ...)

Run per-animal + group permutation tests for each chosen null. Returns
`Dict{Symbol,NamedTuple}` keyed by null with `per_animal` and `group` entries.
"""
function run_all_alignment_tests(unique_animals, best_run;
                                 nulls::Vector{Symbol} = [:circular],
                                 nperm::Int = 2000,
                                 B::Int = 50,
                                 eps::Float64 = 1e-12,
                                 weighted::Bool = true,
                                 rng_seed::Int = 1)

    rng   = MersenneTwister(rng_seed)
    cache = load_posterior_cache(unique_animals, best_run)
    isempty(cache) && error("No animals cached.")

    out = Dict{Symbol, NamedTuple}()

    for null in nulls
        per_animal = NamedTuple[]
        for r in cache
            p_glm = r.p_glm
            p_ddm = r.p_ddm

            C0 = p_glm * transpose(p_ddm)
            w  = weighted ? sum(C0) : 1.0

            test = if null == :circular
                perm_test_alignment_circular(p_glm, p_ddm; nperm = nperm, eps = eps, rng = rng)
            elseif null == :block
                perm_test_alignment_block(p_glm, p_ddm; nperm = min(nperm, 1000), B = B, eps = eps, rng = rng)
            else
                throw(ArgumentError("null must be :circular or :block"))
            end

            push!(per_animal, merge((animal_id = r.animal_id, idx = r.idx,
                                     run_idx = r.run_idx, null = null, w = w), test))
        end

        group = group_perm_from_peranimal(per_animal)
        out[null] = (per_animal = per_animal, group = group)
    end

    return out
end

# ╔═╡ cafe0022-0000-4000-8000-000000000022
res = run_all_alignment_tests(String.(unique_animals), best_run;
    nulls    = [:circular],
    nperm    = 100_000,
    eps      = 1e-12,
    weighted = true,
    rng_seed = 1)

# ╔═╡ cafe0023-0000-4000-8000-000000000023
res[:circular].group

# ╔═╡ cafe0024-0000-4000-8000-000000000024
res[:circular].per_animal[1]

# ╔═╡ cafe0025-0000-4000-8000-000000000025
let
    for i in 1:length(unique_animals)
        r = res[:circular].per_animal[i]
        println("Animal $(r.animal_id): p=$(r.p), z=$(r.z)")
    end
    g = res[:circular].group
    println("Group-level: p=$(g.p), z=$(g.z)")
end

# ╔═╡ cafe0026-0000-4000-8000-000000000026
md"""
## §7 Per-animal alignment figures
"""

# ╔═╡ cafe0027-0000-4000-8000-000000000027
"Benjamini–Hochberg FDR correction. Returns q-values in original order."
function bh_fdr(ps::AbstractVector{<:Real})
    p   = collect(float.(ps))
    n   = length(p)
    ord = sortperm(p)
    q_sorted = similar(p)

    prev = Inf
    for (rank, idx) in Iterators.reverse(enumerate(ord))
        pos = rank
        val = (n / pos) * p[idx]
        prev = min(prev, val)
        q_sorted[pos] = prev
    end

    q = similar(p)
    for (pos, idx) in enumerate(ord)
        q[idx] = min(q_sorted[pos], 1.0)
    end
    return q
end

# ╔═╡ cafe0028-0000-4000-8000-000000000028
function extract_per_animal(per_animal; sortby::Symbol = :z)
    animals = [string(r.animal_id) for r in per_animal]
    ps      = [r.p for r in per_animal]
    zs      = [r.z for r in per_animal]
    qvals   = bh_fdr(ps)

    ord = if sortby == :z
        sortperm(zs)
    elseif sortby == :p
        sortperm(ps)
    else
        1:length(per_animal)
    end

    return (animals = animals[ord], p = ps[ord], z = zs[ord], q = qvals[ord], ord = ord)
end

# ╔═╡ cafe0029-0000-4000-8000-000000000029
function group_null_means(per_animal; weighted::Bool = true)
    keep = [r for r in per_animal if haskey(r, :S_perm) && !isempty(r.S_perm)]
    n    = length(keep)
    n == 0 && error("No animals with S_perm stored.")
    nperm = length(keep[1].S_perm)
    for r in keep
        length(r.S_perm) == nperm || error("All animals must share same nperm.")
    end

    w = if weighted && haskey(keep[1], :w)
        [r.w for r in keep]
    else
        ones(Float64, n)
    end
    wsum     = sum(w)
    mean_obs = sum(w .* [r.S_obs for r in keep]) / wsum

    null_means = Vector{Float64}(undef, nperm)
    @inbounds for b in 1:nperm
        null_means[b] = sum(w .* [r.S_perm[b] for r in keep]) / wsum
    end

    p = (1 + count(>=(mean_obs), null_means)) / (nperm + 1)
    z = (std(null_means) == 0.0) ? 0.0 : (mean_obs - mean(null_means)) / std(null_means)

    return (mean_obs = mean_obs, null_means = null_means, p = p, z = z)
end

# ╔═╡ cafe002a-0000-4000-8000-00000000002a
function fig_zscores(per_animal; outfile::String = "")
    d = extract_per_animal(per_animal; sortby = :z)
    n = length(d.z)
    x = 1:n

    fig = Figure(size = (650, 520))
    ax  = Axis(fig[1, 1],
        title  = "GLM-HMM vs DDM-HMM alignment (circular-shift null): per-animal z-scores",
        xlabel = "Animals (sorted by z)",
        ylabel = "z-score",
    )

    CairoMakie.scatter!(ax, x, d.z, markersize = 10)
    CairoMakie.hlines!(ax, [1.96], linestyle = :dash)

    ax.xticks = (x, d.animals)
    ax.xticklabelrotation = pi / 3
    ax.xticklabelalign    = (:right, :center)

    if !isempty(outfile)
        save(outfile, fig, px_per_unit = 2)
    end
    return fig
end

# ╔═╡ cafe002b-0000-4000-8000-00000000002b
function fig_logp(per_animal; outfile::String = "")
    d = extract_per_animal(per_animal; sortby = :p)
    n = length(d.p)
    x = 1:n

    nperm = haskey(per_animal[1], :S_perm) ? length(per_animal[1].S_perm) : nothing
    pmin  = nperm === nothing ? minimum(d.p) : 1 / (nperm + 1)
    y     = -log10.(max.(d.p, pmin))

    fig = Figure(size = (1100, 520))
    ax  = Axis(fig[1, 1],
        title  = "Permutation p-values by animal (sorted by p)",
        xlabel = "Animals (sorted by p)",
        ylabel = "-log10(p)",
    )

    CairoMakie.barplot!(ax, x, y)
    hlines!(ax, [-log10(0.05)], linestyle = :dash)

    ax.xticks = (x, d.animals)
    ax.xticklabelrotation = pi / 3
    ax.xticklabelalign    = (:right, :center)

    if nperm !== nothing
        CairoMakie.text!(ax, n, y[end] + 0.3,
            text  = "floor p≈$(round(pmin, sigdigits=2))",
            align = (:right, :bottom),
            fontsize = 11)
    end

    if !isempty(outfile)
        save(outfile, fig, px_per_unit = 2)
    end
    return fig
end

# ╔═╡ cafe002c-0000-4000-8000-00000000002c
function fig_group_null(per_animal; outfile::String = "", weighted::Bool = true)
    g     = group_null_means(per_animal; weighted = weighted)
    nulls = g.null_means
    obs   = g.mean_obs

    fig = Figure(size = (900, 520))
    ax  = Axis(fig[1, 1],
        title  = "Group-level alignment: mean NMI vs circular-shift null",
        xlabel = "Mean NMI (across animals)",
        ylabel = "Count",
    )

    h = CairoMakie.hist!(ax, nulls, bins = 50)
    CairoMakie.vlines!(ax, [obs], linewidth = 3)

    ymax       = maximum(h[1][])
    xmin, xmax = extrema(nulls)

    txt = "obs=$(round(obs, sigdigits=4))\n" *
          "perm p=$(g.p < 1e-4 ? "<1e-4" : string(round(g.p, sigdigits=3)))\n" *
          "z=$(round(g.z, sigdigits=3))"

    text!(ax,
          xmin + 0.02 * (xmax - xmin),
          0.95 * ymax,
          text = txt, align = (:left, :top), fontsize = 12)

    if !isempty(outfile)
        save(outfile, fig, px_per_unit = 2)
    end
    return fig
end

# ╔═╡ cafe002d-0000-4000-8000-00000000002d
function fig_example_null(per_animal, animal_name::AbstractString;
                          outfile::String = "")
    idx = findfirst(r -> string(r.animal_id) == animal_name, per_animal)
    idx === nothing && error("Animal '$animal_name' not found.")

    r = per_animal[idx]
    haskey(r, :S_perm) || error("No S_perm stored for this animal.")
    nulls = r.S_perm
    obs   = r.S_obs

    fig = Figure(size = (900, 520))
    ax  = Axis(fig[1, 1],
        title  = "Example animal: $animal_name (NMI vs circular-shift null)",
        xlabel = "NMI",
        ylabel = "Count",
    )

    h = CairoMakie.hist!(ax, nulls, bins = 50)
    CairoMakie.vlines!(ax, [obs], linewidth = 3)

    ymax       = maximum(h[1][])
    xmin, xmax = extrema(nulls)

    txt = "obs=$(round(obs, sigdigits=4))\n" *
          "perm p=$(r.p < 1e-4 ? "<1e-4" : string(round(r.p, sigdigits=3)))\n" *
          "z=$(round(r.z, sigdigits=3))"

    text!(ax,
          xmin + 0.02 * (xmax - xmin),
          0.95 * ymax,
          text = txt, align = (:left, :top), fontsize = 12)

    if !isempty(outfile)
        save(outfile, fig, px_per_unit = 2)
    end
    return fig
end

# ╔═╡ cafe002e-0000-4000-8000-00000000002e
zscores_fig = fig_zscores(res[:circular].per_animal)

# ╔═╡ cafe002f-0000-4000-8000-00000000002f
logp_fig = fig_logp(res[:circular].per_animal)

# ╔═╡ cafe0030-0000-4000-8000-000000000030
group_null_fig = fig_group_null(res[:circular].per_animal; weighted = true)

# ╔═╡ cafe0031-0000-4000-8000-000000000031
example_draco_fig = fig_example_null(res[:circular].per_animal, "Draco")

# ╔═╡ cafe0032-0000-4000-8000-000000000032
example_1065_fig = fig_example_null(res[:circular].per_animal, "1065")

# ╔═╡ cafe0033-0000-4000-8000-000000000033
md"""
## §8 Fraction-of-entropy explained
"""

# ╔═╡ cafe0034-0000-4000-8000-000000000034
function mi_and_entropies_from_C(C; eps = 1e-12)
    P = float.(C)
    Z = sum(P); Z == 0 && return (I = 0.0, H1 = 0.0, H2 = 0.0)
    P ./= Z
    p1 = vec(sum(P, dims = 2))
    p2 = vec(sum(P, dims = 1))
    H1 = -sum(p1 .* log.(p1 .+ eps))
    H2 = -sum(p2 .* log.(p2 .+ eps))
    denom = (p1 .+ eps) * (p2 .+ eps)'
    I = sum(P .* log.((P .+ eps) ./ denom))
    return (I = I, H1 = H1, H2 = H2)
end

# ╔═╡ cafe0035-0000-4000-8000-000000000035
function frac_explained_from_C(C; eps = 1e-12)
    m = mi_and_entropies_from_C(C; eps = eps)
    return (m.H2 <= eps) ? 0.0 : m.I / m.H2
end

# ╔═╡ cafe0036-0000-4000-8000-000000000036
function frac_explained_reverse_from_C(C; eps = 1e-12)
    m = mi_and_entropies_from_C(C; eps = eps)
    return (m.H1 <= eps) ? 0.0 : m.I / m.H1
end

# ╔═╡ cafe0037-0000-4000-8000-000000000037
function aggregate_frac_explained(unique_animals, best_run; eps = 1e-12, weighted = true)
    vals12 = Float64[]
    vals21 = Float64[]
    wts    = Float64[]

    for (idx, animal_id) in pairs(unique_animals)
        run_idx = best_run[idx]
        p_glm = sort_glmmhmm_posterior(animal_id, run_idx)
        p_ddm = sort_ddmhmm_posterior(idx)[:, 2:end]

        size(p_glm, 2) == size(p_ddm, 2) || begin
            @warn "Skipping animal $animal_id: T mismatch"
            continue
        end

        C = confusion_matrix(p_glm, p_ddm; mode = :soft, normalize = nothing)
        push!(vals12, frac_explained_from_C(C; eps = eps))
        push!(vals21, frac_explained_reverse_from_C(C; eps = eps))
        push!(wts,    weighted ? sum(C) : 1.0)
    end

    isempty(vals12) && error("No animals used")

    wsum   = sum(wts)
    mean12 = sum(wts .* vals12) / wsum
    mean21 = sum(wts .* vals21) / wsum

    return (mean_glm_to_ddm = mean12, mean_ddm_to_glm = mean21,
            per_animal_glm_to_ddm = vals12, per_animal_ddm_to_glm = vals21,
            weights = wts, n_used = length(vals12))
end

# ╔═╡ cafe0038-0000-4000-8000-000000000038
frac_res = aggregate_frac_explained(String.(unique_animals), best_run; weighted = true)

# ╔═╡ cafe0039-0000-4000-8000-000000000039
let
    println("GLM→DDM explained entropy: $(round(100*frac_res.mean_glm_to_ddm, digits=2))%")
    println("DDM→GLM explained entropy: $(round(100*frac_res.mean_ddm_to_glm, digits=2))%")
end

# ╔═╡ cafe003a-0000-4000-8000-00000000003a
md"""
## §9 Posterior-only choice prediction
"""

# ╔═╡ cafe003b-0000-4000-8000-00000000003b
function fit_state_bernoulli_rates(post::AbstractMatrix, y::AbstractVector{<:Real}; eps = 1e-12)
    K, T = size(post); @assert length(y) == T
    γ    = float.(post)
    yy   = float.(y)
    denom = vec(sum(γ, dims = 2)) .+ eps
    π    = (γ * yy) ./ denom
    return clamp.(π, eps, 1 - eps)
end

# ╔═╡ cafe003c-0000-4000-8000-00000000003c
predict_from_state_rates(post::AbstractMatrix, π::AbstractVector) = vec(transpose(π) * post)

# ╔═╡ cafe003d-0000-4000-8000-00000000003d
function logloss(y::AbstractVector{<:Real}, p::AbstractVector{<:Real}; eps = 1e-12)
    @assert length(y) == length(p)
    pp = clamp.(float.(p), eps, 1 - eps)
    yy = float.(y)
    return -mean(yy .* log.(pp) .+ (1 .- yy) .* log.(1 .- pp))
end

# ╔═╡ cafe003e-0000-4000-8000-00000000003e
function brier(y::AbstractVector{<:Real}, p::AbstractVector{<:Real})
    @assert length(y) == length(p)
    yy = float.(y); pp = float.(p)
    return mean((pp .- yy) .^ 2)
end

# ╔═╡ cafe003f-0000-4000-8000-00000000003f
function blocked_folds(T::Int, nb::Int)
    edges = round.(Int, range(1, T + 1; length = nb + 1))
    return [edges[i]:(edges[i + 1] - 1) for i in 1:nb]
end

# ╔═╡ cafe0040-0000-4000-8000-000000000040
"""
    posterior_choice_scores(post, y; nb=5)

In-sample and blocked pseudo-CV log-loss / Brier from a posterior-only
state-conditional Bernoulli predictor.
"""
function posterior_choice_scores(post::AbstractMatrix, y::AbstractVector{<:Real}; nb::Int = 5, eps = 1e-12)
    _K, T = size(post); @assert length(y) == T

    π_full = fit_state_bernoulli_rates(post, y; eps = eps)
    p_full = predict_from_state_rates(post, π_full)
    ll_in  = logloss(y, p_full; eps = eps)
    br_in  = brier(y, p_full)

    folds = blocked_folds(T, nb)
    ll_cv = Float64[]
    br_cv = Float64[]
    for test in folds
        train = collect(1:T)
        deleteat!(train, collect(test))
        π = fit_state_bernoulli_rates(@view(post[:, train]), @view(y[train]); eps = eps)
        p̂ = predict_from_state_rates(@view(post[:, test]), π)
        push!(ll_cv, logloss(@view(y[test]), p̂; eps = eps))
        push!(br_cv, brier(@view(y[test]), p̂))
    end

    return (ll_in = ll_in, br_in = br_in,
            ll_cv = mean(ll_cv), br_cv = mean(br_cv),
            fold_ll = ll_cv, fold_br = br_cv)
end

# ╔═╡ cafe0041-0000-4000-8000-000000000041
function get_choices_aligned(idx)
    all_results, _, _ = data_for_ddmhmm(idx)
    y = [r.choice == r.s for r in all_results]
    return y[2:end]
end

# ╔═╡ cafe0042-0000-4000-8000-000000000042
function compare_models_posterior_choice(unique_animals, best_run; nb::Int = 5, eps = 1e-12)
    rows = NamedTuple[]
    for (idx, animal_id) in pairs(unique_animals)
        run_idx = best_run[idx]

        y      = get_choices_aligned(idx)
        p_glm  = sort_glmmhmm_posterior(String(animal_id), run_idx)
        p_ddm  = sort_ddmhmm_posterior(idx)[:, 2:end]

        T = length(y)
        size(p_glm, 2) == T || (@warn "Skip $animal_id: glm T=$(size(p_glm,2)) y T=$T"; continue)
        size(p_ddm, 2) == T || (@warn "Skip $animal_id: ddm T=$(size(p_ddm,2)) y T=$T"; continue)

        s_glm = posterior_choice_scores(p_glm, y; nb = nb, eps = eps)
        s_ddm = posterior_choice_scores(p_ddm, y; nb = nb, eps = eps)

        push!(rows, (
            animal     = string(animal_id),
            T          = T,
            glm_ll_in  = s_glm.ll_in, ddm_ll_in = s_ddm.ll_in,
            glm_ll_cv  = s_glm.ll_cv, ddm_ll_cv = s_ddm.ll_cv,
            glm_br_in  = s_glm.br_in, ddm_br_in = s_ddm.br_in,
            glm_br_cv  = s_glm.br_cv, ddm_br_cv = s_ddm.br_cv,
        ))
    end
    return rows
end

# ╔═╡ cafe0043-0000-4000-8000-000000000043
function plot_choice_comparison(rows; use_cv::Bool = true,
                                outfile_prefix::String = "")
    animals = [r.animal for r in rows]

    glm = use_cv ? [r.glm_ll_cv for r in rows] : [r.glm_ll_in for r in rows]
    ddm = use_cv ? [r.ddm_ll_cv for r in rows] : [r.ddm_ll_in for r in rows]

    lo = min(minimum(glm), minimum(ddm))
    hi = max(maximum(glm), maximum(ddm))

    p1 = Plots.scatter(glm, ddm;
        fontfamily = "helvetica",
        xlabel = "GLM-HMM log loss",
        ylabel = "DDM-HMM log loss",
        title  = use_cv ? "Posterior-only blocked score (lower is better)" :
                          "Posterior-only in-sample score (lower is better)",
        legend = false,
        markersize = 6,
        framestyle = :box,
        size = (650, 650),
    )

    Plots.plot!(p1, [lo, hi], [lo, hi];
        linestyle = :dash, linewidth = 2, label = false)

    if !isempty(outfile_prefix)
        Plots.savefig(p1, "$(outfile_prefix)_scatter.svg")
    end

    Δ    = ddm .- glm
    ord  = sortperm(Δ)
    Δs   = Δ[ord]
    labs = animals[ord]
    x    = 1:length(Δs)

    μ   = mean(Δ)
    sem = std(Δ) / sqrt(length(Δ))

    p2 = Plots.scatter(x, Δs;
        fontfamily = "helvetica",
        xlabel = "Animals (sorted)",
        ylabel = "Δ log loss (DDM − GLM)",
        title  = use_cv ? "Δ log loss (DDM − GLM), blocked posterior-only" :
                          "Δ log loss (DDM − GLM), in-sample posterior-only",
        legend = false,
        markersize = 5,
        framestyle = :box,
        size = (700, 650),
        xticks = (x, labs),
    )

    Plots.hline!(p2, [0.0]; linestyle = :dash, linewidth = 2, label = false)

    ytop = maximum(Δs)
    ybot = minimum(Δs)
    y_annot = ytop - 0.05 * (ytop - ybot)
    Plots.annotate!(p2, 1, y_annot,
        Plots.text("mean Δ = $(round(μ, sigdigits=3)) ± $(round(sem, sigdigits=2)) (SEM)", 10, :left))

    if !isempty(outfile_prefix)
        Plots.savefig(p2, "$(outfile_prefix)_delta.svg")
    end

    return p1, p2
end

# ╔═╡ cafe0044-0000-4000-8000-000000000044
choice_rows = compare_models_posterior_choice(unique_animals, best_run; nb = 5)

# ╔═╡ cafe0045-0000-4000-8000-000000000045
choice_cv_plots = plot_choice_comparison(choice_rows; use_cv = true)

# ╔═╡ cafe0046-0000-4000-8000-000000000046
choice_insample_plots = plot_choice_comparison(choice_rows; use_cv = false)

# ╔═╡ cafe0047-0000-4000-8000-000000000047
md"""
## §10 Save figures
"""

# ╔═╡ cafe0048-0000-4000-8000-000000000048
md"""
Save all figures to `../results/`: $(@bind save_figs CheckBox(default=false))
"""

# ╔═╡ cafe0049-0000-4000-8000-000000000049
begin
    if save_figs
        results_dir = joinpath(@__DIR__, "..", "results")
        mkpath(results_dir)

        # CairoMakie figures
        makie_figs = [
            (lift_heatmap,      "model_comparison_lift_heatmap.eps"),
            (zscores_fig,       "zscores_by_animal.eps"),
            (logp_fig,          "logp_by_animal.eps"),
            (group_null_fig,    "group_null_hist.eps"),
            (example_draco_fig, "null_Draco.eps"),
            (example_1065_fig,  "null_1065.eps"),
        ]
        for (fig, name) in makie_figs
            save(joinpath(results_dir, name), fig, px_per_unit = 2)
        end

        # Plots.jl figure tuples (scatter, delta)
        plots_pairs = [
            (choice_cv_plots,       "choice_cv"),
            (choice_insample_plots, "choice_insample"),
        ]
        for ((p_scatter, p_delta), prefix) in plots_pairs
            Plots.savefig(p_scatter, joinpath(results_dir, "$(prefix)_scatter.svg"))
            Plots.savefig(p_delta,   joinpath(results_dir, "$(prefix)_delta.svg"))
        end

        n_saved = length(makie_figs) + 2 * length(plots_pairs)
        md"Saved $n_saved figures to `$results_dir`."
    else
        md"_Tick the box above to save all figures._"
    end
end

# ╔═╡ Cell order:
# ╟─cafe0001-0000-4000-8000-000000000001
# ╠═cafe0002-0000-4000-8000-000000000002
# ╟─cafe0003-0000-4000-8000-000000000003
# ╠═cafe0004-0000-4000-8000-000000000004
# ╠═cafe0005-0000-4000-8000-000000000005
# ╠═cafe0006-0000-4000-8000-000000000006
# ╠═cafe0007-0000-4000-8000-000000000007
# ╟─cafe0008-0000-4000-8000-000000000008
# ╠═cafe0009-0000-4000-8000-000000000009
# ╠═cafe000a-0000-4000-8000-00000000000a
# ╟─cafe000b-0000-4000-8000-00000000000b
# ╠═cafe000d-0000-4000-8000-00000000000d
# ╠═cafe000e-0000-4000-8000-00000000000e
# ╠═cafe000f-0000-4000-8000-00000000000f
# ╟─cafe0010-0000-4000-8000-000000000010
# ╠═cafe0011-0000-4000-8000-000000000011
# ╠═cafe0012-0000-4000-8000-000000000012
# ╠═cafe0013-0000-4000-8000-000000000013
# ╠═cafe0014-0000-4000-8000-000000000014
# ╟─cafe0015-0000-4000-8000-000000000015
# ╠═cafe0016-0000-4000-8000-000000000016
# ╠═cafe0017-0000-4000-8000-000000000017
# ╠═cafe0018-0000-4000-8000-000000000018
# ╟─cafe0019-0000-4000-8000-000000000019
# ╠═cafe001a-0000-4000-8000-00000000001a
# ╠═cafe001b-0000-4000-8000-00000000001b
# ╠═cafe001c-0000-4000-8000-00000000001c
# ╠═cafe001d-0000-4000-8000-00000000001d
# ╠═cafe001e-0000-4000-8000-00000000001e
# ╠═cafe001f-0000-4000-8000-00000000001f
# ╠═cafe0020-0000-4000-8000-000000000020
# ╠═cafe0021-0000-4000-8000-000000000021
# ╠═cafe0022-0000-4000-8000-000000000022
# ╠═cafe0023-0000-4000-8000-000000000023
# ╠═cafe0024-0000-4000-8000-000000000024
# ╠═cafe0025-0000-4000-8000-000000000025
# ╟─cafe0026-0000-4000-8000-000000000026
# ╠═cafe0027-0000-4000-8000-000000000027
# ╠═cafe0028-0000-4000-8000-000000000028
# ╠═cafe0029-0000-4000-8000-000000000029
# ╠═cafe002a-0000-4000-8000-00000000002a
# ╠═cafe002b-0000-4000-8000-00000000002b
# ╠═cafe002c-0000-4000-8000-00000000002c
# ╠═cafe002d-0000-4000-8000-00000000002d
# ╠═cafe002e-0000-4000-8000-00000000002e
# ╠═cafe002f-0000-4000-8000-00000000002f
# ╠═cafe0030-0000-4000-8000-000000000030
# ╠═cafe0031-0000-4000-8000-000000000031
# ╠═cafe0032-0000-4000-8000-000000000032
# ╟─cafe0033-0000-4000-8000-000000000033
# ╠═cafe0034-0000-4000-8000-000000000034
# ╠═cafe0035-0000-4000-8000-000000000035
# ╠═cafe0036-0000-4000-8000-000000000036
# ╠═cafe0037-0000-4000-8000-000000000037
# ╠═cafe0038-0000-4000-8000-000000000038
# ╠═cafe0039-0000-4000-8000-000000000039
# ╟─cafe003a-0000-4000-8000-00000000003a
# ╠═cafe003b-0000-4000-8000-00000000003b
# ╠═cafe003c-0000-4000-8000-00000000003c
# ╠═cafe003d-0000-4000-8000-00000000003d
# ╠═cafe003e-0000-4000-8000-00000000003e
# ╠═cafe003f-0000-4000-8000-00000000003f
# ╠═cafe0040-0000-4000-8000-000000000040
# ╠═cafe0041-0000-4000-8000-000000000041
# ╠═cafe0042-0000-4000-8000-000000000042
# ╠═cafe0043-0000-4000-8000-000000000043
# ╠═cafe0044-0000-4000-8000-000000000044
# ╠═cafe0045-0000-4000-8000-000000000045
# ╠═cafe0046-0000-4000-8000-000000000046
# ╟─cafe0047-0000-4000-8000-000000000047
# ╟─cafe0048-0000-4000-8000-000000000048
# ╠═cafe0049-0000-4000-8000-000000000049
