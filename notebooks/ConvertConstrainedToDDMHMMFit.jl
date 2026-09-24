using Pkg
Pkg.activate("notebooks")

using DriftDiffusionModels
using HiddenMarkovModels
using BSON
using BSON: @save, @load

# Mirror the constrained-fit structs so BSON can rehydrate `Main.<Type>` tags.
struct ConstrainedDDMHMMFit
    hmm::Any
    tied::Vector{Symbol}
    logL::Float64
    logL_evolution::Vector{Float64}
    n_trials::Int
    n_free_params::Int
    bic::Float64
end

mutable struct TiedPriorHMM{T<:Real} <: HiddenMarkovModels.AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dists::Vector{DriftDiffusionModel}
    α_trans::Matrix{T}
    α_init::Vector{T}
    tied::Vector{Symbol}
end

# Stubs for the direct-gradient script's local types so BSONs that serialized
# them can be loaded.
struct DDMEmission{T<:Real}
    B::T
    v::T
    a₀::T
    τ::T
end

struct DDMHMM{T<:Real} <: HiddenMarkovModels.AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dists::Vector{DDMEmission{T}}
end

# Target struct — matches FitDDMHMMs.jl so downstream loaders see the same type.
struct DDMHMMFit
    hmm::PriorHMM
    logL::Float64
    logL_evolution::Vector{Float64}
end

# θ-space layout from DirectGradientDDMHMM.jl:
#   [ init_logits (K) | trans_logits (K*K) | ddm_y (K*nfree + |tied|) ]
const DDM_PARAMS = (:B, :v, :a₀, :τ)

softmax1(x) = (e = exp.(x .- maximum(x)); e ./ sum(e))
sigmoid_(y) = 1 / (1 + exp(-y))

function build_tied_idx_map(K::Int, tied::Vector{Symbol})
    free = Symbol[p for p in DDM_PARAMS if !(p in tied)]
    nfree = length(free)
    idx = Matrix{Int}(undef, K, length(DDM_PARAMS))
    for (pi, p) in enumerate(DDM_PARAMS)
        if p in tied
            pos = K * nfree + findfirst(==(p), tied)
            for k in 1:K
                idx[k, pi] = pos
            end
        else
            pos = findfirst(==(p), free)
            for k in 1:K
                idx[k, pi] = (k - 1) * nfree + pos
            end
        end
    end
    return idx
end

function θ_to_priorhmm(θ::Vector{Float64}, K::Int, tied::Vector{Symbol})
    n_init = K
    n_trans = K * K
    init_logits = θ[1:n_init]
    trans_logits = θ[(n_init + 1):(n_init + n_trans)]
    ddm_y = θ[(n_init + n_trans + 1):end]

    init = softmax1(init_logits)
    trans = Matrix{Float64}(undef, K, K)
    for k in 1:K
        trans[k, :] .= softmax1(trans_logits[((k - 1) * K + 1):(k * K)])
    end

    idx = build_tied_idx_map(K, tied)
    dists = Vector{DriftDiffusionModel}(undef, K)
    for k in 1:K
        B = exp(ddm_y[idx[k, 1]])
        v = exp(ddm_y[idx[k, 2]])
        a₀ = sigmoid_(ddm_y[idx[k, 3]])
        τ = exp(ddm_y[idx[k, 4]])
        dists[k] = DriftDiffusionModel(; B=B, v=v, a₀=a₀, τ=τ)
    end

    return PriorHMM(init, trans, dists, 1, 1)
end

# Convert a direct-gradient BSON (keys: best_θ, best_direct, K, TIED, RAT, ...).
function convert_direct_gradient(path::AbstractString)
    @load path best_θ best_direct K TIED RAT
    tied = Symbol[s for s in TIED]
    hmm = θ_to_priorhmm(best_θ, K, tied)
    return DDMHMMFit(hmm, best_direct, Float64[]), String(RAT), Int(K)
end

# Convert a constrained Baum–Welch BSON (keys: fit, rat, K_STATES, tied).
function convert_constrained(path::AbstractString)
    @load path fit rat K_STATES
    t = fit.hmm::TiedPriorHMM
    hmm = PriorHMM(
        copy(t.init), copy(t.trans), deepcopy(t.dists), copy(t.α_trans), copy(t.α_init)
    )
    return DDMHMMFit(hmm, fit.logL, fit.logL_evolution), String(rat), Int(K_STATES)
end

# Convert a "rich" constrained BSON (keys: hmm_init/hmm_trans/hmm_dists, best_logL,
# rat, K_STATES, ...). Emissions are stored as `DDMEmission`, not `DriftDiffusionModel`.
function convert_rich_constrained(path::AbstractString)
    @load path hmm_init hmm_trans hmm_dists best_logL rat K_STATES
    dists = [DriftDiffusionModel(; B=d.B, v=d.v, a₀=d.a₀, τ=d.τ) for d in hmm_dists]
    hmm = PriorHMM(Vector{Float64}(hmm_init), Matrix{Float64}(hmm_trans), dists, 1, 1)
    return DDMHMMFit(hmm, Float64(best_logL), Float64[]), String(rat), Int(K_STATES)
end

src_dir = joinpath("results", "final_ddmhmms", "final_ddmhmms")
out_dir = joinpath("results", "final_ddmhmms", "ddmhmmfit_compat")
isdir(out_dir) || mkpath(out_dir)

files = sort(filter(f -> endswith(f, ".bson"), readdir(src_dir; join=true)))
@assert !isempty(files) "no BSONs in $src_dir"
@info "Converting $(length(files)) BSONs → DDMHMMFit format in $out_dir"

n_direct = 0
n_constrained = 0
n_rich = 0
n_skipped = 0

for f in files
    keys_in_file = keys(BSON.load(f))
    converted_fit, rat, K = if :hmm_dists in keys_in_file
        global n_rich += 1
        convert_rich_constrained(f)
    elseif :fit in keys_in_file
        global n_constrained += 1
        convert_constrained(f)
    elseif :best_direct in keys_in_file
        global n_direct += 1
        convert_direct_gradient(f)
    else
        @warn "Unknown BSON schema, skipping" file = basename(f) keys = collect(
            keys_in_file
        )
        global n_skipped += 1
        continue
    end

    fit = converted_fit
    local out = joinpath(out_dir, replace(basename(f), ".bson" => "_compat.bson"))
    @save out fit rat K
    @info "  $(basename(f)) → $(basename(out))"
end

@info "Done. direct=$n_direct rich=$n_rich constrained=$n_constrained skipped=$n_skipped"
