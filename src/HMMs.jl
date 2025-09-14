struct NormalInverseGamma{T}
    μ0::T    # prior mean
    κ0::T    # mean precision scaling
    α0::T    # InvGamma shape
    β0::T    # InvGamma scale
end

struct Priors{T}
    init_α::Vector{T}          # Dirichlet α for π
    trans_α::Matrix{T}         # Dirichlet α for rows of A (K×K)
    emis::Vector{NormalInverseGamma{T}}  # one per state
end

struct GaussianPriorHMM{T,D} <: AbstractHMM
    init::Vector{T}       # π
    trans::Matrix{T}      # A (rows sum to 1)
    dists::Vector{D}      # Normal(μ_k, σ_k)
    priors::Priors{T}
end

# Required accessors
HiddenMarkovModels.initialization(hmm::GaussianPriorHMM)    = hmm.init
HiddenMarkovModels.transition_matrix(hmm::GaussianPriorHMM) = hmm.trans
HiddenMarkovModels.obs_distributions(hmm::GaussianPriorHMM) = hmm.dists

# Optional: log prior over parameters (used in joint objective)
function DensityInterface.logdensityof(hmm::GaussianPriorHMM)
    π = HiddenMarkovModels.initialization(hmm)
    A = HiddenMarkovModels.transition_matrix(hmm)
    ds = HiddenMarkovModels.obs_distributions(hmm)
    pr = hmm.priors

    lp = logpdf(Dirichlet(pr.init_α), π)
    for i in 1:size(A,1)
        lp += logpdf(Dirichlet(pr.trans_α[i, :]), vec(A[i, :]))
    end
    for k in 1:length(ds)
        μ = ds[k].μ
        σ2 = ds[k].σ^2
        p  = pr.emis[k]
        lp += logpdf(InverseGamma(p.α0, p.β0), σ2)
        lp += logpdf(Normal(p.μ0, sqrt(σ2 / p.κ0)), μ)
    end
    return lp
end

# Helpers
function _weighted_mean_var(y::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    N = sum(w)
    μ = sum(@. w * y) / max(N, eps())
    s2 = N > 0 ? sum(@. w * (y - μ)^2) / N : 0.0
    return N, μ, s2
end

# MAP M-step used by Baum–Welch
function StatsAPI.fit!(
    hmm::GaussianPriorHMM,
    fb::HiddenMarkovModels.ForwardBackwardStorage,
    y::AbstractVector{<:Real};
    seq_ends,
)
    K = length(hmm)
    pr = hmm.priors

    init_counts  = zeros(eltype(hmm.init), K)
    trans_counts = zeros(eltype(hmm.trans), K, K)
    γ = fb.γ
    ξ = fb.ξ

    for s in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, s)
        init_counts .+= γ[:, t1]
        trans_counts .+= sum(ξ[t1:t2])
    end

    # π and A (Dirichlet-MAP == normalized α+counts)
    π_post = pr.init_α .+ init_counts
    hmm.init .= π_post ./ sum(π_post)
    for i in 1:K
        row_post = pr.trans_α[i, :] .+ trans_counts[i, :]
        hmm.trans[i, :] .= row_post ./ sum(row_post)
    end

    # Emissions (Normal–Inverse-Gamma posterior → joint MAP)
    for k in 1:K
        w = vec(γ[k, :])
        Nw, ybar, s2 = _weighted_mean_var(y, w)
        p = pr.emis[k]

        κn = p.κ0 + Nw
        μn = (p.κ0*p.μ0 + Nw*ybar) / max(κn, eps())
        αn = p.α0 + Nw/2
        βn = p.β0 + 0.5*(Nw*s2 + (p.κ0*Nw)*(ybar - p.μ0)^2 / max(κn, eps()))

        σ2_map = βn / (αn + 1)
        μ_map  = μn
        hmm.dists[k] = Normal(μ_map, sqrt(σ2_map))
    end

    @assert HiddenMarkovModels.valid_hmm(hmm)
    return nothing
end