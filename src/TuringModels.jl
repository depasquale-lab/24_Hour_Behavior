export GLMObs, BernoulliGLM, glmhmm, simple_ar_model

# GLM-HMM model
struct GLMObs{Tx<:AbstractVector}
    x::Tx          # augmented covariate vector (includes intercept at index 1)
    y::Int         # 0/1
end

# Convenience to augment x with intercept (use when preparing data)
augment_with_intercept(x::AbstractVector) = vcat(one(eltype(x)), x)

# A Bernoulli GLM whose linear predictor is dot(β, x_aug), where β includes intercept
struct BernoulliGLM{Tβ<:AbstractVector} <: Distributions.Distribution{Univariate,Discrete}
    β::Tβ          # includes intercept as β[1]
end

# Log-likelihood, AD-safe with clamping
function Distributions.logpdf(d::BernoulliGLM, o::GLMObs)
    @assert o.y == 0 || o.y == 1
    η = dot(d.β, o.x)               # linear predictor (intercept included)
    p = logistic(η)
    p_safe = clamp(p, eps(typeof(p)), one(typeof(p)) - eps(typeof(p)))
    return o.y == 1 ? log(p_safe) : log1p(-p_safe)
end

# Minimal interface bits HiddenMarkovModels/Distributions expect
Base.eltype(::Type{BernoulliGLM{Tβ}}) where {Tβ} = Int
Base.length(::BernoulliGLM) = 1
Distributions.support(::BernoulliGLM) = Distributions.RealInterval(0, 1)

"""
    glmhmm(data, K)

`data` is a `Vector{Vector{GLMObs}}`, where each `GLMObs.x` is ALREADY augmented
with a leading 1 (intercept). Number of coefficients per state will be `P1 = length(data[1][1].x)`.

Builds a K-state HMM with Bernoulli-GLM emissions (same β dimension in every state).
"""
@model function glmhmm(data::Vector{Vector{GLMObs}}, K::Int)
    # infer dimensionality (already includes intercept)
    P1 = length(data[1][1].x)

    # Coefficients per state: a matrix of size (P1, K), column k is β_k
    β = Matrix{Float64}(undef, P1, K)
    for k in 1:K
        β[:, k] ~ filldist(Normal(0, 5), P1)
    end

    # Initial state distribution
    π₀ ~ Dirichlet(fill(1.0, K))

    # Transition matrix with "sticky" diagonal via α
    α ~ Exponential(1.0)
    Trows = Vector{Vector}(undef, K)
    for k in 1:K
        a = ones(K) .+ zero(α)  # keep AD-friendly element type
        a[k] += α
        Trows[k] ~ Dirichlet(a)
    end
    A = copy(hcat(Trows...)')

    # Build emissions from columns of β
    emissions = [BernoulliGLM(view(β, :, k)) for k in 1:K]

    # HMM and likelihood (parallelized across sequences)
    hmm = HMM(π₀, A, emissions)
    lls = Vector{typeof(zero(α))}(undef, length(data))
    Threads.@threads for s in eachindex(data)
        lls[s] = HiddenMarkovModels.logdensityof(hmm, data[s])
    end
    @addlogprob! sum(lls)
end

function lag_matrix(logRT::AbstractVector{<:Real}, p::Int)
    n = length(logRT)
    @assert n > p "need at least p+1 observations"
    Tobs = n - p

    X = Matrix{Float64}(undef, Tobs, p + 1)
    X[:, 1] .= 1.0
    @inbounds for t in 1:Tobs
        @inbounds for k in 1:p
            X[t, k + 1] = logRT[p + t - k]   # lag k
        end
    end

    y = Vector{Float64}(undef, Tobs)
    @inbounds for t in 1:Tobs
        y[t] = logRT[p + t]
    end
    return X, y
end

function simple_ar_model(data::AbstractVector{<:BehaviorTrial}, p::Int)
    logRT = map(d -> log(d.RT), data)
    X, y = lag_matrix(logRT, p)

    @model function ar_regression(y, X)
        p1 = size(X, 2)
        β ~ MvNormal(zeros(p1), 5.0I)  # intercept + p AR coeffs
        σ2 ~ InverseGamma(2.0, 2.0)     # variance
        σ = sqrt(σ2)

        μ = X * β
        @inbounds for i in eachindex(y)
            y[i] ~ Normal(μ[i], σ)
        end
    end

    return ar_regression(y, X)
end
