export GLMObs, BernoulliGLM, glmhmm, simple_ar_model

###############
#GLM-HMM model#
###############
struct GLMObs{Tx<:AbstractVector}
    x::Tx # Covariate Vector
    y::Int # 0/1
end

struct BernoulliGLM{Tβ<:AbstractVector, F<:Real} <: Distributions.Distribution{Univariate, Discrete}
    β₀::F
    β::Tβ
end

Distributions.logpdf(d::BernoulliGLM, o::GLMObs) = begin
    @assert o.y == 0 || o.y == 1
    p = logistic(d.β₀ + dot(d.β, o.x))
    # Use more robust clamping that works with Dual numbers
    p_safe = clamp(p, eps(typeof(p)), one(typeof(p)) - eps(typeof(p)))
    return o.y == 1 ? log(p_safe) : log1p(-p_safe)
end

# Remove the Base.zero line completely and add these:
Base.eltype(::Type{BernoulliGLM{Tβ, F}}) where {Tβ, F} = Int
Base.length(d::BernoulliGLM) = 1
Distributions.support(d::BernoulliGLM) = Distributions.RealInterval(0, 1)

@model function glmhmm(data::Vector{Vector{GLMObs}}, K::Int)
    # Just intercept and one slope per state
    β₀ ~ filldist(Normal(0, 5), K)
    β₁ ~ filldist(Normal(0, 5), K)
   
    # Initial state distribution
    π₀ ~ Dirichlet(fill(1.0, K))
   
    # Transition matrix - sample each row individually
    α ~ Exponential(1.0)
    Trows = Vector{Vector}(undef, K)
    for k in 1:K
        a = ones(K) .+ zero(α)   # promote to Vector{Dual or Float64} depending on α
        a[k] += α
        Trows[k] ~ Dirichlet(a)
    end
    A = copy(hcat(Trows...)')    
   
    # Emissions per state
    emissions = [BernoulliGLM(β₀[k], [β₁[k]]) for k in 1:K]
   
    # HMM and likelihood - parallelize the computation
    hmm = HMM(π₀, A, emissions)
    
    # Compute likelihoods in parallel, then sum
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
    X, y  = lag_matrix(logRT, p)

    @model function ar_regression(y, X)
        p1 = size(X, 2)
        β  ~ MvNormal(zeros(p1), 5.0I)  # intercept + p AR coeffs
        σ2 ~ InverseGamma(2.0, 2.0)     # variance
        σ   = sqrt(σ2)

        μ = X * β
        @inbounds for i in eachindex(y)
            y[i] ~ Normal(μ[i], σ)
        end
    end

    return ar_regression(y, X)
end