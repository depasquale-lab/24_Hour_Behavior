export GLMObs, BernoulliGLM, glmhmm, simple_ar_model

###############
#GLM-HMM model#
###############
struct GLMObs{Tx<:AbstractVector}
    x::Tx # Covariate Vector
    y::Int # 0/1
end

struct BernoulliGLM{Tβ<:AbstractVector, F<:AbstractFloat} <: Distributions.Distribution{Univariate, Discrete}
    β₀::F
    β::Tβ
end

Distributions.logpdf(d::BernoulliGLM, o::GLMObs) = begin
    @assert o.y == 0 || o.y == 1
    p = logistic(d.β₀ + dot(d.β, o.x))
    o.y == 1 ? log(clamp(p, 1e-12, 1.0)) : log1p(-clamp(p, 0.0, 1-1e-12))
end

@model function glmhmm(data::Vector{Vector{GLMObs}}, K::Int)
    # Priors/Hyperpriors
    β₁ ~ ordered(filldist(Normal(0, 5), K))
    β₀ ~ ordered(filldist(Normal(0, 5), K))

    # Stickyness parameter
    α ~ Exponential(1.0)

    A = Vector{Vector}(undef, K)
    for i in 1:K
        conc = ones(K)
        conc[i] += α
        A[i, :] ~ Dirichlet(conc)
    end
    
    emissions = [BernoulliGLM(β0[k], β[k]) for k in 1:K]

    hmm = HMM(π₀, A, emissions)
    # add sequence log-likelihoods
    for s in eachindex(data)
        @addlogprob! logdensityof(hmm, data[s])
    end
end

@model function simple_ar_model(data::AbstractVector{<:BehaviorTrial}, p::Int)
    n = length(data)
    logRT = [log(d.RT) for d in data]

    # Priors
    β  ~ MvNormal(zeros(p+1), 5.0I)     # intercept + p AR coeffs
    σ2 ~ InverseGamma(2.0, 2.0)         # variance
    σ   = sqrt(σ2)

    # Condition on first p values as given; start likelihood at t = p+1
    for t in (p+1):n
        # lag vector [RT[t-1], RT[t-2], ..., RT[t-p]]
        lags = logRT[(t-1):-1:(t-p)]
        μ = β[1] + dot(β[2:end], lags)
        logRT[t] ~ Normal(μ, σ)
    end
end

