
@model function glmhmm(data::Vector{Vector{BehaviorTrial}}, K::Int)
    N = length(data) # presumes N independent sequences of trials

    # Priors/Hyperpriors
    β₁ ~ ordered(filldist(Normal(0, 5), K))
    β₀ ~ ordered(filldist(Normal(0, 5), K))

    α ~ Exponential(1.0)

    A = Vector{Vector}(undef, K)
    for i in 1:K
        A[i] ~ Dirichlet(ones(K))
        A[i, i] += α
    end

    emissions = [BernoulliGLM(β0[k], β[k]) for k in 1:K]

    hmm = HMM(π₀, A, emissions)
    
end