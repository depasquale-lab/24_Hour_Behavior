@kwdef struct BehaviorTrial{I<:AbstactInt, F<:AbstractFloat}
    ΔFlashes::I # Right Flashes - Left Flashes
    ChooseR::I # 1 -> Chose Right
    Correct::I # 1 -> Correct
    RT::F # # Reaction time
end

struct GLMObs{Tx<:AbstractVector}
    x::Tx # Covariate Vector
    y::Int # 0/1
end

struct BernoulliGLM{Tβ<:AbstractVector, F<:AbstractFloat} <: Distributions.Distribution{Univariate, Discrete}
    β₀::F
    β::Tβ
end

Distributions.logpdf(d::GLMBernoulli, o::GLMObs) = begin
    @assert o.y == 0 || o.y == 1
    p = logistic(d.β0 + dot(d.β, o.x))
    o.y == 1 ? log(clamp(p, 1e-12, 1.0)) : log1p(-clamp(p, 0.0, 1-1e-12))
end