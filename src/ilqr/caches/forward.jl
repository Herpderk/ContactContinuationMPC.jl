mutable struct ForwardCache{T<:AbstractFloat}
    Xprev::Vector{Vector{T}}
    Uprev::Vector{Vector{T}}
    X::Vector{Vector{T}}
    U::Vector{Vector{T}}
    Jprev::T
    ΔJ::T
    α::T
    α_mul::T
    β::T

    function ForwardCache{T}(nx::Integer, nu::Integer, N::Integer) where {T}
        Xprev = [zeros(T, nx) for k in 1:N]
        Uprev = [zeros(T, nu) for k in 1:(N - 1)]
        X = [zeros(T, nx) for k in 1:N]
        U = [zeros(T, nu) for k in 1:(N - 1)]
        Jprev = zero(T)
        ΔJ = zero(T)
        α = zero(T)
        α_mul = zero(T)
        β = zero(T)
        return new{T}(Xprev, Uprev, X, U, Jprev, ΔJ, α, α_mul, β)
    end
end
