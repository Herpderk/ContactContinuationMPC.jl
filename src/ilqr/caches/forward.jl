mutable struct ForwardCache{T<:AbstractFloat}
    X0::Vector{Vector{T}}
    U0::Vector{Vector{T}}
    X1::Vector{Vector{T}}
    U1::Vector{Vector{T}}
    Jprev::T
    ΔJ::T
    α::T
    α_mul::T
    β::T

    function ForwardCache{T}(nx::Integer, nu::Integer, N::Integer) where {T}
        X0 = [zeros(T, nx) for k in 1:N]
        U0 = [zeros(T, nu) for k in 1:(N - 1)]
        X1 = [zeros(T, nx) for k in 1:N]
        U1 = [zeros(T, nu) for k in 1:(N - 1)]
        Jprev = zero(T)
        ΔJ = zero(T)
        α = zero(T)
        α_mul = zero(T)
        β = zero(T)
        return new{T}(X0, U0, X1, U1, Jprev, ΔJ, α, α_mul, β)
    end
end
