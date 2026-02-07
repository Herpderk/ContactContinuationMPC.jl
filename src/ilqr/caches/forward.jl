mutable struct ForwardCache{T<:AbstractFloat}
    X::Vector{Vector{T}}
    U::Vector{Vector{T}}
    ΔJ::T
    α::T
    α_mul::T

    function ForwardCache{T}(
        nx::Integer, nu::Integer, N::Integer
    )::ForwardCache{T} where {T}
        X = [zeros(T, nx) for k in 1:N]
        U = [zeros(T, nu) for k in 1:(N - 1)]
        ΔJ = zero(T)
        α = zero(T)
        α_mul = zero(T)
        return new{T}(X, U, ΔJ, α, α_mul)
    end
end
