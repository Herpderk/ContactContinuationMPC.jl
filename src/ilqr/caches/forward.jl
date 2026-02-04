mutable struct ForwardCache{T<:AbstractFloat}
    X::Vector{Vector{T}}
    U::Vector{Vector{T}}
    α::T
    ΔJ::T
end

function ForwardCache{T}(
    nx::Int, nu::Int, N::Int
)::ForwardCache{T} where {T<:AbstractFloat}
    X = [zeros(T, nx) for k in 1:N]
    U = [zeros(T, nu) for k in 1:(N - 1)]
    α = zero(T)
    ΔJ = zero(T)
    return ForwardCache{T}(X, U, α, ΔJ)
end
