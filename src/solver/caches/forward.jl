mutable struct ForwardCache{T<:AbstractFloat}
    X::Vector{Vector{T}}
    U::Vector{Vector{T}}
    α::T
    ΔJ::T
end

function ForwardCache{T}(nx::Int, nu::Int, N::Int)::ForwardCache{T} where {T}
    X = [zeros(T, nx) for k = 1:N]
    U = [zeros(T, nu) for k = 1:(N-1)]
    α = T(0.0)
    ΔJ = T(0.0)
    return ForwardCache{T}(X, U, α, ΔJ)
end
