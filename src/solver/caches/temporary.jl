mutable struct TemporaryCache{T<:AbstractFloat}
    x::Vector{T}
    u::Vector{T}
    xx::Matrix{T}
    uu::Matrix{T}
    xu::Matrix{T}
    ux::Matrix{T}
    xx_result::DiffResults.DiffResult{2,T,Tuple{Vector{T},Matrix{T}}}
    uu_result::DiffResults.DiffResult{2,T,Tuple{Vector{T},Matrix{T}}}
end

function TemporaryCache{T}(
    nx::Int,
    nu::Int,
)::TemporaryCache{T} where {T<:AbstractFloat}
    x = zeros(T, nx)
    u = zeros(T, nu)
    xx = zeros(T, nx, nx)
    uu = zeros(T, nu, nu)
    xu = zeros(T, nx, nu)
    ux = zeros(T, nu, nx)
    xx_result = DiffResults.HessianResult(zeros(T, nx))
    uu_result = DiffResults.HessianResult(zeros(T, nu))
    return TemporaryCache{T}(x, u, xx, uu, xu, ux, xx_result, uu_result)
end
