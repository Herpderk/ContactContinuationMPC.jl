mutable struct TemporaryCache{T<:AbstractFloat}
    x::Vector{T}
    u::Vector{T}
    xx::Matrix{T}
    uu::Matrix{T}
    xu::Matrix{T}
    ux::Matrix{T}
    hess_xx::DiffResults.DiffResult{2,T,Tuple{Vector{T},Matrix{T}}}
    hess_uu::DiffResults.DiffResult{2,T,Tuple{Vector{T},Matrix{T}}}
    bkws_uu::BunchKaufmanWs
    #luws_uu::LUWs
end

function TemporaryCache{T}(
    nx::Int, nu::Int
)::TemporaryCache{T} where {T<:AbstractFloat}
    x = zeros(T, nx)
    u = zeros(T, nu)
    xx = zeros(T, nx, nx)
    uu = zeros(T, nu, nu)
    xu = zeros(T, nx, nu)
    ux = zeros(T, nu, nx)
    hess_xx = DiffResults.HessianResult(zeros(T, nx))
    hess_uu = DiffResults.HessianResult(zeros(T, nu))
    bkws_uu = BunchKaufmanWs(uu)
    #luws_uu = LUWs(uu)
    return TemporaryCache{T}(x, u, xx, uu, xu, ux, hess_xx, hess_uu, bkws_uu)#luws_uu)
end
