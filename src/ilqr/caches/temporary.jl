mutable struct TemporaryCache{T<:AbstractFloat}
    singleton::Matrix{T}
    x::Vector{T}
    dx::Vector{T}
    u::Vector{T}
    dxdx::Matrix{T}
    uu::Matrix{T}
    dxu::Matrix{T}
    udx::Matrx{T}
    hess_dxdx::DiffResults.DiffResult{2,T,Tuple{Vector{T},Matrix{T}}}
    hess_uu::DiffResults.DiffResult{2,T,Tuple{Vector{T},Matrix{T}}}
    bkws_uu::BunchKaufmanWs
    #luws_uu::LUWs
end

function TemporaryCache{T}(
    nx::Int, ndx::Int, nu::Int
)::TemporaryCache{T} where {T<:AbstractFloat}
    singleton = zeros(T, 1, 1)
    x = zeros(T, nx)
    u = zeros(T, nu)
    dxdx = zeros(T, ndx, ndx)
    uu = zeros(T, nu, nu)
    xu = zeros(T, ndx, nu)
    ux = zeros(T, nu, ndx)
    hess_dxdx = DiffResults.HessianResult(zeros(T, ndx))
    hess_uu = DiffResults.HessianResult(zeros(T, nu))
    bkws_uu = BunchKaufmanWs(uu)
    #luws_uu = LUWs(uu)
    return TemporaryCache{T}(
        singleton, x, u, dxdx, uu, xu, ux, hess_dxdx, hess_uu, bkws_uu
    )#luws_uu)
end
