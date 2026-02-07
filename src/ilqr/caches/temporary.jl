mutable struct TemporaryCache{T<:AbstractFloat}
    singleton::Matrix{T}
    x::Vector{T}
    dx::Vector{T}
    u::Vector{T}
    dxdx::Matrix{T}
    dxdx2::Matrix{T}
    uu::Matrix{T}
    dxu::Matrix{T}
    udx::Matrix{T}
    hess_dxdx::DiffResults.DiffResult{2,T,Tuple{Vector{T},Matrix{T}}}
    hess_uu::DiffResults.DiffResult{2,T,Tuple{Vector{T},Matrix{T}}}
    bkws_uu::BunchKaufmanWs
    #luws_uu::LUWs

    function TemporaryCache{T}(
        nx::Integer, ndx::Integer, nu::Integer
    )::TemporaryCache{T} where {T}
        singleton = zeros(T, 1, 1)
        x = zeros(T, nx)
        dx = zeros(T, ndx)
        u = zeros(T, nu)
        dxdx = zeros(T, ndx, ndx)
        dxdx2 = zeros(T, ndx, ndx)
        uu = zeros(T, nu, nu)
        dxu = zeros(T, ndx, nu)
        udx = zeros(T, nu, ndx)
        hess_dxdx = DiffResults.HessianResult(zeros(T, ndx))
        hess_uu = DiffResults.HessianResult(zeros(T, nu))
        bkws_uu = BunchKaufmanWs(uu)
        #luws_uu = LUWs(uu)
        return new{T}(
            singleton,
            x,
            dx,
            u,
            dxdx,
            dxdx2,
            uu,
            dxu,
            udx,
            hess_dxdx,
            hess_uu,
            bkws_uu,
        )#luws_uu)
    end
end
