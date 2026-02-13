struct SimulatorExpansion{T<:AbstractFloat}
    dx::Transpose{T,Matrix{T}}
    u::Transpose{T,Matrix{T}}
    dx_bwd::Transpose{T,Matrix{T}}
    u_bwd::Transpose{T,Matrix{T}}

    function SimulatorExpansion{T}(ndx::Integer, nu::Integer) where {T}
        Fx = mj_zeros(T, ndx, ndx)
        Fu = mj_zeros(T, ndx, nu)
        Fx_bwd = mj_zeros(T, ndx, ndx)
        Fu_bwd = mj_zeros(T, ndx, nu)
        return new{T}(Fx, Fu, Fx_bwd, Fu_bwd)
    end
end

struct CostFunctionExpansion{T<:AbstractFloat}
    dx::Vector{T}
    u::Vector{T}
    dxdx::Matrix{T}
    uu::Matrix{T}
    dxdx_result::DiffResults.DiffResult{2,T,Tuple{Vector{T},Matrix{T}}}
    uu_result::DiffResults.DiffResult{2,T,Tuple{Vector{T},Matrix{T}}}

    function CostFunctionExpansion{T}(ndx::Integer, nu::Integer) where {T}
        Lx = zeros(T, ndx)
        Lu = zeros(T, nu)
        Lxx = zeros(T, ndx, ndx)
        Luu = zeros(T, nu, nu)
        Lxx_result = DiffResults.HessianResult(zeros(T, ndx))
        Luu_result = DiffResults.HessianResult(zeros(T, nu))
        return new{T}(Lx, Lu, Lxx, Luu, Lxx_result, Luu_result)
    end
end

struct ValueFunctionExpansion{T<:AbstractFloat}
    dx::Vector{T}
    dxdx::Matrix{T}

    function ValueFunctionExpansion{T}(ndx::Integer) where {T}
        Vx = zeros(T, ndx)
        Vxx = zeros(T, ndx, ndx)
        return new{T}(Vx, Vxx)
    end
end

struct ActionValueFunctionExpansion{T<:AbstractFloat}
    dx::Vector{T}
    u::Vector{T}
    dxdx::Matrix{T}
    dxu::Matrix{T}
    udx::Matrix{T}
    uu::Matrix{T}
    #bkws::BunchKaufmanWs

    function ActionValueFunctionExpansion{T}(
        ndx::Integer, nu::Integer
    ) where {T}
        Qx = zeros(T, ndx)
        Qu = zeros(T, nu)
        Qxx = zeros(T, ndx, ndx)
        Qxu = zeros(T, ndx, nu)
        Qux = zeros(T, nu, ndx)
        Quu = zeros(T, nu, nu)
        #bkws = BunchKaufmanWs(Quu)
        return new{T}(Qx, Qu, Qxx, Qxu, Qux, Quu)#, bkws)
    end
end

mutable struct BackwardCache{T<:AbstractFloat}
    F::SimulatorExpansion{T}
    L::CostFunctionExpansion{T}
    V::ValueFunctionExpansion{T}
    Q::ActionValueFunctionExpansion{T}
    Ks::Vector{Matrix{T}}
    ds::Vector{Vector{T}}
    μI::Matrix{T}
    ϵ::T
    ΔJ::T
    ΔJ1::T
    ΔJ2::T
    ΔJmin::T
    ΔJmax::T

    function BackwardCache{T}(ndx::Integer, nu::Integer, N::Integer) where {T}
        F = SimulatorExpansion{T}(ndx, nu)
        L = CostFunctionExpansion{T}(ndx, nu)
        V = ValueFunctionExpansion{T}(ndx)
        Q = ActionValueFunctionExpansion{T}(ndx, nu)
        Ks = [zeros(T, nu, ndx) for k in 1:(N - 1)]
        ds = [zeros(T, nu) for k in 1:(N - 1)]
        μI = Matrix{T}(I(nu))
        ϵ = zero(T)
        ΔJ = zero(T)
        ΔJ1 = zero(T)
        ΔJ2 = zero(T)
        ΔJmin = zero(T)
        ΔJmax = zero(T)
        return new{T}(F, L, V, Q, Ks, ds, μI, ϵ, ΔJ, ΔJ1, ΔJ2, ΔJmin, ΔJmax)
    end
end
