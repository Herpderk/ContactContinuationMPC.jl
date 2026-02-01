
mutable struct SimExpansion{T<:AbstractFloat}
    x::Matrix{T}
    u::Matrix{T}
end

function SimExpansion{T}(nx::Int, nu::Int)::SimExpansion{T} where {T}
    Fx = zeros(T, nx, nx)
    Fu = zeros(T, nx, nu)
    return SimExpansion{T}(Fx, Fu)
end



mutable struct CostExpansion{T<:AbstractFloat}
    x::Vector{T}
    u::Vector{T}
    xx::Matrix{T}
    uu::Matrix{T}
    xx_result::DiffResults.DiffResult{2,T,Tuple{Vector{T},Matrix{T}}}
    uu_result::DiffResults.DiffResult{2,T,Tuple{Vector{T},Matrix{T}}}
end

function CostExpansion{T}(nx::Int, nu::Int)::CostExpansion{T} where {T}
    Lx = zeros(T, nx)
    Lu = zeros(T, nu)
    Lxx = zeros(T, nx, nx)
    Luu = zeros(T, nu, nu)
    Lxx_result = DiffResults.HessianResult(zeros(T, nx))
    Luu_result = DiffResults.HessianResult(zeros(T, nu))
    return CostExpansion{T}(Lx, Lu, Lxx, Luu, Lxx_result, Luu_result)
end



mutable struct ValueExpansion{T<:AbstractFloat}
    x::Vector{T}
    xx::Matrix{T}
end

function ValueExpansion{T}(nx::Int)::ValueExpansion{T} where {T}
    Vx = zeros(T, nx)
    Vxx = zeros(T, nx, nx)
    return ValueExpansion{T}(Vx, Vxx)
end



mutable struct ActionValueExpansion{T<:AbstractFloat}
    x::Vector{T}
    u::Vector{T}
    xx::Matrix{T}
    xu::Matrix{T}
    ux::Matrix{T}
    uu::Matrix{T}
    uu_lu::SparseArrays.UMFPACK.UmfpackLU{T,Int64}
end

function ActionValueExpansion{T}(nx::Int, nu::Int)::ActionValueExpansion{T} where {T}
    Qx = zeros(T, nx)
    Qu = zeros(T, nu)
    Qxx = zeros(T, nx, nx)
    Qxu = zeros(T, nx, nu)
    Qux = zeros(T, nu, nx)
    Quu = zeros(T, nu, nu)
    Quu_lu = lu(sparse(ones(T, nu, nu)))
    return ActionValueExpansion{T}(Qx, Qu, Qxx, Qxu, Qux, Quu, Quu_lu)
end



mutable struct BackwardCache{T<:AbstractFloat}
    Fs::Vector{SimExpansion{T}}
    Ls::Vector{CostExpansion{T}}
    Vs::Vector{ValueExpansion{T}}
    Qs::Vector{ActionValueExpansion{T}}

    Ks::Vector{VecOrMat{T}}
    ds::Vector{Vector{T}}
    μ::Matrix{T}

    ΔJ1::T
    ΔJ2::T
end

function BackwardCache{T}(nx::Int, nu::Int, N::Int)::BackwardCache{T} where {T}
    Fs = [SimExpansion{T}(nx, nu) for k ∈ 1:(N-1)]
    Ls = [CostExpansion{T}(nx, nu) for k ∈ 1:(N-1)]
    Vs = [ValueExpansion{T}(nx) for k ∈ 1:N]
    Qs = [ActionValueExpansion{T}(nx, nu) for k ∈ 1:(N-1)]

    Ks = [zeros(T, nu, nx) for k ∈ 1:(N-1)]
    ds = [zeros(T, nu) for k ∈ 1:(N-1)]
    μ = zeros(T, nu, nu)

    ΔJ1=T(Inf)
    ΔJ2=T(Inf)
    return BackwardCache{T}(Fs, Ls, Vs, Qs, Ks, ds, μ, ΔJ1, ΔJ2)
end
