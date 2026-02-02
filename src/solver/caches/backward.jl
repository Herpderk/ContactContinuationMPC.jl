mutable struct SimulatorExpansion{T<:AbstractFloat}
    x::Matrix{T}
    u::Matrix{T}
end

function SimulatorExpansion{T}(
    nx::Int,
    nu::Int,
)::SimulatorExpansion{T} where {T}
    Fx = zeros(T, nx, nx)
    Fu = zeros(T, nx, nu)
    return SimulatorExpansion{T}(Fx, Fu)
end



mutable struct CostFunctionExpansion{T<:AbstractFloat}
    x::Vector{T}
    u::Vector{T}
    xx::Matrix{T}
    uu::Matrix{T}
end

function CostFunctionExpansion{T}(
    nx::Int,
    nu::Int,
)::CostFunctionExpansion{T} where {T}
    Lx = zeros(T, nx)
    Lu = zeros(T, nu)
    Lxx = zeros(T, nx, nx)
    Luu = zeros(T, nu, nu)
    return CostFunctionExpansion{T}(Lx, Lu, Lxx, Luu)
end



mutable struct ValueFunctionExpansion{T<:AbstractFloat}
    x::Vector{T}
    xx::Matrix{T}
end

function ValueFunctionExpansion{T}(nx::Int)::ValueFunctionExpansion{T} where {T}
    Vx = zeros(T, nx)
    Vxx = zeros(T, nx, nx)
    return ValueFunctionExpansion{T}(Vx, Vxx)
end



mutable struct ActionValueFunctionExpansion{T<:AbstractFloat}
    x::Vector{T}
    u::Vector{T}
    xx::Matrix{T}
    xu::Matrix{T}
    ux::Matrix{T}
    uu::Matrix{T}
    uu_lu::SparseArrays.UMFPACK.UmfpackLU{T,Int64}
end

function ActionValueFunctionExpansion{T}(
    nx::Int,
    nu::Int,
)::ActionValueFunctionExpansion{T} where {T}
    Qx = zeros(T, nx)
    Qu = zeros(T, nu)
    Qxx = zeros(T, nx, nx)
    Qxu = zeros(T, nx, nu)
    Qux = zeros(T, nu, nx)
    Quu = zeros(T, nu, nu)
    Quu_lu = lu(sparse(ones(T, nu, nu)))
    return ActionValueFunctionExpansion{T}(Qx, Qu, Qxx, Qxu, Qux, Quu, Quu_lu)
end



mutable struct BackwardCache{T<:AbstractFloat}
    Fs::StructArray{<:SimulatorExpansion{T}}
    Ls::StructArray{<:CostFunctionExpansion{T}}
    Vs::StructArray{<:ValueFunctionExpansion{T}}
    Qs::StructArray{<:ActionValueFunctionExpansion{T}}

    Ks::Vector{VecOrMat{T}}
    D::Vector{Vector{T}}
    μ::Matrix{T}

    ΔJ::T
end

function BackwardCache{T}(nx::Int, nu::Int, N::Int)::BackwardCache{T} where {T}
    Fs = StructArray([SimulatorExpansion{T}(nx, nu) for k = 1:(N-1)])
    Ls = StructArray([CostFunctionExpansion{T}(nx, nu) for k = 1:(N-1)])
    Vs = StructArray([ValueFunctionExpansion{T}(nx) for k = 1:N])
    Qs = StructArray([ActionValueFunctionExpansion{T}(nx, nu) for k = 1:(N-1)])

    Ks = [zeros(T, nu, nx) for k = 1:(N-1)]
    D = [zeros(T, nu) for k = 1:(N-1)]
    μ = zeros(T, nu, nu)

    ΔJ = T(Inf)
    return BackwardCache{T}(Fs, Ls, Vs, Qs, Ks, D, μ, ΔJ)
end
