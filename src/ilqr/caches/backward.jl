struct SimulatorExpansion{T<:AbstractFloat}
    dx::Matrix{T}
    u::Matrix{T}
end

function SimulatorExpansion{T}(
    ndx::Int, nu::Int
)::SimulatorExpansion{T} where {T}
    Fx = zeros(T, ndx, ndx)
    Fu = zeros(T, ndx, nu)
    return SimulatorExpansion{T}(Fx, Fu)
end

struct CostFunctionExpansion{T<:AbstractFloat}
    dx::Vector{T}
    u::Vector{T}
    dxdx::Matrix{T}
    uu::Matrix{T}
end

function CostFunctionExpansion{T}(
    ndx::Int, nu::Int
)::CostFunctionExpansion{T} where {T}
    Lx = zeros(T, ndx)
    Lu = zeros(T, nu)
    Lxx = zeros(T, ndx, ndx)
    Luu = zeros(T, nu, nu)
    return CostFunctionExpansion{T}(Lx, Lu, Lxx, Luu)
end

struct ValueFunctionExpansion{T<:AbstractFloat}
    dx::Vector{T}
    dxdx::Matrix{T}
end

function ValueFunctionExpansion{T}(
    ndx::Int
)::ValueFunctionExpansion{T} where {T}
    Vx = zeros(T, ndx)
    Vxx = zeros(T, ndx, ndx)
    return ValueFunctionExpansion{T}(Vx, Vxx)
end

struct ActionValueFunctionExpansion{T<:AbstractFloat}
    dx::Vector{T}
    u::Vector{T}
    dxdx::Matrix{T}
    dxu::Matrix{T}
    udx::Matrix{T}
    uu::Matrix{T}
end

function ActionValueFunctionExpansion{T}(
    ndx::Int, nu::Int
)::ActionValueFunctionExpansion{T} where {T}
    Qx = zeros(T, ndx)
    Qu = zeros(T, nu)
    Qxx = zeros(T, ndx, ndx)
    Qxu = zeros(T, ndx, nu)
    Qux = zeros(T, nu, ndx)
    Quu = zeros(T, nu, nu)
    return ActionValueFunctionExpansion{T}(Qx, Qu, Qxx, Qxu, Qux, Quu)
end

mutable struct BackwardCache{T<:AbstractFloat}
    F::SimulatorExpansion{T}
    L::CostFunctionExpansion{T}
    V::ValueFunctionExpansion{T}
    Q::ActionValueFunctionExpansion{T}
    Ks::Vector{Matrix{T}}
    ds::Vector{Vector{T}}
    μ::Matrix{T}
    ϵ::T
    ΔJ::T
end

function BackwardCache{T}(ndx::Int, nu::Int, N::Int)::BackwardCache{T} where {T}
    F = SimulatorExpansion{T}(ndx, nu)
    L = CostFunctionExpansion{T}(ndx, nu)
    V = ValueFunctionExpansion{T}(ndx)
    Q = ActionValueFunctionExpansion{T}(ndx, nu)
    Ks = [zeros(T, nu, ndx) for k in 1:(N - 1)]
    ds = [zeros(T, nu) for k in 1:(N - 1)]
    μ = Matrix{T}(I(nu))
    ϵ = zero(T)
    ΔJ = zero(T)
    return BackwardCache{T}(F, L, V, Q, Ks, ds, μ, ϵ, ΔJ)
end
