struct SimulatorExpansion{T<:AbstractFloat}
    x::Matrix{T}
    u::Matrix{T}
end

function SimulatorExpansion{T}(
    nx::Int, nu::Int
)::SimulatorExpansion{T} where {T<:AbstractFloat}
    Fx = zeros(T, nx, nx)
    Fu = zeros(T, nx, nu)
    return SimulatorExpansion{T}(Fx, Fu)
end

struct CostFunctionExpansion{T<:AbstractFloat}
    x::Vector{T}
    u::Vector{T}
    xx::Matrix{T}
    uu::Matrix{T}
end

function CostFunctionExpansion{T}(
    nx::Int, nu::Int
)::CostFunctionExpansion{T} where {T<:AbstractFloat}
    Lx = zeros(T, nx)
    Lu = zeros(T, nu)
    Lxx = zeros(T, nx, nx)
    Luu = zeros(T, nu, nu)
    return CostFunctionExpansion{T}(Lx, Lu, Lxx, Luu)
end

struct ValueFunctionExpansion{T<:AbstractFloat}
    x::Vector{T}
    xx::Matrix{T}
end

function ValueFunctionExpansion{T}(
    nx::Int
)::ValueFunctionExpansion{T} where {T<:AbstractFloat}
    Vx = zeros(T, nx)
    Vxx = zeros(T, nx, nx)
    return ValueFunctionExpansion{T}(Vx, Vxx)
end

struct ActionValueFunctionExpansion{T<:AbstractFloat}
    x::Vector{T}
    u::Vector{T}
    xx::Matrix{T}
    xu::Matrix{T}
    ux::Matrix{T}
    uu::Matrix{T}
end

function ActionValueFunctionExpansion{T}(
    nx::Int, nu::Int
)::ActionValueFunctionExpansion{T} where {T<:AbstractFloat}
    Qx = zeros(T, nx)
    Qu = zeros(T, nu)
    Qxx = zeros(T, nx, nx)
    Qxu = zeros(T, nx, nu)
    Qux = zeros(T, nu, nx)
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
    ΔJ::T
end

function BackwardCache{T}(
    nx::Int, nu::Int, N::Int
)::BackwardCache{T} where {T<:AbstractFloat}
    F = SimulatorExpansion{T}(nx, nu)
    L = CostFunctionExpansion{T}(nx, nu)
    V = ValueFunctionExpansion{T}(nx)
    Q = ActionValueFunctionExpansion{T}(nx, nu)
    Ks = [zeros(T, nu, nx) for k in 1:(N - 1)]
    ds = [zeros(T, nu) for k in 1:(N - 1)]
    μ = Matrix{T}(I(nu))
    ΔJ = zero(T)
    return BackwardCache{T}(F, L, V, Q, Ks, ds, μ, ΔJ)
end
