using LinearAlgebra
using ForwardDiff
using PreallocationTools
using Printf
using Test
using MjContactImplicit

"""
Struct for storing cartpole system parameters and array caches.
"""
struct CartpoleDynamics{T<:AbstractFloat}
    mc::T
    mp::T
    l::T
    g::T
    xdot::DiffCache{Vector{T},Vector{T}}
    nx::Int
    nu::Int

    function CartpoleDynamics{T}(;
        mc::Real=1.0, mp::Real=0.1, l::Real=0.5, g::Real=9.81
    ) where {T<:AbstractFloat}
        nx = 4
        nu = 1
        xdot = DiffCache(zeros(T, nx))
        return new{T}(T(mc), T(mp), T(l), T(g), xdot, nx, nu)
    end
end

"""
    (system::CartpoleDynamics)(xdot, x, u)

Forward simulation function for the cartpole system. Computes the next state x1 given the current state x and control input u.
"""
@views function (system::CartpoleDynamics{T})(
    xdot::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
)::Nothing where {T}
    mc, mp, l, g = system.mc, system.mp, system.l, system.g

    # Unpack state and input
    θ = x[2]
    sinθ = sin(θ)
    cosθ = cos(θ)
    θdot = x[4]
    f = u[1]

    # Compute dynamics
    xdot[1:2] .= x[3:4]
    xdot[3] = (f + mp * sinθ * (l*θdot^2 + g*cosθ)) / (mc + mp*sinθ^2)
    xdot[4] =
        (-f*cosθ - mp * l * θdot^2 * cosθ * sinθ - (mc+mp)*g*sinθ) /
        (l*(mc+mp*sinθ^2))
    return nothing
end

"""
Struct for storing cartpole system parameters and array caches.
"""
struct CartpoleSimulator{T<:AbstractFloat}
    dt::T
    dynamics::CartpoleDynamics{T}
    xtmp::DiffCache{Vector{T},Vector{T}}
    xdot_tmp::DiffCache{Vector{T},Vector{T}}

    function CartpoleSimulator{T}(;
        mc::Real=1.0, mp::Real=0.1, l::Real=0.5, g::Real=9.81, dt::Real=0.05
    ) where {T<:AbstractFloat}
        dynamics = CartpoleDynamics{T}(; mc=mc, mp=mp, l=l, g=g)
        xtmp = DiffCache(zeros(T, dynamics.nx))
        xdot_tmp = DiffCache(zeros(T, dynamics.nx))
        return new{T}(T(dt), dynamics, xtmp, xdot_tmp)
    end
end

"""
    (sim::CartpoleSimulator)(x1, x0, u0)

Forward simulation function for the cartpole system. Computes the next state x1 given the current state x0 and control input u0.
"""
@views function (sim::CartpoleSimulator{T})(
    x1::AbstractVector{<:Real},
    x0::AbstractVector{<:Real},
    u0::AbstractVector{<:Real},
)::Nothing where {T}
    # Get time step
    dt = sim.dt

    # Get temporary state and dynamics vectors
    super_el = x0[1] + u0[1]
    xtmp = get_tmp(sim.xtmp, super_el)
    xdot_tmp = get_tmp(sim.xdot_tmp, super_el)

    # RK4-integrate in place
    # Set x1 to x0
    x1 .= x0

    # Compute k1
    sim.dynamics(xdot_tmp, x0, u0)
    @. x1 += dt/6 * xdot_tmp

    # Compute k2
    @. xtmp = x0 + dt/2 * xdot_tmp
    sim.dynamics(xdot_tmp, xtmp, u0)
    @. x1 += dt/3 * xdot_tmp

    # Compute k3
    @. xtmp = x0 + dt/2 * xdot_tmp
    sim.dynamics(xdot_tmp, xtmp, u0)
    @. x1 += dt/3 * xdot_tmp

    # Compute k4
    @. xtmp = x0 + dt * xdot_tmp
    sim.dynamics(xdot_tmp, xtmp, u0)
    @. x1 += dt/6 * xdot_tmp
    return nothing
end

"""
    (sim::CartpoleSimulator)(A, B, x1, x0, u0)

Backward differentiation function for the cartpole system. Computes the Jacobians A and B of the next state.
"""
function (sim::CartpoleSimulator{T})(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    x1::AbstractVector{<:Real},
    x0::AbstractVector{<:Real},
    u0::AbstractVector{<:Real},
)::Nothing where {T}
    ForwardDiff.jacobian!(A, (δx1, δx0) -> sim(δx1, δx0, u0), x1, x0)
    ForwardDiff.jacobian!(B, (δx1, δu0) -> sim(δx1, x0, δu0), x1, u0)
    # Revert x1
    sim(x1, x0, u0)
    return nothing
end

"""
Struct for storing quadratic cost function weights.
"""
struct QuadraticCostFunction{T<:AbstractFloat}
    Q::Matrix{T}
    R::Matrix{T}
    Qf::Matrix{T}
    xtmp::DiffCache{Vector{T},Vector{T}}
    utmp::DiffCache{Vector{T},Vector{T}}

    function QuadraticCostFunction{T}(
        Q::AbstractMatrix{<:Real},
        R::AbstractMatrix{<:Real},
        Qf::AbstractMatrix{<:Real},
    ) where {T<:AbstractFloat}
        xtmp = DiffCache(zeros(T, size(Q)[1]))
        utmp = DiffCache(zeros(T, size(R)[1]))
        return new{T}(T.(Q), T.(R), T.(Qf), xtmp, utmp)
    end
end

"""
    (cache::QuadraticCostFunction)(xerr, uerr)

Stage cost function for a quadratic cost. Computes the cost given the state error xerr and control error uerr.
"""
function (cache::QuadraticCostFunction{T})(
    xerr::AbstractVector{<:Real}, uerr::AbstractVector{<:Real}
)::Union{T,ForwardDiff.Dual} where {T}
    super_el = xerr[1] + uerr[1]
    xtmp = get_tmp(cache.xtmp, super_el)
    utmp = get_tmp(cache.utmp, super_el)

    mul!(xtmp, cache.Q, xerr)
    mul!(utmp, cache.R, uerr)
    return 0.5 * (dot(xerr, xtmp) + dot(uerr, utmp))
end

"""
    (cache::QuadraticCostFunction)(xerr)

Terminal cost function for a quadratic cost. Computes the cost given the state error xerr.
"""
function (cache::QuadraticCostFunction{T})(
    xerr::AbstractVector{<:Real}
)::Union{T,ForwardDiff.Dual} where {T}
    xtmp = get_tmp(cache.xtmp, xerr)
    mul!(xtmp, cache.Qf, xerr)
    return 0.5 * dot(xerr, xtmp)
end

function clean_solve(
    params::TrajoptParameters, opts::ILqrOptions; use_time::Bool=false
)::TrajoptSolution
    sol = TrajoptSolution(params)
    cache = ILqrCache(params)
    if use_time
        @time run_ilqr!(sol, cache, params, opts)
    else
        run_ilqr!(sol, cache, params, opts)
    end
    return sol
end

@testset "iLQR Cartpole Test" begin
    dt = 0.05
    sim = CartpoleSimulator{Float64}(; dt=dt)

    Q = diagm([0.1, 1.0, 1.0, 1.0])
    R = 1e-2 * Matrix(I(sim.dynamics.nu))
    Qf = 1e+2 * Q
    costfunc = QuadraticCostFunction{Float64}(Q, R, Qf)

    N = 100
    Xref = [[0.0, pi, 0.0, 0.0] for k in 1:N]
    Uref = [zeros(1) for k in 1:(N - 1)]
    xic = 1e-3 * ones(sim.dynamics.nx)

    # Solve trajectory optimization
    T, S, C = Float64, typeof(sim), typeof(costfunc)
    params = TrajoptParameters{Float64,S,S,C,C}(
        sim, sim, costfunc, costfunc, Xref, Uref, xic
    )
    opts = ILqrOptions(; maxiter_ilqr=1)
    sol = clean_solve(params, opts)
    sol = clean_solve(params, opts; use_time=true)
    @test sol.is_optimal
end
