using LinearAlgebra
using ForwardDiff
using PreallocationTools
using Printf
using Test
using MuJoCo
using MjContactImplicit

"""
Struct for storing quadratic cost function weights and caches.
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
    ) where {T}
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
    xerr::AbstractVector{Tx}, uerr::AbstractVector{Tu}
)::Union{T,ForwardDiff.Dual} where {Tx,Tu,T}
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
    xerr::AbstractVector{Tx}
)::Union{T,ForwardDiff.Dual} where {Tx,T}
    xtmp = get_tmp(cache.xtmp, xerr)
    mul!(xtmp, cache.Qf, xerr)
    return 0.5 * dot(xerr, xtmp)
end

function fresh_solve(
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
    # Mujoco dynamics model
    m = load_model(joinpath(@__DIR__, "..", "assets", "cartpole.xml"))
    d = init_data(m)

    # Set model options
    m.opt.timestep = 0.05
    # m.opt.integrator = MuJoCo.mjINT_RK4

    # Declare cost function
    Q = diagm([0.01, 1.0, 1.0, 1.0])
    R = 1e-2 * Matrix(I(m.nu))
    Qf = 1e+2 * Q
    costfunc = QuadraticCostFunction{Float64}(Q, R, Qf)

    # Declare references and initial conditions
    N = 100
    Xref = [[0.0, pi, 0.0, 0.0] for k in 1:N]
    Uref = [zeros(1) for k in 1:(N - 1)]
    xic = 1e-3 * ones(get_nx(m))

    # Declare parameters and options
    T, C = Float64, typeof(costfunc)
    params = TrajoptParameters{Float64,C,C}(
        m, m, costfunc, costfunc, Xref, Uref, xic
    )
    opts = ILqrOptions{T}()

    # Solve trajectory optimization
    sol = fresh_solve(params, opts)
    sol = fresh_solve(params, opts; use_time=true)
    @test sol.is_optimal
end
