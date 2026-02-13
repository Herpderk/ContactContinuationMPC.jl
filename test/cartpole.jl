using LinearAlgebra
using Test
using MuJoCo
using ContactContinuationMPC

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

    # Set model options
    m.opt.timestep = 0.01

    # Declare references and initial conditions
    N = 400
    Xref = [[0.0, pi, 0.0, 0.0] for k in 1:N]
    Uref = [zeros(1) for k in 1:(N - 1)]
    xic = 1e-2 * ones(get_nx(m))

    # Declare cost function
    Q = 1e-3 * diagm([0.0, 1.0, 1.0, 1.0])
    R = 1e-6 * Matrix(I(m.nu))
    Qf = 1e+2 * Q
    costfunc = QuadraticCostFunction(Q, R, Qf)

    # Declare parameters and options
    params = TrajoptParameters(m, m, costfunc, Xref, Uref, xic)
    opts = ILqrOptions(; tol_converge=1e-2)
    sol = fresh_solve(params, opts; use_time=true)

    # Test solution
    println("\nFinal state: $(sol.X[end])\n")
    @test sol.is_optimal
    return nothing
end
