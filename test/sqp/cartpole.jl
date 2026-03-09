using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using LinearAlgebra
using Test
using MuJoCo
using ContactContinuationMPC
using Plots

@testset "SQP Cartpole Test" begin
    # Mujoco dynamics model
    m = load_model(joinpath(@__DIR__, "../../assets/cartpole.xml"))

    # Set model options
    m.opt.timestep = 0.01

    # Declare references and initial conditions
    N = 400
    Xref = [[0.0, pi, 0.0, 0.0] for k in 1:N]
    Uref = [zeros(1) for k in 1:(N - 1)]
    xic = 1e-2 * ones(Utils.get_nx(m))

    # Declare cost function
    Q = 1e-3 * diagm([0.1, 1.0, 1.0, 1.0])
    R = 1e-6 * Matrix(I(m.nu))
    Qf = 1e+2 * Q
    costfunc = QuadraticCostFunction(Q, R, Qf)

    # Declare parameters and options
    params = TrajoptParameters(m, m, costfunc, Xref, Uref, xic)
    opts = SQPOptions(; maxiter=100, eps_fd=1e-12, eps_reg=1e-4)
    sol = TrajoptSolution(params)
    for k in 1:(N - 1)
        sol.X[k] .= Xref[k]
        sol.U[k] .= Uref[k]
    end
    sol.X[end] .= Xref[end]

    # Trust region bounds
    nx = get_nx(m)
    Δxl = -100.0 * ones(nx)
    Δul = -100.0 * ones(m.nu)
    cache = SQPCache(params; Δxl=Δxl, Δxu=(-Δxl), Δul=Δul, Δuu=(-Δul))
    run_sqp!(sol, cache, params, opts)

    # Test solution
    println("\nInitial state: $(sol.X[1])")
    println("Final state: $(sol.X[end])\n")
    @test sol.is_optimal
    return nothing
end
