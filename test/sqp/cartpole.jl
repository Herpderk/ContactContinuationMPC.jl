using LinearAlgebra
using Test
using MuJoCo
using ContactContinuationMPC
using Plots

init_visualiser()

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

    # Actuator limits
    ul = zeros(m.nu)
    uu = zeros(m.nu)
    for i in 1:m.nu
        if Bool(m.actuator_ctrllimited[i])
            ul[i] = m.actuator_ctrlrange[i, 1]
            uu[i] = m.actuator_ctrlrange[i, 2]
        else
            ul[i] = -Inf
            uu[i] = Inf
        end
    end

    # Solve using SQP
    opts = SQPOptions(; ul=ul, uu=uu, maxiter_sqp=100)
    params = TrajoptParameters(m, m, costfunc, Xref, Uref, xic)
    cache = SQPCache(params)
    sol = TrajoptSolution(params)
    for k in 1:(N - 1)      # warm-start
        sol.X[k] .= Xref[k]
        sol.U[k] .= Uref[k]
    end
    sol.X[end] .= Xref[end]
    run_sqp!(sol, cache, params, opts)

    # Test solution
    println("\nInitial state: $(sol.X[1])")
    println("Final state: $(sol.X[end])\n")
    @test sol.is_optimal
    return nothing
end
