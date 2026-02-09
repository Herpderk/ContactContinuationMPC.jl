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
    d = init_data(m)

    # Set model options
    m.opt.timestep = 0.01

    # Declare references and initial conditions
    N = 400
    Xref = [[0.0, pi, 0.0, 0.0] for k in 1:N]
    Uref = [zeros(1) for k in 1:(N - 1)]
    xic = 1e-2 * ones(get_nx(m))

    # Declare cost function
    T = Float64
    Q = 1e-3 * diagm([0.0, 1.0, 1.0, 1.0])
    R = 1e-6 * Matrix(I(m.nu))
    Qf = 1e+2 * Q
    costfunc = QuadraticCostFunction{T}(Q, R, Qf)

    # Declare parameters and options
    L = typeof(costfunc)
    params = TrajoptParameters{T,L,L}(m, m, costfunc, costfunc, Xref, Uref, xic)
    opts = ILqrOptions{T}()

    # Let the trajpot JIT compile
    opts.is_verbose = false
    sol = fresh_solve(params, opts)

    # Solve a second time after JIT compilation for accurate timing
    opts.is_verbose = true
    sol = fresh_solve(params, opts; use_time=true)

    # Test solution
    println("\nFinal state: $(sol.X[end])\n")
    @test sol.is_optimal
    return nothing
end
