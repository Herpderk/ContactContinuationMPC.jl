using LinearAlgebra
using Test
using MuJoCo
using ContactContinuationMPC
using MuJoCo.LibMuJoCo

# 1. Define the silent handler
function silent_warning_handler(msg_ptr::Ptr{Cchar})::Cvoid
    return nothing
end

# 2. Create the C-compatible function pointer
# Note: We keep this in a constant so it isn't garbage collected
const SILENT_CB = @cfunction(silent_warning_handler, Cvoid, (Ptr{Cchar},))

# 3. Get the memory address of the global 'mju_user_warning' pointer
# and overwrite it with our new function pointer
warning_ptr_addr = cglobal((:mju_user_warning, LibMuJoCo.libmujoco), Ptr{Cvoid})
unsafe_store!(warning_ptr_addr, SILENT_CB)

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
    opts = SQPOptions(; maxiter=50)
    sol = TrajoptSolution(params)

    # Trust region bounds
    nx = get_nx(m)
    Δxl = -ones(nx)
    Δul = -ones(m.nu)
    cache = SQPCache(params; Δxl=Δxl, Δxu=(-Δxl), Δul=Δul, Δuu=(-Δul))
    run_sqp!(sol, cache, params, opts)

    # Test solution
    println("\nFinal state: $(sol.X[end])\n")
    @test sol.is_optimal
    return nothing
end
