using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using MuJoCo
using ContactContinuationMPC

USE_CC = true
HALF_CHEETAH = joinpath(@__DIR__, "../assets/half_cheetah.xml")
HALF_CHEETAH_SMOOTH = joinpath(@__DIR__, "../assets/half_cheetah_smooth.xml")

init_visualiser()

function main(use_cc::Bool)
    # Get smoothed dynamics model
    if use_cc
        mbwd = load_model(HALF_CHEETAH_SMOOTH)
    else
        mbwd = load_model(HALF_CHEETAH)
    end

    # Forward model is always stiff
    mfwd = load_model(HALF_CHEETAH)
    d = init_data(mfwd)
    nx = get_nx(mfwd)
    nu = mfwd.nu

    println("Joint names:")
    joint_names = get_joint_names(mfwd)
    for name in joint_names
        println(name)
    end

    # Declare references and initial conditions
    N = 500

    xidx =
        1 +
        MuJoCo.LibMuJoCo.mj_name2id(mfwd, MuJoCo.LibMuJoCo.mjOBJ_JOINT, "rootx")
    Xref = [zeros(nx) for k in 1:N]
    for k in 1:N
        copy_data_to_state!(Xref[k], d)
        Xref[k][xidx] += 10.0    # Set reference position without changing height
    end

    Uref = [zeros(nu) for k in 1:(N - 1)]
    xic = zeros(nx)
    copy_data_to_state!(xic, d)

    # Declare cost function (Penalize horizontal position)
    zidx =
        1 +
        MuJoCo.LibMuJoCo.mj_name2id(mfwd, MuJoCo.LibMuJoCo.mjOBJ_JOINT, "rootz")
    Q = 1e-5 * Matrix(I(nx))
    Q[xidx, xidx] *= 10.0
    Q[zidx, zidx] *= 2.0

    # Penalize vertical position on the terminal state
    Qf = 1e+0 * Q
    Qf[zidx, zidx] *= 10.0

    R = 1e-3 * Matrix(I(mfwd.nu))
    costfunc = QuadraticCostFunction(Q, R, Qf)

    # Declare parameters and options
    params = TrajoptParameters(mfwd, mbwd, costfunc, Xref, Uref, xic)
    opts = ILqrOptions(;
        maxiter_ilqr=200,
        maxiter_ls=50,
        alpha_mul=0.8,
        tol_converge=5e-1,
        margin_ls=1e-2,
    )

    # Solve trajopt
    sol = TrajoptSolution(params)
    cache = ILqrCache(params)
    @time run_ilqr!(sol, cache, params, opts)

    # Visualize solution
    println("\nFinal state: $(sol.X[end])\n")
    visualise!(mfwd, d; trajectories=stack(sol.X))
end

main(USE_CC)
