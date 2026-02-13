using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using MuJoCo
using ContactContinuationMPC

USE_CC = true
HALF_CHEETAH = joinpath(@__DIR__, "../assets/half_cheetah.xml")
HALF_CHEETAH_CC = joinpath(@__DIR__, "../assets/half_cheetah_cc.xml")

init_visualiser()

function main(use_cc::Bool)
    # Get smoothed dynamics model
    if use_cc
        mbwd = load_model(HALF_CHEETAH_CC)
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
    N = 200     #   4s (timestep is 0.02s)

    xidx =
        1 +
        MuJoCo.LibMuJoCo.mj_name2id(mfwd, MuJoCo.LibMuJoCo.mjOBJ_JOINT, "rootx")
    Xref = [zeros(nx) for k in 1:N]
    for k in 1:N
        copy_data_to_state!(d, Xref[k])
        qref = get_q(d, Xref[k])
        qref[xidx] += 10.0    # Set reference position without changing height
        vref = get_v(d, Xref[k])
        vref[xidx] = 1.0     # Set reference velocity
    end

    Uref = [zeros(nu) for k in 1:(N - 1)]
    xic = zeros(nx)
    copy_data_to_state!(d, xic)

    # Declare cost function (Penalize horizontal pos, vertical pos, and pitch)
    yidx =
        1 +
        MuJoCo.LibMuJoCo.mj_name2id(mfwd, MuJoCo.LibMuJoCo.mjOBJ_JOINT, "rooty")
    zidx =
        1 +
        MuJoCo.LibMuJoCo.mj_name2id(mfwd, MuJoCo.LibMuJoCo.mjOBJ_JOINT, "rootz")
    Q = 1e-5 * Matrix(I(nx))
    Q[xidx, xidx] *= 20.0
    Q[yidx, yidx] *= 5.0
    Q[zidx, zidx] *= 2.0

    # Penalize pitch and vertical position on the terminal state
    Qf = 1e+0 * Q
    Qf[yidx, yidx] *= 500.0
    Qf[zidx, zidx] *= 100.0

    R = 1e-3 * Matrix(I(mfwd.nu))
    costfunc = QuadraticCostFunction(Q, R, Qf)

    # Declare parameters and options
    params = TrajoptParameters(mfwd, mbwd, costfunc, Xref, Uref, xic)
    opts = ILqrOptions(;
        maxiter_ilqr=500,
        maxiter_ls=50,
        alpha_mul=0.8,
        tol_interp=1.0,
        tol_converge=0.5,
        margin_ls=1e-2,
    )

    # Solve trajopt
    sol = TrajoptSolution(params)
    cache = ILqrCache(params)
    run_ilqr!(sol, cache, params, opts)

    # Visualize solution
    println("\nFinal state: $(sol.X[end])\n")
    visualise!(mfwd, d; trajectories=stack(sol.X))
end

main(USE_CC)
