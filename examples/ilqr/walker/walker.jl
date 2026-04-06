using Pkg;
Pkg.activate(joinpath(@__DIR__, "../../.."))
using LinearAlgebra
using MuJoCo
using ContactContinuationMPC

USE_CC = true
WALKER = joinpath(@__DIR__, "../../../assets/walker2d/walker2d.xml")
WALKER_CC = joinpath(@__DIR__, "../../../assets/walker2d/walker2d_cc.xml")

init_visualiser()

function main(use_cc::Bool)
    # Get smoothed dynamics model
    if use_cc
        mbwd = load_model(WALKER_CC)
    else
        mbwd = load_model(WALKER)
    end

    # Forward model is always stiff
    mfwd = load_model(WALKER)
    d = init_data(mfwd)
    nx = get_nx(mfwd)
    ndx = get_ndx(mfwd)
    nu = mfwd.nu

    mfwd.opt.timestep = 0.02
    mbwd.opt.timestep = 0.02
    mfwd.opt.iterations = 5
    mbwd.opt.iterations = 5

    println("Joint names:")
    joint_names = get_joint_names(mfwd)
    for name in joint_names
        println(name)
    end

    # Declare references and initial conditions
    N = 250
    xidx =
        1 +
        MuJoCo.LibMuJoCo.mj_name2id(mfwd, MuJoCo.LibMuJoCo.mjOBJ_JOINT, "rootx")
    Xref = [zeros(nx) for k in 1:N]
    for k in 1:N
        copy_data_to_state!(d, Xref[k])
        qref = get_q(d, Xref[k])
        qref[xidx] += 5.0    # Set reference position without changing height
        vref = get_v(d, Xref[k])
        vref[xidx] = 2.0     # Set reference velocity
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
    Q = 1e-5 * Matrix(I(ndx))
    Q[xidx, xidx] *= 20.0
    Q[yidx, yidx] *= 20.0
    Q[zidx, zidx] *= 1000.0
    Q[mfwd.nq + xidx, mfwd.nq + xidx] *= 50.0
    Q[mfwd.nq + yidx, mfwd.nq + yidx] *= 10.0
    Q[mfwd.nq + zidx, mfwd.nq + zidx] *= 50.0

    # Penalize pitch and vertical position on the terminal state
    Qf = 1e+0 * Q
    Qf[yidx, yidx] *= 10.0
    Qf[zidx, zidx] *= 100.0
    Qf[mfwd.nq + xidx, mfwd.nq + xidx] *= 10.0

    R = 1e-3 * Matrix(I(mfwd.nu))
    costfunc = QuadraticCostFunction(Q, R, Qf)

    # Declare parameters and options
    params = TrajoptParameters(mfwd, mbwd, costfunc, Xref, Uref, xic)
    opts = iLQROptions(;
        maxiter_ilqr=100,
        maxiter_ls=50,
        alpha_mul=0.8,
        tol_interp=2.0,
        tol_converge=0.2,
        margin_ls=5e-2,
        eps_fd=1e-12,
    )

    # Solve trajopt
    sol = TrajoptSolution(params)
    cache = iLQRCache(params)
    run_ilqr!(sol, cache, params, opts)

    # Visualize solution
    println("\nFinal state: $(sol.X[end])\n")
    visualise!(mfwd, d; trajectories=stack(sol.X))
end

main(USE_CC)
