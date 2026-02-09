using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using LinearAlgebra
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

# Mujoco dynamics model
m = load_model(joinpath(@__DIR__, "../../assets/half_cheetah.xml"))
d = init_data(m)
nx = get_nx(m)
nu = m.nu

println("Joint names:")
joint_names = get_joint_names(m)
for name in joint_names
    println(name)
end

# Declare references and initial conditions
N = 500

xidx = 1 + MuJoCo.LibMuJoCo.mj_name2id(m, MuJoCo.LibMuJoCo.mjOBJ_JOINT, "rootx")
Xref = [zeros(nx) for k in 1:N]
for k in 1:N
    copy_data_to_state!(Xref[k], d)
    Xref[k][xidx] += 10.0    # Set reference position without changing height
end

Uref = [zeros(nu) for k in 1:(N - 1)]
xic = zeros(nx)
copy_data_to_state!(xic, d)

# Declare cost function (Penalize horizontal position)
Q = 1e-5 * Matrix(I(nx))
Q[xidx, xidx] *= 10.0

# Penalize vertical position on the terminal state
Qf = 1e+2 * Q
zidx = 1 + MuJoCo.LibMuJoCo.mj_name2id(m, MuJoCo.LibMuJoCo.mjOBJ_JOINT, "rootz")
Qf[zidx, zidx] *= 50.0

R = 1e-7 * Matrix(I(m.nu))
costfunc = QuadraticCostFunction(Q, R, Qf)

# Declare parameters and options
params = TrajoptParameters(m, m, costfunc, Xref, Uref, xic)
opts = ILqrOptions(; maxiter_ilqr=1000, margin_ls=5e-2, tol_converge=1e-2)

# Let the trajpot JIT compile
opts.is_verbose = true
sol = fresh_solve(params, opts)

# Solve a second time after JIT compilation for accurate timing
#opts.is_verbose = true
#sol = fresh_solve(params, opts; use_time=true)

# Visualize solution
println("\nFinal state: $(sol.X[end])\n")

traj = stack(sol.X)
init_visualiser()
visualise!(m, d; trajectories=traj)
