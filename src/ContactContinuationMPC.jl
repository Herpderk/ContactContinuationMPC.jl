module ContactContinuationMPC

using MuJoCo
using OSQP
using SparseArrays
using LinearAlgebra
#using FastLapackInterface
using ForwardDiff
using DiffResults
using PreallocationTools
using Printf
using Plots
using Configurations
using Base.Threads

export QuadraticCostFunction,
    TrajectoryCostFunction,
    TrajoptParameters,
    TrajoptSolution,
    ILqrCache,
    ILqrOptions,
    run_ilqr!,
    run_ilqr,
    get_nx,
    get_ndx,
    get_q,
    get_v,
    get_a,
    get_dq,
    get_dv,
    get_da,
    get_state_diff!,
    copy_data_to_state!,
    copy_state_to_data!,
    get_joint_names

include("utils/exceptions.jl")
include("utils/array.jl")
include("utils/interpolate.jl")
include("utils/mujoco/dims.jl")
include("utils/mujoco/names.jl")
include("utils/mujoco/state.jl")
include("utils/mujoco/finitediff.jl")
include("plot.jl")
include("consts.jl")
include("costfunc.jl")
include("trajopt_interface.jl")
include("ilqr/caches/backward.jl")
include("ilqr/caches/forward.jl")
include("ilqr/caches/temporary.jl")
include("ilqr/contact_params.jl")
include("ilqr/interface.jl")
include("ilqr/backward_pass.jl")
include("ilqr/forward_pass.jl")
include("ilqr/solve.jl")
include("sqp/indexing.jl")
include("sqp/sparsity.jl")
include("sqp/interface.jl")
include("sqp/qp_prep.jl")
include("sqp/solve.jl")

end # module ContactContinuationMPC
