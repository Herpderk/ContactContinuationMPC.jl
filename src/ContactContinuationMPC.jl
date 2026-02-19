module ContactContinuationMPC

using MuJoCo
using Plots
using Reexport

export QuadraticCostFunction,
    TrajectoryCostFunction,
    TrajoptParameters,
    TrajoptSolution,
    iLQRCache,
    iLQROptions,
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
    get_joint_names,
    get_geom_names

include("plot.jl")
include("utils/Utils.jl")
include("interface/Interface.jl")
include("ilqr/iLQR.jl")
include("sqp/SQP.jl")

@reexport using .Utils
@reexport using .Interface
@reexport using .iLQR
@reexport using .SQP

end # module ContactContinuationMPC
