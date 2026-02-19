module Utils

using MuJoCo
using Base.Threads

export throwdim,
    throwarg,
    throwdom,
    null_length,
    copy_nested_array!,
    fill_nested_array!,
    interpolate,
    get_nx,
    get_ndx,
    get_q,
    get_v,
    get_a,
    get_dq,
    get_dv,
    get_da,
    same_dims,
    get_joint_names,
    get_geom_names,
    get_state_diff!,
    add_diff_to_state!,
    copy_data_to_state!,
    copy_state_to_data!,
    FDCache,
    threaded_fd!

include("exceptions.jl")
include("array.jl")
include("interpolate.jl")
include("mujoco/dims.jl")
include("mujoco/names.jl")
include("mujoco/state.jl")
include("mujoco/finitediff.jl")

end # module Utils
