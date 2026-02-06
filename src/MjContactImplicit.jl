module MjContactImplicit

using MuJoCo
using LinearAlgebra
using FastLapackInterface
using ForwardDiff
using DiffResults
using PreallocationTools
using Printf
using Plots
using Configurations

export TrajoptParameters,
    TrajoptSolution, ILqrCache, ILqrOptions, run_ilqr!, run_ilqr, plot_2d_states

include("utils/array.jl")
include("plot.jl")
include("costfunc.jl")
include("ilqr/config/consts.jl")
include("ilqr/caches/backward.jl")
include("ilqr/caches/forward.jl")
include("ilqr/caches/temporary.jl")
include("ilqr/interface.jl")
include("ilqr/backward_pass.jl")
include("ilqr/forward_pass.jl")
include("ilqr/solve.jl")

end # module MjContactImplicit
