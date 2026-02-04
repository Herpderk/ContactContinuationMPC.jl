module MjContactImplicit

using LinearAlgebra
using StaticArrays
using SparseArrays
using StructArrays
using ForwardDiff
using DiffResults
using PreallocationTools
using Printf
using Plots
using Configurations

#= export
		ProblemParameters,
		Solution,
		SolverCache,
		SolverOptions,
		solve!,
		solve,
		plot_2d_states =#

include("plot.jl")
include("costfunc.jl")
include("solver/config/consts.jl")
include("solver/caches/backward.jl")
include("solver/caches/forward.jl")
include("solver/caches/temporary.jl")
include("solver/interface.jl")
include("solver/backward_pass.jl")
include("solver/forward_pass.jl")
include("solver/solve_loop.jl")

end # module MjContactImplicit
