module MjContactImplicit

using LinearAlgebra
using StaticArrays
using SparseArrays
using ForwardDiff
using DiffResults
using Printf
using Plots
import Base: ==, hash

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
include("solver/caches/backward.jl")
include("solver/caches/forward.jl")
include("solver/caches/temporary.jl")
include("solver/interface.jl")
include("solver/backward_pass.jl")
include("solver/forward_pass.jl")
include("solver/main.jl")

end # module MjContactImplicit
