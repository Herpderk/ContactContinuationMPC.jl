module Interface

using MuJoCo
using LinearAlgebra
using ForwardDiff
using PreallocationTools
using ..Utils

export QuadraticCostFunction,
    TrajectoryCostFunction, TrajoptParameters, TrajoptSolution, T_DEFAULT

include("consts.jl")
include("costfunc.jl")
include("interface.jl")

end # module Interface
