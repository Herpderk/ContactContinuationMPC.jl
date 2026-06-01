module iLQR

using MuJoCo
using LinearAlgebra
using ForwardDiff
using DiffResults
using Configurations
using Printf
using CompositeStructs
using Base.Threads
using ..Utils
using ..Interface

export iLQRCache, iLQROptions, run_ilqr!, run_ilqr

include("caches/constraint.jl")
include("caches/backward.jl")
include("caches/forward.jl")
include("caches/temporary.jl")
include("contact_params.jl")
include("interface.jl")
include("constraint.jl")
include("backward_pass.jl")
include("forward_pass.jl")
include("assert.jl")
include("log.jl")
include("solve.jl")

end # module iLQR
