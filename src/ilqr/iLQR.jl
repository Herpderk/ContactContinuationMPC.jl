module iLQR

using MuJoCo
using LinearAlgebra
using ForwardDiff
using DiffResults
using Configurations
using Printf
using Base.Threads
using ..Utils
using ..Interface

include("caches/backward.jl")
include("caches/forward.jl")
include("caches/temporary.jl")
include("contact_params.jl")
include("interface.jl")
include("backward_pass.jl")
include("forward_pass.jl")
include("solve.jl")

export iLQRCache, iLQROptions, run_ilqr!, run_ilqr

end # module iLQR
