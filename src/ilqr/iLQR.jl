module iLQR

using MuJoCo
using LinearAlgebra
using ForwardDiff
using DiffResults
using Configurations
using Printf
using CompositeStructs
using Base.Threads
using Base.Iterators: flatten
using ..Utils
using ..Interface

export iLQRCache, iLQROptions, run_al_ilqr!, run_al_ilqr

include("constraints/abstract.jl")
include("constraints/equality/equality.jl")
include("constraints/inequality/inequality.jl")
include("constraints/inequality/control_bound.jl")
include("caches/backward.jl")
include("caches/forward.jl")
include("caches/temporary.jl")
include("contact_params.jl")
include("interface.jl")
include("backward_pass.jl")
include("forward_pass.jl")
include("assert.jl")
include("log.jl")
include("solve.jl")

end # module iLQR
