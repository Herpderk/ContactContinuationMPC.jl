module SQP

using MuJoCo
using OSQP
using SparseArrays
using LinearAlgebra
using ForwardDiff
using DiffResults
using Configurations
using Printf
using Base.Threads
using ..Utils
using ..Interface

export SQPCache, SQPOptions, run_sqp!

include("indexing.jl")
include("sparsity.jl")
include("caches.jl")
include("interface.jl")
include("qp_prep.jl")
include("solve.jl")

end
