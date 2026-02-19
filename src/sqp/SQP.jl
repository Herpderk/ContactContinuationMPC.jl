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
using ..iLQR: log_converged, log_maxiter

include("indexing.jl")
include("sparsity.jl")
include("interface.jl")
include("qp_prep.jl")
include("solve.jl")

end
