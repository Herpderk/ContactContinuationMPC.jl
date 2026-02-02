mutable struct ProblemParameters{T<:AbstractFloat}
    simfunc_fwd!::Function      # expects simfunc_fwd!(x_next, x_curr, u_curr)
    simfunc_bwd!::Function      # expects simfunc_bwd!(A, B, x, u)::Nothing
    costfunc::TrajectoryCostFunction{T}
    Xref::Vector{Vector{T}}
    Uref::Vector{Vector{T}}
    xic::Vector{T}
end

function ProblemParameters{T}(
    simfunc_fwd!::Function,
    simfunc_bwd!::Function,
    costfunc_stage::Function,
    costfunc_term::Function,
    Xref::AbstractVector{<:AbstractVector{<:Real}},
    Uref::AbstractVector{<:AbstractVector{<:Real}},
    xic::AbstractVector{<:AbstractVector{<:Real}},
)::ProblemParameters{T} where {T}
    # Get problem dimensions
    nx = length(Xref[1])
    nu = length(Uref[1])
    N = length(Xref)

    costfunc =
        TrajectoryCostFunction{T}(costfunc_stage, costfunc_term, nx, nu, N)
    Xref_T = Vector{Vector{T}}(Xref)
    Uref_T = Vector{Vector{T}}(Uref)
    xic_T = Vector{T}(xic)
    return ProblemParameters{T}(
        simfunc_fwd!,
        simfunc_bwd!,
        costfunc,
        Xref_T,
        Uref_T,
        xic_T,
    )
end

# Default type parameter
ProblemParameters(args...) = ProblemParameters{DEFAULT_DTYPE}(args...)



mutable struct Solution{T<:AbstractFloat}
    X::Vector{Vector{T}}
    U::Vector{Vector{T}}
    J::T
end

function Solution{T}(nx::Int, nu::Int, N::Int)::Solution{T} where {T}
    X = [zeros(T, nx) for k = 1:N]
    U = [zeros(T, nu) for k = 1:(N-1)]
    J = T(0.0)
    return Solution{T}(X, U, J)
end

function Solution{T}(params::ProblemParameters{T})::SolverCache{T} where {T}
    nx = length(params.Xref[1])
    nu = length(params.Uref[1])
    N = length(params.Xref)
    return Solution{T}(nx, nu, N)
end

# Default type parameter
Solution(args...) = Solution{DEFAULT_DTYPE}(args...)



mutable struct SolverCache{T<:AbstractFloat}
    fwd::ForwardCache{T}
    bwd::BackwardCache{T}
    tmp::TemporaryCache{T}
end

function SolverCache{T}(nx::Int, nu::Int, N::Int)::SolverCache{T} where {T}
    fwd = ForwardCache{T}(nx, nu, N)
    bwd = BackwardCache{T}(nx, nu, N)
    tmp = TemporaryCache(nx, nu)
    return SolverCache{T}(fwd, bwd, tmp)
end

function SolverCache{T}(params::ProblemParameters{T})::SolverCache{T} where {T}
    nx = length(params.Xref[1])
    nu = length(params.Uref[1])
    N = length(params.Xref)
    return SolverCache{T}(nx, nu, N)
end

# Default type parameter
SolverCache(args...) = SolverCache{DEFAULT_DTYPE}(args...)



mutable struct SolverOptions{T<:AbstractFloat}
    eps_reg::T
    tol_converge::T
    maxiter_solve::Int
    maxiter_ls::Int
    is_verbose::Bool
end

function SolverOptions{T}(;
    eps_reg::AbstractFloat = 1e-6,
    tol_converge::AbstractFloat = 1e-9,
    maxiter_solve::Int = 100,
    maxiter_ls::Int = 20,
    is_verbose::Bool = true,
)::SolverOptions{T} where {T}
    return SolverOptions{T}(
        T(eps_reg),
        T(tol_converge),
        maxiter_solve,
        maxiter_ls,
        is_verbose,
    )
end

# Default type parameter
SolverOptions(args...) = SolverOptions{DEFAULT_DTYPE}(args...)
