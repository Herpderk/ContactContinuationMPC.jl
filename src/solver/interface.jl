mutable struct ProblemParameters{T<:AbstractFloat=Float64}
    simfwd::Function
    simbwd_A::Function
    simbwd_B::Function
    costfunc::TrajectoryCostFunction{T}
    Xref::Vector{Vector{T}}
    Uref::Vector{Vector{T}}
    xic::Vector{T}
    dt::T
end

function ProblemParameters{T}(
    simfwd::Function,
    simbwd_A::Function,
    simbwd_B::Function,
    costfunc_stage::Function,
    costfunc_term::Function,
    Xref::AbstractVector{<:AbstractVector{<:Real}},
    Uref::AbstractVector{<:AbstractVector{<:Real}},
    xic::AbstractVector{<:AbstractVector{<:Real}},
    dt::Real,
)::ProblemParameters{T} where {T}
    # Get problem dimensions
    nx = length(Xref[1])
    nu = length(Uref[1])
    N = length(Xref)

    costfunc = TrajectoryCostFunction{T}(costfunc_stage, costfunc_term, nx, nu, N)
    Xref_T = Vector{Vector{T}}(Xref)
    Uref_T = Vector{Vector{T}}(Uref)
    xic_T = Vector{T}(xic)
    dt_T = T(dt)
    return ProblemParameters{T}(
        simfwd,
        simbwd_A,
        simbwd_B,
        costfunc,
        Xref_T,
        Uref_T,
        xic_T,
        dt_T,
    )
end



mutable struct Solution{T<:AbstractFloat=Float64}
    X::Vector{Vector{T}}
    U::Vector{Vector{T}}
    J::T
end

function Solution{T}(nx::Int, nu::Int, N::Int)::Solution{T} where {T}
    X = [zeros(T, nx) for k ∈ 1:N]
    U = [zeros(T, nu) for k ∈ 1:(N-1)]
    J = T(0.0)
    return Solution{T}(X, U, J)
end

function Solution{T}(params::ProblemParameters{T})::SolverCache{T} where {T}
    nx = length(params.Xref[1])
    nu = length(params.Uref[1])
    N = length(params.Xref)
    return Solution{T}(nx, nu, N)
end



mutable struct SolverCache{T<:AbstractFloat=Float64}
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



mutable struct SolverOptions{T<:AbstractFloat=Float64}
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
