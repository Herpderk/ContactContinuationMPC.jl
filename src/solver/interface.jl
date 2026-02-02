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
    Xref::AbstractVector{V},
    Uref::AbstractVector{V},
    xic::V,
)::ProblemParameters{T} where {T,V<:AbstractVector{<:Real}}
    # Get problem dimensions
    nx = length(Xref[1])
    nu = length(Uref[1])
    N = length(Xref)

    # Assert dimensions
    if length(Uref) != N-1
        throw(
            DimensionMismatch(
                "Number of reference inputs should be 1 less than number of reference states",
            ),
        )
    end
    if length(xic) != nx
        throw(
            DimensionMismatch(
                "Initial conditions dimensions do not match those of reference states",
            ),
        )
    end
    for xref in Xref
        if length(xref) != nx
            throw(
                DimensionMismatch(
                    "Reference state dimensions are not consistent",
                ),
            )
        end
    end
    for uref in Uref
        if length(uref) != nu
            throw(
                DimensionMismatch(
                    "Reference input dimensions are not consistent",
                ),
            )
        end
    end

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

function Solution{T}(params::ProblemParameters{T})::SolverCache{T} where {T}
    # Get problem dims
    nx = length(params.Xref[1])
    nu = length(params.Uref[1])
    N = length(params.Xref)

    # Init solution terms from dims
    X = [zeros(T, nx) for k = 1:N]
    U = [zeros(T, nu) for k = 1:(N-1)]
    J = T(0.0)
    return Solution{T}(X, U, J)
end

# Default type parameter
Solution(args...) = Solution{DEFAULT_DTYPE}(args...)



mutable struct SolverCache{T<:AbstractFloat}
    fwd::ForwardCache{T}
    bwd::BackwardCache{T}
    tmp::TemporaryCache{T}
end

function SolverCache{T}(params::ProblemParameters{T})::SolverCache{T} where {T}
    # Get problem dims
    nx = length(params.Xref[1])
    nu = length(params.Uref[1])
    N = length(params.Xref)

    # Init caches from dims
    fwd = ForwardCache{T}(nx, nu, N)
    bwd = BackwardCache{T}(nx, nu, N)
    tmp = TemporaryCache(nx, nu)
    return SolverCache{T}(fwd, bwd, tmp)
end

# Default type parameter
SolverCache(args...) = SolverCache{DEFAULT_DTYPE}(args...)



@option mutable struct SolverOptions{T<:AbstractFloat}
    eps_reg::T
    tol_converge::T
    maxiter_solve::Int
    maxiter_ls::Int
    is_verbose::Bool
end

function SolverOptions{T}(
    eps_reg::AbstractFloat,
    tol_converge::AbstractFloat,
    maxiter_solve::Int,
    maxiter_ls::Int,
    is_verbose::Bool,
)::SolverOptions{T} where {T}
    return SolverOptions{T}(
        T(eps_reg),
        T(tol_converge),
        maxiter_solve,
        maxiter_ls,
        is_verbose,
    )
end

function SolverOptions{T}(;
    eps_reg::Union{AbstractFloat,Nothing} = nothing,
    tol_converge::Union{AbstractFloat,Nothing} = nothing,
    maxiter_solve::Union{Int,Nothing} = nothing,
    maxiter_ls::Union{Int,Nothing} = nothing,
    is_verbose::Union{Bool,Nothing} = nothing,
)::SolverOptions{T} where {T}
    # Load default options from config
    default =
        from_toml(SolverOptions, joinpath(@__DIR__, "config/default_opts.toml"))

    # Use default options if the corresponding option is nothing
    eps_reg_ = isnothing(eps_reg) ? default.eps_reg : eps_reg
    tol_converge_ =
        isnothing(tol_converge) ? default.tol_converge : tol_converge
    maxiter_solve_ =
        isnothing(maxiter_solve) ? default.maxiter_solve : maxiter_solve
    maxiter_ls_ = isnothing(maxiter_ls) ? default.maxiter_ls : maxiter_ls
    is_verbose_ = isnothing(is_verbose) ? default.is_verbose : is_verbose
    return SolverOptions{T}(
        eps_reg_,
        tol_converge_,
        maxiter_solve_,
        maxiter_ls_,
        is_verbose_,
    )
end

# Default type parameter
SolverOptions(args...) = SolverOptions{DEFAULT_DTYPE}(args...)
