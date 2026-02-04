mutable struct TrajoptParameters{T<:AbstractFloat,S_fwd,S_bwd,C_stage,C_term}
    simfunc_fwd!::S_fwd      # expects simfunc_fwd!(x1, x0, u0)::Nothing
    simfunc_bwd!::S_bwd      # expects simfunc_bwd!(A, B, x1, x0, u0)::Nothing
    costfunc::TrajectoryCostFunction{T,C_stage,C_term}
    Xref::Vector{Vector{T}}
    Uref::Vector{Vector{T}}
    xic::Vector{T}
end

function TrajoptParameters{T,S_fwd,S_bwd,C_stage,C_term}(
    simfunc_fwd!::S_fwd,
    simfunc_bwd!::S_bwd,
    costfunc_stage::C_stage,
    costfunc_term::C_term,
    Xref::AbstractVector{<:AbstractVector{<:Real}},
    Uref::AbstractVector{<:AbstractVector{<:Real}},
    xic::AbstractVector{<:Real},
) where {T<:AbstractFloat,S_fwd,S_bwd,C_stage,C_term}
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
                    "Reference state dimensions are not consistent"
                ),
            )
        end
    end
    for uref in Uref
        if length(uref) != nu
            throw(
                DimensionMismatch(
                    "Reference input dimensions are not consistent"
                ),
            )
        end
    end

    costfunc = TrajectoryCostFunction{T,C_stage,C_term}(
        costfunc_stage, costfunc_term, nx, nu
    )
    Xref_T = Vector{Vector{T}}(Xref)
    Uref_T = Vector{Vector{T}}(Uref)
    xic_T = Vector{T}(xic)
    return TrajoptParameters{T,S_fwd,S_bwd,C_stage,C_term}(
        simfunc_fwd!, simfunc_bwd!, costfunc, Xref_T, Uref_T, xic_T
    )
end

mutable struct TrajoptSolution{T<:AbstractFloat}
    X::Vector{Vector{T}}
    U::Vector{Vector{T}}
    J::T
    is_optimal::Bool
end

function TrajoptSolution{T}(
    params::TrajoptParameters{T,S_fwd,S_bwd,C_stage,C_term}
)::TrajoptSolution{T} where {T<:AbstractFloat,S_fwd,S_bwd,C_stage,C_term}
    # Get problem dims
    nx = length(params.Xref[1])
    nu = length(params.Uref[1])
    N = length(params.Xref)

    # Initialize solution terms from dims
    X = [zeros(T, nx) for k in 1:N]
    U = [zeros(T, nu) for k in 1:(N - 1)]
    J = zero(T)
    is_optimal = false
    return TrajoptSolution{T}(X, U, J, is_optimal)
end

# Default type parameter
TrajoptSolution(args...) = TrajoptSolution{DEFAULT_DTYPE}(args...)

mutable struct ILqrCache{T<:AbstractFloat}
    fwd::ForwardCache{T}
    bwd::BackwardCache{T}
    tmp::TemporaryCache{T}
end

function ILqrCache{T}(
    params::TrajoptParameters{T,S_fwd,S_bwd,C_stage,C_term}
)::ILqrCache{T} where {T<:AbstractFloat,S_fwd,S_bwd,C_stage,C_term}
    # Get problem dims
    nx = length(params.Xref[1])
    nu = length(params.Uref[1])
    N = length(params.Xref)

    # Initialize caches from dims
    fwd = ForwardCache{T}(nx, nu, N)
    bwd = BackwardCache{T}(nx, nu, N)
    tmp = TemporaryCache{T}(nx, nu)
    return ILqrCache{T}(fwd, bwd, tmp)
end

# Default type parameter
ILqrCache(args...) = ILqrCache{DEFAULT_DTYPE}(args...)

@option struct DefaultILqrOptions{T<:AbstractFloat}
    eps_reg::T
    tol_converge::T
    maxiter_ilqr::Int
    maxiter_ls::Int
    is_verbose::Bool
end

mutable struct ILqrOptions{T<:AbstractFloat}
    eps_reg::T
    tol_converge::T
    maxiter_ilqr::Int
    maxiter_ls::Int
    is_verbose::Bool
end

function ILqrOptions{T}(;
    eps_reg::Union{<:AbstractFloat,Nothing}=nothing,
    tol_converge::Union{<:AbstractFloat,Nothing}=nothing,
    maxiter_ilqr::Union{Int,Nothing}=nothing,
    maxiter_ls::Union{Int,Nothing}=nothing,
    is_verbose::Union{Bool,Nothing}=nothing,
)::ILqrOptions{T} where {T<:AbstractFloat}
    # Load default options from config
    default = from_toml(
        DefaultILqrOptions{T}, joinpath(@__DIR__, "config/default_opts.toml")
    )

    # Use default options if the corresponding option is nothing
    eps_reg_ = isnothing(eps_reg) ? default.eps_reg : T(eps_reg)
    tol_converge_ =
        isnothing(tol_converge) ? default.tol_converge : T(tol_converge)
    maxiter_ilqr_ =
        isnothing(maxiter_ilqr) ? default.maxiter_ilqr : maxiter_ilqr
    maxiter_ls_ = isnothing(maxiter_ls) ? default.maxiter_ls : maxiter_ls
    is_verbose_ = isnothing(is_verbose) ? default.is_verbose : is_verbose
    return ILqrOptions{T}(
        eps_reg_, tol_converge_, maxiter_ilqr_, maxiter_ls_, is_verbose_
    )
end

# Default type parameter
ILqrOptions(args...) = ILqrOptions{DEFAULT_DTYPE}(args...)
