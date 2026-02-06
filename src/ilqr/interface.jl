mutable struct TrajoptParameters{T<:AbstractFloat,Lk,Lf}
    mfwd::MuJoCo.Model
    mbwd::MuJoCo.Model
    dfwd::MuJoCo.Data
    dbwd::MuJoCo.Data
    costfunc::TrajectoryCostFunction{T,Lk,Lf}
    Xref::Vector{Vector{T}}
    Uref::Vector{Vector{T}}
    xic::Vector{T}
end

function TrajoptParameters{T,Lk,Lf}(
    mfwd::MuJoCo.Model,
    mbwd::MuJoCo.Model,
    dfwd::MuJoCo.Data,
    dbwd::MuJoCo.Data,
    costfunc_stage::Lk,
    costfunc_term::Lf,
    Xref::AbstractVector{<:AbstractVector{<:Real}},
    Uref::AbstractVector{<:AbstractVector{<:Real}},
    xic::AbstractVector{<:Real},
) where {T<:AbstractFloat,Lk,Lf}
    # Get problem dimensions
    nx = get_nx(mfwd)
    nu = m.nu
    N = length(Xref)

    # Assert dimensions
    if mfwd.nq != mbwd.nq
        throw(
            DimensionMismatch(
                "Forward and backward models do not match in configuration dimensions",
            ),
        )
    end
    if mfwd.nv != mbwd.nv
        throw(
            DimensionMismatch(
                "Forward and backward models do not match in velocity dimensions",
            ),
        )
    end
    if mfwd.na != mbwd.na
        throw(
            DimensionMismatch(
                "Forward and backward models do not match in actuator dimensions",
            ),
        )
    end
    if mbwd.nu != nu
        throw(
            DimensionMismatch(
                "Forward and backward models do not match in control input dimensions",
            ),
        )
    end
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

    costfunc = TrajectoryCostFunction{T,Lk,Lf}(
        mfwd, costfunc_stage, costfunc_term
    )
    Xref_T = Vector{Vector{T}}(Xref)
    Uref_T = Vector{Vector{T}}(Uref)
    xic_T = Vector{T}(xic)
    return TrajoptParameters{T,Lk,Lf}(
        mfwd, mbwd, dfwd, dbwd, costfunc, Xref_T, Uref_T, xic_T
    )
end

mutable struct TrajoptSolution{T<:AbstractFloat}
    X::Vector{Vector{T}}
    U::Vector{Vector{T}}
    J::T
    is_optimal::Bool
end

function TrajoptSolution{T}(
    params::TrajoptParameters{T,Lk,Lf}
)::TrajoptSolution{T} where {T<:AbstractFloat,Lk,Lf}
    # Get problem dims
    nx = get_nx(params.mfwd)
    nu = params.mfwd.nu
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
    params::TrajoptParameters{T,Lk,Lf}
)::ILqrCache{T} where {T<:AbstractFloat,Lk,Lf}
    # Get problem dims
    nx = get_nx(params.mfwd)
    ndx = get_ndx(params.mfwd)
    nu = params.mfwd.nu
    N = length(params.Xref)

    # Initialize caches from dims
    fwd = ForwardCache{T}(nx, nu, N)
    bwd = BackwardCache{T}(ndx, nu, N)
    tmp = TemporaryCache{T}(nx, ndx, nu)
    return ILqrCache{T}(fwd, bwd, tmp)
end

# Default type parameter
ILqrCache(args...) = ILqrCache{DEFAULT_DTYPE}(args...)

@option struct DefaultILqrOptions{T<:AbstractFloat}
    alpha_mul::T
    eps_reg::T
    tol_converge::T
    maxiter_ilqr::Int
    maxiter_ls::Int
    is_verbose::Bool
end

mutable struct ILqrOptions{T<:AbstractFloat}
    alpha_mul::T
    eps_reg::T
    eps_fd::T
    tol_converge::T
    maxiter_ilqr::Int
    maxiter_ls::Int
    is_verbose::Bool
end

function ILqrOptions{T}(;
    alpha_mul::Union{<:AbstractFloat,Nothing}=nothing,
    eps_reg::Union{<:AbstractFloat,Nothing}=nothing,
    eps_fd::Union{<:AbstractFloat,Nothing}=nothing,
    tol_converge::Union{<:AbstractFloat,Nothing}=nothing,
    maxiter_ilqr::Union{Int,Nothing}=nothing,
    maxiter_ls::Union{Int,Nothing}=nothing,
    is_verbose::Union{Bool,Nothing}=nothing,
)::ILqrOptions{T} where {T<:AbstractFloat}
    # Load default options from config
    default = fromtoml(
        DefaultILqrOptions{T}, joinpath(@__DIR__, "config/default_opts.toml")
    )

    # Use default options if the corresponding option is nothing
    alpha_mul_ = isnothing(alpha_mul) ? default.alpha_mul : T(alpha_mul)
    eps_reg_ = isnothing(eps_reg) ? default.eps_reg : T(eps_reg)
    eps_fd_ = isnothing(eps_fd) ? default.eps_fd : T(eps_fd)
    tol_converge_ =
        isnothing(tol_converge) ? default.tol_converge : T(tol_converge)
    maxiter_ilqr_ =
        isnothing(maxiter_ilqr) ? default.maxiter_ilqr : maxiter_ilqr
    maxiter_ls_ = isnothing(maxiter_ls) ? default.maxiter_ls : maxiter_ls
    is_verbose_ = isnothing(is_verbose) ? default.is_verbose : is_verbose
    return ILqrOptions{T}(
        alpha_mul_,
        eps_reg_,
        eps_fd_,
        tol_converge_,
        maxiter_ilqr_,
        maxiter_ls_,
        is_verbose_,
    )
end

# Default type parameter
ILqrOptions(args...) = ILqrOptions{DEFAULT_DTYPE}(args...)
