mutable struct TrajoptParameters{T<:AbstractFloat,Lk,Lf}
    cinterps::Union{ContactParameterInterpolations{T},Nothing}
    mfwd::MuJoCo.Model
    mbwd::MuJoCo.Model
    dfwd::MuJoCo.Data
    dbwd::MuJoCo.Data
    costfunc::TrajectoryCostFunction{T,Lk,Lf}
    Xref::Vector{Vector{T}}
    Uref::Vector{Vector{T}}
    xic::Vector{T}

    function TrajoptParameters{T}(
        mfwd::MuJoCo.Model,
        mbwd::MuJoCo.Model,
        costfunc_stage::Lk,
        costfunc_term::Lf,
        Xref::AbstractVector{<:AbstractVector{<:Real}},
        Uref::AbstractVector{<:AbstractVector{<:Real}},
        xic::AbstractVector{<:Real},
        num_interps::Integer=0,
        geom_name::String="",
    ) where {T,Lk,Lf}
        # Get problem dimensions
        nx = get_nx(mfwd)
        nu = mfwd.nu
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
        if mbwd.nu != mbwd.nu
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

        if num_interps > 0
            cbwd = ContactParameters{T}(geom_name, mbwd)
            cfwd = ContactParameters{T}(geom_name, mfwd)
            cinterps = ContactParameterInterpolations(cbwd, cfwd, num_interps)
        else
            cinterps = nothing
        end
        dfwd, dbwd = init_data(mfwd), init_data(mbwd)
        costfunc = TrajectoryCostFunction{T}(
            mfwd, costfunc_stage, costfunc_term
        )
        Xref_T = Vector{Vector{T}}(Xref)
        Uref_T = Vector{Vector{T}}(Uref)
        xic_T = Vector{T}(xic)
        return new{T,Lk,Lf}(
            cinterps, mfwd, mbwd, dfwd, dbwd, costfunc, Xref_T, Uref_T, xic_T
        )
    end
end

function TrajoptParameters(
    mfwd::MuJoCo.Model,
    mbwd::MuJoCo.Model,
    costfunc_quad::L,
    Xref::AbstractVector{<:AbstractVector{<:Real}},
    Uref::AbstractVector{<:AbstractVector{<:Real}},
    xic::AbstractVector{<:Real},
    num_interps::Integer=0,
    geom_name::String="",
)::TrajoptParameters{T,L,L} where {T,L<:QuadraticCostFunction{T}}
    return TrajoptParameters{T}(
        mfwd,
        mbwd,
        costfunc_quad,
        costfunc_quad,
        Xref,
        Uref,
        xic,
        num_interps,
        geom_name,
    )
end

mutable struct TrajoptSolution{T<:AbstractFloat}
    X::Vector{Vector{T}}
    U::Vector{Vector{T}}
    J::T
    is_optimal::Bool

    function TrajoptSolution(params::TrajoptParameters{T,Lk,Lf}) where {T,Lk,Lf}
        # Get problem dims
        nx = get_nx(params.mfwd)
        nu = params.mfwd.nu
        N = length(params.Xref)

        # Initialize solution terms from dims
        X = [zeros(T, nx) for k in 1:N]
        U = [zeros(T, nu) for k in 1:(N - 1)]
        J = zero(T)
        is_optimal = false
        return new{T}(X, U, J, is_optimal)
    end
end

mutable struct ILqrCache{T<:AbstractFloat}
    fwd::ForwardCache{T}
    bwd::BackwardCache{T}
    tmp::TemporaryCache{T}

    function ILqrCache(params::TrajoptParameters{T,Lk,Lf}) where {T,Lk,Lf}
        # Get problem dims
        nx = get_nx(params.mfwd)
        ndx = get_ndx(params.mfwd)
        nu = params.mfwd.nu
        N = length(params.Xref)

        # Initialize caches from dims
        fwd = ForwardCache{T}(nx, nu, N)
        bwd = BackwardCache{T}(ndx, nu, N)
        tmp = TemporaryCache{T}(nx, ndx, nu)
        return new{T}(fwd, bwd, tmp)
    end
end

@option struct DefaultILqrOptions{T<:AbstractFloat}
    alpha_mul::T
    margin_ls::T
    eps_reg::T
    eps_fd::T
    tol_converge::T
    maxiter_ilqr::Int
    maxiter_ls::Int
    is_verbose::Bool
end

mutable struct ILqrOptions{T<:AbstractFloat}
    alpha_mul::T
    margin_ls::T
    eps_reg::T
    eps_fd::T
    tol_converge::T
    maxiter_ilqr::Int
    maxiter_ls::Int
    is_verbose::Bool

    function ILqrOptions{T}(;
        alpha_mul::Union{<:AbstractFloat,Nothing}=nothing,
        margin_ls::Union{<:AbstractFloat,Nothing}=nothing,
        eps_reg::Union{<:AbstractFloat,Nothing}=nothing,
        eps_fd::Union{<:AbstractFloat,Nothing}=nothing,
        tol_converge::Union{<:AbstractFloat,Nothing}=nothing,
        maxiter_ilqr::Union{Int,Nothing}=nothing,
        maxiter_ls::Union{Int,Nothing}=nothing,
        is_verbose::Union{Bool,Nothing}=nothing,
    )::ILqrOptions{T} where {T}
        # Load default options from config
        default = from_toml(
            DefaultILqrOptions{T},
            joinpath(@__DIR__, "config/default_opts.toml"),
        )

        # Use default options if the corresponding option is nothing
        alpha_mul_ = isnothing(alpha_mul) ? default.alpha_mul : T(alpha_mul)
        margin_ls_ = isnothing(margin_ls) ? default.margin_ls : T(margin_ls)
        eps_reg_ = isnothing(eps_reg) ? default.eps_reg : T(eps_reg)
        eps_fd_ = isnothing(eps_fd) ? default.eps_fd : T(eps_fd)
        tol_converge_ =
            isnothing(tol_converge) ? default.tol_converge : T(tol_converge)
        maxiter_ilqr_ =
            isnothing(maxiter_ilqr) ? default.maxiter_ilqr : maxiter_ilqr
        maxiter_ls_ = isnothing(maxiter_ls) ? default.maxiter_ls : maxiter_ls
        is_verbose_ = isnothing(is_verbose) ? default.is_verbose : is_verbose
        return new{T}(
            alpha_mul_,
            margin_ls_,
            eps_reg_,
            eps_fd_,
            tol_converge_,
            maxiter_ilqr_,
            maxiter_ls_,
            is_verbose_,
        )
    end
end

# Default type parameter
ILqrOptions(; args...) = ILqrOptions{T_DEFAULT}(; args...)
