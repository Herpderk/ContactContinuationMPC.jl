struct ALConstraints{T<:AbstractFloat}
    u::ConstraintSet{T}

    function ALConstraints{T}(
        params::TrajoptParameters{T,Lk,Lf}
    ) where {T,Lk,Lf}
        N, nu = length(params.Xref), params.mfwd.nu
        set_u = ConstraintSet(nu)

        # Lower control bound constraint
        lb = Utils.get_control_bound(params.mfwd, :lower)
        set_u[:ul] = ControlBoundConstraint{T}(N, nu, lb)

        # Upper control bound constraint
        ub = Utils.get_control_bound(params.mfwd, :upper)
        set_u[:uu] = ControlBoundConstraint{T}(N, nu, ub)
        return new{T}(set_u)
    end
end

struct iLQRCache{T<:AbstractFloat}
    fwd::ForwardCache{T}
    bwd::BackwardCache{T}
    tmp::TemporaryCache{T}

    function iLQRCache(params::TrajoptParameters{T,Lk,Lf}) where {T,Lk,Lf}
        # Get problem dims
        nx = Utils.get_nx(params.mfwd)
        ndx = Utils.get_ndx(params.mfwd)
        nu = params.mfwd.nu
        N = length(params.Xref)

        # Initialize caches from dims
        fwd = ForwardCache{T}(nx, nu, N)
        bwd = BackwardCache{T}(params.mbwd, ndx, nu, N)
        tmp = TemporaryCache{T}(nx, ndx, nu)
        return new{T}(fwd, bwd, tmp)
    end
end

@option struct DefaultiLQROptions{T<:AbstractFloat}
    rho_init::T
    rho_mul::T
    alpha_mul::T
    margin_ls::T
    eps_reg::T
    eps_fd::T
    tol_interp::T
    tol_constr::T
    tol_ilqr::T
    maxiter_al::Int
    maxiter_ilqr::Int
    maxiter_ls::Int
    is_verbose::Bool
    save_bestsol::Bool
end

@composite mutable struct iLQROptions{T<:AbstractFloat}
    DefaultiLQROptions{T}...

    function iLQROptions(; kwargs...)
        # Load default options from config
        default = from_toml(
            DefaultiLQROptions{T},
            joinpath(@__DIR__, "config/default_opts.toml"),
        )

        macro symbolof(var)
            return QuoteNode(var)
        end
        function default_opt(opt)
            return getfield(default, @symbolof(opt))
        end

        # Get the names of the fields in the struct
        fnames = fieldnames(DefaultiLQROptions{T})
        fvals = Tuple(
            get(kwargs, fname, default_opt(fname)) for fname in fnames
        )
        return new{T}(fvals...)
    end
end

# Default type parameter
iLQROptions(; args...) = iLQROptions{T_DEFAULT}(; args...)
