struct iLQRCache{T<:AbstractFloat}
    constr::ConstraintCache{T}
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
        constr = ConstraintCache{T}(N, nu)
        fwd = ForwardCache{T}(nx, nu, N)
        bwd = BackwardCache{T}(params.mbwd, ndx, nu, N)
        tmp = TemporaryCache{T}(nx, ndx, nu)
        return new{T}(constr, fwd, bwd, tmp)
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

mutable struct iLQROptions{T<:AbstractFloat}
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

    function iLQROptions{T}(;
        rho_init::Union{<:AbstractFloat,Nothing}=nothing,
        rho_mul::Union{<:AbstractFloat,Nothing}=nothing,
        alpha_mul::Union{<:AbstractFloat,Nothing}=nothing,
        margin_ls::Union{<:AbstractFloat,Nothing}=nothing,
        eps_reg::Union{<:AbstractFloat,Nothing}=nothing,
        eps_fd::Union{<:AbstractFloat,Nothing}=nothing,
        tol_interp::Union{<:AbstractFloat,Nothing}=nothing,
        tol_constr::Union{<:AbstractFloat,Nothing}=nothing,
        tol_ilqr::Union{<:AbstractFloat,Nothing}=nothing,
        maxiter_al::Union{Int,Nothing}=nothing,
        maxiter_ilqr::Union{Int,Nothing}=nothing,
        maxiter_ls::Union{Int,Nothing}=nothing,
        is_verbose::Union{Bool,Nothing}=nothing,
        save_bestsol::Union{Bool,Nothing}=nothing,
    ) where {T}
        # Load default options from config
        default = from_toml(
            DefaultiLQROptions{T},
            joinpath(@__DIR__, "config/default_opts.toml"),
        )

        # Use default options if the corresponding option is nothing
        rho_init_ = isnothing(rho_init) ? default.rho_init : T(rho_init)
        rho_mul_ = isnothing(rho_mul) ? default.rho_mul : T(rho_mul)
        alpha_mul_ = isnothing(alpha_mul) ? default.alpha_mul : T(alpha_mul)
        margin_ls_ = isnothing(margin_ls) ? default.margin_ls : T(margin_ls)
        eps_reg_ = isnothing(eps_reg) ? default.eps_reg : T(eps_reg)
        eps_fd_ = isnothing(eps_fd) ? default.eps_fd : T(eps_fd)
        tol_interp_ = isnothing(tol_interp) ? default.tol_interp : T(tol_interp)
        tol_constr_ = isnothing(tol_constr) ? default.tol_constr : T(tol_constr)
        tol_ilqr_ = isnothing(tol_ilqr) ? default.tol_ilqr : T(tol_ilqr)
        maxiter_al_ = isnothing(maxiter_al) ? default.maxiter_al : maxiter_al
        maxiter_ilqr_ =
            isnothing(maxiter_ilqr) ? default.maxiter_ilqr : maxiter_ilqr
        maxiter_ls_ = isnothing(maxiter_ls) ? default.maxiter_ls : maxiter_ls
        is_verbose_ = isnothing(is_verbose) ? default.is_verbose : is_verbose
        save_bestsol_ =
            isnothing(save_bestsol) ? default.save_bestsol : save_bestsol
        return new{T}(
            rho_init_,
            rho_mul_,
            alpha_mul_,
            margin_ls_,
            eps_reg_,
            eps_fd_,
            tol_interp_,
            tol_constr_,
            tol_ilqr_,
            maxiter_al_,
            maxiter_ilqr_,
            maxiter_ls_,
            is_verbose_,
            save_bestsol_,
        )
    end
end

# Default type parameter
iLQROptions(; args...) = iLQROptions{T_DEFAULT}(; args...)
