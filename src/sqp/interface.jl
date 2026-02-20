mutable struct SQPCache
    m::OSQP.Model
    r::OSQP.Results
    pidx::IndexingParameters
    ∇²L::SparseMatrixCSC{Float64,Int}
    ∇²Jxx::DiffResults.DiffResult{
        2,Float64,Tuple{Vector{Float64},Matrix{Float64}}
    }
    ∇²Juu::DiffResults.DiffResult{
        2,Float64,Tuple{Vector{Float64},Matrix{Float64}}
    }
    ∇J::Vector{Float64}
    ∇h::SparseMatrixCSC{Float64,Int}
    h::Vector{Float64}
    λ::Vector{Float64}
    z::Vector{Float64}
    dz::Vector{Float64}
    xtmp::Vector{Float64}
    dxtmp::Vector{Float64}
    utmp::Vector{Float64}
    FDs::Vector{Utils.FDCache{Float64}}

    function SQPCache(params::TrajoptParameters{T,Lk,Lf}) where {T,Lk,Lf}
        # Get problem dims
        N = length(params.Xref)
        nx = Utils.get_nx(params.mfwd)
        ndx = Utils.get_ndx(params.mfwd)
        nu = params.mfwd.nu

        # Initialize indexing parameters
        pidx = IndexingParameters(N, nx, ndx, nu)
        nz, ndz, nh = pidx.dims.nz, pidx.dims.ndz, pidx.dims.nh

        # Initialize caches from dims
        ∇²L = sparse(costfunc_hessian_pattern(pidx))
        ∇²Jxx = DiffResults.HessianResult(zeros(Float64, nx))
        ∇²Juu = DiffResults.HessianResult(zeros(Float64, nu))
        ∇J = zeros(Float64, ndz)
        ∇h = sparse(equality_jacobian_pattern(pidx))
        h = zeros(Float64, nh)
        λ = zeros(Float64, nh)
        z = zeros(Float64, nz)
        dz = zeros(Float64, ndz)
        xtmp = zeros(Float64, nx)
        dxtmp = zeros(Float64, ndx)
        utmp = zeros(Float64, nu)
        FDs = [Utils.FDCache{64}(m) for t in 1:nthreads()]

        # Initialize OSQP model
        m = OSQP.Model()
        r = OSQP.Results()
        OSQP.setup!(m; P=∇²L, q=∇J, l=zeros(Float64, nh), u=zeros(Float64, nh))
        return new(
            m,
            r,
            idx,
            ∇²L,
            ∇²Jxx,
            ∇²Juu,
            ∇J,
            ∇h,
            h,
            λ,
            z,
            dz,
            xtmp,
            dxtmp,
            utmp,
            FDs,
        )
    end
end
#=
@option struct DefaultQPOptions
    alpha_mul::Float64
    margin_ls::Float64
    eps_reg::Float64
    eps_fd::Float64
    tol_interp::Float64
    tol_converge::Float64
    maxiter_ilqr::Int
    maxiter_ls::Int
    is_verbose::Bool
    save_bestsol::Bool
end

mutable struct QPOptions
    alpha_mul::Float64
    margin_ls::Float64
    eps_reg::Float64
    eps_fd::Float64
    tol_interp::Float64
    tol_converge::T
    maxiter_ilqr::Int
    maxiter_ls::Int
    is_verbose::Bool
    save_bestsol::Bool

    function QPOptions(;
        alpha_mul::Union{<:AbstractFloat,Nothing}=nothing,
        margin_ls::Union{<:AbstractFloat,Nothing}=nothing,
        eps_reg::Union{<:AbstractFloat,Nothing}=nothing,
        eps_fd::Union{<:AbstractFloat,Nothing}=nothing,
        tol_interp::Union{<:AbstractFloat,Nothing}=nothing,
        tol_converge::Union{<:AbstractFloat,Nothing}=nothing,
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
        alpha_mul_ = isnothing(alpha_mul) ? default.alpha_mul : T(alpha_mul)
        margin_ls_ = isnothing(margin_ls) ? default.margin_ls : T(margin_ls)
        eps_reg_ = isnothing(eps_reg) ? default.eps_reg : T(eps_reg)
        eps_fd_ = isnothing(eps_fd) ? default.eps_fd : T(eps_fd)
        tol_interp_ = isnothing(tol_interp) ? default.tol_interp : T(tol_interp)
        tol_converge_ =
            isnothing(tol_converge) ? default.tol_converge : T(tol_converge)
        maxiter_ilqr_ =
            isnothing(maxiter_ilqr) ? default.maxiter_ilqr : maxiter_ilqr
        maxiter_ls_ = isnothing(maxiter_ls) ? default.maxiter_ls : maxiter_ls
        is_verbose_ = isnothing(is_verbose) ? default.is_verbose : is_verbose
        save_bestsol_ =
            isnothing(save_bestsol) ? default.save_bestsol : save_bestsol
        return new{T}(
            alpha_mul_,
            margin_ls_,
            eps_reg_,
            eps_fd_,
            tol_interp_,
            tol_converge_,
            maxiter_ilqr_,
            maxiter_ls_,
            is_verbose_,
            save_bestsol_,
        )
    end
end

# Default type parameter
iLQROptions(; args...) = iLQROptions{T_DEFAULT}(; args...) =#
