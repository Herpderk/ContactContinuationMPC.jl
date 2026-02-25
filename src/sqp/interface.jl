mutable struct SQPCache
    m::OSQP.Model
    r::OSQP.Results
    pidx::IndexingParameters
    ∇²ₓₓL::SparseMatrixCSC{Float64,Int}
    ∇²ₓₓJ::DiffResults.DiffResult
    ∇²ᵤᵤJ::DiffResults.DiffResult
    ∇²ₓᵤJ!_cfg::ForwardDiff.JacobianConfig
    ∇²ᵤₓJ!_cfg::ForwardDiff.JacobianConfig
    ∇²ₓᵤJ::Matrix{Float64}
    ∇²ᵤₓJ::Matrix{Float64}
    ∇ₓJ!_cfg::ForwardDiff.GradientConfig
    ∇ᵤJ!_cfg::ForwardDiff.GradientConfig
    ∇ₓJ!::Function
    ∇ᵤJ!::Function
    ∇ₓL::Vector{Float64}
    ∇J::Vector{Float64}
    ∇g::SparseMatrixCSC{Float64,Int}
    gl::Vector{Float64}
    gu::Vector{Float64}
    Δxl::Vector{Float64}
    Δxu::Vector{Float64}
    Δul::Vector{Float64}
    Δuu::Vector{Float64}
    z::Vector{Float64}
    ztmp::Vector{Float64}
    dztmp::Vector{Float64}
    xtmp::Vector{Float64}
    dxtmp::Vector{Float64}
    dxtmp_ad::Vector{Float64}
    utmp::Vector{Float64}
    utmp_ad::Vector{Float64}
    FDs::Vector{Utils.FDCache{Float64}}

    function SQPCache(
        params::TrajoptParameters{T,Lk,Lf};
        Δxl::Union{Float64,Vector{Float64}}=(-Inf),
        Δxu::Union{Float64,Vector{Float64}}=Inf,
        Δul::Union{Float64,Vector{Float64}}=(-Inf),
        Δuu::Union{Float64,Vector{Float64}}=Inf,
    ) where {T,Lk,Lf}
        # Get problem dims
        N = length(params.Xref)
        nx = Utils.get_nx(params.mfwd)
        ndx = Utils.get_ndx(params.mfwd)
        nu = params.mfwd.nu

        # Initialize indexing parameters
        pidx = IndexingParameters(N, nx, ndx, nu)
        nz, ndz, ng = pidx.dims.nz, pidx.dims.ndz, pidx.dims.ng

        # Initialize trust region bounds
        Δxl_ = zeros(Float64, ndx)
        Δxl_ .= Δxl
        Δxu_ = zeros(Float64, ndx)
        Δxu_ .= Δxu
        Δul_ = zeros(Float64, nu)
        Δul_ .= Δul
        Δuu_ = zeros(Float64, nu)
        Δuu_ .= Δuu

        # Initialize caches from dims
        ∇²ₓₓL = sparse(costfunc_hessian_pattern(pidx))
        ∇²ₓₓJ = DiffResults.HessianResult(zeros(Float64, nx))
        ∇²ᵤᵤJ = DiffResults.HessianResult(zeros(Float64, nu))
        ∇²ₓᵤJ = zeros(Float64, nx, nu)
        ∇²ᵤₓJ = zeros(Float64, nu, nx)
        ∇ₓL = zeros(Float64, ndz)
        ∇J = zeros(Float64, ndz)
        ∇g = sparse(constraint_jacobian_pattern(pidx))
        gl = zeros(Float64, ng)
        gu = zeros(Float64, ng)
        z = zeros(Float64, nz)
        ztmp = zeros(Float64, nz)
        dztmp = zeros(Float64, ndz)
        xtmp = zeros(Float64, nx)
        dxtmp = zeros(Float64, ndx)
        dxtmp_ad = zeros(Float64, ndx)
        utmp = zeros(Float64, nu)
        utmp_ad = zeros(Float64, nu)
        FDs = [Utils.FDCache{Float64}(params.mbwd) for t in 1:nthreads()]

        # Initialize gradient functions and configs for mixed hessians
        ∇ₓJ!(∇ₓJ::Matrix{<:Real}, x::Vector{<:Real}, u::Vector{<:Real}, cfg::ForwardDiff.GradientConfig) = ForwardDiff.gradient!(
            ∇ₓJ, δx -> params.costfunc.stage(δx, u), x, cfg
        )
        ∇ᵤJ!(∇ᵤJ::Matrix{<:Real}, x::Vector{<:Real}, u::Vector{<:Real}, cfg::ForwardDiff.GradientConfig) = ForwardDiff.gradient!(
            ∇ᵤJ, δu -> params.costfunc.stage(x, δu), u, cfg
        )
        ∇ₓJ!_cfg = ForwardDiff.GradientConfig(
            δx -> params.costfunc.stage(δx, utmp_ad), dxtmp_ad
        )
        ∇ᵤJ!_cfg = ForwardDiff.GradientConfig(
            δu -> params.costfunc.stage(dxtmp_ad, δu), utmp_ad
        )
        ∇²ₓᵤJ!_cfg = ForwardDiff.JacobianConfig(
            (y, δu) -> ∇ₓJ!(y, dxtmp_ad, δu, ∇ₓJ!_cfg), dxtmp_ad, utmp_ad
        )
        ∇²ᵤₓJ!_cfg = ForwardDiff.JacobianConfig(
            (y, δx) -> ∇ₓJ!(y, δx, utmp_ad, ∇ᵤJ!_cfg), utmp_ad, dxtmp_ad
        )

        # Initialize OSQP results and model
        r = OSQP.Results()
        r.x = zeros(ndx)
        r.y = zeros(ng)

        l = zeros(Float64, ng)
        m = OSQP.Model()
        OSQP.setup!(m; P=∇²ₓₓL, q=∇J, A=∇g, l=l, u=l, verbose=false)
        return new(
            m,
            r,
            pidx,
            ∇²ₓₓL,
            ∇²ₓₓJ,
            ∇²ᵤᵤJ,
            ∇²ₓᵤJ!_cfg,
            ∇²ᵤₓJ!_cfg,
            ∇²ₓᵤJ,
            ∇²ᵤₓJ,
            ∇ₓJ!_cfg,
            ∇ᵤJ!_cfg,
            ∇ₓJ!,
            ∇ᵤJ!,
            ∇ₓL,
            ∇J,
            ∇g,
            gl,
            gu,
            Δxl_,
            Δxu_,
            Δul_,
            Δuu_,
            z,
            ztmp,
            dztmp,
            xtmp,
            dxtmp,
            dxtmp_ad,
            utmp,
            utmp_ad,
            FDs,
        )
    end
end

@option struct DefaultSQPOptions
    tol_stat::Float64
    tol_eqconstr::Float64
    tol_interp::Float64
    eps_reg::Float64
    eps_fd::Float64
    maxiter::Int
    is_verbose::Bool
    save_bestsol::Bool
end

mutable struct SQPOptions
    tol_stat::Float64
    tol_eqconstr::Float64
    tol_interp::Float64
    eps_reg::Float64
    eps_fd::Float64
    maxiter::Integer
    is_verbose::Bool
    save_bestsol::Bool

    function SQPOptions(;
        tol_stat::Union{AbstractFloat,Nothing}=nothing,
        tol_eqconstr::Union{AbstractFloat,Nothing}=nothing,
        tol_interp::Union{AbstractFloat,Nothing}=nothing,
        eps_reg::Union{AbstractFloat,Nothing}=nothing,
        eps_fd::Union{AbstractFloat,Nothing}=nothing,
        maxiter::Union{Int,Nothing}=nothing,
        is_verbose::Union{Bool,Nothing}=nothing,
        save_bestsol::Union{Bool,Nothing}=nothing,
    )
        # Load default options from config
        default = from_toml(
            DefaultSQPOptions, joinpath(@__DIR__, "config/default_opts.toml")
        )

        # Use default options if the corresponding option is nothing
        tol_stat_ = isnothing(tol_stat) ? default.tol_stat : T(tol_stat)
        tol_eqconstr_ =
            isnothing(tol_eqconstr) ? default.tol_eqconstr : T(tol_eqconstr)
        tol_interp_ = isnothing(tol_interp) ? default.tol_interp : T(tol_interp)
        eps_reg_ = isnothing(eps_reg) ? default.eps_reg : T(eps_reg)
        eps_fd_ = isnothing(eps_fd) ? default.eps_fd : T(eps_fd)
        maxiter_ = isnothing(maxiter) ? default.maxiter : maxiter
        is_verbose_ = isnothing(is_verbose) ? default.is_verbose : is_verbose
        save_bestsol_ =
            isnothing(save_bestsol) ? default.save_bestsol : save_bestsol
        return new(
            tol_stat_,
            tol_eqconstr_,
            tol_interp_,
            eps_reg_,
            eps_fd_,
            maxiter_,
            is_verbose_,
            save_bestsol_,
        )
    end
end
