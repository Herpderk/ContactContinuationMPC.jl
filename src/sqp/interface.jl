function triu_map(A::SparseMatrixCSC)
    Atriu = triu(A)
    mapping = zeros(Int, nnz(Atriu))
    triu_idx = 1
    # Loop over columns
    for j in 1:size(A, 2)
        # Loop over rows in the current column
        for k in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[k]
            # If we are in the upper triangle (row <= col)
            if i <= j
                mapping[triu_idx] = k
                triu_idx += 1
            end
        end
    end
    return Atriu, mapping
end

mutable struct SQPCache
    m::OSQP.Model
    r::OSQP.Results
    pidx::IndexingParameters
    ∇²ₓₓL::SparseMatrixCSC{Float64,Int}
    ∇²ₓₓLtriu::SparseMatrixCSC{Float64,Int}
    ∇²ₓₓLtriu_map::Vector{Int}
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
    gl_pred::Vector{Float64}
    gu_pred::Vector{Float64}
    p::Vector{Float64}
    d::Vector{Float64}
    c::Vector{Float64}
    ul::Vector{Float64}
    uu::Vector{Float64}
    Δxl::Vector{Float64}
    Δxu::Vector{Float64}
    Δul::Vector{Float64}
    Δuu::Vector{Float64}
    λ::Vector{Float64}
    z::Vector{Float64}
    zcand::Vector{Float64}
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
        ul::Union{Float64,Vector{Float64}}=(-Inf),
        uu::Union{Float64,Vector{Float64}}=Inf,
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

        # Initialize control input limits
        ul_ = zeros(Float64, nu)
        ul_ .= ul
        uu_ = zeros(Float64, nu)
        uu_ .= uu

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
        ∇²ₓₓLtriu, ∇²ₓₓLtriu_map = triu_map(∇²ₓₓL)
        ∇²ₓₓJ = DiffResults.HessianResult(zeros(Float64, nx))
        ∇²ᵤᵤJ = DiffResults.HessianResult(zeros(Float64, nu))
        ∇²ₓᵤJ = zeros(Float64, nx, nu)
        ∇²ᵤₓJ = zeros(Float64, nu, nx)
        ∇ₓL = zeros(Float64, ndz)
        ∇J = zeros(Float64, ndz)
        ∇g = sparse(constraint_jacobian_pattern(pidx))
        gl = zeros(Float64, ng)
        gu = zeros(Float64, ng)
        gl_pred = zeros(Float64, ng)
        gu_pred = zeros(Float64, ng)
        p = zeros(Float64, ng)
        d = zeros(Float64, ng)
        c = zeros(Float64, ng)
        λ = zeros(Float64, ng)
        z = zeros(Float64, nz)
        zcand = zeros(Float64, nz)
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
        r.x = zeros(Float64, ndz)
        r.y = zeros(Float64, ng)
        l = zeros(Float64, ng)
        m = OSQP.Model()
        OSQP.setup!(m; P=∇²ₓₓLtriu, q=∇J, A=∇g, l=l, u=l, verbose=false)
        return new(
            m,
            r,
            pidx,
            ∇²ₓₓL,
            ∇²ₓₓLtriu,
            ∇²ₓₓLtriu_map,
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
            gl_pred,
            gu_pred,
            p,
            d,
            c,
            ul_,
            uu_,
            Δxl_,
            Δxu_,
            Δul_,
            Δuu_,
            λ,
            z,
            zcand,
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
    alpha_mul::Float64
    margin_ls::Float64
    tol_stat::Float64
    tol_primal::Float64
    tol_dual::Float64
    tol_comp::Float64
    tol_interp::Float64
    eps_reg::Float64
    eps_fd::Float64
    maxiter_sqp::Int
    maxiter_qp::Int
    maxiter_ls::Int
    is_verbose::Bool
    save_bestsol::Bool
end

mutable struct SQPOptions
    alpha_mul::Float64
    margin_ls::Float64
    tol_stat::Float64
    tol_primal::Float64
    tol_dual::Float64
    tol_comp::Float64
    tol_interp::Float64
    eps_reg::Float64
    eps_fd::Float64
    maxiter_sqp::Int
    maxiter_qp::Int
    maxiter_ls::Int
    is_verbose::Bool
    save_bestsol::Bool

    function SQPOptions(;
        alpha_mul::Union{AbstractFloat,Nothing}=nothing,
        margin_ls::Union{AbstractFloat,Nothing}=nothing,
        tol_stat::Union{AbstractFloat,Nothing}=nothing,
        tol_primal::Union{AbstractFloat,Nothing}=nothing,
        tol_dual::Union{AbstractFloat,Nothing}=nothing,
        tol_comp::Union{AbstractFloat,Nothing}=nothing,
        tol_interp::Union{AbstractFloat,Nothing}=nothing,
        eps_reg::Union{AbstractFloat,Nothing}=nothing,
        eps_fd::Union{AbstractFloat,Nothing}=nothing,
        maxiter_sqp::Union{Int,Nothing}=nothing,
        maxiter_qp::Union{Int,Nothing}=nothing,
        maxiter_ls::Union{Int,Nothing}=nothing,
        is_verbose::Union{Bool,Nothing}=nothing,
        save_bestsol::Union{Bool,Nothing}=nothing,
    )
        # Load default options from config
        default = from_toml(
            DefaultSQPOptions, joinpath(@__DIR__, "config/default_opts.toml")
        )

        # Use default options if the corresponding option is nothing
        alpha_mul_ =
            isnothing(alpha_mul) ? default.alpha_mul : Float64(alpha_mul)
        margin_ls_ =
            isnothing(margin_ls) ? default.margin_ls : Float64(margin_ls)
        tol_stat_ = isnothing(tol_stat) ? default.tol_stat : Float64(tol_stat)
        tol_primal_ =
            isnothing(tol_primal) ? default.tol_primal : Float64(tol_primal)
        tol_dual_ = isnothing(tol_dual) ? default.tol_dual : Float64(tol_dual)
        tol_comp_ = isnothing(tol_comp) ? default.tol_comp : Float64(tol_comp)
        tol_interp_ =
            isnothing(tol_interp) ? default.tol_interp : Float64(tol_interp)
        eps_reg_ = isnothing(eps_reg) ? default.eps_reg : Float64(eps_reg)
        eps_fd_ = isnothing(eps_fd) ? default.eps_fd : Float64(eps_fd)
        maxiter_sqp_ =
            isnothing(maxiter_sqp) ? default.maxiter_sqp : maxiter_sqp
        maxiter_qp_ = isnothing(maxiter_qp) ? default.maxiter_qp : maxiter_qp
        maxiter_ls_ = isnothing(maxiter_ls) ? default.maxiter_ls : maxiter_ls
        is_verbose_ = isnothing(is_verbose) ? default.is_verbose : is_verbose
        save_bestsol_ =
            isnothing(save_bestsol) ? default.save_bestsol : save_bestsol
        return new(
            alpha_mul_,
            margin_ls_,
            tol_stat_,
            tol_primal_,
            tol_dual_,
            tol_comp_,
            tol_interp_,
            eps_reg_,
            eps_fd_,
            maxiter_sqp_,
            maxiter_qp_,
            maxiter_ls_,
            is_verbose_,
            save_bestsol_,
        )
    end
end
