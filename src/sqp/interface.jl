mutable struct SQPCache
    pidx::IndexingParameters
    qp::QPCache
    ad::AutodiffCache
    ls::LineSearchCache
    kkt::KKTCache
    sol::SolutionCache
    tmp::TemporaryCache
    FDs::Vector{Utils.FDCache{Float64}}

    function SQPCache(params::TrajoptParameters{T,Lk,Lf}) where {T,Lk,Lf}
        # Get problem dims
        N = length(params.Xref)
        nx = Utils.get_nx(params.mfwd)
        ndx = Utils.get_ndx(params.mfwd)
        nu = params.mfwd.nu

        # Initialize indexing parameters
        pidx = IndexingParameters(N, nx, ndx, nu)

        # Initialize caches from dims
        qp = QPCache(pidx)
        ad = AutodiffCache(params)
        ls = LineSearchCache(pidx.dims)
        kkt = KKTCache(pidx.dims)
        sol = SolutionCache(pidx.dims)
        tmp = TemporaryCache(pidx.dims)
        FDs = [Utils.FDCache{Float64}(params.mbwd) for t in 1:nthreads()]
        return new(pidx, qp, ad, ls, kkt, sol, tmp, FDs)
    end
end

@option struct DefaultSQPOptions
    ul::Float64
    uu::Float64
    xtr_l::Float64
    xtr_u::Float64
    utr_l::Float64
    utr_u::Float64
    gamma_init::Float64
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
    ul::Union{Float64,Vector{Float64}}
    uu::Union{Float64,Vector{Float64}}
    xtr_l::Union{Float64,Vector{Float64}}
    xtr_u::Union{Float64,Vector{Float64}}
    utr_l::Union{Float64,Vector{Float64}}
    utr_u::Union{Float64,Vector{Float64}}
    gamma_init::Float64
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
        ul::Union{AbstractFloat,AbstractVector{<:AbstractFloat},Nothing}=nothing,
        uu::Union{AbstractFloat,AbstractVector{<:AbstractFloat},Nothing}=nothing,
        xtr_l::Union{AbstractFloat,AbstractVector{<:AbstractFloat},Nothing}=nothing,
        xtr_u::Union{AbstractFloat,AbstractVector{<:AbstractFloat},Nothing}=nothing,
        utr_l::Union{AbstractFloat,AbstractVector{<:AbstractFloat},Nothing}=nothing,
        utr_u::Union{AbstractFloat,AbstractVector{<:AbstractFloat},Nothing}=nothing,
        gamma_init::Union{AbstractFloat,Nothing}=nothing,
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
        ul_ = isnothing(ul) ? default.ul : ul
        ul_ =
            typeof(ul_) <: AbstractVector ? Vector{Float64}(ul_) : Float64(ul_)
        uu_ = isnothing(uu) ? default.uu : uu
        uu_ =
            typeof(uu_) <: AbstractVector ? Vector{Float64}(uu_) : Float64(uu_)

        xtr_l_ = isnothing(xtr_l) ? default.xtr_l : Float64(xtr_l)
        xtr_l_ = if typeof(xtr_l_) <: AbstractVector
            Vector{Float64}(xtr_l_)
        else
            Float64(xtr_l_)
        end
        xtr_u_ = isnothing(xtr_u) ? default.xtr_u : Float64(xtr_u)
        xtr_u_ = if typeof(xtr_u_) <: AbstractVector
            Vector{Float64}(xtr_u_)
        else
            Float64(xtr_u_)
        end
        utr_l_ = isnothing(utr_l) ? default.utr_l : Float64(utr_l)
        utr_l_ = if typeof(utr_l_) <: AbstractVector
            Vector{Float64}(utr_l_)
        else
            Float64(utr_l_)
        end
        utr_u_ = isnothing(utr_u) ? default.utr_u : Float64(utr_u)
        utr_u_ = if typeof(utr_u_) <: AbstractVector
            Vector{Float64}(utr_u_)
        else
            Float64(utr_u_)
        end

        gamma_init_ =
            isnothing(gamma_init) ? default.gamma_init : Float64(gamma_init)
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
            ul_,
            uu_,
            xtr_l_,
            xtr_u_,
            utr_l_,
            utr_u_,
            gamma_init_,
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
