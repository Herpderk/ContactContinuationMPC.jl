struct TemporaryCache
    dz::Vector{Float64}
    dx::Vector{Float64}
    x::Vector{Float64}
    u::Vector{Float64}

    function TemporaryCache(dims::ProblemDimensions)
        dz = zeros(Float64, dims.ndz)
        dx = zeros(Float64, dims.ndx)
        x = zeros(Float64, dims.nx)
        u = zeros(Float64, dims.nu)
        return new(dz, dx, x, u)
    end
end

struct SolutionCache
    z::Vector{Float64}
    λ::Vector{Float64}

    function SolutionCache(dims::ProblemDimensions)
        z = zeros(Float64, dims.nz)
        λ = zeros(Float64, dims.ng)
        return new(z, λ)
    end
end

struct KKTCache
    ∇ₓL::Vector{Float64}
    p::Vector{Float64}
    d::Vector{Float64}
    c::Vector{Float64}

    function KKTCache(dims::ProblemDimensions)
        ∇ₓL = zeros(Float64, dims.ndz)
        p = zeros(Float64, dims.ng)
        d = zeros(Float64, dims.ng)
        c = zeros(Float64, dims.ng)
        return new(∇ₓL, p, d, c)
    end
end

struct LineSearchCache
    zcand::Vector{Float64}
    gl_cand::Vector{Float64}
    gu_cand::Vector{Float64}
    gl_pred::Vector{Float64}
    gu_pred::Vector{Float64}

    function LineSearchCache(dims::ProblemDimensions)
        zcand = zeros(Float64, dims.nz)
        gl_cand = zeros(Float64, dims.ng)
        gu_cand = zeros(Float64, dims.ng)
        gl_pred = zeros(Float64, dims.ng)
        gu_pred = zeros(Float64, dims.ng)
        return new(zcand, gl_cand, gu_cand, gl_pred, gu_pred)
    end
end

struct QPCache
    m::OSQP.Model
    r::OSQP.Results
    ∇²ₓₓL::SparseMatrixCSC{Float64,Int}
    ∇²ₓₓLtriu::SparseMatrixCSC{Float64,Int}
    ∇²ₓₓLtriu_map::Vector{Int}
    ∇J::Vector{Float64}
    ∇g::SparseMatrixCSC{Float64,Int}
    gl::Vector{Float64}
    gu::Vector{Float64}

    function QPCache(pidx::IndexingParameters)
        ∇²ₓₓL = sparse(costfunc_hessian_pattern(pidx))
        ∇²ₓₓLtriu, ∇²ₓₓLtriu_map = triu_map(∇²ₓₓL)

        ndz, ng = pidx.dims.ndz, pidx.dims.ng
        ∇J = zeros(Float64, ndz)
        ∇g = sparse(constraint_jacobian_pattern(pidx))
        gl = zeros(Float64, ng)
        gu = zeros(Float64, ng)

        # Initialize OSQP results and model
        r = OSQP.Results()
        r.x = zeros(Float64, ndz)
        r.y = zeros(Float64, ng)
        m = OSQP.Model()
        OSQP.setup!(m; P=∇²ₓₓLtriu, q=∇J, A=∇g, l=gl, u=gu, verbose=false)
        return new(m, r, ∇²ₓₓL, ∇²ₓₓLtriu, ∇²ₓₓLtriu_map, ∇J, ∇g, gl, gu)
    end
end

struct AutodiffCache
    ∇ₓJ!::Function
    ∇ᵤJ!::Function
    ∇ₓJ!_cfg::ForwardDiff.GradientConfig
    ∇ᵤJ!_cfg::ForwardDiff.GradientConfig
    ∇²ₓₓJ::DiffResults.DiffResult
    ∇²ᵤᵤJ::DiffResults.DiffResult
    ∇²ₓᵤJ!_cfg::ForwardDiff.JacobianConfig
    ∇²ᵤₓJ!_cfg::ForwardDiff.JacobianConfig
    dx::Vector{Float64}
    u::Vector{Float64}

    function AutodiffCache(params::TrajoptParameters)
        ndx = Utils.get_ndx(params.mfwd)
        nu = params.mfwd.nu

        ∇²ₓₓJ = DiffResults.HessianResult(zeros(Float64, ndx))
        ∇²ᵤᵤJ = DiffResults.HessianResult(zeros(Float64, nu))

        # Caches for ForwardDiff
        dx = zeros(Float64, ndx)
        u = zeros(Float64, nu)

        # Initialize gradient functions and configs for mixed hessians
        ∇ₓJ!(∇ₓJ::Matrix{<:Real}, x::Vector{<:Real}, u::Vector{<:Real}, cfg::ForwardDiff.GradientConfig) = ForwardDiff.gradient!(
            ∇ₓJ, δx -> params.costfunc.stage(δx, u), x, cfg
        )
        ∇ᵤJ!(∇ᵤJ::Matrix{<:Real}, x::Vector{<:Real}, u::Vector{<:Real}, cfg::ForwardDiff.GradientConfig) = ForwardDiff.gradient!(
            ∇ᵤJ, δu -> params.costfunc.stage(x, δu), u, cfg
        )
        ∇ₓJ!_cfg = ForwardDiff.GradientConfig(
            δx -> params.costfunc.stage(δx, u), dx
        )
        ∇ᵤJ!_cfg = ForwardDiff.GradientConfig(
            δu -> params.costfunc.stage(dx, δu), u
        )
        ∇²ₓᵤJ!_cfg = ForwardDiff.JacobianConfig(
            (y, δu) -> ∇ₓJ!(y, dx, δu, ∇ₓJ!_cfg), dx, u
        )
        ∇²ᵤₓJ!_cfg = ForwardDiff.JacobianConfig(
            (y, δx) -> ∇ₓJ!(y, δx, u, ∇ᵤJ!_cfg), u, dx
        )
        return new(
            ∇ₓJ!,
            ∇ᵤJ!,
            ∇ₓJ!_cfg,
            ∇ᵤJ!_cfg,
            ∇²ₓₓJ,
            ∇²ᵤᵤJ,
            ∇²ₓᵤJ!_cfg,
            ∇²ᵤₓJ!_cfg,
            dx,
            u,
        )
    end
end
