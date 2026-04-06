function solve_qp!(
    qp::QPCache, Δz::Vector{Float64}, λ::Vector{Float64}
)::Nothing
    ∇²ₓₓLtriu, ∇J, ∇g, gl, gu, m, r = (
        qp.∇²ₓₓLtriu, qp.∇J, qp.∇g, qp.gl, qp.gu, qp.m, qp.r
    )

    # Solve QP
    OSQP.update!(m; Px=∇²ₓₓLtriu.nzval, q=∇J, Ax=∇g.nzval, l=gl, u=gu)
    OSQP.warm_start!(m; y=λ)
    OSQP.solve!(m, r)
    if r.info.status_val != 1
        @warn "QP solver did not converge! Status: $(r.info.status)"
    end
end

function update_qp!(
    qp::QPCache,
    ad::AutodiffCache,
    tmp::TemporaryCache,
    FDs::Vector{Utils.FDCache{Float64}},
    sol::SolutionCache,
    opts::SQPOptions,
    pidx::IndexingParameters,
    params::TrajoptParameters{Float64,Lk,Lf},
)::Nothing where {Lk,Lf}
    ∇²ₓₓL, ∇²ₓₓLtriu, ∇²ₓₓLtriu_map, ∇J, ∇g, gl, gu = (
        qp.∇²ₓₓL, qp.∇²ₓₓLtriu, qp.∇²ₓₓLtriu_map, qp.∇J, qp.∇g, qp.gl, qp.gu
    )
    constraint_residuals!(gl, gu, sol.z, tmp, opts, pidx, params)
    constraint_jacobian!(∇g, sol.z, FDs, opts, pidx, params)

    # Cost function quadraticization (Gauss-Newton)
    fill!(∇²ₓₓL.nzval, 0.0)
    fill!(∇²ₓₓLtriu.nzval, 0.0)
    fill!(∇J, 0.0)
    costfunc_expansion!(∇²ₓₓL, ∇J, sol.z, ad, tmp, pidx, params)
    for (idx_triu, idx) in enumerate(∇²ₓₓLtriu_map)
        ∇²ₓₓLtriu.nzval[idx_triu] = ∇²ₓₓL.nzval[idx]
    end
    return nothing
end

function constraint_residuals!(
    gl::Vector{Float64},
    gu::Vector{Float64},
    z::Vector{Float64},
    tmp::TemporaryCache,
    opts::SQPOptions,
    pidx::IndexingParameters,
    params::TrajoptParameters{Float64,Lk,Lf},
)::Nothing where {Lk,Lf}
    fill!(gl, 0.0)
    fill!(gu, 0.0)
    equality_constraint_residuals!(gl, gu, z, tmp, pidx, params)
    inequality_constraint_residuals!(gl, gu, opts.ul, opts.uu, z, pidx)
    return nothing
end

function equality_constraint_residuals!(
    gl::Vector{Float64},
    gu::Vector{Float64},
    z::Vector{Float64},
    tmp::TemporaryCache,
    pidx::IndexingParameters,
    params::TrajoptParameters{Float64,Lk,Lf},
)::Nothing where {Lk,Lf}
    initcond_residuals!(gl, z, pidx, params)
    dynamics_residuals!(gl, z, tmp, pidx, params)
    copyto!(gu, gl)
    return nothing
end

function inequality_constraint_residuals!(
    gl::Vector{Float64},
    gu::Vector{Float64},
    ul::Union{Float64,Vector{Float64}},
    uu::Union{Float64,Vector{Float64}},
    z::Vector{Float64},
    pidx::IndexingParameters,
)::Nothing
    ctrlinput_bounds!(gl, ul, z, pidx)
    ctrlinput_bounds!(gu, uu, z, pidx)
    #trustregion_bounds!(gl, Δxl, Δul, pidx)
    #trustregion_bounds!(gu, Δxu, Δuu, pidx)
    #trustregion_jacobian!(∇g, pidx)
    return nothing
end

@views function initcond_residuals!(
    g::Vector{Float64},
    z::Vector{Float64},
    pidx::IndexingParameters,
    params::TrajoptParameters{Float64,Lk,Lf},
)::Nothing where {Lk,Lf}
    m = params.mfwd
    zidx, gidx = pidx.z, pidx.g
    Utils.get_state_diff!(m, g[gidx.ic], z[zidx.x[1]], params.xic)
    g[gidx.ic] .*= -1.0 # Flip the sign for OSQP's constraint formulation
    return nothing
end

@views function dynamics_residuals!(
    g::Vector{Float64},
    z::Vector{Float64},
    tmp::TemporaryCache,
    pidx::IndexingParameters,
    params::TrajoptParameters{Float64,Lk,Lf},
)::Nothing where {Lk,Lf}
    m, d = params.mfwd, params.dfwd
    N, zidx, gidx = pidx.dims.N, pidx.z, pidx.g

    # Dynamics residuals
    for k in 1:(N - 1)
        # Step simulator
        x0, u0 = z[zidx.x[k]], z[zidx.u[k]]
        Utils.copy_state_to_data!(d, x0)
        copyto!(d.ctrl, u0)
        step!(m, d)
        Utils.copy_data_to_state!(d, tmp.x)

        # Evaluate constraint
        g0, x1 = g[gidx.dyn[k]], z[zidx.x[k + 1]]
        Utils.get_state_diff!(m, g0, tmp.x, x1)
        g0 .*= -1.0     # Flip the sign for OSQP's constraint formulation
    end
    return nothing
end

@views function ctrlinput_bounds!(
    gb::Vector{Float64},
    ub::Union{Float64,Vector{Float64}},
    z::Vector{Float64},
    pidx::IndexingParameters,
)::Nothing
    N, zidx, gidx = pidx.dims.N, pidx.z, pidx.g
    for k in 1:(N - 1)
        gb[gidx.ub[k]] .= ub
        gb[gidx.ub[k]] .-= z[zidx.u[k]]
    end
    return nothing
end

@views function trustregion_bounds!(
    gb::Vector{Float64},
    Δxb::Vector{Float64},
    Δub::Vector{Float64},
    pidx::IndexingParameters,
)::Nothing
    N, gidx = pidx.dims.N, pidx.g
    for k in 1:(N - 1)
        copyto!(gb[gidx.xtr[k]], Δxb)
        copyto!(gb[gidx.utr[k]], Δub)
    end
    copyto!(gb[gidx.xtr[end]], Δxb)
    return nothing
end

function constraint_jacobian!(
    ∇g::SparseMatrixCSC{Float64,Int},
    z::Vector{Float64},
    FDs::Vector{Utils.FDCache{Float64}},
    opts::SQPOptions,
    pidx::IndexingParameters,
    params::TrajoptParameters{Float64,Lk,Lf};
)::Nothing where {Lk,Lf}
    fill!(∇g.nzval, 0.0)
    initcond_jacobian!(∇g, pidx)
    dynamics_jacobian!(∇g, z, FDs, pidx, params; ϵ=opts.eps_fd)
    ctrlinput_jacobian!(∇g, pidx)
    return nothing
end

@views function initcond_jacobian!(
    ∇g::SparseMatrixCSC{Float64,Int}, pidx::IndexingParameters
)::Nothing
    gidx, dzidx = pidx.g, pidx.dz
    rows_ic = gidx.ic
    cols_ic = dzidx.x[1]
    copyto!(∇g[rows_ic, cols_ic], I)
    return nothing
end

@views function dynamics_jacobian!(
    ∇g::SparseMatrixCSC{Float64,Int},
    z::Vector{Float64},
    FDs::Vector{Utils.FDCache{Float64}},
    pidx::IndexingParameters,
    params::TrajoptParameters{Float64,Lk,Lf};
    ϵ::Float64,
)::Nothing where {Lk,Lf}
    m, d = params.mbwd, params.dbwd
    N, zidx, dzidx, gidx = (pidx.dims.N, pidx.z, pidx.dz, pidx.g)

    for k in 1:(N - 1)
        # Pointers to constraint Jacobians wrt xk and uk
        ∇g0_x0, ∇g0_u0 = (
            ∇g[gidx.dyn[k], dzidx.x[k]], ∇g[gidx.dyn[k], dzidx.u[k]]
        )

        # Compute dynamics Jacobians via FD
        x0, u0 = z[zidx.x[k]], z[zidx.u[k]]
        Utils.copy_state_to_data!(d, x0)
        copyto!(d.ctrl, u0)
        Utils.threaded_fd!(m, d, FDs, ∇g0_x0, ∇g0_u0; ϵ=ϵ)
        #mjd_transitionFD(m, d, ϵ, true, ∇g0_x0, ∇g0_u0, nothing, nothing)

        # Constraint Jacobian wrt xk+1 (negative identity matrix)
        ∇g0_x1 = ∇g[gidx.dyn[k], dzidx.x[k + 1]]
        copyto!(∇g0_x1, -I)
    end
    return nothing
end

@views function ctrlinput_jacobian!(
    ∇g::SparseMatrixCSC{Float64,Int}, pidx::IndexingParameters
)::Nothing
    N, dzidx, gidx = pidx.dims.N, pidx.dz, pidx.g
    for k in 1:(N - 1)
        copyto!(∇g[gidx.ub[k], dzidx.u[k]], I)
    end
end

@views function trustregion_jacobian!(
    ∇g::SparseMatrixCSC{Float64,Int}, pidx::IndexingParameters
)::Nothing
    gidx, ndz = pidx.g, pidx.dims.ndz
    row_start, row_end = gidx.xtr[1][1], gidx.xtr[end][end]
    copyto!(∇g[row_start:row_end, 1:ndz], I)
    return nothing
end

@views function costfunc_expansion!(
    ∇²J::SparseMatrixCSC{Float64,Int},
    ∇J::Vector{Float64},
    z::Vector{Float64},
    ad::AutodiffCache,
    tmp::TemporaryCache,
    pidx::IndexingParameters,
    params::TrajoptParameters{Float64,Lk,Lf},
)::Nothing where {Lk,Lf}
    N, zidx, dzidx = pidx.dims.N, pidx.z, pidx.dz
    dxtmp, utmp = tmp.dx, tmp.u
    Xref, Uref = params.Xref, params.Uref
    ∇ₓJ!, ∇ᵤJ!, ∇ₓJ!_cfg, ∇ᵤJ!_cfg, ∇²ₓᵤJ!_cfg, ∇²ᵤₓJ!_cfg = (
        ad.∇ₓJ!, ad.∇ᵤJ!, ad.∇ₓJ!_cfg, ad.∇ᵤJ!_cfg, ad.∇²ₓᵤJ!_cfg, ad.∇²ᵤₓJ!_cfg
    )

    # Stage costfunc gradients and hessians
    for k in 1:(N - 1)
        # Get x and u errors
        x, xref, u, uref = z[zidx.x[k]], Xref[k], z[zidx.u[k]], Uref[k]
        Utils.get_state_diff!(params.mfwd, dxtmp, x, xref)
        @. utmp = u - uref

        # Get gradients and hessians of stage cost wrt x and u
        ∇ₓJ = ∇J[dzidx.x[k]]
        ∇ᵤJ = ∇J[dzidx.u[k]]
        ∇²ₓₓJ = ∇²J[dzidx.x[k], dzidx.x[k]]
        ∇²ᵤᵤJ = ∇²J[dzidx.u[k], dzidx.u[k]]
        ∇²ₓᵤJ = ∇²J[dzidx.x[k], dzidx.u[k]]
        ∇²ᵤₓJ = ∇²J[dzidx.u[k], dzidx.x[k]]

        if Lk <: QuadraticCostFunction
            # Compute stage costfunc gradient and hessian via analytic expression
            # Ignore mixed hessians for quadratic costfunc
            Q, R = params.costfunc.stage.Q, params.costfunc.stage.R
            mul!(∇ₓJ, Q, dxtmp)
            mul!(∇ᵤJ, R, utmp)
            copyto!(∇²ₓₓJ, Q)
            copyto!(∇²ᵤᵤJ, R)
            fill!(∇²ₓᵤJ, 0.0)
            fill!(∇²ᵤₓJ, 0.0)
        else
            # Compute stage costfunc gradient and hessian via autodiff
            ForwardDiff.hessian!(
                ad.∇²ₓₓJ, δx -> params.costfunc.stage(δx, utmp), dxtmp
            )
            ForwardDiff.hessian!(
                ad.∇²ᵤᵤJ, δu -> params.costfunc.stage(dxtmp, δu), utmp
            )
            copyto!(∇ₓJ, DiffResults.gradient(ad.∇²ₓₓJ))
            copyto!(∇ᵤJ, DiffResults.gradient(ad.∇²ᵤᵤJ))
            copyto!(∇²ₓₓJ, DiffResults.hessian(ad.∇²ₓₓJ))
            copyto!(∇²ᵤᵤJ, DiffResults.hessian(ad.∇²ᵤᵤJ))

            # Compute mixed costfunc hessians
            ForwardDiff.jacobian!(
                ∇²ₓᵤJ,
                (y, δu) -> ad.∇ₓJ!(y, dxtmp, δu, ad.∇ₓJ!_cfg),
                utmp,
                ad.∇²ₓᵤJ!_cfg,
            )
            ForwardDiff.jacobian!(
                ∇²ᵤₓJ,
                (y, δx) -> ad.∇ᵤJ!(y, δx, utmp, ad.∇ᵤJ!_cfg),
                dxtmp,
                ad.∇²ᵤₓJ!_cfg,
            )
        end
    end

    # Terminal state error
    x, xref = z[zidx.x[end]], Xref[end]
    Utils.get_state_diff!(params.mfwd, dxtmp, x, xref)

    # Terminal costfunc gradient and hessian
    ∇ₓJ = ∇J[dzidx.x[end]]
    ∇²ₓₓJ = ∇²J[dzidx.x[end], dzidx.x[end]]

    if Lf <: QuadraticCostFunction
        # Compute terminal costfunc gradient and hessian via analytic expression
        Qf = params.costfunc.term.Qf
        mul!(∇ₓJ, Qf, dxtmp)
        copyto!(∇²ₓₓJ, Qf)
    else
        # Compute terminal costfunc gradient and hessian via autodiff.
        # Use the terminal cost function (not the stage version) and the
        # terminal state `dxtmp` as the point of evaluation.
        ForwardDiff.hessian!(ad.∇²ₓₓJ, δx -> params.costfunc.term(δx), dxtmp)
        copyto!(∇ₓJ, DiffResults.gradient(ad.∇²ₓₓJ))
        copyto!(∇²ₓₓJ, DiffResults.hessian(ad.∇²ₓₓJ))
    end
    return nothing
end
