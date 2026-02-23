@views function initcond_residuals!(
    g::Vector{Float64},
    z::Vector{Float64},
    pidx::IndexingParameters,
    params::TrajoptParameters{Float64,Lk,Lf},
)::Nothing where {Lk,Lf}
    m = params.mfwd
    zidx, gidx = pidx.z, pidx.g
    Utils.get_state_diff!(m, g[gidx.ic], z[zidx.x[1]], params.xic)
    return nothing
end

@views function dynamics_residuals!(
    g::Vector{Float64},
    z::Vector{Float64},
    cache::SQPCache,
    params::TrajoptParameters{Float64,Lk,Lf},
)::Nothing where {Lk,Lf}
    m, d = params.mfwd, params.dfwd
    N, zidx, gidx = cache.pidx.dims.N, cache.pidx.z, cache.pidx.g

    # Dynamics residuals
    reset!(m, d)
    for k in 1:(N - 1)
        # Step simulator
        x0, u0 = z[zidx.x[k]], z[zidx.u[k]]
        Utils.copy_state_to_data!(d, x0)
        copyto!(d.ctrl, u0)
        step!(m, d)
        Utils.copy_data_to_state!(d, cache.xtmp)

        # Evaluate constraint
        g0, x1 = g[gidx.dyn[k]], z[zidx.x[k + 1]]
        Utils.get_state_diff!(m, g0, cache.xtmp, x1)
        g0 .*= -1.0     # Flip the sign for OSQP's constraint formulation
    end
    return nothing
end

@views function trustregion_bounds!(
    g::Vector{Float64},
    Δxb::Vector{Float64},
    Δub::Vector{Float64},
    pidx::IndexingParameters,
)::Nothing
    N, gidx = pidx.dims.N, pidx.g
    for k in 1:(N - 1)
        copyto!(g[gidx.xtr[k]], Δxb)
        copyto!(g[gidx.utr[k]], Δub)
    end
    copyto!(g[gidx.xtr[end]], Δxb)
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
    cache::SQPCache,
    params::TrajoptParameters{Float64,Lk,Lf};
    ϵ::Float64,
)::Nothing where {Lk,Lf}
    m, d, FDs = params.mbwd, params.dbwd, cache.FDs
    N, zidx, dzidx, gidx = (
        cache.pidx.dims.N, cache.pidx.z, cache.pidx.dz, cache.pidx.g
    )

    for k in 1:(N - 1)
        # Pointers to constraint Jacobians wrt xk and uk
        ∇g0_x0, ∇g0_u0 = ∇g[gidx.dyn[k], dzidx.x[k]],
        ∇g[gidx.dyn[k], dzidx.u[k]]

        # Compute dynamics Jacobians via FD
        x0, u0 = z[zidx.x[k]], z[zidx.u[k]]
        Utils.copy_state_to_data!(d, x0)
        copyto!(d.ctrl, u0)
        Utils.threaded_fd!(m, d, FDs, ∇g0_x0, ∇g0_u0; ϵ=ϵ)

        # Constraint Jacobian wrt xk+1 (negative identity matrix)
        ∇g0_x1 = ∇g[gidx.dyn[k], dzidx.x[k + 1]]
        copyto!(∇g0_x1, -I)
    end
    return nothing
end

@views function trustregion_jacobian!(
    ∇g::SparseMatrixCSC{Float64,Int}, pidx::IndexingParameters
)::Nothing
    gidx, ndz = pidx.g, pidx.dims.ndz
    startrow, endrow = gidx.xtr[1][1], gidx.xtr[end][end]
    copyto!(∇g[startrow:endrow, 1:ndz], I)
    return nothing
end

@views function costfunc_expansion!(
    ∇²J::SparseMatrixCSC{Float64,Int},
    ∇J::Vector{Float64},
    z::Vector{Float64},
    cache::SQPCache,
    params::TrajoptParameters{Float64,Lk,Lf},
)::Nothing where {Lk,Lf}
    N, zidx, dzidx = cache.pidx.dims.N, cache.pidx.z, cache.pidx.dz
    dxtmp, utmp = cache.dxtmp, cache.utmp
    Xref, Uref = params.Xref, params.Uref
    ∇ₓJ!, ∇ᵤJ!, ∇ₓJ!_cfg, ∇ᵤJ!_cfg, ∇²ₓᵤJ!_cfg, ∇²ᵤₓJ!_cfg = (
        cache.∇ₓJ!,
        cache.∇ᵤJ!,
        cache.∇ₓJ!_cfg,
        cache.∇ᵤJ!_cfg,
        cache.∇²ₓᵤJ!_cfg,
        cache.∇²ᵤₓJ!_cfg,
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
                cache.∇²ₓₓJ, δx -> params.costfunc.stage(δx, utmp), dxtmp
            )
            ForwardDiff.hessian!(
                cache.∇²ᵤᵤJ, δu -> params.costfunc.stage(dxtmp, δu), utmp
            )
            copyto!(∇ₓJ, DiffResults.gradient(cache.∇²ₓₓJ))
            copyto!(∇ᵤJ, DiffResults.gradient(cache.∇²ᵤᵤJ))
            copyto!(∇²ₓₓJ, DiffResults.hessian(cache.∇²ₓₓJ))
            copyto!(∇²ᵤᵤJ, DiffResults.hessian(cache.∇²ᵤᵤJ))

            # Compute mixed costfunc hessians
            ForwardDiff.jacobian!(
                ∇²ₓᵤJ, (y, δu) -> ∇ₓJ!(y, dxtmp, δu, ∇ₓJ!_cfg), utmp, ∇²ₓᵤJ!_cfg
            )
            ForwardDiff.jacobian!(
                ∇²ᵤₓJ, (y, δx) -> ∇ᵤJ!(y, δx, utmp, ∇ᵤJ!_cfg), dxtmp, ∇²ᵤₓJ!_cfg
            )
        end
    end

    # Terminal state error
    x, xref = z[zidx.x[end]], Xref[end]
    Utils.get_state_diff!(params.mfwd, dxtmp, x, xref)

    # Terminal costfunc gradient and hessian
    ∇ₓJ = ∇J[dzidx.x[end]]
    ∇²ₓₓJ = ∇²J[dzidx.x[end], dzidx.x[end]]

    if Lk <: QuadraticCostFunction
        # Compute terminal costfunc gradient and hessian via analytic expression
        Qf = params.costfunc.term.Qf
        mul!(∇ₓJ, Qf, dxtmp)
        copyto!(∇²ₓₓJ, Qf)
    else
        # Compute terminal costfunc gradient and hessian via autodiff
        ForwardDiff.hessian!(
            cache.∇²ₓₓJ, δx -> params.costfunc.stage(δx, utmp), dxtmp
        )
        copyto!(∇ₓJ, DiffResults.gradient(cache.∇²ₓₓJ))
        copyto!(∇²ₓₓJ, DiffResults.hessian(cache.∇²ₓₓJ))
    end
    return nothing
end
