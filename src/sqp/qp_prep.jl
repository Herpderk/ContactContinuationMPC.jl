@views function equality_residuals!(
    h::Vector{Float64},
    z::Vector{Float64},
    cache::SQPCache,
    params::TrajoptParameters{Float64,Lk,Lf},
)::Nothing where {Lk,Lf}
    m, d = params.mfwd, params.dfwd
    N, zidx, hidx = cache.pidx.dims.N, cache.pidx.z, cache.pidx.h

    # Initial condition residuals
    hic, xic = h[hidx.ic], z[zidx.x[1]]
    Utils.get_state_diff!(m, hic, xic, params.xic)

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
        x1, h0 = h[hidx.dyn[k]], z[zidx.x[k + 1]]
        Utils.get_state_diff!(m, h0, cache.xtmp, x1)
    end
    return nothing
end

@views function equality_jacobian!(
    ∇h::SparseMatrixCSC{Float64,Int},
    z::Vector{Float64},
    cache::SQPCache,
    params::TrajoptParameters{Float64,Lk,Lf};
    ϵ::Float64,
)::Nothing where {Lk,Lf}
    m, d, FDs = params.mbwd, params.dbwd, cache.FDs
    N, zidx, dzidx, hidx = (
        cache.pidx.dims.N, cache.pidx.z, cache.pidx.dz, cache.pidx.h
    )

    # Initial conditions Jacobian
    rows_ic = hidx.ic
    cols_ic = dzidx.x[1]
    copyto!(∇h[rows_ic, cols_ic], I)

    for k in 1:(N - 1)
        # Dynamics residual rows
        rows_dyn = hidx.dyn[k]

        # Pointers to constraint Jacobians wrt xk and uk
        ∇h0_x0, ∇h0_u0 = ∇h[rows_dyn, dzidx.x[k]], ∇h[rows_dyn, dzidx.u[k]]

        # Compute dynamics Jacobians via FD
        x0, u0 = z[zidx.x[k]], z[zidx.u[k]]
        Utils.copy_state_to_data!(d, x0)
        copyto!(d.ctrl, u0)
        Utils.threaded_fd!(m, d, FDs, ∇h0_x0, ∇h0_u0; ϵ=ϵ)

        # Constraint Jacobian wrt xk+1 (negative identity matrix)
        ∇h0_x1 = ∇h[rows_dyn, dzidx.x[k + 1]]
        copyto!(∇h0_x1, -I)
    end
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
