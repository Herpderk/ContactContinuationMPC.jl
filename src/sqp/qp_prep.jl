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
    get_state_diff!(m, hic, xic, params.xic)

    # Dynamics residuals
    reset!(m, d)
    for k in 1:(N - 1)
        # Step simulator
        x0, u0 = z[zidx.x[k]], z[zidx.u[k]]
        copy_state_to_data!(d, x0)
        copyto!(d.ctrl, u0)
        step!(m, d)
        copy_data_to_state!(d, cache.xtmp)

        # Evaluate constraint
        x1, h0 = h[hidx.dyn[k]], z[zidx.x[k + 1]]
        get_state_diff!(m, h0, cache.xtmp, x1)
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
    N, zidx, dzidx, hidx = cache.pidx.dims.N, cache.pidx.z, cache.pidx.h

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
        copy_state_to_data!(d, x0)
        copyto!(d.ctrl, u0)
        threaded_fd!(m, d, FDs, ∇h0_x0, ∇h0_u0; ϵ=ϵ)

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

    # Stage costfunc gradients and hessians
    for k in 1:(N - 1)
        # Get x and u errors
        x, xref, u, uref = z[zidx.x[k]], Xref[k], z[zidx.u[k]], Uref[k]
        get_state_diff!(params.mfwd, dxtmp, x, xref)
        @. utmp = u - uref

        # Get gradients and hessians of stage cost wrt x and u
        # Assume x and u costs are separable
        ∇Jx = ∇J[dzidx.x[k]]
        ∇Ju = ∇J[dzidx.u[k]]
        ∇²Jxx = ∇²J[dzidx.x[k], dzidx.x[k]]
        ∇²Juu = ∇²J[dzidx.u[k], dzidx.u[k]]

        if Lk <: QuadraticCostFunction
            # Compute stage costfunc gradient and hessian via analytic expression
            Q, R = params.costfunc.stage.Q, params.costfunc.stage.R
            mul!(∇Jx, Q, dxtmp)
            mul!(∇Ju, R, utmp)
            copyto!(∇²Jxx, Q)
            copyto!(∇²Juu, R)
        else
            # Compute stage costfunc gradient and hessian via autodiff
            ForwardDiff.hessian!(
                cache.∇J²xx, δx -> params.costfunc.stage(δx, utmp), dxtmp
            )
            ForwardDiff.hessian!(
                cache.∇J²uu, δu -> params.costfunc.stage(dxtmp, δu), utmp
            )
            copyto!(∇Jx, DiffResults.gradient(cache.∇J²xx))
            copyto!(∇Ju, DiffResults.gradient(cache.∇J²uu))
            copyto!(∇Jxx, DiffResults.hessian(cache.∇J²xx))
            copyto!(∇Juu, DiffResults.hessian(cache.∇J²uu))
        end
    end

    # Terminal state error
    x, xref = z[zidx.x[k]], Xref[k]
    get_state_diff!(params.mfwd, dxtmp, x, xref)

    # Terminal costfunc gradient and hessian
    ∇Jx = ∇J[dzidx.x[end]]
    ∇²Jxx = ∇²J[dzidx.x[end], dzidx.x[end]]

    if Lk <: QuadraticCostFunction
        # Compute terminal costfunc gradient and hessian via analytic expression
        Qf = params.costfunc.term.Qf
        mul!(∇Jx, Qf, dxtmp)
        copyto!(∇²Jxx, Qf)
    else
        # Compute terminal costfunc gradient and hessian via autodiff
        ForwardDiff.hessian!(
            cache.∇J²xx, δx -> params.costfunc.stage(δx, utmp), dxtmp
        )
        copyto!(∇Jx, DiffResults.gradient(cache.∇J²xx))
        copyto!(∇Jxx, DiffResults.hessian(cache.∇J²xx))
    end
    return nothing
end
