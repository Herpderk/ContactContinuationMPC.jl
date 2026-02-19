@views function equality_jacobian_pattern(
    pidx::IndexingParameters
)::Matrix{Float64}
    dims, dzidx, hidx = pidx.dims, pidx.dz, pidx.h
    ∇h = zeros(Float64, dims.nh, dims.ndz)

    # Constraint Jacobian wrt xic (identity matrix)
    rows_ic = hidx.ic
    cols_ic = dzidx.x[1]
    copyto!(∇h[rows_ic, cols_ic], I)

    for k in 1:(dims.N - 1)
        rows_dyn = hidx.dyn[k]
        # Constraint Jacobian wrt xk (dynamics Jacobian A)
        cols_dyn = dzidx.x[k]
        ∇h[rows_dyn, cols_dyn] .= 1.0
        # Constraint Jacobian wrt uk (dynamics Jacobian B)
        cols_dyn = dzidx.u[k]
        ∇h[rows_dyn, cols_dyn] .= 1.0
        # Constraint Jacobian wrt xk+1 (negative identity matrix)
        cols_dyn = dzidx.x[k + 1]
        copyto!(∇h[rows_dyn, cols_dyn], I)
    end
    return ∇h
end

@views function costfunc_hessian_pattern(
    pidx::IndexingParameters
)::Matrix{Float64}
    dims, dzidx = pidx.dims, pidx.dz
    ∇²J = zeros(Float64, dims.ndz, dims.ndz)

    # Stage costfunc hessians; Assume x and u costs are separable
    for k in 1:(dims.N - 1)
        idx_xk = dzidx.x[k]
        idx_uk = dzidx.u[k]
        ∇²J[idx_xk, idx_xk] .= 1.0
        ∇²J[idx_uk, idx_uk] .= 1.0
    end

    # Terminal costfunc hessian
    idx_xterm = dzidx.x[end]
    ∇²J[idx_xterm, idx_xterm] .= 1.0
    return ∇²J
end

@views function lagrangian_hessian_pattern(
    pidx::IndexingParameters
)::Matrix{Float64}
    dims = pidx.dims
    ∇²L = zeros(Float64, dims.ndz, dims.ndz)
    ∇²L += costfunc_hessian_pattern(pidx)
    return ∇²L
end
