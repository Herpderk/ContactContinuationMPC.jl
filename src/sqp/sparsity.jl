@views function constraint_jacobian_pattern(
    pidx::IndexingParameters
)::Matrix{Float64}
    dims, dzidx, gidx = pidx.dims, pidx.dz, pidx.g
    ∇g = zeros(Float64, dims.ng, dims.ndz)

    # Initial conditions constraint Jacobian wrt xic (identity)
    rows_ic = gidx.ic
    cols_ic = dzidx.x[1]
    copyto!(∇g[rows_ic, cols_ic], I)

    for k in 1:(dims.N - 1)
        # Dynamics constraint Jacobians
        ∇g[gidx.dyn[k], dzidx.x[k]] .= 1.0          # wrt xk (dynamics Jacobian A)
        ∇g[gidx.dyn[k], dzidx.u[k]] .= 1.0          # wrt uk (dynamics Jacobian B)
        copyto!(∇g[gidx.dyn[k], dzidx.x[k + 1]], I)    # wrt xk+1 (negative identity)
    end

    # Trust region constraint Jacobian (identity)
    startrow, endrow = gidx.xtr[1][1], gidx.xtr[end][end]
    copyto!(∇g[startrow:endrow, 1:dims.ndz], I)
    return ∇g
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
        ∇²J[idx_xk, idx_uk] .= 1.0
        ∇²J[idx_uk, idx_xk] .= 1.0
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
    ∇²ₓₓL = zeros(Float64, dims.ndz, dims.ndz)
    ∇²ₓₓL += costfunc_hessian_pattern(pidx)
    return ∇²ₓₓL
end
