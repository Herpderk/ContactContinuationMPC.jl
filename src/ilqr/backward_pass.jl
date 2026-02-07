function expand_term_L!(
    bwd::BackwardCache,
    tmp::TemporaryCache,
    fwd::ForwardCache,
    params::TrajoptParameters,
)::Nothing
    # Get terminal x error
    get_state_diff!(params.mfwd, tmp.dx, fwd.X[end], params.Xref[end])

    # Get terminal costfunc hessian wrt x
    tmp.hess_dxdx = ForwardDiff.hessian!(
        tmp.hess_dxdx, params.costfunc.term, tmp.dx
    )

    # Save terminal costfunc gradient and hessian
    V = bwd.V
    copyto!(V.dx, DiffResults.gradient(tmp.hess_dxdx))
    copyto!(V.dxdx, DiffResults.hessian(tmp.hess_dxdx))
    return nothing
end

function expand_stage_L!(
    bwd::BackwardCache,
    tmp::TemporaryCache,
    fwd::ForwardCache,
    params::TrajoptParameters,
    k::Int,
)::Nothing
    # Get k-th x and u errors
    get_state_diff!(params.mfwd, tmp.dx, fwd.X[k], params.Xref[k])
    @. tmp.u = fwd.U[k] - params.Uref[k]

    # Get gradients and hessians of stage cost wrt x and u
    tmp.hess_dxdx = ForwardDiff.hessian!(
        tmp.hess_dxdx, δx -> params.costfunc.stage(δx, tmp.u), tmp.dx
    )
    tmp.hess_uu = ForwardDiff.hessian!(
        tmp.hess_uu, δu -> params.costfunc.stage(tmp.dx, δu), tmp.u
    )

    # Save stage cost gradients and hessians wrt x and u
    L = bwd.L
    copyto!(L.dx, DiffResults.gradient(tmp.hess_dxdx))
    copyto!(L.dxdx, DiffResults.hessian(tmp.hess_dxdx))
    copyto!(L.u, DiffResults.gradient(tmp.hess_uu))
    copyto!(L.uu, DiffResults.hessian(tmp.hess_uu))
    return nothing
end

function expand_F!(
    bwd::BackwardCache, fwd::ForwardCache, params::TrajoptParameters, k::Int
)::Nothing
    # Reference backward model
    m, d = params.mbwd, params.dbwd

    # Reference k-th dynamics jacobians, state, and control input
    F = bwd.F

    # Evaluate dynamics jacobians at xk, uk
    copy_state_to_data!(d, fwd.X[k])
    copyto!(d.ctrl, fwd.U[k])
    mjd_transitionFD(m, d, bwd.ϵ, true, F.dx, F.u, nothing, nothing)

    println("Fx: $(F.dx)")
    println("Fu: $(F.u)")
    return nothing
end

function expand_Q!(bwd::BackwardCache, tmp::TemporaryCache)::Nothing
    # Reference k+1-th value expansion and k-th expansions
    V, L, F, Q = bwd.V, bwd.L, bwd.F, bwd.Q

    # Action-value gradients
    # Q.dx = L.dx + F.dx'*V.dx
    # Since F.dx is row-major, copying to col-major is equivalent to transpose
    copyto!(tmp.dxdx, F.dx)
    mul!(Q.dx, tmp.dxdx, V.dx)
    @. Q.dx += L.dx
    copyto!(Q.dx, L.dx)

    # Q.u = L.u + F.u'*V.dx
    copyto!(Q.u, L.u)
    BLAS.gemv!('T', 1.0, F.u, V.dx, 1.0, Q.u)

    # Action-value hessians
    # Q.dxdx = L.dxdx + F.dx'*V.dxdx*F.dx
    # `tmp.dxdx` is storing F.dx'
    mul!(tmp.dxdx2, tmp.dxdx, V.dxdx)
    mul!(Q.dxdx, tmp.dxdx2, F.dx)
    @. Q.dxdx += L.dxdx

    # Q.dxu = F.dx'*V.dxdx*F.u
    # `tmp.dxdx2` is storing F.dx'*V.dxdx
    mul!(Q.dxu, tmp.dxdx2, F.u)

    # Q.uu = L.uu + F.u'*V.dxdx*F.u + μ*I
    BLAS.gemm!('T', 'N', 1.0, F.u, V.dxdx, 0.0, tmp.udx)
    mul!(Q.uu, tmp.udx, F.u)
    @. Q.uu += L.uu + bwd.μ

    # Q.udx = F.u'*V.dxdx*F.dx
    # `tmp.udx` is storing F.u'*V.dxdx
    mul!(Q.udx, tmp.udx, F.dx)
    return nothing
end

function expand_V!(bwd::BackwardCache, tmp::TemporaryCache, k::Int)::Nothing
    # Reference k-th value and action-value expansion
    V, Q = bwd.V, bwd.Q

    # Reference k-th gains
    d, K = bwd.ds[k], bwd.Ks[k]

    # Cost-to-go gradient
    # V.dx = Q.dx - K'*Q.u + K'*Q.uu*d - Q.dxu*d
    copyto!(V.dx, Q.dx)
    BLAS.gemv!('T', -1.0, K, Q.u, 1.0, V.dx)

    mul!(tmp.uu, Q.uu, d)
    BLAS.gemm!('T', 'N', 1.0, K, tmp.uu, 1.0, V.dx)

    mul!(tmp.dx, Q.dxu, d)
    @. V.dx -= tmp.dx

    # Cost-to-go hessian
    # V.dxdx = Q.dxdx - K'*Q.udx + K'*Q.uu*K - Q.dxu*K
    copyto!(V.dxdx, Q.dxdx)
    BLAS.gemm!('T', 'N', -1.0, K, Q.udx, 1.0, V.dxdx)

    mul!(tmp.udx, Q.uu, K)
    BLAS.gemm!('T', 'N', 1.0, K, tmp.udx, 1.0, V.dxdx)

    mul!(tmp.dxdx, Q.dxu, K)
    @. V.dxdx -= tmp.dxdx
    return nothing
end

function update_gains!(bwd::BackwardCache, tmp::TemporaryCache, k::Int)::Nothing
    # Reference k-th action-value expansion and matrix inverse helpers
    Q = bwd.Q
    Quu_tmp, bkws = tmp.uu, tmp.bkws_uu
    #luws = tmp.luws_uu

    # Reference k-th control gains
    d, K = bwd.ds[k], bwd.Ks[k]

    # Perform lower-triangular Bunch-Kaufman factorization in place
    # Overwrite Quu_tmp with Bunch-Kaufman factors
    copyto!(Quu_tmp, Q.uu)
    LAPACK.sytrf!(bkws, 'L', Quu_tmp)
    #LAPACK.getrf!(luws, Quu_tmp)

    # Feedforward gains: d = Q.uu \ Q.u
    # sytrs! directly overwrites Q.u
    copyto!(d, Q.u)
    LAPACK.sytrs!('L', Quu_tmp, bkws.ipiv, d)
    #LAPACK.getrs!('N', Quu_tmp, luws.ipiv, d)

    # Feedback gains: K = Q.uu \ Q.udx
    # sytrs! directly overwrites Q.udx
    copyto!(K, Q.udx)
    LAPACK.sytrs!('L', Quu_tmp, bkws.ipiv, K)
    #LAPACK.getrs!('N', Quu_tmp, luws.ipiv, K)
    return nothing
end

function update_cost_prediction!(
    bwd::BackwardCache, tmp::TemporaryCache, k::Int
)::Nothing
    Q = bwd.Q
    d = bwd.ds[k]
    singleton = tmp.singleton

    # Predicted change in cost: ΔJ += Q.u' * d
    BLAS.gemm!('T', 'N', 1.0, Q.u, d, 0.0, singleton)
    bwd.ΔJ += tmp.singleton[1]
    return nothing
end

function backward_pass!(cache::ILqrCache, params::TrajoptParameters)::Nothing
    # Get references to ILqrCache structs
    fwd, bwd, tmp = cache.fwd, cache.bwd, cache.tmp

    # Reset predicted change in cost
    bwd.ΔJ = 0.0

    # Initialize value expansion
    expand_term_L!(bwd, tmp, fwd, params)

    # Backward Riccati
    @inbounds for k in length(params.Uref):-1:1
        expand_stage_L!(bwd, tmp, fwd, params, k) # Stage cost expansion
        expand_F!(bwd, fwd, params, k)  # Dynamics expansion
        expand_Q!(bwd, tmp)          # Action-value expansion
        update_gains!(bwd, tmp, k)      # Update feedback and feedforward
        expand_V!(bwd, tmp, k)          # Value expansion
        update_cost_prediction!(bwd, tmp, k)
    end
    return nothing
end
