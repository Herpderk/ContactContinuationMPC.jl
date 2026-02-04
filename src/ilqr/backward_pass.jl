function expand_term_L!(
    bwd::BackwardCache{T_bwd},
    tmp::TemporaryCache{T_tmp},
    fwd::ForwardCache{T_fwd},
    params::TrajoptParameters{T_params},
)::Nothing where {T_bwd,T_tmp,T_fwd,T_params}
    # Get terminal x error
    copy!(tmp.x, fwd.X[end])
    axpy!(-1.0, params.Xref[end], tmp.x)

    # Get terminal costfunc hessian wrt x
    tmp.hess_xx = ForwardDiff.hessian!(tmp.hess_xx, params.costfunc.term, tmp.x)

    # Reference terminal value expansion
    V = bwd.V

    # Save terminal costfunc gradient and hessian
    copy!(V.x, DiffResults.gradient(tmp.hess_xx))
    copy!(V.xx, DiffResults.hessian(tmp.hess_xx))
    return nothing
end

function expand_stage_L!(
    bwd::BackwardCache{T_bwd},
    tmp::TemporaryCache{T_tmp},
    fwd::ForwardCache{T_fwd},
    params::TrajoptParameters{T_params},
    k::Int,
)::Nothing where {T_bwd,T_tmp,T_fwd,T_params}
    # Get k-th x and u errors
    copy!(tmp.x, fwd.X[k])
    axpy!(-1.0, params.Xref[k], tmp.x)

    copy!(tmp.u, fwd.U[k])
    axpy!(-1.0, params.Uref[k], tmp.u)

    # Get gradients and hessians of stage cost wrt x and u
    tmp.hess_xx = ForwardDiff.hessian!(
        tmp.hess_xx, δx -> params.costfunc.stage(δx, tmp.u), tmp.x
    )
    tmp.hess_uu = ForwardDiff.hessian!(
        tmp.hess_uu, δu -> params.costfunc.stage(tmp.x, δu), tmp.u
    )

    # Reference k-th stage costfunc expansion
    L = bwd.L

    # Save stage cost gradients and hessians wrt x and u
    copy!(L.x, DiffResults.gradient(tmp.hess_xx))
    copy!(L.xx, DiffResults.hessian(tmp.hess_xx))

    copy!(L.u, DiffResults.gradient(tmp.hess_uu))
    copy!(L.uu, DiffResults.hessian(tmp.hess_uu))
    return nothing
end

function expand_F!(
    bwd::BackwardCache{T_bwd},
    fwd::ForwardCache{T_fwd},
    params::TrajoptParameters{T_params},
    k::Int,
)::Nothing where {T_bwd,T_fwd,T_params}
    # Reference k-th dynamics jacobians, state, and control input
    F = bwd.F
    x1, x0, u0 = fwd.X[k + 1], fwd.X[k], fwd.U[k]
    # Get simulator jacobians
    params.simfunc_bwd!(F.x, F.u, x1, x0, u0)
    return nothing
end

function expand_Q!(
    bwd::BackwardCache{T_bwd}, tmp::TemporaryCache{T_tmp}
)::Nothing where {T_bwd,T_tmp}
    # Reference k+1-th value expansion and k-th expansions
    V, L, F, Q = bwd.V, bwd.L, bwd.F, bwd.Q

    # Action-value gradients
    # Q.x = L.x + F.x'*V.x
    #mul!(Q.x, F.x', V.x)
    copy!(Q.x, L.x)
    BLAS.gemv!('T', 1.0, F.x, V.x, 1.0, Q.x)

    # Q.u = L.u + F.u'*V.x
    copy!(Q.u, L.u)
    BLAS.gemv!('T', 1.0, F.u, V.x, 1.0, Q.u)

    # Action-value hessians
    # Q.xx = L.xx + F.x'*V.xx*F.x
    BLAS.gemm!('T', 'N', 1.0, F.x, V.xx, 0.0, tmp.xx)
    mul!(Q.xx, tmp.xx, F.x)
    axpy!(1.0, L.xx, Q.xx)

    # Q.uu = L.uu + F.u'*V.xx*F.u + μ*I
    BLAS.gemm!('T', 'N', 1.0, F.u, V.xx, 0.0, tmp.ux)
    mul!(Q.uu, tmp.ux, F.u)
    axpy!(1.0, L.uu, Q.uu)
    axpy!(1.0, bwd.μ, Q.uu)

    # Q.xu = F.x'*V.xx*F.u
    BLAS.gemm!('T', 'N', 1.0, F.x, V.xx, 0.0, tmp.xx)
    mul!(Q.xu, tmp.xx, F.u)

    # Q.ux = F.u'*V.xx*F.x
    BLAS.gemm!('T', 'N', 1.0, F.u, V.xx, 0.0, tmp.ux)
    mul!(Q.ux, tmp.ux, F.x)
    return nothing
end

function expand_V!(
    bwd::BackwardCache{T_bwd}, tmp::TemporaryCache{T_tmp}, k::Int
)::Nothing where {T_bwd,T_tmp}
    # Reference k-th value and action-value expansion
    V, Q = bwd.V, bwd.Q

    # Reference k-th gains
    d, K = bwd.ds[k], bwd.Ks[k]

    # Cost-to-go gradient
    # V.x = Q.x - K'*Q.u + K'*Q.uu*d - Q.xu*d
    copy!(V.x, Q.x)
    BLAS.gemv!('T', -1.0, K, Q.u, 1.0, V.x)

    mul!(tmp.uu, Q.uu, d)
    BLAS.gemm!('T', 'N', 1.0, K, tmp.uu, 1.0, V.x)

    mul!(tmp.x, Q.xu, d)
    axpy!(-1.0, tmp.x, V.x)

    # Cost-to-go hessian
    # V.xx = Q.xx - K'*Q.ux + K'*Q.uu*K - Q.xu*K
    copy!(V.xx, Q.xx)
    BLAS.gemm!('T', 'N', -1.0, K, Q.ux, 1.0, V.xx)

    mul!(tmp.ux, Q.uu, K)
    BLAS.gemm!('T', 'N', 1.0, K, tmp.ux, 1.0, V.xx)

    mul!(tmp.xx, Q.xu, K)
    axpy!(-1.0, tmp.xx, V.xx)
    return nothing
end

function update_gains!(
    bwd::BackwardCache{T_bwd}, tmp::TemporaryCache{T_tmp}, k::Int
)::Nothing where {T_bwd,T_tmp}
    # Reference k-th action-value expansion and matrix inverse helpers
    Q = bwd.Q
    Quu_tmp, bkws = tmp.uu, tmp.bkws_uu
    #luws = tmp.luws_uu

    # Reference k-th control gains
    d, K = bwd.ds[k], bwd.Ks[k]

    # Perform lower-triangular Bunch-Kaufman factorization in place
    # Overwrite Quu_tmp with Bunch-Kaufman factors
    copy!(Quu_tmp, Q.uu)
    LAPACK.sytrf!(bkws, 'L', Quu_tmp)
    #LAPACK.getrf!(luws, Quu_tmp)

    # Feedforward gains: d = Q.uu \ Q.u
    # sytrs! directly overwrites Q.u
    copy!(d, Q.u)
    LAPACK.sytrs!('L', Quu_tmp, bkws.ipiv, d)
    #LAPACK.getrs!('N', Quu_tmp, luws.ipiv, d)

    # Feedback gains: K = Q.uu \ Q.ux
    # sytrs! directly overwrites Q.ux
    copy!(K, Q.ux)
    LAPACK.sytrs!('L', Quu_tmp, bkws.ipiv, K)
    #LAPACK.getrs!('N', Quu_tmp, luws.ipiv, K)
    return nothing
end

function update_cost_prediction!(
    bwd::BackwardCache{T_bwd}, tmp::TemporaryCache{T_tmp}, k::Int
)::Nothing where {T_bwd,T_tmp}
    Q = bwd.Q
    d = bwd.ds[k]
    singleton = tmp.singleton

    # Predicted change in cost: ΔJ += Q.u' * d
    BLAS.gemm!('T', 'N', 1.0, Q.u, d, 0.0, singleton)
    bwd.ΔJ += tmp.singleton[1]
    return nothing
end

function backward_pass!(
    cache::ILqrCache, params::TrajoptParameters{T_params}
)::Nothing where {T_params}
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
