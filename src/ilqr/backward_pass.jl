function expand_term_L!(
    bwd::BackwardCache{Tc},
    tmp::TemporaryCache{Tc},
    fwd::ForwardCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
)::Nothing where {Tc,Tp,Lk,Lf}
    # Get terminal x error
    get_state_diff!(params.mfwd, tmp.dx, fwd.X[end], params.Xref[end])

    # Get terminal costfunc hessian wrt x
    ForwardDiff.hessian!(bwd.L.dxdx_result, params.costfunc.term, tmp.dx)

    # Initialize value function expansion
    copyto!(bwd.V.dx, DiffResults.gradient(bwd.L.dxdx_result))
    copyto!(bwd.V.dxdx, DiffResults.hessian(bwd.L.dxdx_result))
    return nothing
end

function expand_stage_L!(
    bwd::BackwardCache{Tc},
    tmp::TemporaryCache{Tc},
    fwd::ForwardCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    k::Int,
)::Nothing where {Tc,Tp,Lk,Lf}
    # Get k-th x and u errors
    get_state_diff!(params.mfwd, tmp.dx, fwd.X[k], params.Xref[k])
    @. tmp.u = fwd.U[k] - params.Uref[k]

    # Get gradients and hessians of stage cost wrt x and u
    ForwardDiff.hessian!(
        bwd.L.dxdx_result, δx -> params.costfunc.stage(δx, tmp.u), tmp.dx
    )
    ForwardDiff.hessian!(
        bwd.L.uu_result, δu -> params.costfunc.stage(tmp.dx, δu), tmp.u
    )

    # Save stage cost gradients and hessians wrt x and u
    copyto!(bwd.L.dx, DiffResults.gradient(bwd.L.dxdx_result))
    copyto!(bwd.L.dxdx, DiffResults.hessian(bwd.L.dxdx_result))
    copyto!(bwd.L.u, DiffResults.gradient(bwd.L.uu_result))
    copyto!(bwd.L.uu, DiffResults.hessian(bwd.L.uu_result))
    return nothing
end

function expand_F!(
    bwd::BackwardCache{Tc},
    fwd::ForwardCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    k::Int,
)::Nothing where {Tc,Tp,Lk,Lf}
    # Reference backward model
    m, d = params.mbwd, params.dbwd

    # Pre-process MuJoCo data
    reset!(m, d)
    copy_state_to_data!(d, fwd.X[k])
    copyto!(d.ctrl, fwd.U[k])
    forward!(m, d)

    # Evaluate dynamics jacobians at xk, uk
    mjd_transitionFD(m, d, bwd.ϵ, true, bwd.F.dx, bwd.F.u, nothing, nothing)
    return nothing
end

function expand_Q!(
    bwd::BackwardCache{T}, tmp::TemporaryCache{T}
)::Nothing where {T}
    # Reference k+1-th value expansion and k-th expansions
    V, L, F, Q = bwd.V, bwd.L, bwd.F, bwd.Q

    # Action-value gradients
    # Q.dx = L.dx + F.dx'*V.dx
    transpose!(tmp.dxdx, F.dx)
    mul!(Q.dx, tmp.dxdx, V.dx)
    @. Q.dx += L.dx

    # Action-value hessians
    # Q.dxdx = L.dxdx + F.dx'*V.dxdx*F.dx
    # `tmp.dxdx` is storing F.dx'
    mul!(tmp.dxdx2, tmp.dxdx, V.dxdx)
    mul!(Q.dxdx, tmp.dxdx2, F.dx)
    @. Q.dxdx += L.dxdx

    # Q.dxu = F.dx'*V.dxdx*F.u
    # `tmp.dxdx2` is storing F.dx'*V.dxdx
    mul!(Q.dxu, tmp.dxdx2, F.u)

    # Q.u = L.u + F.u'*V.dx
    transpose!(tmp.udx, F.u)
    mul!(Q.u, tmp.udx, V.dx)
    @. Q.u += L.u

    # Q.uu = L.uu + F.u'*V.dxdx*F.u + μI
    # `tmp.udx` is storing F.u'
    mul!(tmp.udx2, tmp.udx, V.dxdx)
    mul!(Q.uu, tmp.udx2, F.u)
    @. Q.uu += L.uu + bwd.μI

    # Q.udx = F.u'*V.dxdx*F.dx
    # `tmp.udx` is storing F.u'*V.dxdx
    mul!(Q.udx, tmp.udx2, F.dx)
    return nothing
end

function expand_V!(
    bwd::BackwardCache{T}, tmp::TemporaryCache{T}, k::Int
)::Nothing where {T}
    # Reference cache variables
    V, Q, K, d = bwd.V, bwd.Q, bwd.Ks[k], bwd.ds[k]

    # Cost-to-go gradient
    # V.dx = Q.dx - K'*Q.u + K'*Q.uu*d - Q.dxu*d
    copyto!(V.dx, Q.dx)
    BLAS.gemv!('T', -1.0, K, Q.u, 1.0, V.dx)

    mul!(tmp.u, Q.uu, d)
    BLAS.gemm!('T', 'N', 1.0, K, tmp.u, 1.0, V.dx)

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

function update_gains!(
    bwd::BackwardCache{T}, tmp::TemporaryCache{T}, k::Int
)::Nothing where {T}
    # Reference cache variables
    Q, d, K = bwd.Q, bwd.ds[k], bwd.Ks[k]

    # Upper-triangular Cholesky factorization
    copyto!(tmp.uu, Q.uu)
    LAPACK.potrf!('U', tmp.uu)
    #LAPACK.sytrf!(Q.bkws, 'U', Q.uu)

    # Feedforward gains: d = Q.uu \ Q.u
    copyto!(d, Q.u)
    LAPACK.potrs!('U', tmp.uu, d)
    #LAPACK.sytrs!('U', Q.uu, Q.bkws.ipiv, d)

    # Feedback gains: K = Q.uu \ Q.udx
    copyto!(K, Q.udx)
    LAPACK.potrs!('U', tmp.uu, K)
    #LAPACK.sytrs!('U', Q.uu, Q.bkws.ipiv, K)
    return nothing
end

function update_cost_prediction!(
    bwd::BackwardCache{T}, tmp::TemporaryCache{T}, k::Int
)::Nothing where {T}
    # Reference cache variables
    Q, d = bwd.Q, bwd.ds[k]

    # First-order predicted change in cost: ΔJ1 += d'*Q.u
    bwd.ΔJ1 += dot(d, Q.u)

    # Second-order predicted change in cost: ΔJ2 += d'*Q.uu*d
    mul!(tmp.u, Q.uu, d)
    bwd.ΔJ2 += dot(d, tmp.u)
    return nothing
end

function backward_pass!(
    cache::ILqrCache{Tc}, params::TrajoptParameters{Tp,Lk,Lf}
)::Nothing where {Tc,Tp,Lk,Lf}
    # Get references to ILqrCache structs
    fwd, bwd, tmp = cache.fwd, cache.bwd, cache.tmp

    # Reset predicted change in cost
    bwd.ΔJ1 = 0.0
    bwd.ΔJ2 = 0.0

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
