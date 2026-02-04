function expand_term_L!(
    bwd::BackwardCache,
    tmp::TemporaryCache,
    fwd::ForwardCache,
    params::TrajoptParameters,
)::Nothing
    # Get terminal x error
    copy!(tmp.x, fwd.X[end])
    axpy!(-1.0, params.Xref[end], tmp.x)

    # Get terminal costfunc hessian wrt x
    tmp.xx_result = ForwardDiff.hessian!(
        tmp.xx_result, params.costfunc.term, tmp.x
    )

    # Reference terminal value expansion
    Vx, Vxx = bwd.Vs.x[end], bwd.Vs.xx[end]

    # Save terminal costfunc gradient and hessian
    copy!(Vx, DiffResults.gradient(tmp.xx_result))
    copy!(Vxx, DiffResults.hessian(tmp.xx_result))
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
    copy!(tmp.x, fwd.X[k])
    axpy!(-1.0, params.Xref[k], tmp.x)

    copy!(tmp.u, fwd.U[k])
    axpy!(-1.0, params.Uref[k], tmp.u)

    # Get gradients and hessians of stage cost wrt x and u
    tmp.xx_result = ForwardDiff.hessian!(
        tmp.xx_result, δx -> params.costfunc.stage(δx, tmp.u), tmp.x
    )
    tmp.uu_result = ForwardDiff.hessian!(
        tmp.uu_result, δu -> params.costfunc.stage(tmp.x, δu), tmp.u
    )

    # Reference k-th stage costfunc expansion
    Lx, Lu, Lxx, Luu = bwd.Ls.x[k], bwd.Ls.u[k], bwd.Ls.xx[k], bwd.Ls.uu[k]

    # Save stage cost gradients and hessians wrt x and u
    copy!(Lx, DiffResults.gradient(tmp.xx_result))
    copy!(Lxx, DiffResults.hessian(tmp.xx_result))

    copy!(Lu, DiffResults.gradient(tmp.uu_result))
    copy!(Luu, DiffResults.hessian(tmp.uu_result))
    return nothing
end

function expand_F!(
    bwd::BackwardCache, fwd::ForwardCache, params::TrajoptParameters, k::Int
)::Nothing
    # Reference k-th dynamics jacobians, state, and control input
    Fx, Fu = bwd.Fs.x[k], bwd.Fs.u[k]
    x1, x, u = fwd.X[k + 1], fwd.X[k], fwd.U[k]
    # Get simulator jacobians
    params.simfunc_bwd!(Fx, Fu, x1, x, u)
    return nothing
end

function expand_Q!(bwd::BackwardCache, tmp::TemporaryCache, k::Int)::Nothing
    # Reference k+1-th value expansion and k-th expansions
    Vx, Vxx = bwd.Vs.x[k + 1], bwd.Vs.xx[k + 1]
    Lx, Lu, Lxx, Luu = bwd.Ls.x[k], bwd.Ls.u[k], bwd.Ls.xx[k], bwd.Ls.uu[k]
    Fx, Fu = bwd.Fs.x[k], bwd.Fs.u[k]
    Qx, Qu = bwd.Qs.x[k], bwd.Qs.u[k]
    Qxx, Quu, Qxu, Qux = bwd.Qs.xx[k], bwd.Qs.uu[k], bwd.Qs.xu[k], bwd.Qs.ux[k]

    # Action-value gradients
    # Qx = Lx + Fx'*Vx
    mul!(Qx, Fx', Vx)
    axpy!(1.0, Lx, Qx)

    # Qu = Lu + Fu'*Vx
    mul!(Qu, Fu', Vx)
    axpy!(1.0, Lu, Qu)

    # Action-value hessians
    # Qxx = Lxx + Fx'*Vxx*Fx
    mul!(tmp.xx, Fx', Vxx)
    mul!(Qxx, tmp.xx, Fx)
    axpy!(1.0, Lxx, Qxx)

    # Quu = Luu + Fu'*Vxx*Fu + μ*I
    mul!(tmp.ux, Fu', Vxx)
    mul!(Quu, tmp.ux, Fu)
    axpy!(1.0, Luu, Quu)
    axpy!(1.0, bwd.eps_reg, Quu)

    # Qxu = Fx'*Vxx*Fu
    mul!(tmp.xx, Fx', Vxx)
    mul!(Qxu, tmp.xx, Fu)

    # Qux = Fu'*Vxx*Fx
    mul!(tmp.ux, Fu', Vxx)
    mul!(Qux, tmp.ux, Fx)
    return nothing
end

function expand_V!(bwd::BackwardCache, tmp::TemporaryCache, k::Int)::Nothing
    # Reference k-th value and action-value expansion
    Vx, Vxx = bwd.Vs.x[k], bwd.Vs.xx[k]
    Qx, Qu = bwd.Qs.x[k], bwd.Qs.u[k]
    Qxx, Quu, Qxu, Qux = bwd.Qs.xx[k], bwd.Qs.uu[k], bwd.Qs.xu[k], bwd.Qs.ux[k]

    # Reference k-th gains
    K = bwd.Ks[k]
    d = bwd.D[k]

    # Cost-to-go hessian
    # Vxx = Qxx - K'*Qux + K'*Quu*K - Qxu*K
    copy!(Vxx, Qxx)
    mul!(tmp.xx, K', Qux)
    axpy!(-1.0, tmp.xx, Vxx)
    mul!(tmp.xu, K', Quu)
    mul!(tmp.xx, tmp.xu, K)
    axpy!(1.0, tmp.xx, Vxx)
    mul!(tmp.xx, Qxu, K)
    axpy!(-1.0, tmp.xx, Vxx)

    # Cost-to-go gradient
    # Vx = Qx - K'*u + K'*uu*d - xu*d
    copy!(Vx, Qx)
    mul!(tmp.x, K', Qu)
    axpy!(-1.0, tmp.x, Vx)
    mul!(tmp.xu, K', Quu)
    mul!(tmp.x, tmp.xu, d)
    axpy!(1.0, tmp.x, Vx)
    mul!(tmp.x, Qxu, d)
    axpy!(-1.0, tmp.x, Vx)
    return nothing
end

function update_gains!(bwd::BackwardCache, tmp::TemporaryCache, k::Int)::Nothing
    # Reference k-th action-value expansion
    Qu, Quu, Qux = bwd.Qs.u[k], bwd.Qs.uu[k], bwd.Qs.ux[k]
    Quu_temp = tmp.uu
    bkws = tmp.bkws_uu
    #luws = tmp.luws_uu

    # Reference k-th control gains
    d, K = bwd.D[k], bwd.Ks[k]

    # Perform lower-triangular Bunch-Kaufman factorization in place
    # Overwrite Quu_temp with Bunch-Kaufman factors
    copy!(Quu_temp, Quu)
    LAPACK.sytrf!(bkws, 'L', Quu_temp)
    #LAPACK.getrf!(luws, Quu_temp)

    # Feedforward gains: d = Quu \ Qu
    # sytrs! directly overwrites Qu
    copy!(d, Qu)
    LAPACK.sytrs!('L', Quu_temp, bkws.ipiv, d)
    #LAPACK.getrs!('N', Quu_temp, luws.ipiv, d)

    # Feedback gains: K = Quu \ Qux
    # sytrs! directly overwrites Qux
    copy!(K, Qux)
    LAPACK.sytrs!('L', Quu_temp, bkws.ipiv, K)
    #LAPACK.getrs!('N', Quu_temp, luws.ipiv, K)
    return nothing
end

function update_cost_prediction!(bwd::BackwardCache, k::Int)::Nothing
    # Reference k-th/k+1-th action-value and value expansion
    Qu = bwd.Qs.u[k]
    d = bwd.D[k]

    # Predicted change in cost
    # ΔJ += Qu' * d
    bwd.ΔJ += Qu' * d
    return nothing
end

function backward_pass!(cache::ILqrCache, params::TrajoptParameters)::Nothing
    # Get references to ILqrCache structs
    fwd = cache.fwd
    bwd = cache.bwd
    tmp = cache.tmp

    # Reset predicted change in cost
    bwd.ΔJ = 0.0

    # Initialize value expansion
    expand_term_L!(bwd, tmp, fwd, params)

    # Backward Riccati
    @inbounds for k in length(params.Uref):-1:1
        expand_stage_L!(bwd, tmp, fwd, params, k) # Stage cost expansion
        expand_F!(bwd, fwd, params, k) # Dynamics expansion
        expand_Q!(bwd, tmp, k)              # Action-value expansion
        update_gains!(bwd, tmp, k)               # Update feedback and feedforward
        expand_V!(bwd, tmp, k)         # Value expansion
        update_cost_prediction!(bwd, k)
    end
    return nothing
end
