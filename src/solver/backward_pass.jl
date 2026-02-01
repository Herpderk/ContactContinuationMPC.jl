function expand_term_L!(
    bwd::BackwardCache,
    tmp::TemporaryCache,
    fwd::ForwardCache,
    params::ProblemParameters,
)::Nothing
    # Get terminal x error
    BLAS.copy!(tmp.x, fwd.X[end])
    BLAS.axpy!(-1.0, params.Xref[end], tmp.x)

    # Get terminal costfunc hessian wrt x
    tmp.xx_result =
        ForwardDiff.hessian!(tmp.xx_result, params.costfunc.term, tmp.x)

    # Reference terminal value expansion
    Vx, Vxx = bwd.Vs.x[end], bwd.Vs.xx[end]

    # Save terminal costfunc gradient and hessian
    BLAS.copy!(Vx, DiffResults.gradient(tmp.xx_result))
    BLAS.copy!(Vxx, DiffResults.hessian(tmp.xx_result))
end


function expand_stage_L!(
    bwd::BackwardCache,
    tmp::TemporaryCache,
    fwd::ForwardCache,
    params::ProblemParameters,
    k::Int,
)::Nothing
    # Get k-th x and u errors
    BLAS.copy!(tmp.x, fwd.X[k])
    BLAS.axpy!(-1.0, params.Xref[k], tmp.x)

    BLAS.copy!(tmp.u, fwd.U[k])
    BLAS.axpy!(-1.0, params.Uref[k], tmp.u)

    # Get gradients and hessians of stage cost wrt x and u
    tmp.xx_result = ForwardDiff.hessian!(
        tmp.xx_result,
        δx -> params.costfunc.stage(δx, tmp.u),
        tmp.x,
    )
    tmp.uu_result = ForwardDiff.hessian!(
        tmp.uu_result,
        δu -> params.costfunc.stage(tmp.x, δu),
        tmp.u,
    )

    # Reference k-th stage costfunc expansion
    Lx, Lu, Lxx, Luu = bwd.Ls.x[k], bwd.Ls.u[k], bwd.Ls.xx[k], bwd.Ls.uu[k]

    # Save stage cost gradients and hessians wrt x and u
    BLAS.copy!(Lx, DiffResults.gradient(tmp.xx_result))
    BLAS.copy!(Lxx, DiffResults.hessian(tmp.xx_result))

    BLAS.copy!(Lu, DiffResults.gradient(tmp.uu_result))
    BLAS.copy!(Luu, DiffResults.hessian(tmp.uu_result))
end


function expand_F!(
    bwd::BackwardCache,
    fwd::ForwardCache,
    params::ProblemParameters,
    k::Int,
)::Nothing
    # Reference k-th dynamics jacobians, state, and control input
    Fx, Fu = bwd.Fs.x[k], bwd.Fs.u[k]
    x = fwd.X[k]
    u = fwd.U[k]
    # Get simulator jacobians
    params.simfunc_bwd!(Fx, Fu, x, u)
end


function expand_Q!(bwd::BackwardCache, tmp::TemporaryCache, k::Int)::Nothing
    # Reference k+1-th value expansion and k-th expansions
    Vx, Vxx = bwd.Vs.x[k+1], bwd.Vs.xx[k+1]
    Lx, Lu, Lxx, Luu = bwd.Ls.x[k], bwd.Ls.u[k], bwd.Ls.xx[k], bwd.Ls.uu[k]
    Fx, Fu = bwd.Fs.x[k], bwd.Fs.u[k]
    Qx, Qu, Qxx, Quu, Qxu, Qux = bwd.Qs.x[k],
    bwd.Qs.u[k],
    bwd.Qs.xx[k],
    bwd.Qs.uu[k],
    bwd.Qs.xu[k],
    bwd.Qs.ux[k]

    # Action-value gradients
    # Qx = Lx + Fx'*Vx
    mul!(Qx, Fx', Vx)
    BLAS.axpy!(1.0, Lx, Qx)

    # Qu = Lu + Fu'*Vx
    mul!(Qu, Fu', Vx)
    BLAS.axpy!(1.0, Lu, Qu)

    # Action-value hessians
    # Qxx = Lxx + Fx'*Vxx*Fx
    mul!(tmp.xx, Fx', Vxx)
    mul!(Qxx, tmp.xx, Fx)
    BLAS.axpy!(1.0, Lxx, Qxx)

    # Quu = Luu + Fu'*Vxx*Fu + μ*I
    mul!(tmp.ux, Fu', Vxx)
    mul!(Quu, tmp.ux, Fu)
    BLAS.axpy!(1.0, Luu, Quu)
    BLAS.axpy!(1.0, bwd.μ, Quu)

    # Qxu = Fx'*Vxx*Fu
    mul!(tmp.xx, Fx', Vxx)
    mul!(Qxu, tmp.xx, Fu)

    # Qux = Fu'*Vxx*Fx
    mul!(tmp.ux, Fu', Vxx)
    mul!(Qux, tmp.ux, Fx)
end


function expand_V!(bwd::BackwardCache, tmp::TemporaryCache, k::Int)::Nothing
    # Reference k-th value and action-value expansion
    Vx, Vxx = bwd.Vs.x[k+1], bwd.Vs.xx[k+1]
    Qx, Qu, Qxx, Quu, Qxu, Qux = bwd.Qs.x[k],
    bwd.Qs.u[k],
    bwd.Qs.xx[k],
    bwd.Qs.uu[k],
    bwd.Qs.xu[k],
    bwd.Qs.ux[k]

    # Reference k-th gains
    K = bwd.Ks[k]
    d = bwd.D[k]

    # Cost-to-go hessian
    # Vxx = Qxx - K'*Qux + K'*Quu*K - Qxu*K
    BLAS.copy!(Vxx, Qxx)
    mul!(tmp.xx, K', Qux)
    BLAS.axpy!(-1.0, tmp.xx, Vxx)
    mul!(tmp.xu, K', Quu)
    mul!(tmp.xx, tmp.xu, K)
    BLAS.axpy!(1.0, tmp.xx, Vxx)
    mul!(tmp.xx, Qxu, K)
    BLAS.axpy!(-1.0, tmp.xx, Vxx)

    # Cost-to-go gradient
    # Vx = Qx - K'*u + K'*uu*d - xu*d
    BLAS.copy!(Vx, Qx)
    mul!(tmp.x, K', Qu)
    BLAS.axpy!(-1.0, tmp.x, Vx)
    mul!(tmp.xu, K', Quu)
    mul!(tmp.x, tmp.xu, d)
    BLAS.axpy!(1.0, tmp.x, Vx)
    mul!(tmp.x, Qxu, d)
    BLAS.axpy!(-1.0, tmp.x, Vx)
end


function update_gains!(bwd::BackwardCache, k::Int)::Nothing
    # Reference k-th action-value expansion
    Qu, Quu, Qux, Quu_lu =
        bwd.Qs.u[k], bwd.Qs.uu[k], bwd.Qs.ux[k], bwd.Qs.uu_lu[k]

    # Get sparse LU factorization
    lu!(Quu_lu, sparse(Quu))

    # Feedforward gains: d = Quu \ Qu
    ldiv!(bwd.D[k], Quu_lu, Qu)

    # Feedback gains: K = Quu \ Qux
    ldiv!(bwd.Ks[k], Quu_lu, Qux)
end


function update_cost_prediction!(bwd::BackwardCache, k::Int)::Nothing
    # Reference k-th/k+1-th action-value and value expansion
    Qu = bwd.Qs.u[k]
    d = bwd.D[k]

    # Predicted change in cost
    # ΔJ += Qu' * d
    bwd.ΔJ += Qu' * d
end


function backward_pass!(cache::SolverCache, params::ProblemParameters)::Nothing
    # Get references to SolverCache structs
    fwd = cache.fwd
    bwd = cache.bwd
    tmp = cache.tmp

    # Reset predicted change in cost
    bwd.ΔJ = 0.0

    # Initialize value expansion
    expand_term_L!(bwd, tmp, fwd, params)

    # Backward Riccati
    @inbounds for k = length(params.Uref):-1:1
        expand_stage_L!(bwd, tmp, fwd, params, k) # Stage cost expansion
        expand_F!(bwd, fwd, params, k) # Dynamics expansion
        expand_Q!(bwd, tmp, k)              # Action-value expansion
        update_gains!(bwd, k)               # Update feedback and feedforward
        expand_V!(bwd, tmp, k)         # Value expansion
        update_cost_prediction!(bwd, k)
    end
end
