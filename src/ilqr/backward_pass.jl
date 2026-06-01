function expand_term_L!(
    bwd::BackwardCache{Tc},
    tmp::TemporaryCache{Tc},
    fwd::ForwardCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
)::Nothing where {Tc,Tp,Lk,Lf}
    # Get terminal x error
    Utils.get_state_diff!(params.mfwd, tmp.dx, fwd.X0[end], params.Xref[end])

    # Initialize value expansion with terminal costfunc gradient and hessian wrt xf
    if Lf <: QuadraticCostFunction
        # Compute terminal costfunc gradient and hessian via analytic expression
        Qf = params.costfunc.term.Qf
        mul!(bwd.V.dx, Qf, tmp.dx)
        copyto!(bwd.V.dxdx, Qf)
    else
        # Compute terminal costfunc gradient and hessian via autodiff
        ForwardDiff.hessian!(bwd.L.dxdx_result, params.costfunc.term, tmp.dx)
        copyto!(bwd.V.dx, DiffResults.gradient(bwd.L.dxdx_result))
        copyto!(bwd.V.dxdx, DiffResults.hessian(bwd.L.dxdx_result))
    end
    return nothing
end

function expand_stage_L!(   # TODO add "meta" AL struct as argument
    bwd::BackwardCache{Tc},
    tmp::TemporaryCache{Tc},
    constr::ConstraintCache{Tc},
    fwd::ForwardCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    k::Int,
)::Nothing where {Tc,Tp,Lk,Lf}
    # Get k-th x and u errors
    Utils.get_state_diff!(params.mfwd, tmp.dx, fwd.X0[k], params.Xref[k])
    @. tmp.u1 = fwd.U0[k] - params.Uref[k]

    # Get gradients and hessians of stage cost wrt x and u
    if Lk <: QuadraticCostFunction
        # Compute stage costfunc gradient and hessian via analytic expression
        Q, R = params.costfunc.stage.Q, params.costfunc.stage.R
        mul!(bwd.L.dx, Q, tmp.dx)
        copyto!(bwd.L.dxdx, Q)
        mul!(bwd.L.u, R, tmp.u1)
        copyto!(bwd.L.uu, R)
    else
        # Compute stage costfunc gradient and hessian via autodiff
        ForwardDiff.hessian!(
            bwd.L.dxdx_result, δx -> params.costfunc.stage(δx, tmp.u1), tmp.dx
        )
        ForwardDiff.hessian!(
            bwd.L.uu_result, δu -> params.costfunc.stage(tmp.dx, δu), tmp.u1
        )
        copyto!(bwd.L.dx, DiffResults.gradient(bwd.L.dxdx_result))
        copyto!(bwd.L.dxdx, DiffResults.hessian(bwd.L.dxdx_result))
        copyto!(bwd.L.u, DiffResults.gradient(bwd.L.uu_result))
        copyto!(bwd.L.uu, DiffResults.hessian(bwd.L.uu_result))
    end

    # Add AL cost expansion
    expand_control_bound_cost!(bwd, tmp, constr.ul, :l, k)
    expand_control_bound_cost!(bwd, tmp, constr.uu, :u, k)
    return nothing
end

function expand_control_bound_cost!(
    bwd::BackwardCache,
    tmp::TemporaryCache,
    constr::ControlBoundCache,
    l_or_u::Symbol,
    k::Int,
)::Nothing
    ∇c = tmp.uu1
    control_bound_jacobian!(∇c, l_or_u)

    inequality_constraint_cost_gradient!(tmp.u1, ∇c, constr.F[k])
    bwd.L.u .+= tmp.u1

    inequality_constraint_cost_hessian!(
        tmp.uu2, tmp.dxdx1, ∇c, constr.I[k], constr.Ρ[k]
    )
    bwd.L.uu .+= tmp.uu2
    return nothing
end

function expand_F!(
    bwd::BackwardCache{Tc},
    fwd::ForwardCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    k::Int,
)::Nothing where {Tc,Tp,Lk,Lf}
    # Get state and control input to differentiate at
    x, u = fwd.X0[k], fwd.U0[k]

    if bwd.ΔJ > bwd.ΔJmax
        # Don't interpolate if too far from local minima
        m, d = params.mbwd, params.dbwd
        A, B = bwd.F.dx, bwd.F.u
        fd_dynamics!(m, d, bwd.FDs, A, B, x, u; ϵ=bwd.ϵ)
    else
        # Get forward dynamics jacobian
        mfwd, dfwd = params.mfwd, params.dfwd
        A, B = bwd.F.dx, bwd.F.u
        fd_dynamics!(mfwd, dfwd, bwd.FDs, A, B, x, u; ϵ=bwd.ϵ)

        # Get backward dynamics jacobian
        mbwd, dbwd = params.mbwd, params.dbwd
        Abwd, Bbwd = bwd.F.dx_bwd, bwd.F.u_bwd
        fd_dynamics!(mbwd, dbwd, bwd.FDs, Abwd, Bbwd, x, u; ϵ=bwd.ϵ)

        # Exponentially interpolate dynamics jacobian
        #γ = 0.1^((bwd.ΔJ - bwd.ΔJmax) / (1e-1 - bwd.ΔJmax))

        # Logarithmically interpolate dynamics jacobian
        #log_floor, log_ceil = log(1e-1), log(bwd.ΔJmax)
        #γ = 1.0 - (log(bwd.ΔJ) - log_ceil) / (log_floor - log_ceil)
        #println("ΔJ: $(bwd.ΔJ), ΔJmax: $(bwd.ΔJmax), γ: $γ")

        # Linearly interpolate dynamics jacobian based on convergence criteria ceiling
        γ = (bwd.ΔJ - bwd.ΔJmin) / (bwd.ΔJmax - bwd.ΔJmin)
        γ = clamp(γ, 0.0, 1.0)
        @. A += γ*(Abwd - A)
        @. B += γ*(Bbwd - B)
    end
    return nothing
end

function fd_dynamics!(
    m::Model,
    d::Data,
    c::Vector{Utils.FDCache{T}},
    A::Matrix{T},
    B::Matrix{T},
    x::Vector{T},
    u::Vector{T};
    ϵ::T,
)::Nothing where {T}
    # Pre-process MuJoCo data
    #reset!(m, d)
    Utils.copy_state_to_data!(d, x)
    copyto!(d.ctrl, u)
    #forward!(m, d)

    # Evaluate forward dynamics jacobians at xk, uk
    #mjd_transitionFD(m, d, ϵ, true, A, B, nothing, nothing)
    Utils.threaded_fd!(m, d, c, A, B; ϵ=ϵ)
    return nothing
end

function expand_Q!(
    bwd::BackwardCache{T}, tmp::TemporaryCache{T}
)::Nothing where {T}
    # Reference k+1-th value expansion and k-th expansions
    V, L, F, Q = bwd.V, bwd.L, bwd.F, bwd.Q

    # Action-value gradients
    # Q.dx = L.dx + F.dx'*V.dx
    transpose!(tmp.dxdx1, F.dx)
    mul!(Q.dx, tmp.dxdx1, V.dx)
    @. Q.dx += L.dx

    # Action-value hessians
    # Q.dxdx = L.dxdx + F.dx'*V.dxdx*F.dx
    # `tmp.dxdx1` is storing F.dx'
    mul!(tmp.dxdx2, tmp.dxdx1, V.dxdx)
    mul!(Q.dxdx, tmp.dxdx2, F.dx)
    @. Q.dxdx += L.dxdx

    # Q.dxu = F.dx'*V.dxdx*F.u
    # `tmp.dxdx2` is storing F.dx'*V.dxdx
    mul!(Q.dxu, tmp.dxdx2, F.u)

    # Q.u = L.u + F.u'*V.dx
    transpose!(tmp.udx1, F.u)
    mul!(Q.u, tmp.udx1, V.dx)
    @. Q.u += L.u

    # Q.uu = L.uu + F.u'*V.dxdx*F.u + μI
    # `tmp.udx1` is storing F.u'
    mul!(tmp.udx2, tmp.udx1, V.dxdx)
    mul!(Q.uu, tmp.udx2, F.u)
    @. Q.uu += L.uu + bwd.μI

    # Q.udx = F.u'*V.dxdx*F.dx
    # `tmp.udx1` is storing F.u'*V.dxdx
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

    mul!(tmp.u1, Q.uu, d)
    BLAS.gemm!('T', 'N', 1.0, K, tmp.u1, 1.0, V.dx)

    mul!(tmp.dx, Q.dxu, d)
    @. V.dx -= tmp.dx

    # Cost-to-go hessian
    # V.dxdx = Q.dxdx - K'*Q.udx + K'*Q.uu*K - Q.dxu*K
    copyto!(V.dxdx, Q.dxdx)
    BLAS.gemm!('T', 'N', -1.0, K, Q.udx, 1.0, V.dxdx)

    mul!(tmp.udx1, Q.uu, K)
    BLAS.gemm!('T', 'N', 1.0, K, tmp.udx1, 1.0, V.dxdx)

    mul!(tmp.dxdx1, Q.dxu, K)
    @. V.dxdx -= tmp.dxdx1
    return nothing
end

function update_gains!(
    bwd::BackwardCache{T}, tmp::TemporaryCache{T}, k::Int
)::Nothing where {T}
    # Reference cache variables
    Q, d, K = bwd.Q, bwd.ds[k], bwd.Ks[k]

    # Upper-triangular Cholesky factorization
    copyto!(tmp.uu1, Q.uu)
    LAPACK.potrf!('U', tmp.uu1)
    #LAPACK.sytrf!(Q.bkws, 'U', Q.uu)

    # Feedforward gains: d = Q.uu \ Q.u
    copyto!(d, Q.u)
    LAPACK.potrs!('U', tmp.uu1, d)
    #LAPACK.sytrs!('U', Q.uu, Q.bkws.ipiv, d)

    # Feedback gains: K = Q.uu \ Q.udx
    copyto!(K, Q.udx)
    LAPACK.potrs!('U', tmp.uu1, K)
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
    mul!(tmp.u1, Q.uu, d)
    bwd.ΔJ2 += dot(d, tmp.u1)
    return nothing
end

function backward_pass!(
    cache::iLQRCache{Tc}, params::TrajoptParameters{Tp,Lk,Lf}
)::Nothing where {Tc,Tp,Lk,Lf}
    # Get references to iLQRCache structs
    al, fwd, bwd, tmp = cache.constr, cache.fwd, cache.bwd, cache.tmp

    # Reset predicted change in cost
    bwd.ΔJ1 = 0.0
    bwd.ΔJ2 = 0.0

    # Initialize value expansion
    expand_term_L!(bwd, tmp, fwd, params)

    # Backward Riccati
    @inbounds for k in length(params.Uref):-1:1
        expand_stage_L!(bwd, tmp, al, fwd, params, k) # Stage cost expansion
        expand_F!(bwd, fwd, params, k)  # Dynamics expansion
        expand_Q!(bwd, tmp)          # Action-value expansion
        update_gains!(bwd, tmp, k)      # Update feedback and feedforward
        expand_V!(bwd, tmp, k)          # Value expansion
        update_cost_prediction!(bwd, tmp, k)
    end
    # Total predicted change in cost
    bwd.ΔJ = bwd.ΔJ1 + 0.5 * bwd.ΔJ2
    return nothing
end
