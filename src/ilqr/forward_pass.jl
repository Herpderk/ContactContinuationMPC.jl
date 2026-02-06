function roll_out!(
    fwd::ForwardCache,
    bwd::BackwardCache,
    tmp::TemporaryCache,
    sol::TrajoptSolution,
    params::TrajoptParameters,
)::Nothing
    # Initialize trajectory with previous solution
    copy_nested_array!(fwd.X, sol.X)
    copy_nested_array!(fwd.U, sol.U)

    # Forward rollout
    @inbounds for k in 1:length(params.Uref)
        # Reference simulator model
        m = params.mfwd

        # Update control input
        #fwd.U[k] = sol.U[k] - α*ds[k] - Ks[k]*(fwd.X[k] - sol.X[k])
        #=
        mul!(tmp.u, fwd.α, bwd.ds[k])
        axpy!(-1.0, tmp.u, fwd.U[k])
        copyto!(tmp.x, fwd.X[k])
        axpy!(-1.0, sol.X[k], tmp.x)
        mul!(tmp.u, bwd.Ks[k], tmp.x)
        axpy!(-1.0, tmp.u, fwd.U[k])
        =#
        mul!(tmp.u, fwd.α, bwd.ds[k])
        @. fwd.U[k] -= tmp.u

        # Compute state difference in tangent space
        get_state_diff!(m, tmp.dx, fwd.X[k], sol.X[k])
        mul!(tmp.u, bwd.Ks[k], tmp.dx)
        @. fwd.U[k] -= tmp.u

        # Step simulator
        d = params.dfwd
        @. d.ctrl = fwd.U[k]
        step!(m, d)

        # Save next state
        x1 = fwd.X[k + 1]
        q1, v1, a1 = get_q(m, x1), get_v(m, x1), get_a(m, x1)
        @. q1 = d.qpos
        @. v1 = d.qvel
        @. a1 = d.act
    end
    return nothing
end

function forward_pass!(
    sol::TrajoptSolution,
    cache::ILqrCache,
    params::TrajoptParameters,
    maxiter_ls::Int,
)::Nothing
    # Get references to ILqrCache structs
    fwd = cache.fwd
    bwd = cache.bwd
    tmp = cache.tmp

    # Initialize line search step size and trajectory cost
    fwd.α = 1.0
    J_ls = 0.0

    # Iterate backtracking line search
    @inbounds for i in 1:maxiter_ls
        # Roll out new gains
        roll_out!(fwd, bwd, tmp, sol, params)

        # Evaluate trajectory cost
        J_ls = params.costfunc(fwd.X, fwd.U, params.Xref, params.Uref)

        # Use decreasing cost as line search criteria
        J_ls < sol.J ? break : nothing

        # Shrink step size
        fwd.α *= fwd.α_mul
    end

    # Save solver iteration data
    fwd.ΔJ = abs(J_ls - sol.J)
    sol.J = J_ls
    copy_nested_array!(sol.X, fwd.X)
    copy_nested_array!(sol.U, fwd.U)
    return nothing
end
