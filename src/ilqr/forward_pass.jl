function roll_out!(
    fwd::ForwardCache,
    bwd::BackwardCache,
    tmp::TemporaryCache,
    sol::TrajoptSolution,
    params::TrajoptParameters,
)::Nothing
    # Reference forward model
    m, d = params.mfwd, params.dfwd
    reset!(m, d)
    copy_state_to_data!(d, params.xic)

    # Initialize trajectory with previous solution
    copy_nested_array!(fwd.X, sol.X)
    copy_nested_array!(fwd.U, sol.U)

    # Forward rollout
    @inbounds for k in 1:length(params.Uref)
        # Update control input
        mul!(tmp.u, fwd.α, bwd.ds[k])
        @. fwd.U[k] -= tmp.u

        # Compute state difference in tangent space
        get_state_diff!(m, tmp.dx, fwd.X[k], sol.X[k])
        mul!(tmp.u, bwd.Ks[k], tmp.dx)
        @. fwd.U[k] -= tmp.u

        # Step simulator
        copyto!(d.ctrl, fwd.U[k])
        step!(m, d)
        copy_data_to_state!(fwd.X[k + 1], d)
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

    # Iterate backtracking line search
    fwd.α = 1.0
    J_ls = 0.0

    @inbounds for i in 1:maxiter_ls
        # Roll out new trajectory
        roll_out!(fwd, bwd, tmp, sol, params)
        J_ls = params.costfunc(fwd.X, fwd.U, params.Xref, params.Uref)

        # Line-search criteria
        # Actual change in cost must be as good as β*predicted change
        ΔJ_pred = bwd.ΔJ1*fwd.α + 0.5*bwd.ΔJ2*fwd.α^2
        fwd.ΔJ = J_ls - sol.J
        fwd.ΔJ < -fwd.β * ΔJ_pred ? break : nothing

        # Shrink step size
        fwd.α *= fwd.α_mul
    end

    # Save new solution
    sol.J = J_ls
    copy_nested_array!(sol.X, fwd.X)
    copy_nested_array!(sol.U, fwd.U)
    return nothing
end
