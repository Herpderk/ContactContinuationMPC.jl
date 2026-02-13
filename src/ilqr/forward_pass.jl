function roll_out!(
    fwd::ForwardCache{Tc},
    bwd::BackwardCache{Tc},
    tmp::TemporaryCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
)::Nothing where {Tc,Tp,Lk,Lf}
    # Reference forward model
    m, d = params.mfwd, params.dfwd
    reset!(m, d)
    copy_state_to_data!(d, params.xic)

    # Initialize trajectory with previous solution
    copy_nested_array!(fwd.X1, fwd.X0)
    copy_nested_array!(fwd.U1, fwd.U0)

    # Forward rollout
    @inbounds for k in 1:length(params.Uref)
        # Update control input
        mul!(tmp.u, fwd.α, bwd.ds[k])
        @. fwd.U1[k] -= tmp.u

        # Compute state difference in tangent space
        get_state_diff!(m, tmp.dx, fwd.X1[k], fwd.X0[k])
        mul!(tmp.u, bwd.Ks[k], tmp.dx)
        @. fwd.U1[k] -= tmp.u

        # Step simulator
        copyto!(d.ctrl, fwd.U1[k])
        step!(m, d)
        copy_data_to_state!(fwd.X1[k + 1], d)
    end
    return nothing
end

function forward_pass!(
    sol::TrajoptSolution{Ts},
    cache::ILqrCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    maxiter_ls::Int,
    save_bestsol::Bool,
)::Nothing where {Ts,Tc,Tp,Lk,Lf}
    # Get references to ILqrCache structs
    fwd = cache.fwd
    bwd = cache.bwd
    tmp = cache.tmp

    # Iterate backtracking line search
    fwd.α = 1.0
    J_ls = 0.0

    @inbounds for i in 1:maxiter_ls
        # Roll out new trajectory
        roll_out!(fwd, bwd, tmp, params)
        J_ls = params.costfunc(fwd.X1, fwd.U1, params.Xref, params.Uref)

        # Line-search criteria
        # Actual change in cost must be as good as β*predicted change
        ΔJ_pred = bwd.ΔJ1*fwd.α + 0.5*bwd.ΔJ2*fwd.α^2
        fwd.ΔJ = J_ls - fwd.Jprev
        fwd.ΔJ < -fwd.β * ΔJ_pred ? break : nothing

        # Shrink step size
        fwd.α *= fwd.α_mul
    end

    # Carry solver state
    fwd.Jprev = J_ls
    copy_nested_array!(fwd.X0, fwd.X1)
    copy_nested_array!(fwd.U0, fwd.U1)

    # Update solution if new one is better
    if (!save_bestsol) || (save_bestsol && fwd.Jprev < sol.J)
        sol.J = fwd.Jprev
        copy_nested_array!(sol.X, fwd.X0)
        copy_nested_array!(sol.U, fwd.U0)
    end
    return nothing
end
