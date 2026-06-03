function roll_out!(
    fwd::ForwardCache{Tc},
    bwd::BackwardCache{Tc},
    tmp::TemporaryCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
)::Nothing where {Tc,Tp,Lk,Lf}
    # Reference forward model
    m, d = params.mfwd, params.dfwd
    reset!(m, d)
    Utils.copy_state_to_data!(d, params.xic)

    # Initialize trajectory with previous solution
    Utils.copy_nested_array!(fwd.X1, fwd.X0)
    Utils.copy_nested_array!(fwd.U1, fwd.U0)

    # Forward rollout
    @inbounds for k in 1:length(params.Uref)
        # Update control input
        mul!(tmp.u1, fwd.α, bwd.ds[k])
        fwd.U1[k] .-= tmp.u1

        # Compute state difference in tangent space
        Utils.get_state_diff!(m, tmp.dx, fwd.X1[k], fwd.X0[k])
        mul!(tmp.u1, bwd.Ks[k], tmp.dx)
        fwd.U1[k] .-= tmp.u1

        # Step simulator
        copyto!(d.ctrl, fwd.U1[k])
        step!(m, d)
        Utils.copy_data_to_state!(d, fwd.X1[k + 1])
    end
    return nothing
end

function forward_pass!(
    sol::TrajoptSolution{Ts},
    constrs::ALConstraints{Ta},
    cache::iLQRCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    maxiter_ls::Int,
    save_bestsol::Bool,
)::Nothing where {Ts,Ta,Tc,Tp,Lk,Lf}
    # Get references to iLQRCache structs
    fwd, bwd, tmp = cache.fwd, cache.bwd, cache.tmp

    # Iterate backtracking line search
    fwd.α = 1.0
    J_ls = 0.0

    @inbounds for i in 1:maxiter_ls
        # Roll out new trajectory
        roll_out!(fwd, bwd, tmp, params)
        J_ls = params.costfunc(fwd.X1, fwd.U1, params.Xref, params.Uref)
        J_ls += evaluate_constraints!(constrs.u, fwd)   # Update constraint evaluations and costs

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
    Utils.copy_nested_array!(fwd.X0, fwd.X1)
    Utils.copy_nested_array!(fwd.U0, fwd.U1)

    # Update solution if new one is better
    if (!save_bestsol) || (save_bestsol && fwd.Jprev < sol.J)
        sol.J = fwd.Jprev
        Utils.copy_nested_array!(sol.X, fwd.X0)
        Utils.copy_nested_array!(sol.U, fwd.U0)
    end
    return nothing
end

"""
Update constraint residuals, forces, and indicators for a given constraint set and return its AL cost.
"""
function evaluate_constraints!(
    constrs::ConstraintSet, fwd::ForwardCache{T}
)::T where {T}
    if constrs.n_input != length(fwd.U1[1])
        throw(
            ArgumentError(
                "Constraint set dimensions must match that of states or controls",
            ),
        )
    end
    inputs = fwd.U1

    @inbounds for key in keys(constrs)
        constr = constrs[key]
        update_residuals!(constr, inputs)
        update_forces!(constr)
        update_indicators!(constr)
    end

    J_al = T(0)
    @inbounds for key in keys(constrs)
        constr = constrs[key]
        J_al += get_trajectory_cost(constr)
    end
    return J_al
end
