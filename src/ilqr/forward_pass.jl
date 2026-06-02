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
    cache::iLQRCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    maxiter_ls::Int,
    save_bestsol::Bool,
)::Nothing where {Ts,Tc,Tp,Lk,Lf}
    # Get references to iLQRCache structs
    al, fwd, bwd, tmp = cache.constr, cache.fwd, cache.bwd, cache.tmp

    # Iterate backtracking line search
    fwd.α = 1.0
    J_ls = 0.0

    @inbounds for i in 1:maxiter_ls
        # Roll out new trajectory
        roll_out!(fwd, bwd, tmp, params)
        J_ls = params.costfunc(fwd.X1, fwd.U1, params.Xref, params.Uref)

        # Update constraint forces and costs
        evaluate_constraints!(al, fwd)
        J_ls += trajectory_constraint_cost(al)

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

function evaluate_control_bound!(
    cache::ControlBoundCache{T}, U::Vector{Vector{T}}, l_or_u::Symbol
)::Nothing where {T}
    C, F, Λ, Ρ, I, B = cache.C, cache.F, cache.Λ, cache.Ρ, cache.I, cache.B
    @inbounds @simd for k in eachindex(U)
        control_bound_residual!(C[k], U[k], B[k], l_or_u)
        inequality_constraint_force!(F[k], C[k], Λ[k], Ρ[k])
        inequality_constraint_indicator!(I[k], F[k])
    end
    return nothing
end

function evaluate_constraints!(
    constr::ConstraintCache{T}, fwd::ForwardCache{T}
)::Nothing where {T}
    ul, uu = constr.ul, constr.uu
    evaluate_control_bound!(ul, fwd.U1, :l)
    evaluate_control_bound!(uu, fwd.U1, :u)
    return nothing
end

function trajectory_constraint_cost(constr::ConstraintCache{T})::T where {T}
    ul, uu = constr.ul, constr.uu
    J_al = T(0)
    @inbounds for k in eachindex(ul.F)
        J_al += inequality_constraint_cost(ul.F[k], ul.Λ[k], ul.Ρ[k])
        J_al += inequality_constraint_cost(uu.F[k], uu.Λ[k], uu.Ρ[k])
    end
    return J_al
end
