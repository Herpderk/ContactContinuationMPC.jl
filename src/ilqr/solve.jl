function init_penalty_parameters!(
    cache::InequalityConstraintCache{T}, ρ0::T
)::Nothing where {T}
    Utils.fill_nested_array!(cache.Ρ, ρ0)
    return nothing
end

function init_control_bound!(
    cache::ControlBoundCache{T},
    params::TrajoptParameters{Tp,Lk,Lf},
    l_or_u::Symbol,
)::Nothing where {T,Tp,Lk,Lf}
    for i in 1:params.mbwd.nu
        if Bool(params.mbwd.actuator_ctrllimited[i])
            cache.B[1][i] = params.mbwd.actuator_ctrlrange[i, 1]
        else
            if l_or_u == :u
                cache.B[1][i] = Inf
            elseif l_or_u == :l
                cache.B[1][i] = -Inf
            else
                throwarg(
                    "Control bounds can only be evaluated as lower (l) or upper (u)!",
                )
            end
        end
    end
    cache.B[2:end] .= cache.B[1]    # Copy control bounds for all time steps
    return nothing
end

function init_al!(
    al::ConstraintCache{T},
    params::TrajoptParameters{Tp,Lk,Lf},
    opts::iLQROptions{T},
)::Nothing where {T}
    ul, uu = al.ul, al.uu

    init_penalty_parameters!(ul, opts.rho_init)
    init_penalty_parameters!(uu, opts.rho_init)

    init_control_bound!(ul, params, :l)
    init_control_bound!(uu, params, :u)
    return nothing
end

function init_ilqr!(
    sol::TrajoptSolution{Ts},
    cache::iLQRCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    opts::iLQROptions{To},
)::Nothing where {Ts,Tc,To,Tp,Lk,Lf}
    # Get references to iLQRCache structs
    al, fwd, bwd = cache.al, cache.fwd, cache.bwd

    # Set line-search contraction rate and merit function tolerance
    fwd.α_mul = opts.alpha_mul
    fwd.β = opts.margin_ls

    # Set regularizer matrix, FD epsilon, and gains
    bwd.ΔJmin = opts.tol_ilqr
    bwd.ΔJmax = opts.tol_interp
    bwd.ϵ = opts.eps_fd
    mul!(bwd.μI, opts.eps_reg, I)
    Utils.fill_nested_array!(bwd.Ks, 0.0)
    Utils.fill_nested_array!(bwd.ds, 0.0)

    # Set initial conditions and solution terms
    sol.is_optimal = false
    sol.J = Inf
    copyto!(sol.X[1], params.xic)

    # Roll out warm-start
    forward_pass!(sol, cache, params, 1, opts.save_bestsol)
    return nothing
end

function run_ilqr!(
    sol::TrajoptSolution{Ts},
    cache::iLQRCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    opts::iLQROptions{To}=iLQROptions{Tp}(),
    iter::Int,
)::Int where {Ts,Tc,To,Tp,Lk,Lf}
    while iter < opts.maxiter_ilqr
        backward_pass!(cache, params)
        forward_pass!(sol, cache, params, opts.maxiter_ls, opts.save_bestsol)
        iter += 1
        opts.is_verbose ? log_iter(cache, iter) : nothing
        cache.bwd.ΔJ < opts.tol_ilqr ? break : nothing
    end
    return iter
end

function iterate_lagrange_multipliers!(
    al::ConstraintCache{T}
)::Nothing where {T}
    ul, uu = al.ul, al.uu
    update_lagrange_multipliers!(ul)
    update_lagrange_multipliers!(uu)
    return nothing
end

function iterate_penalty_parameters!(
    al::ConstraintCache{T}, opts::iLQROptions{T}
)::Nothing where {T}
    ul, uu = al.ul, al.uu
    update_penalty_parameters!(ul, opts.rho_mul)
    update_penalty_parameters!(uu, opts.rho_mul)
end

function constraint_violation(al::ConstraintCache{T})::T where {T}
    ul, uu = al.ul, al.uu
    return max(norm(ul.C, Inf), norm(uu.C, Inf))
end

function run_al_ilqr!(
    sol::TrajoptSolution{Ts},
    cache::iLQRCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    opts::iLQROptions{To}=iLQROptions{Tp}(),
)::Nothing where {Ts,Tc,To,Tp,Lk,Lf}
    assert_opts!(opts)
    init_al!(cache.al, params, opts)
    init_ilqr!(sol, cache, params, opts)

    # Main solve loop
    iter_ilqr = 0
    iter_al = 0
    try
        while iter_al < opts.maxiter_al
            iter_ilqr = run_ilqr!(sol, cache, params, opts, iter_ilqr)

            # Update Lagrange multipliers and penalty parameters
            iterate_lagrange_multipliers!(cache.al)
            iterate_penalty_parameters!(cache.al, opts)

            iter_al += 1
            opts.is_verbose ? log_al(iter_al) : nothing
            if constraint_violation(cache.al) < opts.tol_al
                sol.is_optimal = true
                break
            end
        end
    catch e
        e isa InterruptException ? log_interrupted() : rethrow(e)
    end

    if opts.is_verbose
        if sol.is_optimal
            log_converged()
        elseif iter == opts.maxiter_ilqr
            log_maxiter()
        end
    end
    return nothing
end

function run_al_ilqr(
    params::TrajoptParameters{Tp,Lk,Lf}, opts::iLQROptions{To}=iLQROptions{Tp}()
)::TrajoptSolution where {To,Tp,Lk,Lf}
    sol = TrajoptSolution(params)
    cache = iLQRCache(params)
    run_al_ilqr!(sol, cache, params, opts)
    return sol
end
