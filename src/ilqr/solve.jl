function init_al!(
    constrs::ALConstraints{Ta}, opts::iLQROptions{To}
)::Nothing where {Ta,To}
    @inbounds @simd for key in keys(constrs)
        constr = constrs[key]
        Utils.fill_nested_array!(constr.Ρ, opts.rho_init)   # Init penalty parameters
    end
    return nothing
end

function init_ilqr!(
    sol::TrajoptSolution{Ts},
    constrs::ALConstraints{Ta},
    cache::iLQRCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    opts::iLQROptions{To},
)::Nothing where {Ts,Ta,Tc,To,Tp,Lk,Lf}
    # Get references to iLQRCache structs
    fwd, bwd = cache.fwd, cache.bwd

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
    forward_pass!(sol, constrs, cache, params, 1, opts.save_bestsol)
    return nothing
end

function run_ilqr!(
    sol::TrajoptSolution{Ts},
    constrs::ALConstraints{Ta},
    cache::iLQRCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    opts::iLQROptions{To}=iLQROptions{Tp}(),
    iter::Int,
)::Int where {Ts,Ta,Tc,To,Tp,Lk,Lf}
    while iter < iter + opts.maxiter_ilqr
        backward_pass!(constrs, cache, params)
        forward_pass!(
            sol, constrs, cache, params, opts.maxiter_ls, opts.save_bestsol
        )
        iter += 1
        opts.is_verbose ? log_iter(cache, iter) : nothing
        cache.bwd.ΔJ < opts.tol_ilqr ? break : nothing
    end
    return iter
end

function iterate_lagrange_multipliers!(
    constrs::ALConstraints{T}
)::Nothing where {T}
    @inbounds @simd for key in keys(constrs.u)
        constr = constrs.u[key]
        update_lagrange_multipliers!(constr)
    end
    return nothing
end

function iterate_penalty_parameters!(
    constrs::ALConstraints{T}, opts::iLQROptions{T}
)::Nothing where {T}
    @inbounds @simd for key in keys(constrs.u)
        constr = constrs.u[key]
        update_penalty_parameters!(constr, opts.rho_mul)
    end
    return nothing
end

function constraint_violation_norm(constrs::ALConstraints{T})::T where {T}
    viol = T(0)
    @inbounds @simd for key in keys(constrs.u)
        constr = constr[key]
        if typeof(constr) <: AbstractEqualityConstraint
            viol_new = norm(flatten(constr.C), Inf)
        elseif typeof(constr) <: AbstractInequalityConstraint
            viol_new = maximum(flatten(constr.C))
        else
            Utils.throwarg("Invalid constraint type in constraint set")
        end
        viol = max(viol, viol_new)
    end
    return viol
end

function run_al_ilqr!(
    sol::TrajoptSolution{Ts},
    constrs::ALConstraints{Ta},
    cache::iLQRCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    opts::iLQROptions{To}=iLQROptions{Tp}(),
)::Nothing where {Ts,Ta,Tc,To,Tp,Lk,Lf}
    assert_opts!(opts)
    init_al!(constrs, opts)
    init_ilqr!(sol, constrs, cache, params, opts)

    iter_ilqr = 0
    iter_al = 0
    try     # Main solve loop
        while iter_al < opts.maxiter_al
            iter_ilqr = run_ilqr!(sol, constrs, cache, params, opts, iter_ilqr)
            iterate_lagrange_multipliers!(constrs)
            iterate_penalty_parameters!(constrs, opts)

            iter_al += 1
            opts.is_verbose ? log_al(iter_al) : nothing
            if constraint_violation_norm(constrs) < opts.tol_constr
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
        elseif iter == opts.maxiter_al
            log_maxiter()
        end
    end
    return nothing
end

function run_al_ilqr(
    params::TrajoptParameters{Tp,Lk,Lf}, opts::iLQROptions{To}=iLQROptions{Tp}()
)::TrajoptSolution where {To,Tp,Lk,Lf}
    sol = TrajoptSolution(params)
    constrs = ALConstraints(params)
    cache = iLQRCache(params)
    run_al_ilqr!(sol, constrs, cache, params, opts)
    return sol
end
