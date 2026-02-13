function is_converged(cache::ILqrCache, tol_converge::AbstractFloat)::Bool
    #return abs(cache.fwd.ΔJ) < tol_converge
    return cache.bwd.ΔJ < tol_converge
end

function log_converged()::Nothing
    println("-------------------------------------")
    println("       Optimal solution found!")
    println("-------------------------------------")
    return nothing
end

function log_maxiter()::Nothing
    println("-------------------------------------")
    println("Maximum number of iterations reached!")
    println("-------------------------------------")
    return nothing
end

function log_interrupted()::Nothing
    println("-------------------------------------")
    println("          iLQR interrupted!")
    println("-------------------------------------")
    return nothing
end

function log_iter(cache::ILqrCache, iter::Int)::Nothing
    if rem(iter-1, 20) == 0
        println("-------------------------------------")
        println("iter       J         ΔJ          α")
        println("-------------------------------------")
    end
    @printf(
        "%4.04i   %8.2e   %8.2e   %8.2e\n",
        iter,
        cache.fwd.Jprev,
        cache.bwd.ΔJ,
        cache.fwd.α,
    )
    return nothing
end

function assert_opts!(opts::ILqrOptions)::Nothing
    if !(0.0 < opts.alpha_mul < 1.0)
        throwdom(
            opts.alpha_mul,
            "The backtracking contraction rate must be between 0 and 1",
        )
    end
    if opts.eps_reg < 0.0
        throwdom(
            opts.eps_reg,
            "The regularizer coefficient must be greater than or equal to 0",
        )
    end
    if opts.eps_fd <= 0.0
        throwdom(
            opts.eps_fd,
            "The finite-difference coefficient must be greater than 0",
        )
    end
    if opts.tol_converge <= 0.0
        throwdom(
            opts.tol_converge,
            "The stationarity tolerance must be greater than 0",
        )
    end
    if opts.margin_ls < 0.0
        throwdom(
            opts.margin_ls,
            "The merit function margin factor must be greater than or equal to 0",
        )
    end
    if opts.maxiter_ilqr <= 0
        throwdom(
            opts.maxiter_ilqr,
            "The max number of iterations must be greater than 0",
        )
    end
    if opts.maxiter_ls <= 0
        throwdom(
            opts.maxiter_ls,
            "The max number of line-search iterations must be greater than 0",
        )
    end
    return nothing
end

function init_ilqr!(
    sol::TrajoptSolution{Ts},
    cache::ILqrCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    opts::ILqrOptions{To},
)::Nothing where {Ts,Tc,To,Tp,Lk,Lf}
    # Get references to ILqrCache structs
    fwd = cache.fwd
    bwd = cache.bwd

    # Set line-search contraction rate and merit function tolerance
    fwd.α_mul = opts.alpha_mul
    fwd.β = opts.margin_ls

    # Set regularizer matrix, FD epsilon, and gains
    bwd.ΔJmin = opts.tol_converge
    bwd.ΔJmax = opts.tol_interp
    bwd.ϵ = opts.eps_fd
    mul!(bwd.μI, opts.eps_reg, I)
    fill_nested_array!(bwd.Ks, 0.0)
    fill_nested_array!(bwd.ds, 0.0)

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
    cache::ILqrCache{Tc},
    params::TrajoptParameters{Tp,Lk,Lf},
    opts::ILqrOptions{To}=ILqrOptions{Tp}(),
)::Nothing where {Ts,Tc,To,Tp,Lk,Lf}
    assert_opts!(opts)
    init_ilqr!(sol, cache, params, opts)

    # Main solve loop
    iter = 0
    try
        while iter < opts.maxiter_ilqr
            iter += 1
            backward_pass!(cache, params)
            forward_pass!(
                sol, cache, params, opts.maxiter_ls, opts.save_bestsol
            )

            opts.is_verbose ? log_iter(cache, iter) : nothing

            if is_converged(cache, opts.tol_converge)
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

function run_ilqr(
    params::TrajoptParameters{Tp,Lk,Lf}, opts::ILqrOptions{To}=ILqrOptions{Tp}()
)::TrajoptSolution where {To,Tp,Lk,Lf}
    sol = TrajoptSolution(params)
    cache = ILqrCache(params)
    run_ilqr!(sol, cache, params, opts)
    return sol
end
