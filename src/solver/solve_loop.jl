function is_converged(cache::SolverCache, tol_converge::AbstractFloat)::Bool
    return cache.bwd.ΔJ < tol_converge
end

function log(sol::Solution, cache::SolverCache, iter::Int)::Nothing
    if rem(iter-1, 20) == 0
        println("-----------------------------------")
        println("iter      J          ΔJ         α")
        println("-----------------------------------")
    end

    @printf(
        "%4.04i   %8.2e   %8.2e   %6.4f\n",
        iter,
        sol.J,
        cache.fwd.ΔJ,
        cache.fwd.α,
    )
    return nothing
end

function assert_opts!(opts::SolverOptions)::Nothing
    if opts.tol_converge <= 0.0
        throw(
            ArgumentError("The stationarity tolerance should be greater than 0")
        )
    end
    if opts.maxiter_solve <= 0
        throw(
            ArgumentError(
                "The max number of iterations should be greater than 0"
            ),
        )
    end
    if opts.maxiter_ls <= 0
        throw(
            ArgumentError(
                "The max number of line search iterations should be greater than 0",
            ),
        )
    end
    return nothing
end

function init_solver!(
    sol::Solution,
    cache::SolverCache,
    params::ProblemParameters,
    eps_reg::Float64,
)::Nothing
    # Get references to SolverCache structs
    fwd = cache.fwd
    bwd = cache.bwd

    # Set regularizer matrix
    mul!(bwd.μ, eps_reg, I)

    # Initialize gains
    fill!.(bwd.Ks, 0.0)
    fill!.(bwd.D, 0.0)

    # Set initial conditions
    copy!(sol.X[1], params.xic)

    # Initialize solution terms
    sol.J = Inf
    sol.is_optimal = false

    # Roll out warm-start
    forward_pass!(sol, cache, params, 1)
    return nothing
end

function solve!(
    sol::Solution,
    cache::SolverCache,
    params::ProblemParameters,
    opts::SolverOptions=SolverOptions(),
)::Nothing
    # Verify options are valid
    assert_opts!(opts)

    # Initialize solver variables
    init_solver!(sol, cache, params, opts.eps_reg)

    # Main solve loop
    for i in 1:opts.maxiter_solve
        backward_pass!(cache, params)
        forward_pass!(sol, cache, params, opts.maxiter_ls)

        opts.is_verbose ? log(sol, cache, i) : nothing
        if is_converged(cache, opts.tol_converge)
            sol.is_optimal = true
            opts.is_verbose ? println("\nOptimal solution found!") : nothing
            return nothing
        end
    end

    opts.is_verbose ? println("\nMaximum iterations exceeded!") : nothing
    return nothing
end

function solve(
    params::ProblemParameters, opts::SolverOptions=SolverOptions()
)::Solution
    sol = Solution(params)
    cache = SolverCache(params)
    solve!(sol, cache, params, opts)
    return sol
end
