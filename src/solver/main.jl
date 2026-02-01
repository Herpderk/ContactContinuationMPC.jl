
function terminate(cache::SolverCache, tol_stat::Float64)::Bool
    return cache.fwd.ΔJ < tol_stat
end


function log(sol::Solution, cache::SolverCache, iter::Int)::Nothing
    if rem(iter-1, 20) == 0
        println("------------------------------------------")
        println("iter        J          ΔJ        α       τ")
        println("------------------------------------------")
    end

    τ = 0
    for trn in cache.fwd.trn_syms
        τ = trn != NULL_TRANSITION ? τ+1 : τ
    end

    @printf(
        "%4.04i     %8.2e   %8.2e   %7.5f   %3.03i\n",
        iter,
        sol.J,
        cache.fwd.ΔJ,
        cache.fwd.α,
        τ
    )
end


function assert_opts!(opts::SolverOptions)::Nothing
    if opts.max_step <= 0.0 || opts.max_step > 1.0
        ArgumentError("The max step size should be between 0 and 1")
    end
    if opts.tol_stat <= 0.0
        ArgumentError("The stationarity tolerance should be greater than 0")
    end
    if opts.maxiter_opt <= 0
        ArgumentError("The max number of iterations should be greater than 0")
    end
    if opts.maxiter_ls <= 0
        ArgumentError("The max number of line search iterations should be greater than 0")
    end
end


function init_solver!(
    sol::Solution,
    cache::SolverCache,
    params::ProblemParameters,
    regularizer::Float64,
)::Nothing
    # Get references to SolverCache structs
    fwd = cache.fwd
    bwd = cache.bwd

    # Set regularizer matrix
    mul!(bwd.μ, regularizer, I)

    # Initialize gains
    fill!.(bwd.Ks, 0.0)
    fill!.(bwd.ds, 0.0)

    # Set initial conditions
    fwd.modes[1] = params.fwd_sys.modes[params.mI]
    BLAS.copy!(sol.xs[1], params.x0)

    # Initialize trajectory cost
    sol.J = Inf

    # Roll out with a full newton step
    forward_pass!(sol, cache, params, 1.0, 1.0, 1)
    return
end


function solve!(
    sol::Solution,
    cache::SolverCache,
    params::ProblemParameters,
    opts::SolverOptions,
)::Nothing
    # Verify options are valid
    assert_opts!(opts)

    # Initialize solver variables
    init_solver!(sol, cache, params, opts.regularizer)

    # Main solve loop
    for i = 1:opts.maxiter_opt
        backward_pass!(cache, params)
        forward_pass!(sol, cache, params, opts.max_step, opts.maxiter_ls)

        opts.verbose ? log(sol, cache, i) : nothing
        if terminate(cache, opts.tol_stat)
            opts.verbose ? println("\nOptimal solution found!") : nothing
            return
        end
    end

    opts.verbose ? println("\nMaximum iterations exceeded!") : nothing
    return
end

function solve(params::ProblemParameters, opts::SolverOptions = SolverOptions())::Solution
    sol = Solution(params)
    cache = SolverCache(params)
    solve!(sol, cache, params, opts)
    return sol
end
