
function terminate(cache::SolverCache, tol_converge::Float64)::Bool
    return cache.fwd.ΔJ < tol_converge
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
    if opts.tol_converge <= 0.0
        ArgumentError("The stationarity tolerance should be greater than 0")
    end
    if opts.maxiter_solve <= 0
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
    eps_reg::Float64,
)::Nothing
    # Get references to SolverCache structs
    fwd = cache.fwd
    bwd = cache.bwd

    # Set regularizer matrix
    mul!(bwd.μ, eps_reg, I)

    # Initialize gains
    fill!.(bwd.Ks, 0.0)
    fill!.(bwd.ds, 0.0)

    # Set initial conditions
    fwd.modes[1] = params.fwd_sys.modes[params.mI]
    BLAS.copy!(sol.xs[1], params.x0)

    # Initialize trajectory cost
    sol.J = Inf

    # Roll out with a full newton step
    forward_pass!(sol, cache, params, 1)
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
    init_solver!(sol, cache, params, opts.eps_reg)

    # Main solve loop
    for i = 1:opts.maxiter_solve
        backward_pass!(cache, params)
        forward_pass!(sol, cache, params, opts.maxiter_ls)

        opts.is_verbose ? log(sol, cache, i) : nothing
        if terminate(cache, opts.tol_converge)
            opts.is_verbose ? println("\nOptimal solution found!") : nothing
            return
        end
    end

    opts.is_verbose ? println("\nMaximum iterations exceeded!") : nothing
    return
end

function solve(params::ProblemParameters, opts::SolverOptions = SolverOptions())::Solution
    sol = Solution(params)
    cache = SolverCache(params)
    solve!(sol, cache, params, opts)
    return sol
end
