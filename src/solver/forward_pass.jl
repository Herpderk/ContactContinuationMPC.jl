function roll_out!(
    fwd::ForwardCache,
    bwd::BackwardCache,
    tmp::TemporaryCache,
    sol::Solution,
    params::ProblemParameters,
)::Nothing
    # Initialize trajectory with previous solution
    BLAS.copy!.(fwd.X, sol.X)
    BLAS.copy!.(fwd.U, sol.U)

    # Forward rollout
    @inbounds for k = 1:length(params.Uref)
        # Update control input
        #fwd.U[k] = sol.U[k] - α*ds[k] - Ks[k]*(fwd.X[k] - sol.X[k])
        mul!(tmp.u, fwd.α, bwd.D[k])
        BLAS.axpy!(-1.0, tmp.u, fwd.U[k])
        BLAS.copy!(tmp.x, fwd.X[k])
        BLAS.axpy!(-1.0, sol.X[k], tmp.x)
        mul!(tmp.u, bwd.Ks[k], tmp.x)
        BLAS.axpy!(-1.0, tmp.u, fwd.U[k])

        # Step simulator
        params.simfunc_fwd!(fwd.X[k+1], fwd.X[k], fwd.U[k])
    end
end


function forward_pass!(
    sol::Solution,
    cache::SolverCache,
    params::ProblemParameters,
    maxiter_ls::Int,
)::Nothing
    # Get references to SolverCache structs
    fwd = cache.fwd
    bwd = cache.bwd
    tmp = cache.tmp

    # Initialize line search step size and trajectory cost
    fwd.α = 1.0
    J_ls = 0.0

    # Iterate backtracking line search
    @inbounds for i = 1:maxiter_ls
        # Roll out new gains
        roll_out!(fwd, bwd, tmp, sol, params)

        # Evaluate trajectory cost
        J_ls = params.costfunc(fwd.X, fwd.U, params.Xref, params.Uref)

        # Use decreasing cost as line search criteria
        J_ls < sol.J ? break : nothing

        # Shrink step size
        fwd.α *= 0.5
    end

    # Save solver iteration data
    fwd.ΔJ = abs(J_ls - sol.J)
    sol.J = J_ls
    BLAS.copy!.(sol.X, fwd.X)
    BLAS.copy!.(sol.U, fwd.U)
end
