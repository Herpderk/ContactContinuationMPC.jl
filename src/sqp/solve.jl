@views function is_converged(
    snorm::Float64,
    pnorm::Float64,
    dnorm::Float64,
    cnorm::Float64;
    tol_stat::Float64,
    tol_primal::Float64,
    tol_dual::Float64,
    tol_comp::Float64,
)::Bool
    return (snorm < tol_stat) &&
           (pnorm < tol_primal) &&
           (dnorm < tol_dual) &&
           (cnorm < tol_comp)
end

@views function primal_infeasibility!(
    p::Vector{Float64},
    gl::Vector{Float64},
    gu::Vector{Float64},
    cache::SQPCache,
)::Float64
    N, gidx = cache.pidx.dims.N, cache.pidx.g

    # Get equality constraint violation
    copyto!(p[gidx.eq], gl[gidx.eq])

    # Get inequality constraint violation
    utmp = cache.utmp
    for k in 1:(N - 1)
        @. utmp = -1.0 * gu[gidx.ub[k]] # Flip the sign of ceiling violation to get a positive value
        @. p[gidx.ub[k]] = max(0.0, gl[gidx.ub[k]], utmp)
    end
    return norm(p, Inf)
end

@views function dual_infeasibility!(
    d::Vector{Float64}, λ::Vector{Float64}, pidx::IndexingParameters
)::Float64
    fill!(d, 0.0)   # Reset dual feasibility violation
    @. d[pidx.g.ineq] = abs(min(0.0, λ[pidx.g.ineq])) # Only evaluate for inequality constraints
    return norm(d, Inf)
end

@views function complementarity_error!(
    c::Vector{Float64}, p::Vector{Float64}, λ::Vector{Float64}
)::Float64
    @. c = p * λ
    return norm(c, Inf)
end

@views function stationarity!(
    ∇ₓL::Vector{Float64},
    ztmp::Vector{Float64},
    ∇J::Vector{Float64},
    ∇g::SparseMatrixCSC{Float64,Int},
    λ::Vector{Float64},
)::Float64
    mul!(ztmp, ∇g', λ)
    @. ∇ₓL = ztmp + ∇J
    return norm(∇ₓL, Inf)
end

function log_converged()::Nothing
    println("------------------------------------------------")
    println("            Optimal solution found!")
    println("------------------------------------------------")
    return nothing
end

function log_maxiter()::Nothing
    println("------------------------------------------------")
    println("     Maximum number of iterations reached!")
    println("------------------------------------------------")
    return nothing
end

function log_interrupted()::Nothing
    println("------------------------------------------------")
    println("                SQP interrupted!")
    println("------------------------------------------------")
    return nothing
end

function log_iter(
    iter::Int,
    J::Float64,
    snorm::Float64,
    pnorm::Float64,
    dnorm::Float64,
    cnorm::Float64,
    α::Float64,
)::Nothing
    if rem(iter, 20) == 1
        println(
            "----------------------------------------------------------------------",
        )
        println(
            "iter       J        ‖∇ₓL‖       ‖p‖        ‖d‖        ‖c‖         α",
        )
        println(
            "----------------------------------------------------------------------",
        )
    end
    @printf(
        "%4.04i   %8.2e   %8.2e   %8.2e   %8.2e   %8.2e   %8.2e\n",
        iter,
        J,
        snorm,
        pnorm,
        dnorm,
        cnorm,
        α
    )
    return nothing
end

function init_sqp!(
    sol::TrajoptSolution{Float64},
    cache::SQPCache,
    params::TrajoptParameters{Float64,Lk,Lf},
    opts::SQPOptions,
)::Nothing where {Lk,Lf}
    # Primal warm-start
    copy_solution_to_primals!(cache.z, sol, cache.pidx)
    update_qparrays!(cache.z, cache, params; ϵfd=opts.eps_fd)

    # Initialize cost
    sol.J = params.costfunc(sol.X, sol.U, params.Xref, params.Uref)
    sol.is_optimal = false
    return nothing
end

@views function copy_solution_to_primals!(
    z::Vector{Float64}, sol::TrajoptSolution{Float64}, pidx::IndexingParameters
)::Nothing
    N, zidx = pidx.dims.N, pidx.z
    @inbounds for k in 1:(N - 1)
        copyto!(z[zidx.x[k]], sol.X[k])
        copyto!(z[zidx.u[k]], sol.U[k])
    end
    copyto!(z[zidx.x[end]], sol.X[end])
    return nothing
end

@views function copy_primals_to_solution!(
    sol::TrajoptSolution{Float64}, z::Vector{Float64}, pidx::IndexingParameters
)::Nothing
    N, zidx = pidx.dims.N, pidx.z
    @inbounds for k in 1:(N - 1)
        copyto!(sol.X[k], z[zidx.x[k]])
        copyto!(sol.U[k], z[zidx.u[k]])
    end
    copyto!(sol.X[end], z[zidx.x[end]])
    return nothing
end

@views function step_primals!(
    z::Vector{Float64},
    Δz::Vector{Float64},
    cache::SQPCache,
    params::TrajoptParameters{Float64,Lk,Lf};
    α::Float64=1.0,
)::Nothing where {Lk,Lf}
    N, zidx, dzidx, dztmp = (
        cache.pidx.dims.N, cache.pidx.z, cache.pidx.dz, cache.dztmp
    )
    @. dztmp = α * Δz

    for k in 1:(N - 1)
        Utils.add_diff_to_state!(params.mfwd, z[zidx.x[k]], dztmp[dzidx.x[k]])
        z[zidx.u[k]] .+= dztmp[dzidx.u[k]]
    end
    Utils.add_diff_to_state!(params.mfwd, z[zidx.x[end]], dztmp[dzidx.x[end]])
    return nothing
end

function run_sqp!(
    sol::TrajoptSolution{Float64},
    cache::SQPCache,
    params::TrajoptParameters{Float64,Lk,Lf},
    opts::SQPOptions,
)::Nothing where {Lk,Lf}
    # References to useful objects
    ∇²ₓₓL, ∇²ₓₓLtriu, ∇ₓL, ∇J, ∇g, gl, gu, gl_pred, gu_pred, p, d, c, z, zcand, dztmp, λ, pidx = (
        cache.∇²ₓₓL,
        cache.∇²ₓₓLtriu,
        cache.∇ₓL,
        cache.∇J,
        cache.∇g,
        cache.gl,
        cache.gu,
        cache.gl_pred,
        cache.gu_pred,
        cache.p,
        cache.d,
        cache.c,
        cache.z,
        cache.zcand,
        cache.dztmp,
        cache.λ,
        cache.pidx,
    )

    # Set OSQP settings
    m, r = cache.m, cache.r
    eps_osqp = min(opts.tol_stat, opts.tol_primal)
    OSQP.update_settings!(
        m; eps_abs=eps_osqp, max_iter=opts.maxiter_qp, warm_start=true
    )

    # Initialize solver state
    init_sqp!(sol, cache, params, opts)
    pnorm = primal_infeasibility!(p, gl, gu, cache)
    γ = 1.0

    # Start SQP loop
    iter = 0
    try
        while iter < opts.maxiter_sqp
            iter += 1

            # Solve QP
            OSQP.update!(m; Px=∇²ₓₓLtriu.nzval, q=∇J, Ax=∇g.nzval, l=gl, u=gu)
            OSQP.warm_start!(m; y=λ)
            OSQP.solve!(m, r)
            if r.info.status_val != 1
                @warn "QP solver did not converge! Status: $(r.info.status)"
            end

            # Precompute QP predicted-decrease terms for Δz = r.x
            # predicted(α) = α * q'Δz + 0.5 * α^2 * Δz' * P * Δz
            ΔJ1 = dot(∇J, r.x)
            mul!(dztmp, ∇²ₓₓL, r.x)
            ΔJ2 = dot(r.x, dztmp)

            # Predicted (linearized) primal infeasibility
            mul!(gl_pred, ∇g, z)
            gl_pred .+= gl
            copyto!(gu_pred, gl_pred)
            inequality_constraint_residuals!(gl_pred, gu_pred, z, cache)
            pnorm_pred = primal_infeasibility!(p, gl_pred, gu_pred, cache)
            Δpnorm_pred = pnorm_pred - pnorm

            # Backtracking line-search
            J_ls = 0.0
            pnorm_ls = 0.0
            α = 1.0
            γ = max(γ, norm(λ, Inf) * 1.01)
            for i in 1:opts.maxiter_ls
                # Step along new search direction
                copyto!(zcand, z) # Candidate primal variables for line-search
                step_primals!(zcand, r.x, cache, params; α=α)

                # Candidate constraint violation
                constraint_residuals!(gl, gu, zcand, cache, params)
                pnorm_ls = primal_infeasibility!(p, gl, gu, cache)

                # Candidate trajectory cost
                copy_primals_to_solution!(sol, zcand, pidx)
                J_ls = params.costfunc(sol.X, sol.U, params.Xref, params.Uref)

                # Evaluate merit function
                merit_pred = α * (ΔJ1 + 0.5*α*ΔJ2 + γ * Δpnorm_pred)
                merit_ls = J_ls - sol.J + γ * (pnorm_ls - pnorm)
                merit_ls < opts.margin_ls*merit_pred ? break : nothing

                # Update step length
                α *= opts.alpha_mul
            end

            # Update solution
            @. r.x *= α
            @. λ = (1-α)*λ + α*(r.y-λ) # Candidate duals for line-search
            @. @views λ[pidx.g.ineq] = max(λ[pidx.g.ineq], 0.0) # Project negative inequality duals to 0

            copyto!(z, zcand)
            copy_primals_to_solution!(sol, z, pidx)
            sol.J = params.costfunc(sol.X, sol.U, params.Xref, params.Uref)
            pnorm = pnorm_ls
            dnorm = dual_infeasibility!(d, λ, pidx)
            cnorm = complementarity_error!(c, p, λ)
            snorm = stationarity!(∇ₓL, dztmp, ∇J, ∇g, λ)

            # Log and check for convergence
            if opts.is_verbose
                log_iter(iter, sol.J, snorm, pnorm, dnorm, cnorm, α)
            else
                nothing
            end
            sol.is_optimal = is_converged(
                snorm,
                pnorm,
                dnorm,
                cnorm;
                tol_stat=opts.tol_stat,
                tol_primal=opts.tol_primal,
                tol_dual=opts.tol_dual,
                tol_comp=opts.tol_comp,
            )
            sol.is_optimal ? break : nothing

            # Set up for next QP solve
            update_qparrays!(z, cache, params; ϵfd=opts.eps_fd)
        end
    catch e
        e isa InterruptException ? log_interrupted() : rethrow(e)
    end

    if sol.is_optimal && opts.is_verbose
        log_converged()
    elseif iter == opts.maxiter && opts.is_verbose
        log_maxiter()
    end
    return nothing
end
