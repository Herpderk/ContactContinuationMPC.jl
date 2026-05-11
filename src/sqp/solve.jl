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
    tmp::TemporaryCache,
    pidx::IndexingParameters,
)::Float64
    # Get equality constraint violation
    copyto!(p[pidx.g.eq], gl[pidx.g.eq])

    # Get inequality constraint violation
    for k in 1:(pidx.dims.N - 1)
        @. tmp.u = -1.0 * gu[pidx.g.ub[k]] # Flip the sign of ceiling violation to get a positive value
        @. p[pidx.g.ub[k]] = max(0.0, gl[pidx.g.ub[k]], tmp.u)
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
    dztmp::Vector{Float64},
    ∇J::Vector{Float64},
    ∇g::SparseMatrixCSC{Float64,Int},
    λ::Vector{Float64},
)::Float64
    mul!(dztmp, ∇g', λ)
    @. ∇ₓL = dztmp + ∇J
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
    dztmp::Vector{Float64},
    Δz::Vector{Float64},
    pidx::IndexingParameters,
    params::TrajoptParameters{Float64,Lk,Lf};
    α::Float64=1.0,
)::Nothing where {Lk,Lf}
    N, zidx, dzidx = pidx.dims.N, pidx.z, pidx.dz
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
    pidx, qp, ad, ls, kkt, sol_sqp, tmp, FDs = (
        cache.pidx,
        cache.qp,
        cache.ad,
        cache.ls,
        cache.kkt,
        cache.sol,
        cache.tmp,
        cache.FDs,
    )
    ∇²ₓₓL, ∇J, ∇g, gl, gu = (qp.∇²ₓₓL, qp.∇J, qp.∇g, qp.gl, qp.gu)
    zcand, gl_cand, gu_cand, gl_pred, gu_pred = (
        ls.zcand, ls.gl_cand, ls.gu_cand, ls.gl_pred, ls.gu_pred
    )
    z, λ = sol_sqp.z, sol_sqp.λ
    ∇ₓL, p, d, c = kkt.∇ₓL, kkt.p, kkt.d, kkt.c

    # Initialize cost
    sol.J = params.costfunc(sol.X, sol.U, params.Xref, params.Uref)
    sol.is_optimal = false

    γ0 = opts.gamma_init

    # Initialize primal infeasibility
    pnorm = primal_infeasibility!(p, gl, gu, tmp, pidx)

    # Primal warm-start
    copy_solution_to_primals!(z, sol, pidx)
    update_qp!(qp, ad, tmp, FDs, sol_sqp, opts, pidx, params)

    # Set up QP solver
    eps_osqp = min(opts.tol_stat, opts.tol_primal)
    OSQP.update_settings!(
        qp.m; eps_abs=eps_osqp, max_iter=opts.maxiter_qp, warm_start=true
    )

    # Start SQP loop
    iter = 0
    try
        while iter < opts.maxiter_sqp
            iter += 1
            solve_qp!(qp, qp.r.x, λ)

            # Precompute QP predicted-decrease terms for Δz = r.x
            # predicted(α) = α * q'Δz + 0.5 * α^2 * Δz' * P * Δz
            ΔJ1 = dot(∇J, qp.r.x)
            mul!(tmp.dz, ∇²ₓₓL, qp.r.x)
            ΔJ2 = dot(qp.r.x, tmp.dz)

            # Predicted (linearized) primal infeasibility for Δz = r.x
            # Use linearization for both lower and upper residuals:
            mul!(gl_pred, ∇g, qp.r.x)
            copyto!(gu_pred, gl_pred)
            gl_pred .+= gl
            gu_pred .+= gu
            #inequality_constraint_residuals!(gl_pred, gu_pred, opts.ul, opts.uu, z, pidx)
            pnorm_pred = primal_infeasibility!(p, gl_pred, gu_pred, tmp, pidx)
            Δpnorm_pred = pnorm_pred - pnorm

            # Backtracking line-search
            J_ls = 0.0
            pnorm_ls = 0.0
            α = 1.0
            γ = max(γ0, norm(λ, Inf) * 1.01)
            for i in 1:opts.maxiter_ls
                # Step along new search direction
                copyto!(zcand, z) # Candidate primal variables for line-search
                step_primals!(zcand, tmp.dz, qp.r.x, pidx, params; α=α)

                # Candidate constraint violation
                constraint_residuals!(
                    gl_cand, gu_cand, zcand, tmp, opts, pidx, params
                )
                pnorm_ls = primal_infeasibility!(p, gl_cand, gu_cand, tmp, pidx)

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
            @. qp.r.x *= α                  # Primal step from line-search
            @. λ = (1-α)*λ + α*qp.r.y       # Duals from line-search
            @. @views λ[pidx.g.ineq] = max(λ[pidx.g.ineq], 0.0) # Project negative inequality duals to 0

            # Update solution and QP arrays before checking convergence
            copyto!(z, zcand)
            copy_primals_to_solution!(sol, z, pidx)
            sol.J = params.costfunc(sol.X, sol.U, params.Xref, params.Uref)
            update_qp!(qp, ad, tmp, FDs, sol_sqp, opts, pidx, params)

            #= Adapt merit weight if infeasibility did not decrease
            if pnorm_ls > pnorm + 1e-12
                γ0 *= 2.0
                if opts.is_verbose
                    @printf("Increased gamma to %8.2e due to infeasibility stagnation\n", γ0)
                end
            elseif pnorm_ls < max(1e-12, pnorm*0.1)
                γ0 /=2.0
                if opts.is_verbose
                    @printf("Reduced gamma to %8.2e after strong infeasibility decrease\n", γ0)
                end
            end
            γ0 = clamp(γ0, opts.gamma_init, 1e+6) =#

            # Check KKT conditions for convergence
            pnorm = pnorm_ls
            dnorm = dual_infeasibility!(d, λ, pidx)
            cnorm = complementarity_error!(c, p, λ)
            snorm = stationarity!(∇ₓL, tmp.dz, ∇J, ∇g, λ)
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
        end
    catch e
        e isa InterruptException ? log_interrupted() : rethrow(e)
    end

    if sol.is_optimal && opts.is_verbose
        log_converged()
    elseif iter == opts.maxiter_sqp && opts.is_verbose
        log_maxiter()
    end
    return nothing
end
