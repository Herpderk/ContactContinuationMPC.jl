@views function is_converged(
    statnorm::Float64, viol::Float64; tol_stat::Float64, tol_eq::Float64
)::Bool
    return statnorm < tol_stat && viol < tol_eq
end

@views function constraint_violation(
    g::Vector{Float64}, pidx::IndexingParameters
)::Float64
    start_eq = pidx.g.ic[1][1]
    end_eq = pidx.g.dyn[end][end]
    geq = g[start_eq:end_eq]
    return norm(geq, Inf)
end

@views function predicted_constraint_violation(
    ∇g::SparseMatrixCSC{Float64,Int},
    g::Vector{Float64},
    gtmp::Vector{Float64},
    z::Vector{Float64},
    pidx::IndexingParameters,
)::Float64
    mul!(gtmp, ∇g, z)
    gtmp .+= g
    start_eq = first(pidx.g.ic)
    endidx_eq = pidx.g.dyn[end][end]
    return norm(gtmp[start_eq:endidx_eq], Inf)
end

@views function stationarity!(
    ∇ₓL::Vector{Float64},
    ztmp::Vector{Float64},
    ∇J::Vector{Float64},
    ∇g::SparseMatrixCSC{Float64,Int},
    λ::Vector{Float64},
    pidx::IndexingParameters,
)::Float64
    start_eq = pidx.g.ic[1][1]
    endidx_eq = pidx.g.dyn[end][end]
    ∇geq = ∇g[start_eq:endidx_eq, :]
    λeq = λ[start_eq:endidx_eq]
    mul!(ztmp, ∇geq', λeq)
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
    iter::Int, J::Float64, statnorm::Float64, viol::Float64, α::Float64
)::Nothing
    if rem(iter, 20) == 1
        println("------------------------------------------------")
        println("iter       J        ‖∇ₓL‖       ‖g‖         α")
        println("------------------------------------------------")
    end
    @printf(
        "%4.04i   %8.2e   %8.2e   %8.2e   %8.2e\n", iter, J, statnorm, viol, α
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
    update_qparrays!(cache.z, cache, params; ϵfd=opts.eps_fd, ϵreg=opts.eps_reg)

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
    # References to OSQP structs
    m, r = cache.m, cache.r

    # References to useful objects
    ∇²ₓₓL, ∇²ₓₓLtriu, ∇ₓL, ∇J, ∇g, gl, gu, vl, vu, z, zcand, dztmp, gtmp, λ, λcand, pidx = (
        cache.∇²ₓₓL,
        cache.∇²ₓₓLtriu,
        cache.∇ₓL,
        cache.∇J,
        cache.∇g,
        cache.gl,
        cache.gu,
        cache.vl,
        cache.vu,
        cache.z,
        cache.zcand,
        cache.dztmp,
        cache.gtmp,
        cache.λ,
        cache.λcand,
        cache.pidx,
    )

    # Initialize solver state
    init_sqp!(sol, cache, params, opts)
    viol = constraint_violation(gl, pidx)
    γ = 1.0

    # Start SQP loop
    iter = 0
    try
        while iter < opts.maxiter
            iter += 1

            # Solve QP
            OSQP.update_settings!(m; eps_abs=1e-6, eps_rel=1e-6, max_iter=1000)

            l = 1*gl
            @views l[1:pidx.g.dyn[end][end]] .*= -1.0
            u = 1*gu
            @views u[1:pidx.g.dyn[end][end]] .*= -1.0
            OSQP.update!(m; Px=∇²ₓₓLtriu.nzval, q=∇J, Ax=∇g.nzval, l=l, u=u)
            OSQP.solve!(m, r)

            if r.info.status_val != 1
                println("OSQP status: ", r.info.status)
                println("P condition number: ", cond(Array(∇²ₓₓL), 2))
                println(
                    "Pmax: ",
                    maximum(abs, ∇²ₓₓL),
                    ", Pmin: ",
                    minimum(abs, ∇²ₓₓL),
                )
                println("max step: ", maximum(r.x))
                # 1. Check for literal NaNs or Infs
                println("--- Pre-Solve Sanity Check ---")
                println(
                    "P (Hessian) contains NaN/Inf? : ",
                    any(isnan, ∇²ₓₓL.nzval) || any(isinf, ∇²ₓₓL.nzval),
                )
                println(
                    "q (Gradient) contains NaN/Inf?: ",
                    any(isnan, ∇J) || any(isinf, ∇J),
                )
                println(
                    "A (Jacobian) contains NaN/Inf?: ",
                    any(isnan, ∇g.nzval) || any(isinf, ∇g.nzval),
                )
                println("l (Lower Bnd) contains NaN?   : ", any(isnan, gl)) # Inf is okay for bounds, NaN is not
                println("u (Upper Bnd) contains NaN?   : ", any(isnan, gu))

                # 2. Check for astronomical numbers (The FD Explosion check)
                println("Max value in P: ", maximum(abs, ∇²ₓₓL.nzval))
                println("Max value in A: ", maximum(abs, ∇g.nzval))
                println("Max value in q: ", maximum(abs, ∇J))
                println("------------------------------")

                is_sym = issymmetric(∇²ₓₓL)
                println("Is P perfectly symmetric?: ", is_sym)

                if !is_sym
                    # Find the maximum asymmetry error
                    P_dense = Matrix(∇²ₓₓL)
                    asym_error = maximum(abs.(P_dense - P_dense'))
                    println("Maximum asymmetry error: ", asym_error)
                end

                # Try to Cholesky factorize a dense, perfectly symmetric version of P
                P_sym = Symmetric(Matrix(∇²ₓₓL))
                is_psd = isposdef(P_sym)
                println("Is P strictly positive definite?: ", is_psd)

                if !is_psd
                    # If it fails, let's look at the worst eigenvalue
                    eigenvalues = eigvals(P_sym)
                    println("Minimum eigenvalue of P: ", minimum(eigenvalues))
                end
            end

            # Precompute QP predicted-decrease terms for Δz = r.x
            # predicted(α) = α * q'Δz + 0.5 * α^2 * Δz' * P * Δz
            pP = ∇²ₓₓL * r.x
            base_q = dot(∇J, r.x)
            base_quad = dot(r.x, pP)

            # Backtracking line-search
            J_ls = 0.0
            viol_ls = 0.0
            maxiter_ls = 20
            αmul = 0.5
            α = 1.0
            β = 1e-2

            endidx_eq = pidx.g.dyn[end][end]
            @views λeq = r.y[1:endidx_eq]
            γ = max(γ, norm(λeq, Inf) * 1.1)

            # The linearized constraint is: ∇g * Δz + g(z)
            # Since you pass -g(z) to OSQP as `gl`, then g(z) = -gl
            predicted_viol = predicted_constraint_violation(
                ∇g, gl, gtmp, z, pidx
            )
            predicted_viol_change = predicted_viol - viol

            # 3. The TRUE predicted merit change at a full step (α = 1.0)
            # Notice we add `γ * predicted_viol_change` instead of `- γ * viol`
            predicted_merit_change_full = base_q + γ * predicted_viol_change

            for i in 1:maxiter_ls
                # Step along new search direction
                copyto!(zcand, z) # Candidate primal variables for line-search
                step_primals!(zcand, r.x, cache, params; α=α)

                # Get new constraint residuals
                initcond_residuals!(gl, zcand, pidx, params)
                dynamics_residuals!(gl, zcand, cache, params)

                # Evalute merit function ingredients
                viol_ls = constraint_violation(gl, pidx)
                copy_primals_to_solution!(sol, zcand, pidx)
                J_ls = params.costfunc(sol.X, sol.U, params.Xref, params.Uref)

                predicted_merit_change = α * predicted_merit_change_full

                # The sufficient decrease condition
                merit_cand = J_ls + γ * viol_ls
                merit_base = sol.J + γ * viol
                if merit_cand < merit_base + β * predicted_merit_change
                    break
                else
                    nothing
                end

                # Update step length
                α *= αmul
            end

            # Update solution
            @. λ = (1-α)*λ + α*(r.y-λ) # Candidate duals for line-search
            copyto!(z, zcand)
            copy_primals_to_solution!(sol, z, pidx)
            sol.J = params.costfunc(sol.X, sol.U, params.Xref, params.Uref)
            viol = viol_ls
            statnorm = stationarity!(∇ₓL, dztmp, ∇J, ∇g, λ, pidx) # statnorm_ls

            update_qparrays!(
                z, cache, params; ϵfd=opts.eps_fd, ϵreg=opts.eps_reg
            )

            # Log and check for convergence
            opts.is_verbose ? log_iter(iter, sol.J, statnorm, viol, α) : nothing
            sol.is_optimal = is_converged(
                statnorm, viol; tol_stat=opts.tol_stat, tol_eq=opts.tol_eqconstr
            )
            sol.is_optimal ? break : nothing
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
