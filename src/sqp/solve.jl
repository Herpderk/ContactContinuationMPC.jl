@views function is_converged(
    statnorm::Float64, viol::Float64; tol_stat::Float64, tol_eq::Float64
)::Bool
    return statnorm < tol_stat && viol < tol_eq
end

@views function constraint_violation(
    g::Vector{Float64}, pidx::IndexingParameters
)::Float64
    geq = g[pidx.g.ic[1][1]:pidx.g.dyn[end][end]]
    return norm(geq, Inf)
end

@views function stationarity!(
    ∇ₓL::Vector{Float64},
    ztmp::Vector{Float64},
    ∇J::Vector{Float64},
    ∇g::SparseMatrixCSC{Float64,Int},
    λ::Vector{Float64},
    pidx::IndexingParameters,
)::Float64
    #println("ztmp length:", length(ztmp))
    #println("∇g shape:", size(∇g))
    #println("λ length:", length(λ))
    endidx_eq = pidx.g.dyn[end][end]
    ∇geq = ∇g[1:endidx_eq, :]
    λeq = λ[1:endidx_eq]
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
    if rem(iter, 20) == 0
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
)::Nothing where {Lk,Lf}
    # Primal warm-start
    copy_solution_to_primals!(cache.z, sol, cache.pidx)
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
    ∇²ₓₓL, ∇ₓL, ∇J, ∇g, gl, gu, Δxl, Δxu, Δul, Δuu, z, ztmp, pidx = (
        cache.∇²ₓₓL,
        cache.∇ₓL,
        cache.∇J,
        cache.∇g,
        cache.gl,
        cache.gu,
        cache.Δxl,
        cache.Δxu,
        cache.Δul,
        cache.Δuu,
        cache.z,
        cache.ztmp,
        cache.pidx,
    )

    # Initialize solver state
    init_sqp!(sol, cache, params)
    initcond_residuals!(gl, z, pidx, params)
    dynamics_residuals!(gl, z, cache, params)
    viol = constraint_violation(gl, pidx)

    # Start SQP loop
    iter = 0
    try
        while iter < opts.maxiter
            iter += 1

            # Update inequality constraints
            copyto!(gu, gl)
            trustregion_bounds!(gl, Δxl, Δul, pidx)
            trustregion_bounds!(gu, Δxu, Δuu, pidx)

            # Update constraint Jacobians
            initcond_jacobian!(∇g, pidx)
            dynamics_jacobian!(∇g, z, cache, params; ϵ=opts.eps_fd)
            trustregion_jacobian!(∇g, pidx)

            # Update quadratic cost function
            costfunc_expansion!(∇²ₓₓL, ∇J, z, cache, params)    # Gauss-Newton

            # Visualize the sparsity pattern
            #display(spy(∇g, title="constraint jacobian"))
            #display(spy(∇²ₓₓL, title="cost function hessian"))
            #readline()

            # Solve QP
            OSQP.update!(m; Px=∇²ₓₓL.nzval, q=∇J, Ax=∇g.nzval, l=gl, u=gu)
            OSQP.solve!(m, r)

            #=
            if r.info.status_val == 1 || r.info.status_val == 2
                # Expand trust region if QP does not error
                @. cache.dxtmp = 0.5 * (Δxu - Δxl)
                Δxl .-= cache.dxtmp
                Δxu .+= cache.dxtmp
                @. cache.utmp = 0.5 * (Δuu - Δul)
                Δul .-= cache.utmp
                Δuu .+= cache.utmp
            else
                # Contract trust region and re-solve if QP does error
                iter = 0
                while iter < 10 && (r.info.status_val != 1 || r.info.status_val != 2)
                    iter += 1

                    @. cache.dxtmp = 0.5 * (Δxu - Δxl)
                    Δxl .+= cache.dxtmp
                    Δxu .-= cache.dxtmp

                    @. cache.utmp = 0.5 * (Δuu - Δul)
                    Δul .+= cache.utmp
                    Δuu .-= cache.utmp

                    trustregion_bounds!(gl, Δxl, Δul, pidx)
                    trustregion_bounds!(gu, Δxu, Δuu, pidx)

                    OSQP.update!(m; Px=∇²ₓₓL.nzval, q=∇J, Ax=∇g.nzval, l=gl, u=gu)
                    OSQP.solve!(m, r)
                end
            end
            =#

            # Backtracking line-search
            J_ls = 0.0
            viol_ls = 0.0
            maxiter_ls = 1
            αmul = 0.5
            α = 0.01
            for i in 1:maxiter_ls
                # Copy current solution
                copyto!(ztmp, z)
                step_primals!(ztmp, r.x, cache, params; α=α)

                # Evaluate new cost
                copy_primals_to_solution!(sol, ztmp, pidx)
                J_ls = params.costfunc(sol.X, sol.U, params.Xref, params.Uref)

                # Evaluate new constraint residuals
                initcond_residuals!(gl, ztmp, pidx, params)
                dynamics_residuals!(gl, ztmp, cache, params)
                viol_ls = constraint_violation(gl, pidx)

                # Evalute merit function
                J_ls - sol.J + viol_ls < viol ? break : nothing

                α *= αmul
            end

            # Update solution
            sol.J = J_ls
            viol = viol_ls
            copyto!(z, ztmp)

            # Log and check for convergence
            statnorm = stationarity!(∇ₓL, ztmp, ∇J, ∇g, r.y, pidx)
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
