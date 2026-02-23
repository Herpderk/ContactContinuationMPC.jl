@views function is_converged(
    statnorm::Float64, viol::Float64; tol_stat::Float64, tol_eq::Float64
)::Bool
    return statnorm < tol_stat && viol < tol_eq
end

function constraint_violation(
    g::Vector{Float64}, pidx::IndexingParameters
)::Float64
    geq = g[pidx.g.ic[1][1]:pidx.g.dyn[end][end]]
    return norm(geq, Inf)
end

function stationarity!(
    ∇ₓL::Vector{Float64},
    ztmp::Vector{Float64},
    ∇J::Vector{Float64},
    ∇g::SparseMatrixCSC{Float64,Int},
    λ::Vector{Float64},
)::Nothing
    #println("ztmp length:", length(ztmp))
    #println("∇g shape:", size(∇g))
    #println("λ length:", length(λ))
    mul!(ztmp, ∇g', λ)
    @. ∇ₓL = ztmp + ∇J
    return nothing
end

function log_interrupted()::Nothing
    println("-------------------------------------")
    println("          SQP interrupted!")
    println("-------------------------------------")
    return nothing
end

function log_iter(
    iter::Int, J::Float64, statnorm::Float64, viol::Float64
)::Nothing
    if rem(iter, 20) == 0
        println("-------------------------------------")
        println("iter       J        ‖∇ₓL‖       ‖h‖")
        println("-------------------------------------")
    end
    @printf("%4.04i   %8.2e   %8.2e   %8.2e\n", iter, J, statnorm, viol,)
    return nothing
end

function init_sqp!(
    sol::TrajoptSolution{Float64}, cache::SQPCache
)::Nothing where {Lk,Lf}
    # Primal warm-start
    copy_solution_to_primals!(cache.z, sol, cache.pidx)
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
    pidx::IndexingParameters,
    params::TrajoptParameters{Float64,Lk,Lf},
)::Nothing where {Lk,Lf}
    N, zidx, dzidx = pidx.dims.N, pidx.z, pidx.dz
    for k in 1:(N - 1)
        Utils.add_diff_to_state!(params.mfwd, z[zidx.x[k]], Δz[dzidx.x[k]])
        z[zidx.u[k]] .+= Δz[dzidx.u[k]]
    end
    Utils.add_diff_to_state!(params.mfwd, z[zidx.x[end]], Δz[dzidx.x[end]])
    return nothing
end

function run_sqp!(
    Δxl::Vector{Float64},
    Δxu::Vector{Float64},
    Δul::Vector{Float64},
    Δuu::Vector{Float64},
    sol::TrajoptSolution{Float64},
    cache::SQPCache,
    params::TrajoptParameters{Float64,Lk,Lf},
    opts::SQPOptions,
)::Nothing where {Lk,Lf}
    init_sqp!(sol, cache)

    # References to OSQP structs
    m, r = cache.m, cache.r

    # References to optimization arrays
    ∇²ₓₓL, ∇ₓL, ∇J, ∇g, gl, gu, z = (
        cache.∇²ₓₓL, cache.∇ₓL, cache.∇J, cache.∇g, cache.gl, cache.gu, cache.z
    )

    # Start SQP loop
    iter = 0
    try
        while iter < opts.maxiter
            # Update solution
            iter > 0 ? copy_primals_to_solution!(sol, z, cache.pidx) : nothing
            sol.J = params.costfunc(sol.X, sol.U, params.Xref, params.Uref)

            # Update QP constraints
            initcond_residuals!(gl, z, cache.pidx, params)
            dynamics_residuals!(gl, z, cache, params)
            copyto!(gu, gl)
            trustregion_bounds!(gl, Δxl, Δul, cache.pidx)
            trustregion_bounds!(gu, Δxu, Δuu, cache.pidx)

            # Update QP constraint Jacobians
            initcond_jacobian!(∇g, cache.pidx)
            dynamics_jacobian!(∇g, z, cache, params; ϵ=opts.eps_fd)
            trustregion_jacobian!(∇g, cache.pidx)

            # Update QP cost function
            costfunc_expansion!(∇²ₓₓL, ∇J, z, cache, params)    # Gauss-Newton

            # Log and check for convergence
            stationarity!(∇ₓL, cache.ztmp, ∇J, ∇g, r.y)
            statnorm = norm(∇ₓL, Inf)
            viol = constraint_violation(gl, cache.pidx)
            opts.is_verbose ? log_iter(iter, sol.J, statnorm, viol) : nothing
            sol.is_optimal = is_converged(
                statnorm, viol; tol_stat=opts.tol_stat, tol_eq=opts.tol_eqconstr
            )
            sol.is_optimal ? break : nothing

            # Solve QP
            OSQP.update!(m; Px=∇²ₓₓL.nzval, q=∇J, Ax=∇g.nzval, l=gl, u=gu)
            #iter == 1 ? OSQP.warm_start!(m; x=z) : nothing
            OSQP.solve!(m, r)

            # The QP primals are in tangent space.
            # We need to update the "manifold states" correctly
            step_primals!(z, r.x, cache.pidx, params)
            iter += 1
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
