function is_converged(∇J::Vector{Float64}, h::Vector{Float64})::Bool
    return norm(∇J, 2) < 1e-2 && norm(h, Inf) < 1e-2
end

function log_sqp_interrupted()::Nothing
    println("-------------------------------------")
    println("          SQP interrupted!")
    println("-------------------------------------")
    return nothing
end

function log_iter(
    iter::Int, J::Float64, ∇J::Vector{Float64}, h::Vector{Float64}
)::Nothing
    if rem(iter-1, 20) == 0
        println("-------------------------------------------------")
        println("iter       J         ‖∇J‖         ‖h‖          α")
        println("-------------------------------------------------")
    end
    @printf(
        "%4.04i   %8.2e   %8.2e   %8.2e   %8.2e\n",
        iter,
        J,
        norm(∇J, 2),
        norm(h, Inf),
        1.0,
    )
    return nothing
end

@views function init_sqp!(
    sol::TrajoptSolution{Float64}, cache::SQPCache
)::Nothing
    N, zidx, z = cache.pidx.dims.N, cache.pidx.z, cache.z

    # Copy trajectory to decision variables
    for k in 1:(N - 1)
        copyto!(z[zidx.x[k]], sol.X[k])
        copyto!(z[zidx.u[k]], sol.U[k])
    end
    copyto!(z[zidx.x[end]], sol.X[end])

    # Initialize solution terms
    sol.is_optimal = false
    sol.J = Inf
    return nothing
end

@views function get_trajectory_cost(
    z::Vector, cache::SQPCache, params::TrajoptParameters
)::Float64
    N, zidx = cache.pidx.dims.N, cache.pidx.z, pidx.dz
    dxtmp, utmp = cache.dxtmp, cache.utmp
    Xref, Uref = params.Xref, params.Uref

    J = 0.0
    @inbounds for k in 1:(N - 1)
        # Get x and u errors
        x, xref, u, uref = z[zidx.x[k]], Xref[k], z[zidx.u[k]], Uref[k]
        get_state_diff!(params.mfwd, dxtmp, x, xref)
        @. utmp = u - uref
        # Add stage cost
        J += params.costfunc.stage(dxtmp, utmp)
    end

    # Add terminal cost
    J += params.costfunc.term(z[zidx.x[end]])
    return J
end

@views function update_primals!(
    z::Vector, Δz::Vector, pidx::IndexingParameters, params::TrajoptParameters
)::Nothing
    N, zidx, dzidx = pidx.dims.N, pidx.z, pidx.dz
    for k in 1:(N - 1)
        add_diff_to_state!(params.mfwd, z[zidx.x[k]], Δz[dzidx.x[k]])
        z[zidx.u[k]] .+= Δz[dzidx.u[k]]
    end
    add_diff_to_state!(params.mfwd, z[zidx.x[end]], ΔZ[dzidx.x[end]])
    return nothing
end

function run_sqp!(
    sol::TrajoptSolution{Float64},
    cache::SQPCache,
    params::TrajoptParameters{Float64,Lk,Lf};
    ϵ::Float64,
    tol_cost::Float64,
    tol_eq::Float64,
    maxiter_sqp::Int,
    is_verbose::Bool,
)::Nothing where {Lk,Lf}
    init_sqp!(sol, cache)

    # References to OSQP structs
    m, r = cache.m, cache.r

    # References to QP arrays
    ∇²L, ∇J, ∇h, h, λ, z = cache.∇²L,
    cache.∇J, cache.∇h, cache.h, cache.λ,
    cache.z

    # Start SQP loop
    iter = 0
    try
        while iter < maxiter_sqp
            # Update QP arrays
            # Gauss-Newton (only put the costfunc hessian into the Lagrangian)
            costfunc_expansion!(∇²L, ∇J, z, cache, params)
            equality_jacobian!(∇h, z, cache, params; ϵ=ϵ)
            equality_residuals!(h, z, cache, params)

            # Update solution and log
            sol.J = get_trajectory_cost(z, cache, params)
            # copy z to sol.X and sol,U
            is_verbose ? log_iter(iter, J, ∇J, h) : nothing

            # Check for convergence
            sol.is_optimal = is_converged(∇J, h)
            sol.is_optimal ? break : nothing
            iter += 1

            # Solve QP
            h .*= -1.0
            OSQP.update!(m; Px=∇²L.nzval, q=∇J, Ax=∇h.nzval, l=h, u=h)
            iter == 1 ? OSQP.warm_start!(m; x=z, y=λ) : nothing
            OSQP.solve!(m, r)

            # Update decision variables
            copyto!(λ, r.y)
            # The QP primals are in tangent space. We need to update the "manifold states" correctly
            update_primals!(z, r.x, cache.pidx, params)
        end
    catch e
        e isa InterruptException ? log_sqp_interrupted() : rethrow(e)
    end

    if opts.is_verbose
        if sol.is_optimal
            log_converged()
        elseif iter == maxiter_sqp
            log_maxiter()
        end
    end
    return nothing
end
