function is_converged(
    ∇ₓL::Vector{Float64}, h::Vector{Float64}; tol_stat::Float64, tol_eq::Float64
)::Bool
    return norm(∇ₓL, Inf) < tol_stat && norm(h, Inf) < tol_eq
end

function stationarity!(
    ∇ₓL::Vector{Float64},
    ztmp::Vector{Float64},
    ∇J::Vector{Float64},
    ∇h::SparseMatrixCSC{Float64,Int},
    λ::Vector{Float64},
)::Nothing
    mul!(ztmp, ∇h', λ)
    @. ∇ₓL += ztmp + ∇J
end

function log_interrupted()::Nothing
    println("-------------------------------------")
    println("          SQP interrupted!")
    println("-------------------------------------")
    return nothing
end

function log_iter(
    iter::Int, J::Float64, ∇ₓL::Vector{Float64}, h::Vector{Float64}
)::Nothing
    if rem(iter-1, 20) == 0
        println("-------------------------------------------------")
        println("iter       J         ‖∇ₓL‖         ‖h‖          α")
        println("-------------------------------------------------")
    end
    @printf(
        "%4.04i   %8.2e   %8.2e   %8.2e   %8.2e\n",
        iter,
        J,
        norm(∇ₓL, Inf),
        norm(h, Inf),
        1.0,
    )
    return nothing
end

@views function init_sqp!(
    sol::TrajoptSolution{Float64},
    cache::SQPCache,
    params::TrajoptParameters{Float64,Lk,Lf},
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
    Utils.add_diff_to_state!(params.mfwd, z[zidx.x[end]], ΔZ[dzidx.x[end]])
    return nothing
end

function run_sqp!(
    sol::TrajoptSolution{Float64},
    cache::SQPCache,
    params::TrajoptParameters{Float64,Lk,Lf},
    opts::SQPOptions,
)::Nothing where {Lk,Lf}
    init_sqp!(sol, cache, params)

    # References to OSQP structs
    m, r = cache.m, cache.r

    # References to optimization arrays
    ∇²ₓₓL, ∇ₓL, ∇J, ∇h, h, z = (
        cache.∇²ₓₓL, cache.∇J, cache.∇h, cache.h, cache.z
    )

    # Start SQP loop
    iter = 0
    try
        while iter < opts.maxiter
            # Update solution
            iter > 0 ? copy_primals_to_solution!(sol, z, cache.pidx) : nothing
            sol.J = params.costfunc(sol.X, sol.U, params.Xref, params.Uref)

            # Update QP arrays
            # Gauss-Newton (only put the costfunc hessian into the Lagrangian)
            costfunc_expansion!(∇²ₓₓL, ∇J, z, cache, params)
            equality_jacobian!(∇h, z, cache, params; ϵ=opts.eps_fd)
            equality_residuals!(h, z, cache, params)

            # Log and check for convergence
            stationarity!(∇ₓL, cache.ztmp, ∇J, ∇h, r.y)
            opts.is_verbose ? log_iter(iter, sol.J, ∇ₓL, h) : nothing
            sol.is_optimal = is_converged(
                ∇ₓL, h; tol_stat=opts.tol_stationarity, tol_eq=opts.tol_eqconstr
            )
            sol.is_optimal ? break : nothing
            iter += 1

            # Solve QP
            h .*= -1.0
            OSQP.update!(m; Px=∇²ₓₓL.nzval, q=∇J, Ax=∇h.nzval, l=h, u=h)
            iter == 1 ? OSQP.warm_start!(m; x=z) : nothing
            OSQP.solve!(m, r)

            # Update decision variables
            # The QP primals are in tangent space. We need to update the "manifold states" correctly
            step_primals!(z, r.x, cache.pidx, params)
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
