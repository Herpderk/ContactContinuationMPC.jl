function log_converged()::Nothing
    println("-------------------------------------")
    println("       Optimal solution found!")
    println("-------------------------------------")
    return nothing
end

function log_maxiter()::Nothing
    println("-------------------------------------")
    println("Maximum number of iterations reached!")
    println("-------------------------------------")
    return nothing
end

function log_interrupted()::Nothing
    println("-------------------------------------")
    println("          iLQR interrupted!")
    println("-------------------------------------")
    return nothing
end

function log_iter(cache::iLQRCache, iter::Int)::Nothing
    if rem(iter-1, 20) == 0
        println("-------------------------------------")
        println("iter       J         ΔJ          α")
        println("-------------------------------------")
    end
    @printf(
        "%4.04i   %8.2e   %8.2e   %8.2e\n",
        iter,
        cache.fwd.Jprev,
        cache.bwd.ΔJ,
        cache.fwd.α,
    )
    return nothing
end

function log_al(iter::Int)::Nothing
    println("-----------AL iter: $iter------------")
end
