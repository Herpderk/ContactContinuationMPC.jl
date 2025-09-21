"""
	TrajectoryCostFunction(stage_cost, term_cost, nx, nu, N)

Callable struct containing a given problem's dimensions, indices, and cost functions.
"""
mutable struct TrajectoryCostFunction
    stage_costfunc::Function
    term_costfunc::Function
    Xerr::Vector{Vector{<:Real}}
    Uerr::Vector{Vector{<:Real}}
    L::Vector{<:Real}

    function TrajectoryCostFunction(
        stage_cost::Function,
        term_cost::Function,
        nx::Int,
        nu::Int,
        N::Int,
    )::TrajectoryCostFunction
        stage_costfunc(x::Vector{<:Real}, u::Vector{<:Real}) = stage_cost(x, u)::Real
        term_costfunc(x::Vector{<:Real}) = term_cost(x)::Real

        Xerr = [zeros(nx) for k ∈ 1:N]
        Uerr = [zeros(nu) for k ∈ 1:(N-1)]
        L = zeros(N)
        return new(stage_costfunc, term_costfunc, Xerr, Uerr, L)
    end
end

"""
	costfunc(X, U, Xref, Uref)

Callable struct method for the `TrajectoryCostFunction` struct that computes the accumulated cost over a trajectory given a sequence of references.
"""
function (cache::TrajectoryCostFunction)(
    X::Vector{Vector{Float64}},
    U::Vector{Vector{Float64}},
    Xref::Vector{Vector{Float64}},
    Uref::Vector{Vector{Float64}},
)::Float64
    # Broadcast x - xref
    BLAS.copy!.(cache.Xerr, X)
    BLAS.axpy!.(-1.0, Xref, cache.Xerr)

    # Broadcast u - uref
    BLAS.copy!.(cache.Uerr, U)
    BLAS.axpy!.(-1.0, Uref, cache.Uerr)

    # Broadcast stage cost
    Xerr_stage = @view cache.Xerr[1:(end-1)]
    Lstage = @view cache.L[1:(end-1)]
    Lstage_new = cache.stage_costfunc.(Xerr_stage, cache.Uerr)
    BLAS.copy!(Lstage, Lstage_new)

    # Get terminal cost
    Xerr_term = cache.Xerr[end]
    Lterm = cache.L[end]
    Lterm_new = cache.term_costfunc(Xerr_term)
    BLAS.copy!(Lterm, Lterm_new)
    return sum(cache.L)
end
