"""
	TrajectoryCostFunction(stage, term, nx, nu, N)

Callable struct containing a given problem's dimensions, indices, and cost functions.
"""
mutable struct TrajectoryCostFunction{T<:AbstractFloat}
    stage::Function
    term::Function
    Xerr::Vector{Vector{T}}
    Uerr::Vector{Vector{T}}
    L::Vector{T}

    function TrajectoryCostFunction{T}(
        costfunc_stage::Function,
        costfunc_term::Function,
        nx::Int,
        nu::Int,
        N::Int,
    ) where {T}
        Xerr = [zeros(T, nx) for k = 1:N]
        Uerr = [zeros(T, nu) for k = 1:(N-1)]
        L = zeros(T, N)
        return new{T}(costfunc_stage, costfunc_term, Xerr, Uerr, L)
    end
end

# Default type parameter
TrajectoryCostFunction(args...) = TrajectoryCostFunction{DEFAULT_DTYPE}(args...)

"""
	costfunc(X, U, Xref, Uref)

Callable struct method for the `TrajectoryCostFunction` struct that computes the accumulated cost over a trajectory given a sequence of references.
"""
function (cache::TrajectoryCostFunction{T})(
    X::AbstractVector{<:AbstractVector{<:Real}},
    U::AbstractVector{<:AbstractVector{<:Real}},
    Xref::AbstractVector{<:AbstractVector{<:Real}},
    Uref::AbstractVector{<:AbstractVector{<:Real}},
)::T where {T}
    # Broadcast x - xref
    BLAS.copy!.(cache.Xerr, X)
    BLAS.axpy!.(-1.0, Xref, cache.Xerr)

    # Broadcast u - uref
    BLAS.copy!.(cache.Uerr, U)
    BLAS.axpy!.(-1.0, Uref, cache.Uerr)

    # Broadcast stage cost
    Xerr_stage = @view cache.Xerr[1:(end-1)]
    Lstage = @view cache.L[1:(end-1)]
    Lstage_new = cache.stage.(Xerr_stage, cache.Uerr)
    BLAS.copy!(Lstage, Lstage_new)

    # Get terminal cost
    Xerr_term = cache.Xerr[end]
    Lterm = cache.L[end]
    Lterm_new = cache.term(Xerr_term)
    BLAS.copy!(Lterm, Lterm_new)
    return sum(cache.L)
end
