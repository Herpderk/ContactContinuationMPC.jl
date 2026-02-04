"""
	TrajectoryCostFunction(stage, term, nx, nu, N)
Callable struct containing a given problem's dimensions, indices, and cost functions.
"""
mutable struct TrajectoryCostFunction{T<:AbstractFloat}
    stage::Function
    term::Function
    xerr::DiffCache{Vector{T},Vector{T}}
    uerr::DiffCache{Vector{T},Vector{T}}

    function TrajectoryCostFunction{T}(
        costfunc_stage::Function,
        costfunc_term::Function,
        nx::Int,
        nu::Int,
    ) where {T<:AbstractFloat}
        xerr = DiffCache(zeros(T, nx))
        uerr = DiffCache(zeros(T, nu))
        return new{T}(costfunc_stage, costfunc_term, xerr, uerr)
    end
end

"""
	costfunc(X, U, Xref, Uref)
Callable struct method for the `TrajectoryCostFunction` struct that computes the accumulated cost over a trajectory given a sequence of references.
"""
@views function (cache::TrajectoryCostFunction{T})(
    X::AbstractVector{V},
    U::AbstractVector{V},
    Xref::AbstractVector{V},
    Uref::AbstractVector{V},
)::Union{T,ForwardDiff.Dual} where {T,V<:AbstractVector{<:Real}}
    # Get temporary error vectors
    xerr = get_tmp(cache.xerr, X[1])
    uerr = get_tmp(cache.uerr, U[1])

    # Sum costs
    super_el = X[1][1] + U[1][1]
    J = zero(super_el)

    @inbounds @simd for k = 1:(length(Uref))
        @. xerr = X[k] - Xref[k]
        @. uerr = U[k] - Uref[k]
        J += cache.stage(xerr, uerr)
    end

    @. xerr = X[end] - Xref[end]
    J += cache.term(xerr)
    return J
end

# Default type parameter
TrajectoryCostFunction(args...) = TrajectoryCostFunction{DEFAULT_DTYPE}(args...)
