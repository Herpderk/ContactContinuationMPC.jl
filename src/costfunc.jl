"""
    TrajectoryCostFunction(stage, term, nx, nu, N)

Callable struct containing a given problem's dimensions, indices, and cost functions.
"""
mutable struct TrajectoryCostFunction{T<:AbstractFloat,C_stage,C_term}
    stage::C_stage
    term::C_term
    xerr::DiffCache{Vector{T},Vector{T}}
    uerr::DiffCache{Vector{T},Vector{T}}

    function TrajectoryCostFunction{T,C_stage,C_term}(
        costfunc_stage::C_stage, costfunc_term::C_term, nx::Int, nu::Int
    ) where {T<:AbstractFloat,C_stage,C_term}
        xerr = DiffCache(zeros(T, nx))
        uerr = DiffCache(zeros(T, nu))
        return new{T,C_stage,C_term}(costfunc_stage, costfunc_term, xerr, uerr)
    end
end

"""
    costfunc(X, U, Xref, Uref)

Callable struct method for the `TrajectoryCostFunction` struct that computes the accumulated cost over a trajectory given a sequence of references.
"""
function (cache::TrajectoryCostFunction{T,C_stage,C_term})(
    X::AbstractVector{<:AbstractVector{<:Real}},
    U::AbstractVector{<:AbstractVector{<:Real}},
    Xref::AbstractVector{<:AbstractVector{<:Real}},
    Uref::AbstractVector{<:AbstractVector{<:Real}},
)::Union{T,ForwardDiff.Dual} where {T,C_stage,C_term}
    # Get temporary error vectors
    xerr = get_tmp(cache.xerr, X[1])
    uerr = get_tmp(cache.uerr, U[1])

    # Sum costs
    super_el = X[1][1] + U[1][1]
    J = zero(super_el)

    @inbounds @simd for k in 1:(length(Uref))
        @. xerr = X[k] - Xref[k]
        @. uerr = U[k] - Uref[k]
        J += cache.stage(xerr, uerr)
    end

    @. xerr = X[end] - Xref[end]
    J += cache.term(xerr)
    return J
end
