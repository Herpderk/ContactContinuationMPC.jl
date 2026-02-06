"""
    TrajectoryCostFunction(stage, term, nx, nu, N)

Callable struct containing a given problem's dimensions, indices, and cost functions.
"""
mutable struct TrajectoryCostFunction{T<:AbstractFloat,Lk,Lf}
    m::MuJoCo.Model
    stage::Lk
    term::Lf
    xerr::DiffCache{Vector{T},Vector{T}}
    uerr::DiffCache{Vector{T},Vector{T}}

    function TrajectoryCostFunction{T,Lk,Lf}(
        m::MuJoCo.Model, costfunc_stage::Lk, costfunc_term::Lf
    ) where {T<:AbstractFloat,Lk,Lf}
        xerr = DiffCache(zeros(T, get_ndx(m)))
        uerr = DiffCache(zeros(T, m.nu))
        return new{T,Lk,Lf}(m, costfunc_stage, costfunc_term, xerr, uerr)
    end
end

"""
    costfunc(X, U, Xref, Uref)

Callable struct method for the `TrajectoryCostFunction` struct that computes the accumulated cost over a trajectory given a sequence of references.
"""
function (cache::TrajectoryCostFunction{T,Lk,Lf})(
    X::AbstractVector{V},
    U::AbstractVector{V},
    Xref::AbstractVector{V},
    Uref::AbstractVector{V},
)::Union{T,ForwardDiff.Dual} where {T,Lk,Lf,V<:AbstractVector{<:Real}}
    # Reference model
    m = cache.m

    # Get temporary error vectors
    xerr = get_tmp(cache.xerr, X[1])
    uerr = get_tmp(cache.uerr, U[1])

    # Initialize trajectory cost
    el_super = X[1][1] + U[1][1]
    J = zero(el_super)

    @inbounds @simd for k in 1:(length(Uref))
        get_state_diff!(m, xerr, X[k], Xref[k])     # Compute state error
        @. uerr = U[k] - Uref[k]                    # Compute control error
        J += cache.stage(xerr, uerr)                # Add stage cost
    end

    # Add terminal cost
    get_state_diff!(m, xerr, X[end], Xref[end])
    J += cache.term(xerr)
    return J
end
