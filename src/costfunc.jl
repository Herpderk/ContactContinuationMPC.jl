"""
    QuadraticCostFunction{T}(Q, R, Qf)

Callable struct containing quadratic cost weights. Call method is overloaded
with stage and terminal cost functions.
"""
struct QuadraticCostFunction{T<:AbstractFloat}
    Q::Matrix{T}
    R::Matrix{T}
    Qf::Matrix{T}
    xtmp::DiffCache{Vector{T},Vector{T}}
    utmp::DiffCache{Vector{T},Vector{T}}

    function QuadraticCostFunction{T}(
        Q::AbstractMatrix{<:Real},
        R::AbstractMatrix{<:Real},
        Qf::AbstractMatrix{<:Real},
    ) where {T}
        if size(Q) != size(Qf)
            throwdim("Q and Qf matrices must have the same size")
        end
        xtmp = DiffCache(zeros(T, size(Q)[1]))
        utmp = DiffCache(zeros(T, size(R)[1]))
        return new{T}(T.(Q), T.(R), T.(Qf), xtmp, utmp)
    end
end

"""
    QuadraticCostFunction(Q, R, Qf)

Convenience constructor for initialization from dtype of weight matrices.
"""
function QuadraticCostFunction(
    Q::AbstractMatrix{T}, R::AbstractMatrix{T}, Qf::AbstractMatrix{T}
)::QuadraticCostFunction{T} where {T}
    return QuadraticCostFunction{T}(Q, R, Qf)
end

"""
    (cache::QuadraticCostFunction)(xerr, uerr)

Quadratic stage cost function. Computes the cost given the state error xerr and
control error uerr.
"""
function (cache::QuadraticCostFunction{T})(
    xerr::AbstractVector{Tx}, uerr::AbstractVector{Tu}
)::Union{T,ForwardDiff.Dual} where {Tx,Tu,T}
    super_el = xerr[1] + uerr[1]
    xtmp = get_tmp(cache.xtmp, super_el)
    utmp = get_tmp(cache.utmp, super_el)
    mul!(xtmp, cache.Q, xerr)
    mul!(utmp, cache.R, uerr)
    return 0.5 * (dot(xerr, xtmp) + dot(uerr, utmp))
end

"""
    (cache::QuadraticCostFunction)(xerr)

Quadratic terminal cost function. Computes the cost given the state error xerr.
"""
function (cache::QuadraticCostFunction{T})(
    xerr::AbstractVector{Tx}
)::Union{T,ForwardDiff.Dual} where {Tx,T}
    xtmp = get_tmp(cache.xtmp, xerr)
    mul!(xtmp, cache.Qf, xerr)
    return 0.5 * dot(xerr, xtmp)
end

"""
    TrajectoryCostFunction(m, costfunc_stage, costfunc_term)

Callable struct containing a given problem's dimensions, indices, and cost functions.
"""
mutable struct TrajectoryCostFunction{T<:AbstractFloat,Lk,Lf}
    m::MuJoCo.Model
    stage::Lk
    term::Lf
    xerr::DiffCache{Vector{T},Vector{T}}
    uerr::DiffCache{Vector{T},Vector{T}}

    function TrajectoryCostFunction{T}(
        m::MuJoCo.Model, costfunc_stage::Lk, costfunc_term::Lf
    ) where {T,Lk,Lf}
        if isempty(methods(costfunc_stage))
            throwarg("The provided stage cost function is not callable")
        end
        if isempty(methods(costfunc_term))
            throwarg("The provided terminal cost function is not callable")
        end
        xerr = DiffCache(zeros(T, get_ndx(m)))
        uerr = DiffCache(zeros(T, m.nu))
        return new{T,Lk,Lf}(m, costfunc_stage, costfunc_term, xerr, uerr)
    end
end

"""
    TrajectoryCostFunction(m, costfunc_quad)

Convenience constructor for initialization from a quadratic cost function.
"""
function TrajectoryCostFunction(
    m::MuJoCo.Model, costfunc_quad::L
)::TrajectoryCostFunction{T,L,L} where {T,L<:QuadraticCostFunction{T}}
    return TrajectoryCostFunction{T,L,L}(m, costfunc_quad, costfunc_quad)
end

"""
    costfunc(X, U, Xref, Uref)

Computes the accumulated cost over a trajectory given a sequence of references.
"""
function (cache::TrajectoryCostFunction{T,Lk,Lf})(
    X::AbstractVector{Tx},
    U::AbstractVector{Tu},
    Xref::AbstractVector{Txr},
    Uref::AbstractVector{Tur},
)::Union{T,ForwardDiff.Dual} where {Tx,Tu,Txr,Tur,T,Lk,Lf}
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
