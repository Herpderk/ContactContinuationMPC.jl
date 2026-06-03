@composite struct ControlBoundConstraint{T} <: AbstractInequalityConstraint{T}
    InequalityConstraint{T}...
    B::Vector{T}
    d::Symbol

    function ControlBoundConstraint{T}(
        N::Integer, m::Integer, b::Vector{T}, d::Symbol
    ) where {T}
        constr = InequalityConstraint{T}(N, m)
        B = Utils.vector(b, N - 1)
        if d != :lower && d != :upper
            throwarg("Control bounds can only be evaluated as lower or upper")
        end
        return new{T}(constr..., B, d)
    end
end

function update_residuals!(
    constr::ControlBoundConstraint{T}, inputs::Vector{Vector{T}}
)::Nothing where {T}
    @inbounds @simd for k in eachindex(inputs)
        u, b, c = inputs[k], constr.B[k], constr.C[k]
        if constr.d == :upper
            @. c = u - b
        elseif constr.d == :lower
            @. c = b - u
        end
    end
    return nothing
end

function get_stage_jacobian!(
    ∇c::Matrix{T},
    constr::ControlBoundConstraint{T},
    inputs::Vector{Vector{T}},
    k::Int,
)::Nothing where {T}
    ∇c .= T(0)
    if constr.d == :upper
        @view ∇c[diagind(∇c)] .= T(1)
    elseif constr.d == :lower
        @view ∇c[diagind(∇c)] .= T(-1)
    end
    return nothing
end
