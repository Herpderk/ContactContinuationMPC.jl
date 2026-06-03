@composite struct InequalityConstraint{T<:AbstractFloat} <:
                  AbstractInequalityConstraint{T}
    EqualityConstraint{T}...

    function InequalityConstraint{T}(N::Integer, m::Integer) where {T}
        constr = EqualityConstraint{T}(N, m)
        return new{T}(constr...)
    end
end

function update_forces!(
    constr::AbstractInequalityConstraint{T}
)::Nothing where {T}
    F, Λ, Ρ = constr.F, constr.Λ, constr.Ρ
    @inbounds @simd for k in eachindex(F)
        @. F[k] = Λ[k] + Ρ[k] * C[k]
        @. F[k] = max(T(0), F[k])
    end
    return nothing
end

function update_indicators!(
    constr::AbstractInequalityConstraint{T}
)::Nothing where {T}
    @inbounds @simd for k in eachindex(constr.I)
        I, f = constr.I[k], constr.F[k]
        @view @. I[diagind(I)] = T(f > T(0)) # 1 if constraint is violated, 0 otherwise
    end
    return nothing
end

function get_trajectory_cost(
    constr::AbstractInequalityConstraint{T}
)::T where {T}
    F, Λ, Ρ = constr.F, constr.Λ, constr.Ρ
    L = T(0)
    @inbounds for k in eachindex(F)
        for i in eachindex(F[k])
            L += (F[k][i]^2 - Λ[k][i]^2) / (2 * Ρ[k][i])
        end
    end
    return L
end
