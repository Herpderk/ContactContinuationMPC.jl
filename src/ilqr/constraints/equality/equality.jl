struct EqualityConstraint{T<:AbstractFloat} <: AbstractEqualityConstraint{T}
    C::Vector{Vector{T}}    # Constraint residuals for c(x) = 0
    F::Vector{Vector{T}}    # Constraint forces (what gets saved in the next Lagrange multipliers)
    Λ::Vector{Vector{T}}    # Current Lagrange multipliers (previous constraint forces)
    Ρ::Vector{Vector{T}}    # Current penalty parameters
    I::Vector{Diagonal{T,Vector{T}}}    # Indicator matrices for active constraints

    function EqualityConstraint{T}(N::Integer, m::Integer) where {T}
        vec_zeros = zeros(T, m)
        C = Utils.vector(vec_zeros, N)
        F = Utils.vector(vec_zeros, N)
        Λ = Utils.vector(vec_zeros, N)
        Ρ = Utils.vector(vec_zeros, N)
        diag_zeros = Diagonal(zeros(T, m))
        I = Utils.vector(diaǵ_zeros, N - 1)
        return new{T}(C, F, Λ, Ρ, I)
    end
end

function update_lagrange_multipliers!(
    constr::AbstractConstraint{T}
)::Nothing where {T}
    Utils.copy_nested_array!(constr.Λ, constr.F)
    return nothing
end

function update_penalty_parameters!(
    constr::AbstractConstraint{T}, ρ_mul::AbstractFloat
)::Nothing where {T}
    @inbounds @simd for k in eachindex(constr.Ρ)
        constr.Ρ[k] .*= ρ_mul
    end
    return nothing
end

function update_forces!(constr::AbstractConstraint{T})::Nothing where {T}
    F, Λ, Ρ, C = constr.F, constr.Λ, constr.Ρ, constr.C
    @inbounds @simd for k in eachindex(constr.F)
        @. F[k] = Λ[k] + Ρ[k] * C[k]
    end
    return nothing
end

function update_indicators!(
    constr::AbstractEqualityConstraint{T}
)::Nothing where {T}
    @inbounds @simd for k in eachindex(constr.I)
        fill!(constr.I[k], T(1))
    end
    return nothing
end

function get_trajectory_cost(constr::AbstractEqualityConstraint{T})::T where {T}
    Λ, Ρ, C = constr.Λ, constr.Ρ, constr.C
    L = T(0)
    @inbounds for k in eachindex(constr.F)
        for i in eachindex(constr.F[k])
            L += Λ[k][i] * C[k][i] + 0.5 * Ρ[k][i] * C[k][i]^2
        end
    end
    return L
end

function get_stage_cost_gradient!(
    ∇L::Vector{T}, ∇c::Matrix{T}, constr::AbstractConstraint{T}, k::Integer
)::Nothing where {T}
    f = constr.F[k]
    BLAS.gemv!('T', T(1), ∇c, f, T(0), ∇L)# ∇L = ∇c'*f
    return nothing
end

function get_stage_cost_hessian!(
    ∇²L::Matrix{T},
    tmp::Matrix{T},
    ∇c::Matrix{T},
    constr::AbstractConstraint{T},
    k::Integer,
)::Nothing where {T}
    I, ρ = constr.I[k], constr.Ρ[k]

    # ∇²L = ρ * ∇c' * I * ∇c
    BLAS.gemm!('N', 'N', T(1), I, ∇c, T(0), tmp)
    BLAS.gemm!('T', 'N', ρ, ∇c, tmp, T(0), ∇²L)
    return nothing
end
