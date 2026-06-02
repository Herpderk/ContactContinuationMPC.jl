function update_lagrange_multipliers!(
    constr::InequalityConstraintCache{T}
)::Nothing where {T}
    Utils.copy_nested_array!(constr.Λ, constr.F)
    return nothing
end

function update_penalty_parameters!(
    constr::InequalityConstraintCache{T}, ρ_mul::T
)::Nothing where {T}
    @inbounds @simd for k in eachindex(constr.Ρ)
        constr.Ρ[k] .*= ρ_mul
    end
    return nothing
end

function inequality_constraint_force!(
    f::Vector{T}, c::Vector{T}, λ::Vector{T}, ρ::Vector{T}
)::Nothing where {T}
    @. f = λ + ρ*c
    @. f = max(T(0), f)
    return nothing
end

function inequality_constraint_indicator!(
    I::Diagonal{T}, f::Vector{T}
)::Nothing where {T}
    @view @. I[diagind(I)] = T(f > T(0))# 1 if constraint is violated, 0 otherwise
    return nothing
end

function inequality_constraint_cost(
    f::Vector{T}, λ::Vector{T}, ρ::Vector{T}
)::T where {T}
    L = T(0)
    @inbounds for i in eachindex(f)
        L += (f[i]^2 - λ[i]^2) / (2 * ρ[i])
    end
    return L
end

function inequality_constraint_cost_gradient!(
    ∇L::Vector{T}, ∇c::Matrix{T}, f::Vector{T}
)::Nothing where {T}
    # ∇L = ∇c'*f
    BLAS.gemv!('T', T(1), ∇c, f, T(0), ∇L)
    return nothing
end

function inequality_constraint_cost_hessian!(
    ∇²L::Matrix{T}, tmp::Matrix{T}, ∇c::Matrix{T}, I::Diagonal{T}, ρ::Vector{T}
)::Nothing where {T}
    # ∇²L = ρ * ∇c' * I * ∇c
    BLAS.gemm!('N', 'N', T(1), I, ∇c, T(0), tmp)
    BLAS.gemm!('T', 'N', ρ, ∇c, tmp, T(0), ∇²L)
    return nothing
end

function control_bound_violation!(
    c::Vector{Vector{T}}, u::Vector{Vector{T}}, b::Vector{T}, l_or_u::Symbol
)::Nothing where {T}
    if l_or_u == :u
        @. c = u - b
    elseif l_or_u == :l
        @. c = b - u
    else
        throwarg(
            "Control bounds can only be evaluated as lower (l) or upper (u)!"
        )
    end
    @. c = max(T(0), c)
    return nothing
end

function control_bound_jacobian!(
    ∇ₓc::Matrix{T}, l_or_u::Symbol
)::Nothing where {T}
    ∇ₓc .= T(0)
    if l_or_u == :u
        @view ∇ₓc[diagind(∇ₓc)] .= T(1)
    elseif l_or_u == :l
        @view ∇ₓc[diagind(∇ₓc)] .= T(-1)
    else
        throwarg(
            "Control bound Jacobian can only be evaluated as lower (l) or upper (u)!",
        )
    end
    return nothing
end
