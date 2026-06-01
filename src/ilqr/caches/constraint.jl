struct InequalityConstraintCache{T<:AbstractFloat}
    C::Vector{Vector{T}}
    F::Vector{Vector{T}}
    Λ::Vector{Vector{T}}
    Ρ::Vector{Vector{T}}
    I::Vector{Diagonal{T,Vector{T}}}

    function InequalityConstraintCache{T}(N::Integer, m::Integer) where {T}
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

@composite struct ControlBoundCache{T<:AbstractFloat}
    InequalityConstraintCache{T}...
    B::Vector{Vector{T}}

    function ControlBoundCache{T}(N::Integer, nu::Integer) where {T}
        cache = InequalityConstraintCache{T}(N-1, nu)
        B = Utils.vector(zeros(T, nu), N - 1)# Jacobian of control bound constraints (constant)
        return new{T}(cache..., B)
    end
end

struct ConstraintCache{T<:AbstractFloat}
    ul::ControlBoundCache{T}
    uu::ControlBoundCache{T}

    function ConstraintCache{T}(N::Integer, nu::Integer) where {T}
        ul = ControlBoundCache{T}(N, nu)
        uu = ControlBoundCache{T}(N, nu)
        return new{T}(ul, uu)
    end
end
