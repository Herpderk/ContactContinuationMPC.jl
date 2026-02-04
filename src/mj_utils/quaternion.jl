"""
Assortment of quaternion multiplication utilities based on the conventions in
the following paper: https://ieeexplore.ieee.org/document/9326337.
We assume quaternions of the following form: q = [s, v1, v2, v3], where s is the
scalar component and v = [v1, v2, v3] is the vector component.
"""


function skew!(
    A::AbstractMatrix{T},
    v::AbstractVector{T},
)::Nothing where {T<:Real}
    if size(A) != (3, 3)
        throw(DimensionMismatch("Input matrix must be of size 3x3"))
    end
    if length(v) != 3
        throw(DimensionMismatch("Input vector must be of length 3"))
    end

    A[1, 1] = 0.0
    A[1, 2] = -v[3]
    A[1, 3] = v[2]
    A[2, 1] = v[3]
    A[2, 2] = 0.0
    A[2, 3] = -v[1]
    A[3, 1] = -v[2]
    A[3, 2] = v[1]
    A[3, 3] = 0.0
    return
end


"""
We define the kinematic mapping matrix K such that:

    q̇ = K(q) * ω

where q is a quaternion and ω is the angular velocity vector. K(q) is the
result of the following left quaternion multiplication:

    K(q) = L(q) * H
"""
@views function kinematic_mapping_matrix!(
    K::AbstractMatrix{T},
    q::AbstractVector{T},
)::Nothing where {T<:AbstractFloat}
    if size(K) != (4, 3)
        throw(DimensionMismatch("Input matrix must be of size 4x3"))
    end
    if length(q) != 4
        throw(DimensionMismatch("Input quaternion must be of length 4"))
    end

    s = q[1]
    v = q[2:4]
    K[1, :] .= -v'
    skew!(K[2:4, :], v)
    axpy!(s, I(3), K[2:4, :])
    K .*= 0.5
    return
end
