"""
Assortment of quaternion multiplication utilities based on the conventions in
the following paper: https://ieeexplore.ieee.org/document/9326337
"""


ZERO_SCALAR_MAPPING_MATRIX = [zeros(3); I(3)]


function skew!(A::Matrix{T}, v::Vector{T})::Nothing where {T<:Real}
    if length(v) != 3
        throw(DimensionMismatch("Input vector must be of length 3"))
    end
    if size(A) != (3, 3)
        throw(DimensionMismatch("Input matrix must be of size 3x3"))
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
end


@views function base_multiplication_matrix!(
    B::Matrix{T},
    q::Vector{T},
    is_right::Bool,
)::Nothing where {T<:AbstractFloat}
    if length(q) != 4
        throw(DimensionMismatch("Input quaternion must be of length 4"))
    end
    if size(B) != (4, 4)
        throw(DimensionMismatch("Input matrix must be of size 4x4"))
    end

    s = q[1]
    v = q[2:end]

    B[1, 1] = s
    B[1, 2:4] = v'
    B[1, 2:4] .*= -1.0
    B[2:4, 1] = v

    # Add or substract the skew-symmetric vector component from the 3x3 block
    skew!(B[2:4, 2:4], v)
    if is_right
        B[2:4, 2:4] .*= -1.0
    end

    # Add the scalar component to the diagonal of the 3x3 block
    B[2:4, 2:4] .+= s * I(3)
end


function left_multiplication_matrix!(
    L::Matrix{T},
    q::Vector{T},
)::Nothing where {T<:AbstractFloat}
    base_multiplication_matrix!(L, q, false)
end


function right_multiplication_matrix!(
    R::Matrix{T},
    q::Vector{T},
)::Nothing where {T<:AbstractFloat}
    base_multiplication_matrix!(R, q, true)
end


function kinematic_mapping_matrix!(
    K::Matrix{T},
    L::Matrix{T},
    q::Vector{T},
)::Nothing where {T<:AbstractFloat}
    left_multiplication_matrix!(L, q)
    mul!(K, L, ZERO_SCALAR_MAPPING_MATRIX)
    K .*= 0.5
end
