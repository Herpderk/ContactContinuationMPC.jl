function get_nx(m::Model)::Int
    return m.nq + m.nv + m.na
end

function get_ndx(m::Model)::Int
    return 2 * m.nv + m.na
end

"""
Assume x = [q, v, a]
"""
function get_q(m::Model, x::AbstractVector{<:Real})::SubArray
    return view(x, 1:m.nq)
end

"""
Assume x = [q, v, a]
"""
function get_v(m::Model, x::AbstractVector{<:Real})::SubArray
    return view(x, (m.nq + 1):(m.nq + m.nv))
end

"""
Assume x = [q, v, a]
"""
function get_a(m::Model, x::AbstractVector{<:Real})::SubArray
    return view(x, (m.nq + m.nv + 1):(m.nq + m.nv + m.na))
end

"""
Assume dx = [dq, dv, da]
"""
function get_dq(m::Model, dx::AbstractVector{<:Real})::SubArray
    return view(dx, 1:m.nv)
end

"""
Assume dx = [dq, dv, da]
"""
function get_dv(m::Model, dx::AbstractVector{<:Real})::SubArray
    return view(dx, (m.nv + 1):(m.nv + m.nv))
end

"""
Assume dx = [dq, dv, da]
"""
function get_da(m::Model, dx::AbstractVector{<:Real})::SubArray
    return view(dx, (2 * m.nv + 1):(2 * m.nv + m.na))
end

"""
    subtract_states!(m, Δx, x1, x2)

Get the state difference (x1 - x2) in tangent space coordinates.
"""
function subtract_states!(
    m::Model,
    Δx::AbstractVector{<:Real},
    x1::AbstractVector{<:Real},
    x2::AbstractVector{<:Real},
)::Nothing
    # Compute configuration difference
    Δq, q1, q2 = get_dq(m, Δx), get_q(m, x1), get_q(m, x2)
    mj_differentiatePos(m, Δq, 1.0, q2, q1)

    # Compute velocity difference
    Δv, v1, v2 = get_dv(m, Δx), get_v(m, x1), get_v(m, x2)
    @. Δv = v1 - v2

    # Compute actuator difference
    Δa, a1, a2 = get_da(m, Δx), get_a(m, x1), get_a(m, x2)
    @. Δa = a1 - a2
    return nothing
end
