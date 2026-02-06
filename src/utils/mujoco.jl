function get_nx(m::MuJoCo.Model)::Int
    return m.nq + m.nv + m.na
end

function get_ndx(m::MuJoCo.Model)::Int
    return 2 * m.nv + m.na
end

"""
Assume x = [q, v, a]
"""
function get_q(m::MuJoCo.Model, x::AbstractVector{T})::SubArray{T} where {T}
    return view(x, 1:m.nq)
end

"""
Assume x = [q, v, a]
"""
function get_v(m::MuJoCo.Model, x::AbstractVector{T})::SubArray{T} where {T}
    return view(x, (m.nq + 1):(m.nq + m.nv))
end

"""
Assume x = [q, v, a]
"""
function get_a(m::MuJoCo.Model, x::AbstractVector{T})::SubArray{T} where {T}
    return view(x, (m.nq + m.nv + 1):(m.nq + m.nv + m.na))
end

"""
Assume dx = [dq, dv, da]
"""
function get_dq(m::MuJoCo.Model, dx::AbstractVector{T})::SubArray{T} where {T}
    return view(dx, 1:m.nv)
end

"""
Assume dx = [dq, dv, da]
"""
function get_dv(m::MuJoCo.Model, dx::AbstractVector{T})::SubArray{T} where {T}
    return view(dx, (m.nv + 1):(m.nv + m.nv))
end

"""
Assume dx = [dq, dv, da]
"""
function get_da(m::MuJoCo.Model, dx::AbstractVector{T})::SubArray{T} where {T}
    start = 2 * m.nv
    return view(dx, (start + 1):(start + m.na))
end

"""
    get_state_diff!(m, Δx, x1, x2)

Get the state difference (x1 - x2) in tangent space coordinates.
"""
function get_state_diff!(
    m::MuJoCo.Model,
    Δx::AbstractVector{TΔ},
    x1::AbstractVector{T1},
    x2::AbstractVector{T2},
)::Nothing where {TΔ,T1,T2}
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
