function get_nx(m::MuJoCo.Model)::Int
    return m.nq + m.nv + m.na
end

function get_nx(d::MuJoCo.Data)::Int
    return null_length(d.qpos) + null_length(d.qvel) + null_length(d.act)
end

function get_ndx(m::MuJoCo.Model)::Int
    return 2 * m.nv + m.na
end

function get_ndx(d::MuJoCo.Data)::Int
    return 2 * null_length(d.qvel) + null_length(d.act)
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
function get_q(d::MuJoCo.Data, x::AbstractVector{T})::SubArray{T} where {T}
    nq = null_length(d.qpos)
    return view(x, 1:nq)
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
function get_v(d::MuJoCo.Data, x::AbstractVector{T})::SubArray{T} where {T}
    nq = null_length(d.qpos)
    nv = null_length(d.qvel)
    return view(x, (nq + 1):(nq + nv))
end

"""
Assume x = [q, v, a]
"""
function get_a(m::MuJoCo.Model, x::AbstractVector{T})::SubArray{T} where {T}
    return view(x, (m.nq + m.nv + 1):(m.nq + m.nv + m.na))
end

"""
Assume x = [q, v, a]
"""
function get_a(d::MuJoCo.Data, x::AbstractVector{T})::SubArray{T} where {T}
    nq = null_length(d.qpos)
    nv = null_length(d.qvel)
    na = null_length(d.act)
    return view(x, (nq + nv + 1):(nq + nv + na))
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
function get_dq(d::MuJoCo.Data, dx::AbstractVector{T})::SubArray{T} where {T}
    nv = null_length(d.qvel)
    return view(dx, 1:nv)
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
function get_dv(d::MuJoCo.Data, dx::AbstractVector{T})::SubArray{T} where {T}
    nv = null_length(d.qvel)
    return view(dx, (nv + 1):(2 * nv))
end

"""
Assume dx = [dq, dv, da]
"""
function get_da(m::MuJoCo.Model, dx::AbstractVector{T})::SubArray{T} where {T}
    start = 2 * m.nv
    return view(dx, (start + 1):(start + m.na))
end

"""
Assume dx = [dq, dv, da]
"""
function get_da(d::MuJoCo.Data, dx::AbstractVector{T})::SubArray{T} where {T}
    nv = null_length(d.qvel)
    na = null_length(d.act)
    start = 2*nv
    return view(dx, (start + 1):(start + na))
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

function copy_data_to_state!(
    x::AbstractVector{T}, d::MuJoCo.Data
)::Nothing where {T}
    if !isnothing(d.qpos)
        copyto!(get_q(d, x), d.qpos)
    end
    if !isnothing(d.qvel)
        copyto!(get_v(d, x), d.qvel)
    end
    if !isnothing(d.act)
        copyto!(get_a(d, x), d.act)
    end
    return nothing
end

function copy_state_to_data!(
    d::MuJoCo.Data, x::AbstractVector{T}
)::Nothing where {T}
    if !isnothing(d.qpos)
        copyto!(d.qpos, get_q(d, x))
    end
    if !isnothing(d.qvel)
        copyto!(d.qvel, get_v(d, x))
    end
    if !isnothing(d.act)
        copyto!(d.act, get_a(d, x))
    end
    return nothing
end

function get_joint_names(m::MuJoCo.Model)::Vector{String}
    names = ["" for i in 1:m.nq]
    for i in 1:m.nq
        ptr = mj_id2name(m, MuJoCo.mjOBJ_JOINT, i)
        if ptr != C_NULL
            name = unsafe_string(ptr)
        else
            name = "unnamed"
        end
        names[i] = name
    end
    return names
end
