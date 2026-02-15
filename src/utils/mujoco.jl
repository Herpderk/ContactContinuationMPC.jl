function get_nx(m::Model)::Int
    return m.nq + m.nv + m.na
end

function get_nx(d::Data)::Int
    return null_length(d.qpos) + null_length(d.qvel) + null_length(d.act)
end

function get_ndx(m::Model)::Int
    return 2 * m.nv + m.na
end

function get_ndx(d::Data)::Int
    return 2 * null_length(d.qvel) + null_length(d.act)
end

"""
Assume x = [q, v, a]
"""
function get_q(m::Model, x::AbstractVector{T})::SubArray{T} where {T}
    return view(x, 1:m.nq)
end

"""
Assume x = [q, v, a]
"""
function get_q(d::Data, x::AbstractVector{T})::SubArray{T} where {T}
    nq = null_length(d.qpos)
    return view(x, 1:nq)
end

"""
Assume x = [q, v, a]
"""
function get_v(m::Model, x::AbstractVector{T})::SubArray{T} where {T}
    return view(x, (m.nq + 1):(m.nq + m.nv))
end

"""
Assume x = [q, v, a]
"""
function get_v(d::Data, x::AbstractVector{T})::SubArray{T} where {T}
    nq = null_length(d.qpos)
    nv = null_length(d.qvel)
    return view(x, (nq + 1):(nq + nv))
end

"""
Assume x = [q, v, a]
"""
function get_a(m::Model, x::AbstractVector{T})::SubArray{T} where {T}
    return view(x, (m.nq + m.nv + 1):(m.nq + m.nv + m.na))
end

"""
Assume x = [q, v, a]
"""
function get_a(d::Data, x::AbstractVector{T})::SubArray{T} where {T}
    nq = null_length(d.qpos)
    nv = null_length(d.qvel)
    na = null_length(d.act)
    return view(x, (nq + nv + 1):(nq + nv + na))
end

"""
Assume dx = [dq, dv, da]
"""
function get_dq(m::Model, dx::AbstractVector{T})::SubArray{T} where {T}
    return view(dx, 1:m.nv)
end

"""
Assume dx = [dq, dv, da]
"""
function get_dq(d::Data, dx::AbstractVector{T})::SubArray{T} where {T}
    nv = null_length(d.qvel)
    return view(dx, 1:nv)
end

"""
Assume dx = [dq, dv, da]
"""
function get_dv(m::Model, dx::AbstractVector{T})::SubArray{T} where {T}
    return view(dx, (m.nv + 1):(m.nv + m.nv))
end

"""
Assume dx = [dq, dv, da]
"""
function get_dv(d::Data, dx::AbstractVector{T})::SubArray{T} where {T}
    nv = null_length(d.qvel)
    return view(dx, (nv + 1):(2 * nv))
end

"""
Assume dx = [dq, dv, da]
"""
function get_da(m::Model, dx::AbstractVector{T})::SubArray{T} where {T}
    start = 2 * m.nv
    return view(dx, (start + 1):(start + m.na))
end

"""
Assume dx = [dq, dv, da]
"""
function get_da(d::Data, dx::AbstractVector{T})::SubArray{T} where {T}
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
    m::Model,
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

function copy_data_to_state!(d::Data, x::AbstractVector{T})::Nothing where {T}
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

function copy_state_to_data!(d::Data, x::AbstractVector{T})::Nothing where {T}
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

function get_joint_names(m::Model)::Vector{String}
    names = ["" for i in 1:m.njnt]
    for i in 1:m.njnt
        ptr = mj_id2name(m, MuJoCo.mjOBJ_JOINT, i-1)
        names[i] = ptr == C_NULL ? "unnamed" : unsafe_string(ptr)
    end
    return names
end

function get_geom_names(m::Model)::Vector{String}
    names = ["" for i in 1:m.ngeom]
    for i in 1:m.ngeom
        ptr = mj_id2name(m, MuJoCo.mjOBJ_GEOM, i-1)
        names[i] = ptr == C_NULL ? "unnamed" : unsafe_string(ptr)
    end
    return names
end

"""
    same_dims(m1::Model, m2::Model)

Checks if two MuJoCo models have the same counts for all attributes starting with 'n'.
Returns (true, []) if they match, or (false, mismatched_fields) if they don't.
"""
function same_dims(m1::Model, m2::Model)::Tuple{Bool,Vector{String}}
    mismatches = Vector{String}()

    # Get all field names from the Model struct
    fields = String.(fieldnames(Model))

    for field in fields
        # Check if the field starts with 'n'
        if startswith(field, "n")
            val1 = m1.field
            val2 = m2.field

            # Only compare if they are integer dimensions
            if val1 isa Integer && val2 isa Integer
                if val1 != val2
                    push!(mismatches, field)
                end
            end
        end
    end

    flag = isempty(mismatches) ? true : false
    return flag, mismatches
end
