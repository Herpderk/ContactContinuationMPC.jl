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
