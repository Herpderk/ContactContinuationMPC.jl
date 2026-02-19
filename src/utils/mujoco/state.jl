"""
    get_state_diff!(m, Δx, x1, x2)

Get the state difference (x1 - x2) in tangent space coordinates.
"""
function get_state_diff!(
    m::Union{Model,Data},
    Δx::AbstractVector{TΔ},
    x1::AbstractVector{T1},
    x2::AbstractVector{T2},
)::Nothing where {TΔ,T1,T2}
    if length(Δx) != get_ndx(m)
        throwdim("Δx does not have the correct number of dimensions")
    end
    if length(x1) != get_nx(m)
        throwdim("x1 does not have the correct number of dimensions")
    end
    if length(x2) != get_nx(m)
        throwdim("x2 does not have the correct number of dimensions")
    end

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

function add_diff_to_state!(
    m::Union{Model,Data}, x::AbstractVector{Tx}, Δx::AbstractVector{TΔ}
)::Nothing where {Tx,TΔ}
    if length(x) != get_nx(m)
        throwdim("x does not have the correct number of dimensions")
    end
    if length(Δx) != get_ndx(m)
        throwdim("Δx does not have the correct number of dimensions")
    end

    # Compute configuration sum
    q, Δq = get_q(m, x), get_dq(m, Δx)
    mj_integratePos(m, q, Δq, 1.0)

    # Compute velocity sum
    v, Δv = get_v(m, x), get_dv(m, Δx)
    v .+= Δv

    # Compute actuator sum
    a, Δa = get_a(m, x), get_da(m, Δx)
    a .+= Δa
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
