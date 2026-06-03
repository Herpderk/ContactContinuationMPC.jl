function get_control_bound!(
    m::Model, b::AbstractVector{T}, direction::Symbol
)::Nothing where {T}
    for i in 1:m.nu
        if Bool(m.actuator_ctrllimited[i])
            b[i] = m.actuator_ctrlrange[i, 1]
        else
            if direction == :lower
                b[i] = -Inf
            elseif direction == :upper
                b[i] = Inf
            else
                throwarg(
                    "Control bounds can only be evaluated as lower or upper!"
                )
            end
        end
    end
    return nothing
end

function get_control_bound(m::Model, direction::Symbol)::Vector{T} where {T}
    b = zeros(T, m.nu)
    get_control_bound!(m, b, direction)
    return b
end
