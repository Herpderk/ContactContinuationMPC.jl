function nx(m::Model)::Int
    return m.nq + m.nv + m.na
end

function ndx(m::Model)::Int
    return 2 * m.nv + m.na
end
