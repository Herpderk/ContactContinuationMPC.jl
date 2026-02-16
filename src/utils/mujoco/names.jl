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
