struct ContactParameters{T<:AbstractFloat}
    margin::T
    gap::T
    dmin::T
    dmax::T
    width::T
    midpoint::T
    power::T
    timeconst::T
    dampratio::T

    function ContactParameters{T}(;
        margin::Real,
        gap::Real,
        dmin::Real,
        dmax::Real,
        width::Real,
        midpoint::Real,
        power::Real,
        timeconst::Real,
        dampratio::Real,
    ) where {T}
        if !(0 <= dmin <= 1)
            throwdom(dmin, "dmin must be between 0 and 1")
        end
        if !(0 <= dmax <= 1)
            throwdom(dmax, "dmax must be between 0 and 1")
        end
        if !(0 < width)
            throwdom(width, "width must be greater than 0")
        end
        if !(0 < midpoint < 1)
            throwdom(midpoint, "midpoint must be between 0 and 1")
        end
        if !(1 <= power)
            throwdom(power, "power must be greater than or equal to 1")
        end
        if !(0 < timeconst)
            throwdom(timeconst, "timeconst must be greater than 0")
        end
        if !(0 < dampratio)
            throwdom(dampratio, "dampratio must be greater than 0")
        end
        return new{T}(
            margin,
            gap,
            dmin,
            dmax,
            width,
            midpoint,
            power,
            timeconst,
            dampratio,
        )
    end
end

function ContactParameters{T}(
    geomname::AbstractString, m::MuJoCo.Model
) where {T}
    id = 1 + mj_name2id(m, MuJoCo.mjOBJ_GEOM, geomname)
    if id == 0
        throwarg("Geometry name $geomname is invalid")
    end

    # Grab contact parameters from specified id
    margin = m.geom_margin[id]
    gap = m.geom_gap[id]

    # Solimp is stored as pointer array. Wrap in Julia array first
    solimp_T = unsafe_wrap(Array, m.geom_solimp, Int.((5, m.ngeom)))
    solimp = solimp_T'  # Need to transpose because the pointer is row-major
    dmin, dmax, width, midpoint, power = solimp[id, :]

    # Repeat for solref
    solref_T = unsafe_wrap(Array, m.geom_solref, Int.((2, m.ngeom)))
    solref = solref_T'  # Need to transpose because the pointer is row-major
    timeconst, dampratio = solref[id, :]
    return ContactParameters{T}(;
        margin=T(margin),
        gap=T(gap),
        dmin=T(dmin),
        dmax=T(dmax),
        width=T(width),
        midpoint=T(midpoint),
        power=T(power),
        timeconst=T(timeconst),
        dampratio=T(dampratio),
    )
end

ContactParameters(; args...) = ContactParameters{T_DEFAULT}(; args...)
ContactParameters(args...) = ContactParameters{T_DEFAULT}(args...)

struct ContactParameterInterpolations{T<:AbstractFloat}
    margin::Vector{T}
    gap::Vector{T}
    dmin::Vector{T}
    dmax::Vector{T}
    width::Vector{T}
    midpoint::Vector{T}
    power::Vector{T}
    timeconst::Vector{T}
    dampratio::Vector{T}

    function ContactParameterInterpolations{T}(
        c1::ContactParameters{T1},
        c2::ContactParameters{T2},
        num_interps::Integer,
    ) where {T,T1,T2}
        return new{T1}(
            interpolate(T(c1.margin), T(c2.margin), num_interps),
            interpolate(T(c1.gap), T(c2.gap), num_interps),
            interpolate(T(c1.dmin), T(c2.dmin), num_interps),
            interpolate(T(c1.dmax), T(c2.dmax), num_interps),
            interpolate(T(c1.width), T(c2.width), num_interps),
            interpolate(T(c1.midpoint), T(c2.midpoint), num_interps),
            interpolate(T(c1.power), T(c2.power), num_interps),
            interpolate(T(c1.timeconst), T(c2.timeconst), num_interps),
            interpolate(T(c1.dampratio), T(c2.dampratio), num_interps),
        )
    end
end

function ContactParameterInterpolations(
    c1::ContactParameters{T}, c2::ContactParameters{T}, num_interps::Integer
) where {T}
    return ContactParameterInterpolations{T}(c1, c2, num_interps)
end

function ContactParameterInterpolations(args...)
    ContactParameterInterpolations{T_DEFAULT}(args...)
end
