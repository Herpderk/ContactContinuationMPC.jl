mutable struct ContactParameters{T<:AbstractFloat}
    geom_name::String
    margin::T
    gap::T
    dmin::T
    dmax::T
    width::T
    midpoint::T
    power::Int
    timeconst::T
    dampratio::T

    function ContactParameters{T}(
        geom_name::String;
        margin::Real,
        gap::Real,
        dmin::Real,
        dmax::Real,
        width::Real,
        midpoint::Real,
        power::Integer,
        timeconst::Real,
        dampratio::Real,
    ) where {T}
        if !(0 <= dmin <= 1)
            throw(DomainError("dmin must be between 0 and 1"))
        end
        if !(0 <= dmax <= 1)
            throw(DomainError("dmax must be between 0 and 1"))
        end
        if !(0 < width)
            throw(DomainError("width must be greater than 0"))
        end
        if !(0 < midpoint < 1)
            throw(DomainError("midpoint must be between 0 and 1"))
        end
        if !(1 <= power)
            throw(DomainError("power must be greater than or equal to 1"))
        end
        if !(0 < timeconst)
            throw(DomainError("timeconst must be greater than 0"))
        end
        if !(0 < dampratio)
            throw(DomainError("dampration must be greater than 0"))
        end
        return new{T}(
            geom_name,
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
    geom_name::String, m::MuJoCo.Model
)::ContactParameters{T} where {T}
    geom_id = 1 + mj_name2id(m, mjtObj.mjOBJ_GEOM, geom_name)
    margin = m.geom_margin[geom_id]
    gap = m.geom_gap[geom_id]
    dmin, dmax, width, midpoint, power = m.geom_solimp[:, geom_id]
    timeconst, dampratio = m.geom_solref[:, geom_id]
    return ContactParameters{T}(
        geom_name;
        margin=margin,
        gap=gap,
        dmin=dmin,
        dmax=dmax,
        width=width,
        midpoint=midpoint,
        power=power,
        timeconst=timeconst,
        dampratio=dampratio,
    )
end

function ContactParameters(geom_name::String; args...)
    ContactParameters{T_DEFAULT}(geom_name::String; args...)
end
ContactParameters(args...) = ContactParameters{T_DEFAULT}(args...)

mutable struct ContactParameterInterpolations{T<:AbstractFloat}
    margin::Vector{T}
    gap::Vector{T}
    dmin::Vector{T}
    dmax::Vector{T}
    width::Vector{T}
    midpoint::Vector{T}
    power::Vector{Int}
    timeconst::Vector{T}
    dampratio::Vector{T}

    function ContactParameterInterpolations{T}(
        c1::ContactParameters{T1},
        c2::ContactParameters{T2},
        num_interps::Integer,
    ) where {T,T1,T2}
        return new{T}(
            interpolate(c1.margin, c2.margin, num_interps),
            interpolate(c1.gap, c2.gap, num_interps),
            interpolate(c1.dmin, c2.dmin, num_interps),
            interpolate(c1.dmax, c2.dmax, num_interps),
            interpolate(c1.width, c2.width, num_interps),
            interpolate(c1.midpoint, c2.midpoint, num_interps),
            interpolate(c1.power, c2.power, num_interps),
            interpolate(c1.timeconst, c2.timeconst, num_interps),
            interpolate(c1.dampratio, c2.dampratio, num_interps),
        )
    end
end

function ContactParameterInterpolations(
    c1::ContactParameters{T}, c2::ContactParameters{T}, num_interps::Integer
) where {T}
    return ContactParameterInterpolations{T}(c1, c2, num_interps)
end
