function interpolate(
    a::T1, b::T2, N::Integer
)::Vector{T1} where {T1<:Number,T2<:Number}
    if N < 0
        throw(DomainError("Number of interpolations must be >= 0"))
    end

    # Generate N interpolations (N+2 including start and end)
    interps = [a + t*(b-a) for t in range(0, 1, N+2)]

    if T1 <: Integer
        return T1.(round.(interps))
    else
        return T1.(interps)
    end
end
