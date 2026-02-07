function copy_nested_array!(
    dest::AbstractArray{Td}, src::AbstractArray{Ts}
)::Nothing where {Td,Ts}
    if size(dest) != size(src)
        throw(
            DimensionMismatch(
                "Size of destination array must equal that of source array."
            ),
        )
    end
    @inbounds @simd for i in eachindex(dest)
        copyto!(dest[i], src[i])
    end
    return nothing
end

function fill_nested_array!(A::AbstractArray{Ta}, x::Tx)::Nothing where {Ta,Tx}
    @inbounds @simd for i in eachindex(A)
        fill!(A[i], x)
    end
    return nothing
end

function null_length(arr::Union{AbstractArray{T},Nothing})::Int where {T}
    if isnothing(arr)
        return 0
    else
        return length(arr)
    end
end
