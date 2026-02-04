function copy_nested_array!(
    dst::AbstractArray{T_dst}, src::AbstractArray{T_src}
)::Nothing where {T_dst,T_src}
    if size(dst) != size(src)
        throw(
            DimensionMismatch(
                "Size of destination array must equal that of source array."
            ),
        )
    end
    @inbounds @simd for i in eachindex(dst)
        copy!(dst[i], src[i])
    end
    return nothing
end

function fill_nested_array!(
    A::AbstractArray{T_A}, x::T_x
)::Nothing where {T_A,T_x}
    @inbounds @simd for i in eachindex(A)
        fill!(A[i], x)
    end
    return nothing
end
