function copy_nested_array!(
    dest::AbstractArray{Td}, src::AbstractArray{Ts}
)::Nothing where {Td,Ts}
    if size(dest) != size(src)
        throwdim("Size of destination array must equal that of source array")
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

function vector(a::T, N::Integer)::Vector{T}
    return [a for _ in 1:N]
end
