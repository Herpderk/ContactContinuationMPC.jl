
mutable struct TemporaryCache{T<:AbstractFloat}
    x::Vector{T}
    u::Vector{T}

    xx1::Matrix{T}
    xx2::Matrix{T}

    uu::Matrix{T}
    xu::Matrix{T}
    ux::Matrix{T}
end

function TemporaryCache{T}(nx::Int, nu::Int)::TemporaryCache{T} where {T}
    x = zeros(T, nx)
    u = zeros(T, nu)
    xx1 = zeros(T, nx, nx)
    xx2 = zeros(T, nx, nx)
    uu = zeros(T, nu, nu)
    xu = zeros(T, nx, nu)
    ux = zeros(T, nu, nx)
    return TemporaryCache(x, u, xx1, xx2, uu, xu, ux)
end
