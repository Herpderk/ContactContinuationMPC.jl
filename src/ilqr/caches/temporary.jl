mutable struct TemporaryCache{T<:AbstractFloat}
    singleton::Vector{T}
    x::Vector{T}
    dx::Vector{T}
    u::Vector{T}
    dxdx::Matrix{T}
    dxdx2::Matrix{T}
    uu::Matrix{T}
    dxu::Matrix{T}
    udx::Matrix{T}
    udx2::Matrix{T}

    function TemporaryCache{T}(
        nx::Integer, ndx::Integer, nu::Integer
    )::TemporaryCache{T} where {T}
        singleton = zeros(T, 1)
        x = zeros(T, nx)
        dx = zeros(T, ndx)
        u = zeros(T, nu)
        dxdx = zeros(T, ndx, ndx)
        dxdx2 = zeros(T, ndx, ndx)
        uu = zeros(T, nu, nu)
        dxu = zeros(T, ndx, nu)
        udx = zeros(T, nu, ndx)
        udx2 = zeros(T, nu, ndx)
        return new{T}(singleton, x, dx, u, dxdx, dxdx2, uu, dxu, udx, udx2)
    end
end
