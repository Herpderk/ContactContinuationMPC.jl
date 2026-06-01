mutable struct TemporaryCache{T<:AbstractFloat}
    singleton::Vector{T}
    x::Vector{T}
    dx::Vector{T}
    u::Vector{T}
    u2::Vector{T}
    dxdx::Matrix{T}
    dxdx2::Matrix{T}
    uu::Matrix{T}
    uu2::Matrix{T}
    uu3::Matrix{T}
    dxu::Matrix{T}
    udx::Matrix{T}
    udx2::Matrix{T}

    function TemporaryCache{T}(nx::Integer, ndx::Integer, nu::Integer) where {T}
        singleton = zeros(T, 1)
        x = zeros(T, nx)
        dx = zeros(T, ndx)
        u = zeros(T, nu)
        u2 = zeros(T, nu)
        dxdx = zeros(T, ndx, ndx)
        dxdx2 = zeros(T, ndx, ndx)
        uu = zeros(T, nu, nu)
        uu2 = zeros(T, nu, nu)
        uu3 = zeros(T, nu, nu)
        dxu = zeros(T, ndx, nu)
        udx = zeros(T, nu, ndx)
        udx2 = zeros(T, nu, ndx)
        return new{T}(
            singleton, x, dx, u, u2, dxdx, dxdx2, uu, uu2, uu3, dxu, udx, udx2
        )
    end
end
