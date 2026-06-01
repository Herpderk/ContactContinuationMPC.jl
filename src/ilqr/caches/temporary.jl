mutable struct TemporaryCache{T<:AbstractFloat}
    x::Vector{T}
    dx::Vector{T}
    u1::Vector{T}
    u2::Vector{T}
    dxdx1::Matrix{T}
    dxdx2::Matrix{T}
    uu1::Matrix{T}
    uu2::Matrix{T}
    uu3::Matrix{T}
    dxu::Matrix{T}
    udx1::Matrix{T}
    udx2::Matrix{T}

    function TemporaryCache{T}(nx::Integer, ndx::Integer, nu::Integer) where {T}
        x = zeros(T, nx)
        dx = zeros(T, ndx)
        u1 = zeros(T, nu)
        u2 = zeros(T, nu)
        dxdx1 = zeros(T, ndx, ndx)
        dxdx2 = zeros(T, ndx, ndx)
        uu1 = zeros(T, nu, nu)
        uu2 = zeros(T, nu, nu)
        uu3 = zeros(T, nu, nu)
        dxu = zeros(T, ndx, nu)
        udx1 = zeros(T, nu, ndx)
        udx2 = zeros(T, nu, ndx)
        return new{T}(
            x, dx, u1, u2, dxdx1, dxdx2, uu1, uu2, uu3, dxu, udx1, udx2
        )
    end
end
