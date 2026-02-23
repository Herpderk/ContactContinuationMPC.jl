struct ProblemDimensions
    N::Int
    nx::Int
    ndx::Int
    nu::Int
    ny::Int
    ndy::Int
    nz::Int
    ndz::Int
    nh::Int

    function ProblemDimensions(
        N::Integer, nx::Integer, ndx::Integer, nu::Integer
    )
        ny = nx + nu
        ndy = ndx + nu
        nz = (N-1)*ny + nx
        ndz = (N-1)*ndy + ndx
        nh = N*ndx
        return new(N, nx, ndx, nu, ny, ndy, nz, ndz, nh)
    end
end

struct PrimalVariableIndices
    x::Vector{UnitRange}
    u::Vector{UnitRange}

    function PrimalVariableIndices(N::Integer, nx::Integer, nu::Integer)
        # y: (x,u) pair; z: all decision variables
        ny = nx + nu
        nz = (N-1)*ny + nx
        xidx = [start:(start + nx - 1) for start in 1:ny:nz]
        uidx = [(start + nx):(start + nx + nu - 1) for start in 1:ny:(nz - nx)]
        return new(xidx, uidx)
    end
end

struct EqualityConstraintIndices
    ic::UnitRange
    dyn::Vector{UnitRange}

    function EqualityConstraintIndices(N::Integer, nx::Integer)
        icidx = 1:nx
        ndyn = N*nx
        dynidx = [start:(start + nx - 1) for start in (1 + nx):nx:ndyn]
        return new(icidx, dynidx)
    end
end

mutable struct IndexingParameters
    dims::ProblemDimensions
    z::PrimalVariableIndices
    dz::PrimalVariableIndices
    h::EqualityConstraintIndices

    function IndexingParameters(
        N::Integer, nx::Integer, ndx::Integer, nu::Integer
    )
        dims = ProblemDimensions(N, nx, ndx, nu)
        zidx = PrimalVariableIndices(N, nx, nu)
        dzidx = PrimalVariableIndices(N, ndx, nu)
        hidx = EqualityConstraintIndices(N, ndx)
        return new(dims, zidx, dzidx, hidx)
    end
end
