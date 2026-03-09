struct ProblemDimensions
    N::Int
    nx::Int
    ndx::Int
    nu::Int
    ny::Int
    ndy::Int
    nz::Int
    ndz::Int
    ng::Int

    function ProblemDimensions(
        N::Integer, nx::Integer, ndx::Integer, nu::Integer
    )
        ny = nx + nu
        ndy = ndx + nu
        nz = (N-1)*ny + nx
        ndz = (N-1)*ndy + ndx
        ng = N*ndx #+ ndz    # eq + tr constr
        return new(N, nx, ndx, nu, ny, ndy, nz, ndz, ng)
    end
end

struct PrimalVariableIndices
    x::Vector{UnitRange{Int}}
    u::Vector{UnitRange{Int}}

    function PrimalVariableIndices(N::Integer, nx::Integer, nu::Integer)
        # y: (x,u) pair; z: all decision variables
        ny = nx + nu
        nz = (N-1)*ny + nx
        xidx = [start:(start + nx - 1) for start in 1:ny:nz]
        uidx = [(start + nx):(start + ny - 1) for start in 1:ny:(nz - nx)]
        return new(xidx, uidx)
    end
end

struct ConstraintIndices
    ic::UnitRange{Int}
    dyn::Vector{UnitRange{Int}}
    xtr::Vector{UnitRange{Int}}
    utr::Vector{UnitRange{Int}}

    function ConstraintIndices(N::Integer, nx::Integer, nu::Integer)
        # ic and dynamics eq constrs
        nic = nx
        icidx = 1:nx
        ndyn = (N-1)*nx
        dynidx = [start:(start + nx - 1) for start in (nic + 1):nx:(nic + ndyn)]

        # trust region box constraints
        ny = nx + nu
        nz = (N-1)*ny + nx
        start_tr = nic + ndyn + 1
        end_tr = nic + ndyn + nz
        xidx = [start:(start + nx - 1) for start in start_tr:ny:end_tr]
        uidx = [
            (start + nx):(start + ny - 1) for start in start_tr:ny:(end_tr - nx)
        ]
        return new(icidx, dynidx, xidx, uidx)
    end
end

mutable struct IndexingParameters
    dims::ProblemDimensions
    z::PrimalVariableIndices
    dz::PrimalVariableIndices
    g::ConstraintIndices

    function IndexingParameters(
        N::Integer, nx::Integer, ndx::Integer, nu::Integer
    )
        dims = ProblemDimensions(N, nx, ndx, nu)
        zidx = PrimalVariableIndices(N, nx, nu)
        dzidx = PrimalVariableIndices(N, ndx, nu)
        gidx = ConstraintIndices(N, ndx, nu)
        return new(dims, zidx, dzidx, gidx)
    end
end
