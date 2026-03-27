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
        ng = N*ndx + (N-1)*nu    # eq constrs + ctrl bds
        return new(N, nx, ndx, nu, ny, ndy, nz, ndz, ng)
    end
end

struct PrimalIndices
    x::Vector{UnitRange{Int}}
    u::Vector{UnitRange{Int}}

    function PrimalIndices(N::Integer, nx::Integer, nu::Integer)
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
    ub::Vector{UnitRange{Int}}
    xtr::Vector{UnitRange{Int}}
    utr::Vector{UnitRange{Int}}
    eq::UnitRange{Int}
    ineq::UnitRange{Int}

    function ConstraintIndices(N::Integer, nx::Integer, nu::Integer)
        # Initial conditions
        n_ic = nx
        end_ic = n_ic
        idx_ic = 1:end_ic

        # Dynamics
        n_dyn = (N-1)*nx
        step_dyn = nx
        start_dyn = end_ic + 1
        end_dyn = end_ic + n_dyn
        idx_dyn = [
            start:(start + step_dyn - 1) for start in start_dyn:step_dyn:end_dyn
        ]

        # Control input limits
        n_ub = (N-1)*nu
        step_ub = nu
        start_ub = end_dyn + 1
        end_ub = end_dyn + n_ub
        idx_ub = [
            start:(start + step_ub - 1) for start in start_ub:step_ub:end_ub
        ]

        # Trust region box constraints
        ny = nx + nu
        n_tr = (N-1)*ny + nx
        step_tr = ny
        start_tr = end_ub + 1
        end_tr = end_ub + n_tr
        idx_xtr = [
            start:(start + step_tr - 1) for start in start_tr:step_tr:end_tr
        ]
        idx_utr = [
            (start + nx):(start + step_tr - 1) for
            start in start_tr:step_tr:(end_tr - nx)
        ]

        # Total equality and inequality constraint indices
        idx_eq = 1:end_dyn
        idx_ineq = start_ub:end_ub
        return new(idx_ic, idx_dyn, idx_ub, idx_xtr, idx_utr, idx_eq, idx_ineq)
    end
end

mutable struct IndexingParameters
    dims::ProblemDimensions
    z::PrimalIndices
    dz::PrimalIndices
    g::ConstraintIndices

    function IndexingParameters(
        N::Integer, nx::Integer, ndx::Integer, nu::Integer
    )
        dims = ProblemDimensions(N, nx, ndx, nu)
        zidx = PrimalIndices(N, nx, nu)
        dzidx = PrimalIndices(N, ndx, nu)
        gidx = ConstraintIndices(N, ndx, nu)
        return new(dims, zidx, dzidx, gidx)
    end
end
