mutable struct TrajoptParameters{T<:AbstractFloat,Lk,Lf}
    #cinterps::Dict{String,ContactParameterInterpolations{T}}
    mfwd::Model
    mbwd::Model
    dfwd::Data
    dbwd::Data
    costfunc::TrajectoryCostFunction{T,Lk,Lf}
    Xref::Vector{Vector{T}}
    Uref::Vector{Vector{T}}
    xic::Vector{T}

    function TrajoptParameters{T}(
        mfwd::Model,
        mbwd::Model,
        costfunc_stage::Lk,
        costfunc_term::Lf,
        Xref::AbstractVector{<:AbstractVector{<:Real}},
        Uref::AbstractVector{<:AbstractVector{<:Real}},
        xic::AbstractVector{<:Real};
        #geomnames_interp::AbstractVector{<:AbstractString}=Vector{String}(),
        #num_interps::Integer=0,
    ) where {T,Lk,Lf}
        # Get problem dimensions
        nx = get_nx(mfwd)
        nu = mfwd.nu
        N = length(Xref)

        # Assert dimensions
        dims_same, mismatches = same_dims(mfwd, mbwd)
        if !dims_same
            throwdim("Models have mismatched dimensions:\n$mismatches")
        end
        if length(Uref) != N-1
            throwdim(
                "Number of reference inputs should be 1 less than number of reference states",
            )
        end
        if length(xic) != nx
            throwdim(
                "Initial conditions dimensions do not match those of reference states",
            )
        end
        for xref in Xref
            if length(xref) != nx
                throwdim("Reference state dimensions are not consistent")
            end
        end
        for uref in Uref
            if length(uref) != nu
                throwdim("Reference input dimensions are not consistent")
            end
        end

        #=
        # Check geometry names
        all_geomnames = get_geom_names(mfwd)
        if get_geom_names(mbwd) != all_geomnames
            throwarg("Geometry names between models are not consistent")
        end
        for geomname_interp in geomnames_interp
            if !(geomname_interp in all_geomnames)
                throwarg("Geometry $geomname_interp is not in the models")
            end
        end

        # Populate contact parameter interpolation dict
        cinterps = Dict{String,ContactParameterInterpolations{T}}()
        for geomname in all_geomnames
            cfwd = ContactParameters{T}(geomname, mfwd)
            cbwd = ContactParameters{T}(geomname, mbwd)
            if geomname in geomnames_interp # Interpolate for specified geoms
                cinterp = ContactParameterInterpolations(
                    cbwd, cfwd, num_interps
                )
            else                            # No interpolated values otherwise
                cinterp = ContactParameterInterpolations(cbwd, cfwd, 0)
            end
            cinterps[geomname] = cinterp
        end
        =#

        dfwd, dbwd = init_data(mfwd), init_data(mbwd)
        costfunc = TrajectoryCostFunction{T}(
            mfwd, costfunc_stage, costfunc_term
        )
        Xref_T = Vector{Vector{T}}(Xref)
        Uref_T = Vector{Vector{T}}(Uref)
        xic_T = Vector{T}(xic)
        return new{T,Lk,Lf}(
            mfwd, mbwd, dfwd, dbwd, costfunc, Xref_T, Uref_T, xic_T
        )
    end
end

function TrajoptParameters(
    mfwd::Model,
    mbwd::Model,
    costfunc_quad::QuadraticCostFunction{T},
    Xref::AbstractVector{<:AbstractVector{<:Real}},
    Uref::AbstractVector{<:AbstractVector{<:Real}},
    xic::AbstractVector{<:Real};
    #geomnames_interp::AbstractVector{<:AbstractString}=Vector{String}(),
    #num_interps::Integer=0,
) where {T}
    return TrajoptParameters{T}(
        mfwd,
        mbwd,
        costfunc_quad,
        costfunc_quad,
        Xref,
        Uref,
        xic;
        #geomnames_interp=geomnames_interp,
        #num_interps=num_interps,
    )
end

function TrajoptParameters(args...; kwargs...)
    TrajoptParameters{T_DEFAULT}(args...; kwargs...)
end

mutable struct TrajoptSolution{T<:AbstractFloat}
    X::Vector{Vector{T}}
    U::Vector{Vector{T}}
    J::T
    is_optimal::Bool

    function TrajoptSolution(params::TrajoptParameters{T,Lk,Lf}) where {T,Lk,Lf}
        # Get problem dims
        nx = get_nx(params.mfwd)
        nu = params.mfwd.nu
        N = length(params.Xref)

        # Initialize solution terms from dims
        X = [zeros(T, nx) for k in 1:N]
        U = [zeros(T, nu) for k in 1:(N - 1)]
        J = zero(T)
        is_optimal = false
        return new{T}(X, U, J, is_optimal)
    end
end
